"""
Stage 4 — Label step-wise correctness with an LLM judge (GPT-4o-mini).

Reads the transformed graphs, asks the judge to mark every edge-node pair of
every sampled reasoning graph as correct (1) or incorrect (0), and writes the
labels to ``evaluation.json``. These labels are the ground truth used by
``evaluation_metrics.py`` to score every confidence-estimation method.

The OpenAI API key is read from the OPENAI_API_KEY environment variable (or the
``--api-key`` flag). Leave it unset to do a dry run that produces the same
output schema with placeholder labels.

Examples
--------
    export OPENAI_API_KEY=sk-...
    python gpt_eval.py --dataset morehopqa --model llama
    python gpt_eval.py --dataset morehopqa --model llama --source data/source/morehopqa_sample.json
"""

import os
import re
import sys
import json
import time
import argparse
import pickle
from typing import Dict, List, Tuple

# Reuse the canonical path layout
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
from dataset_config import transformed_path, evaluation_path  # noqa: E402


def load_pickle(file_path: str) -> Dict:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def build_context_index(source_path: str) -> Dict[str, str]:
    """Map each problem _id -> stringified context (for datasets that have one)."""
    if not source_path or not os.path.exists(source_path):
        return {}
    with open(source_path, 'r') as f:
        source_data = json.load(f)
    index = {}
    for item in source_data:
        if isinstance(item, dict) and '_id' in item:
            index[item['_id']] = str(item.get('context', ''))
    return index


def parse_structure_retrieve(structure_str: str) -> List[Tuple[str, str, str, str]]:
    """Extract (edge_name, edge_desc, node_name, node_value) tuples from string_q_a."""
    pattern = r'(Edge\d+|ResultEdge): ([^?]+)\?, (Node\d+|NodeResult): ([^;]+);'
    return [(e, d.strip(), n, v.strip()) for e, d, n, v in re.findall(pattern, structure_str)]


def create_batch_prompt(question, answer, context, edge_node_pairs) -> str:
    pairs_text = ""
    for i, (edge_name, edge_desc, node_name, node_value) in enumerate(edge_node_pairs):
        pairs_text += f"{i+1}. {edge_name}: {edge_desc}? -> {node_name}: {node_value}\n"

    context_block = f"Context Passages: {context}\n" if context else ""
    return f"""You are evaluating a multi-step reasoning graph for correctness. Given a question that requires multiple reasoning steps, {('the provided context passages, ' if context else '')}the correct answer, and a series of reasoning steps (edge-node pairs), determine if each step is logically and factually correct.

Question: {question}
{context_block}Correct Answer: {answer}

Reasoning Graph (Edge-Node Pairs):
{pairs_text}
Instructions:
1. Evaluate each edge-node pair in the context of solving this question.
2. Verify the value is correct given the available information, the edge description, and any previous reasoning steps.
3. For steps involving entity extraction, fact retrieval, bridge reasoning, arithmetic/counting, or commonsense reasoning, check the corresponding kind of correctness.

For each pair, respond with:
- 1 if the edge-node pair is correct
- 0 if the edge-node pair is incorrect

Format your response as a comma-separated list of digits (no spaces), one digit per pair, without any explanation.
Example for 5 pairs: 1,0,1,1,0

Your evaluation:"""


def call_judge(client, prompt, model_name, n_pairs, max_retries=2):
    """Return a list of per-pair labels (1/0/-1) and the raw judge output string."""
    if client is None:
        # Dry run: no API key supplied. Emit placeholders so the schema is valid.
        return [-1] * n_pairs, ""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You evaluate the correctness of reasoning graphs with precision."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=100,
            )
            result_str = response.choices[0].message.content.strip()
            results = [int(x) for x in result_str.split(',') if x.strip() in ['0', '1']]
            if len(results) != n_pairs:
                results.extend([-1] * (n_pairs - len(results)))
                results = results[:n_pairs]
            return results, result_str
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Judge call failed after {max_retries} attempts: {e}")
                return [-1] * n_pairs, ""


def evaluate_single_generation(client, model_name, generation_data, problem_id,
                               generation_id, context_index):
    question = generation_data['question']
    answer = generation_data['answer']
    structure_retrieve = generation_data.get('string_q_a', '')
    context = context_index.get(problem_id, '')

    edge_node_pairs = parse_structure_retrieve(structure_retrieve)
    prompt = create_batch_prompt(question, answer, context, edge_node_pairs)
    results, result_str = call_judge(client, prompt, model_name, len(edge_node_pairs))

    evaluations = []
    correct = incorrect = error = 0
    for i, (edge_name, edge_desc, node_name, node_value) in enumerate(edge_node_pairs):
        is_correct = results[i] if i < len(results) else -1
        evaluations.append({
            'index': i + 1,
            'edge_name': edge_name,
            'edge_desc': edge_desc,
            'node_name': node_name,
            'node_value': node_value,
            'is_correct': is_correct,
        })
        correct += is_correct == 1
        incorrect += is_correct == 0
        error += is_correct not in (0, 1)

    return {
        'problem_id': problem_id,
        'generation_id': generation_id,
        'question': question,
        'answer': answer,
        'total_pairs': len(edge_node_pairs),
        'evaluations': evaluations,
        'gpt_output': result_str,
        'summary': {
            'correct': correct, 'incorrect': incorrect, 'error': error,
            'accuracy': correct / len(edge_node_pairs) if edge_node_pairs else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='LLM-judge step-wise correctness labeling.')
    parser.add_argument('--dataset', required=True, choices=['morehopqa', 'gsm8k', 'math'])
    parser.add_argument('--model', required=True, choices=['llama', 'deepseek', 'phi4'])
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--input', default=None,
                        help='Override transformed .pkl (default: derived from dataset/model).')
    parser.add_argument('--output', default=None,
                        help='Override evaluation.json (default: derived from dataset/model).')
    parser.add_argument('--source', default=None,
                        help='Original source JSON, used for context (morehopqa). Optional.')
    parser.add_argument('--judge-model', default='gpt-4o-mini', help='OpenAI judge model.')
    parser.add_argument('--api-key', default=None,
                        help='OpenAI API key. Defaults to the OPENAI_API_KEY env var. '
                             'If absent, runs a dry run with placeholder labels.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only label the first N problems (handy for demos).')
    args = parser.parse_args()

    in_path = args.input or transformed_path(args.output_root, args.dataset, args.model)
    out_path = args.output or evaluation_path(args.output_root, args.dataset, args.model)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    client = None
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    else:
        print("WARNING: no OpenAI API key found. Running a DRY RUN: evaluation.json "
              "will have the right schema but every label is -1 (unknown).")

    data = load_pickle(in_path)
    context_index = build_context_index(args.source)

    problem_ids = list(data.keys())
    if args.limit is not None:
        problem_ids = problem_ids[:args.limit]
    print(f"Labeling {len(problem_ids)} problems from {in_path}")

    all_results = {}
    for idx, problem_id in enumerate(problem_ids, 1):
        print(f"[{idx}/{len(problem_ids)}] {problem_id}")
        problem_results = {}
        for generation_id, generation_data in data[problem_id].items():
            try:
                problem_results[generation_id] = evaluate_single_generation(
                    client, args.judge_model, generation_data, problem_id,
                    generation_id, context_index)
            except Exception as e:
                print(f"  Error on {generation_id}: {e}")
        all_results[problem_id] = problem_results
        if client is not None:
            time.sleep(0.3)  # be gentle with rate limits

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved correctness labels to {out_path}")


if __name__ == "__main__":
    main()
