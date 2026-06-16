import os
import re
import math
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse

def load_model_and_tokenizer(model_name, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer


def logprob_to_prob(logprob):
    if logprob is None:
        return None
    return math.exp(logprob)


def parse_steps(string_q_a):
    steps = []
    for part in string_q_a.split(";"):
        part = part.strip()
        if not part:
            continue
 
        m = re.match(r"((?:Edge\d+|ResultEdge)):", part)
        if m:
            edge_label = m.group(1)  
            steps.append({"label": edge_label, "text": part})
    return steps



def build_prompts(question, steps, dataset, context=None):
    prompts = []
    for step in steps:
        if dataset == "morehopqa" and context is not None:
            prefix = f"Context: {context}\nQuestion: {question}\n"
        else:
            prefix = f"Question: {question}\n"
        step_text = f"{step['text']}\nIs this step:\n(A) True\n(B) False\nThe answer is: ("
        prompts.append(prefix + step_text)
    return prompts


def get_ptrue_scores(model, tokenizer, prompts, device="cuda:0"):
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # [batch, seq_len, vocab_size]
    scores = []

    for i, prompt in enumerate(prompts):
        length = inputs['attention_mask'][i].sum().item()
        last_logits = logits[i, length - 1]

        A_tok = tokenizer.encode("(A")[-1]
        B_tok = tokenizer.encode("(B")[-1]

        logprobs = torch.nn.functional.log_softmax(last_logits, dim=-1)
        score_A = logprob_to_prob(logprobs[A_tok].item())
        score_B = logprob_to_prob(logprobs[B_tok].item())

        scores.append({"True": score_A, "False": score_B})
    return scores


def process_dataset(data_path, model_name, dataset, device="cuda:0"):
    with open(data_path, "rb") as f:
        bert_data = pickle.load(f)

    all_problem_ids = list(bert_data.keys())
    last_500_ids = all_problem_ids[-500:]

    if model_name == "llama3":
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    elif model_name == "phi4":
        model_name = "microsoft/Phi-4-reasoning"
    elif model_name == "deepseek":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model, tokenizer = load_model_and_tokenizer(model_name, device)

    results = []

    for problem_id in tqdm(last_500_ids, desc="Problems"):
        for reference_id, content in bert_data[problem_id].items():
            question = content["question"]
            string_q_a = content["string_q_a"]

            #step
            steps = parse_steps(string_q_a)
            if not steps:
                continue

            #context
            context = content.get("context", None) if dataset == "morehopqa" else None

            #prompt
            prompts = build_prompts(question, steps, dataset, context=context)

            #P(true) results
            scores = get_ptrue_scores(model, tokenizer, prompts, device)

            # save results
            edge_results = []
            for step, score in zip(steps, scores):
                edge_results.append({
                    "edge_label": step["label"],
                    "ptrue": score
                })

            results.append({
                "problem_id": problem_id,
                "graph_id": reference_id,
                "edge_details": edge_results
            })

    return results


# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="llama3 / phi4 / deepseek")
    parser.add_argument("--dataset", type=str, required=True,
                        help="gsm8k / math / morehopqa")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_path", type=str, default="results.pkl")
    args = parser.parse_args()

    results = process_dataset(args.data_path, args.model, args.dataset, args.device)

    with open(args.save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved results to {args.save_path}")
