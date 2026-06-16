"""
Stage 3 — Compute BERT embeddings for every reasoning step.

For each transformed graph this embeds the per-step Q&A text, every Node/Edge
label, the final response and the raw question with ``bert-base-uncased``. The
result feeds the NIBS / GIBS / white-box methods.

Dataset/model are parameters; paths follow ``dataset_config.py``.

Example:
    python bert_graph.py --dataset morehopqa --model llama          # GPU
    python bert_graph.py --dataset morehopqa --model llama --device cpu
"""

import os
import sys
import re
import pickle
import argparse

import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_config import transformed_path, bert_path


def build_embedder(device):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    return get_bert_embedding


def build_embeddings(questions, get_bert_embedding):
    embeddings_dict = {}
    for question_category, answers in tqdm(questions.items(), desc="Processing categories"):
        embeddings_dict.setdefault(question_category, {})
        for sample_id, answer_item in answers.items():
            string_q_a = answer_item.get("string_q_a", "")
            structure_list = answer_item.get("structure_retrieve", [])
            response_str = answer_item.get("response_str", "")
            logprobs = answer_item.get("logprobs", [])
            probs = answer_item.get("probs", [])
            structure_str = answer_item.get("structure_retrieve")

            entry = {
                "embeddings": {},
                "structure_str": structure_str,
                "sub_structure_str": structure_str,
                "logprobs": logprobs,
                "probs": probs,
                "reasoning_string": {},
            }
            embeddings_dict[question_category][sample_id] = entry

            # Per-step Q&A text
            for segment in re.split(r'[;,]', string_q_a):
                if ':' in segment:
                    key, value = segment.split(':', 1)
                    key, value = key.strip(), value.strip()
                    if value:
                        entry["embeddings"][key] = get_bert_embedding(value)
                        entry["reasoning_string"][key] = value

            # Node / Edge labels
            for structure_item in structure_list:
                structure_item_str = str(structure_item).strip('[]')
                for element in (e.strip() for e in structure_item_str.split(',')):
                    if element.startswith("Node") or element.startswith("Edge"):
                        if element not in entry["embeddings"]:
                            entry["embeddings"][element] = get_bert_embedding(element)

            # Final response + raw question
            if response_str:
                entry["embeddings"]["ResultEdge"] = get_bert_embedding(response_str)
            if 'question' in answer_item:
                entry["embeddings"]["NodeRaw"] = get_bert_embedding(answer_item['question'])

    return embeddings_dict


def main():
    parser = argparse.ArgumentParser(description='Compute BERT embeddings for reasoning graphs.')
    parser.add_argument('--dataset', required=True, choices=['morehopqa', 'gsm8k', 'math'])
    parser.add_argument('--model', required=True, choices=['llama', 'deepseek', 'phi4'])
    parser.add_argument('--output-root', default='output',
                        help='Root directory for artifacts (default: output).')
    parser.add_argument('--input', default=None,
                        help='Override input transformed .pkl (default: derived from dataset/model).')
    parser.add_argument('--output', default=None,
                        help='Override output embeddings .pkl (default: derived from dataset/model).')
    parser.add_argument('--device', default='cuda', help="Device: 'cuda' or 'cpu' (default: cuda).")
    args = parser.parse_args()

    device = args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    if device != args.device:
        print(f"CUDA not available; falling back to {device}.")

    in_path = args.input or transformed_path(args.output_root, args.dataset, args.model)
    out_path = args.output or bert_path(args.output_root, args.dataset, args.model)

    with open(in_path, 'rb') as f:
        questions = pickle.load(f)
    print("Total question categories:", len(questions))

    get_bert_embedding = build_embedder(device)
    embeddings_dict = build_embeddings(questions, get_bert_embedding)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(embeddings_dict, f)
    print(f"\nBERT embeddings with probability data saved to {out_path}")


if __name__ == "__main__":
    main()
