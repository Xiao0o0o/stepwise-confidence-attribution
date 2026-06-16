import argparse
import torch
import pickle
import json
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd

from GIB_MCS_wo_mcs_feature import GIBErrorDetectionModel, build_graph, load_evaluation_results, is_correct_graph

def load_trained_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint['model_config']
    model = GIBErrorDetectionModel(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    return model, device


def inference_one_graph(model, graph_info, problem_id, graph_id, is_correct, threshold=0.3):
    with torch.no_grad():
        outputs = model(graph_info, [])
        edge_masks = outputs['edge_masks']
        edge_ids = outputs['edge_ids']
        G = build_graph(graph_info['structure_str'])

        edge_details = []
        for i, (u, v) in enumerate(edge_ids):
            edge_label = G.get_edge_data(u, v).get("label", f"{u}->{v}")
            edge_text = graph_info.get("embeddings", {}).get(edge_label, None)
            edge_text = graph_info.get("reasoning_string", "") if edge_text is None else str(edge_label)

            score = float(edge_masks[i].item())
            edge_details.append({
                "edge_index": i,
                "edge_nodes": [u, v],
                "edge_label": edge_label,
                "edge_text": edge_text,
                "mask_score": score,
                "prediction": 1 if score >= threshold else 0
            })

        return {
            "problem_id": problem_id,
            "graph_id": graph_id,
            "is_correct": int(is_correct),
            "edge_details": edge_details,
            "num_edges": len(edge_ids),
            "success": True
        }


def batch_inference(model_path, pkl_path, eval_results_path,
                    output_file, threshold=0.3, device=None):
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1. Load model
    model, device = load_trained_model(model_path, device)

    # 2. Load graph data
    with open(pkl_path, "rb") as f:
        graph_data = pickle.load(f)

    # 3. Load evaluation results
    eval_results = load_evaluation_results(eval_results_path)

    all_problem_ids = list(graph_data.keys())
    print(f"Total problems: {len(all_problem_ids)}")

    all_results = []
    for pid in tqdm(all_problem_ids, desc="Inference"):
        problem_data = graph_data[pid]
        for gid, ginfo in problem_data.items():
            is_corr = is_correct_graph(ginfo, pid, gid, eval_results)
            result = inference_one_graph(model, ginfo, pid, gid, is_corr, threshold)
            all_results.append(result)


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_file}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="GIBS inference/evaluation: run a trained GIB checkpoint over graph "
                    "data and write per-edge mask scores/predictions to a JSON file."
    )
    parser.add_argument("--model-path", required=True,
                        help="Trained GIB checkpoint (.pth) produced by the training script.")
    parser.add_argument("--embedding-file", required=True,
                        help="Path to bert_embeddings_*.pkl (graph data).")
    parser.add_argument("--label-file", required=True,
                        help="Path to evaluation.json (evaluation results).")
    parser.add_argument("--output-file", required=True,
                        help="Full path to write the GIBS results JSON (e.g. .../GIBS_results.json).")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Mask->prediction threshold used in inference_one_graph (default: 0.3).")
    parser.add_argument("--device", default="cuda",
                        help="Device to run inference on (default: cuda).")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    batch_inference(
        model_path=args.model_path,
        pkl_path=args.embedding_file,
        eval_results_path=args.label_file,
        output_file=args.output_file,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
