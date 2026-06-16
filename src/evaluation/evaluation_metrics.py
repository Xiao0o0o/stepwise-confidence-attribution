import os
import sys
import argparse
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ====================================================
# Enum for method names
# ====================================================



class Method(Enum):
    GIB = "GIB-based"
    COS_MAX = "Cos-Max"
    COS_MEAN = "Cos-Mean"
    COS_RAND = "Cos-Rand"
    NLI_MAX = "NLI-Max"
    NLI_MEAN = "NLI-Mean"
    NLI_RAND = "NLI-Rand"
    PTRUE = "Ptrue"  
    SEQ = "sl_norm"
    ENTROPY = "entropy"
    LECO = "leco"

    

# ====================================================
# Extra metrics: Risk–Coverage, Accuracy@Coverage
# ====================================================


def risk_coverage_curve(y_true, y_scores, n_points=20):
    """
    Compute risk–coverage curve
    Risk = 1 - accuracy
    Coverage = proportion of samples retained (sorted by confidence descending)
    """
    order = np.argsort(-y_scores)
    y_true_sorted = y_true[order]
    y_scores_sorted = y_scores[order]

    risks, coverages = [], []
    n = len(y_true)
    for k in range(1, n_points + 1):
        cutoff = int(n * (k / n_points))
        if cutoff == 0:
            continue
        selected = y_true_sorted[:cutoff]
        acc = selected.mean()
        risk = 1 - acc
        coverage = cutoff / n
        risks.append(risk)
        coverages.append(coverage)
    return np.array(coverages), np.array(risks)


def accuracy_at_coverage(y_true, y_scores, coverage=0.8):
    """
    Compute accuracy at given coverage.
    Keep top (coverage * N) samples by confidence.
    """
    n = len(y_true)
    k = int(n * coverage)
    order = np.argsort(-y_scores)
    selected = y_true[order[:k]]
    return selected.mean()



class GraphFilter(Enum):
    WRONG_ONLY = "Wrong Graphs Only"  # Includes both 0 and -1


# ====================================================
# Data loader helpers
# ====================================================
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_graph_correctness(generation_data):
    """
    Determine if a graph is correct based on ResultEdge
    Returns: 1 for correct, 0 or -1 for wrong
    """
    for eval_item in generation_data['evaluations']:
        if eval_item['edge_name'] == 'ResultEdge':
            return eval_item['is_correct']
    return -1  # No ResultEdge found → wrong


# ====================================================
# Metrics
# ====================================================
def calculate_auroc(y_true, y_scores):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_scores)


def calculate_aucpr(y_true, y_scores):
    if len(np.unique(y_true)) < 2:
        return None
    return average_precision_score(y_true, y_scores)



def extract_labels_and_scores(label_data, score_data, method: Method, last_n=500):
    y_true, y_scores = [], []
    
    all_problem_ids = list(label_data.keys())
    problem_ids_to_process = all_problem_ids
    
    if method == Method.PTRUE:
        score_index = {}
        for item in score_data:
            score_index.setdefault(item["problem_id"], {})[item["graph_id"]] = item
    else:
        score_index = {}
        for item in score_data:
            score_index.setdefault(item["problem_id"], {})[item["graph_id"]] = item
    
    for problem_id in problem_ids_to_process:
        if problem_id not in label_data:
            continue
        generations = label_data[problem_id]
        for generation_id, generation_data in generations.items():
            graph_correctness = get_graph_correctness(generation_data)
            
            if graph_correctness == 1:
                continue
            
            if problem_id not in score_index or generation_id not in score_index[problem_id]:
                continue
            score_item = score_index[problem_id][generation_id]
            
            if method == Method.GIB:
                edge_score_map = {
                    edge["edge_label"]: edge["mask_score"]
                    for edge in score_item["edge_details"]
                    if "mask_score" in edge
                }
            elif method == Method.PTRUE:
                edge_score_map = {
                    edge["edge_label"]: edge["ptrue"]["True"]
                    for edge in score_item["edge_details"]
                    if "ptrue" in edge and "True" in edge["ptrue"]
                }
            else:
                sim_key = method.value
                edge_score_map = {}
                for edge in score_item["edge_details"]:
                    if "scores" in edge and sim_key in edge["scores"]:
                        val = edge["scores"][sim_key]
                        if val != -1.0:  # skip invalid
                            edge_score_map[edge["edge_label"]] = val
            
            # match all edges
            for eval_item in generation_data["evaluations"]:
                edge_name = eval_item["edge_name"]
                is_correct = eval_item["is_correct"]
                if edge_name in edge_score_map and is_correct != -1:
                    score_value = float(edge_score_map[edge_name])
                    
                    # Skip NaN or infinite values
                    if not np.isnan(score_value) and not np.isinf(score_value):
                        y_true.append(int(is_correct))
                        y_scores.append(score_value)
    
    return np.array(y_true), np.array(y_scores)
# ====================================================
# Evaluation pipeline
# ====================================================
def evaluate_method(label_data, score_data, method: Method):
    y_true, y_scores = extract_labels_and_scores(label_data, score_data, method)

    if len(y_true) == 0:
        print(f"No samples for {method.value}")
        return None
    if method == Method.ENTROPY:
        # For entropy, lower is more confident, so invert scores
        y_scores = -y_scores
    auroc = calculate_auroc(y_true, y_scores)
    aucpr = calculate_aucpr(y_true, y_scores)
    print(f"\n{method.value} Results:")
    print(f"  Samples: {len(y_true)}, Positives: {y_true.sum()}, Negatives: {len(y_true) - y_true.sum()}")
    print(f"  AUROC = {auroc:.4f}" if auroc is not None else "  AUROC = N/A")
    print(f"  AUCPR = {aucpr:.4f}" if aucpr is not None else "  AUCPR = N/A")


    # Risk–Coverage Curve (just return arrays)
    coverages, risks = risk_coverage_curve(y_true, y_scores, n_points=20)

    # Accuracy@Coverage (比如 80%)
    acc_cov80 = accuracy_at_coverage(y_true, y_scores, coverage=0.8)

    # print(f"  Brier Score = {brier:.4f}")
    print(f"  Accuracy@80% coverage = {acc_cov80:.4f}")

    return {
        "method": method.value,
        "auroc": auroc,
        "aucpr": aucpr,
        "acc_cov80": acc_cov80,
        "coverage_curve": (coverages, risks),
        "n_samples": len(y_true)
    }

# ====================================================
# Main
# ====================================================
def _default(args, derive):
    """Pick the explicit flag if given, else derive from --output-root/dataset/model."""
    return derive


def main():
    parser = argparse.ArgumentParser(
        description="Score every confidence-estimation method against the GPT correctness labels.")
    parser.add_argument('--dataset', choices=['morehopqa', 'gsm8k', 'math'],
                        help='Used (with --model/--output-root) to derive default file paths.')
    parser.add_argument('--model', choices=['llama', 'deepseek', 'phi4'])
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--label-file', default=None, help='evaluation.json (GPT labels).')
    parser.add_argument('--gib-file', default=None, help='GIBS_results.json.')
    parser.add_argument('--nibs-file', default=None, help='NIBS_results.json (similarity).')
    parser.add_argument('--ptrue-file', default=None, help='ptrue_*.pkl.')
    parser.add_argument('--white-box-file', default=None, help='white_box_baseline_*.json.')
    args = parser.parse_args()

    # Derive any path not explicitly provided from the canonical layout.
    if args.dataset and args.model:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
        import dataset_config as dc
        r, d, m = args.output_root, args.dataset, args.model
        label_file = args.label_file or dc.evaluation_path(r, d, m)
        wo_mcs_gib_score_file = args.gib_file or dc.gibs_path(r, d, m)
        similarity_score_file = args.nibs_file or dc.nibs_path(r, d, m)
        ptrue_file = args.ptrue_file or dc.ptrue_path(r, d, m)
        white_box_baseline = args.white_box_file or dc.white_box_path(r, d, m)
    else:
        label_file = args.label_file
        wo_mcs_gib_score_file = args.gib_file
        similarity_score_file = args.nibs_file
        ptrue_file = args.ptrue_file
        white_box_baseline = args.white_box_file
        missing = [n for n, v in [('--label-file', label_file), ('--gib-file', wo_mcs_gib_score_file),
                                  ('--nibs-file', similarity_score_file), ('--ptrue-file', ptrue_file),
                                  ('--white-box-file', white_box_baseline)] if v is None]
        if missing:
            parser.error("Provide --dataset and --model to auto-derive paths, or pass all of: "
                         + ", ".join(missing))

    # Legacy hardcoded examples (kept for reference):
    #morehopqa_72B
    # label_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/qwen2.5-72b/evaluation_results_last500/detailed_results_20260326_171200.json"
    # wo_mcs_gib_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/qwen2.5-72b/ablation_study/inference_results_ablation/results_20260327_180035.json"
    # similarity_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/qwen2.5-72b/similarity_results.json"
    # #ptrue_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/qwen2.5-72b/ptrue_qwen2.5-72b_morehopqa.pkl"
    # white_box_baseline = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/qwen2.5-72b/white_box_baseline.json"
    

    #morehopqa_phi4 (example absolute paths — now supplied via CLI args above)
    # label_file = ".../phi4/evaluation_results_last500/detailed_results.json"
    # wo_mcs_gib_score_file = ".../phi4/ablation_study/inference_results_ablation/results_*.json"
    # similarity_score_file = ".../phi4/similarity_results.json"
    # ptrue_file = ".../phi4/ptrue_phi4_morehopqa_.pkl"
    # white_box_baseline = ".../phi4/white_box_baseline_phi4.json"

    # morehopqa_deepseek done
    # label_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/deepseek/evaluation_results_last500/detailed_results_20250819_234706.json"
    # wo_mcs_gib_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/deepseek/ablation_study/inference_results_ablation/results_20250923_164205.json"
    # similarity_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/deepseek/similarity_results.json"
    # ptrue_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/deepseek/ptrue_deepseek_morehopqa.pkl"
    # white_box_baseline = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/deepseek/white_box_baseline_deepseek_r1.json"

    #morehopqa_llama3 done
    # label_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/llama31_8b/detailed_results_20250820_132339.json"
    # wo_mcs_gib_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/llama31_8b/ablation_study/inference_results_ablation/results_20250923_210450.json"
     # similarity_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/llama31_8b/similarity_results.json"
    # ptrue_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/llama31_8b/ptrue_llama3_morehopqa.pkl"
    # white_box_baseline = "/scratch/xiaoouli/project/LLM_reasoning/dataset/MoreHopQA/output/llama31_8b/white_box_baseline_llama31_8b.json"

    # Math_phi4  done
    # label_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/phi4/evaluation_results_last500/detailed_results_20250916_142000.json"
    # wo_mcs_gib_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/phi4/inference_results_last_500_all_graphs/detailed_results_20250916_150702.json"
    # similarity_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/phi4/similarity_results.json"
    # ptrue_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/phi4/ptrue_phi4_math.pkl"
    # white_box_baseline = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/phi4/white_box_baseline_phi4.json"


    # Math_llama3 done
    # label_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/llama3/evaluation_detailed_results_20250916_144109.json"
    # wo_mcs_gib_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/llama3/inference_detailed_results_20250916_144555.json"
    # similarity_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/llama3/similarity_results.json"
    # ptrue_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/llama3/ptrue_llama3_math.pkl"
    # white_box_baseline = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/llama3/white_box_baseline_llama31_8b.json"

    # Math_deepseek done
    # label_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/deepseek/evaluation_results_last500/detailed_results_20250916_164516.json"
    # wo_mcs_gib_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/deepseek/ablation_study/inference_results_ablation/results_20250924_104905.json"
    # similarity_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/deepseek/similarity_results.json"
    # ptrue_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/deepseek/ptrue_deepseek_math.pkl"
    # white_box_baseline = "/scratch/xiaoouli/project/LLM_reasoning/dataset/Math/deepseek/white_box_baseline_deepseek_r1.json"


    #GSM8K phi4 done
    # label_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/phi4/evaluation_results_last500/detailed_results_20250813_015034.json"
    # wo_mcs_gib_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/phi4/src/inference_results_last_500/detailed_results_20250814_165739.json"
    # similarity_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/phi4/similarity_results.json"
    # ptrue_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/phi4/ptrue_phi4_gsm8k.pkl"
    # white_box_baseline = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/phi4/white_box_baseline_phi4.json"

    #gsm8k deepseek done
    # label_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/deepseek-r1/evaluation_results_last500/detailed_results_20250810_120709.json"
    # wo_mcs_gib_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/deepseek-r1/inference_deepseek_detailed_results_20250810_113717.json"
    # similarity_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/deepseek-r1/similarity_results.json"
    # ptrue_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/deepseek-r1/ptrue_deepseek_gsm8k.pkl"
    # white_box_baseline = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/deepseek-r1/white_box_baseline_deepseek_r1.json"

    #gsm8k llama3
    # label_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/llama3-8b/evaluation_detailed_results_20250807_194324.json"
    # wo_mcs_gib_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/llama3-8b/inference_detailed_results.json"
    # similarity_score_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/llama3-8b/similarity_results.json"
    # ptrue_file = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/llama3-8b/ptrue_llama3_gsm8k.pkl"
    # white_box_baseline = "/scratch/xiaoouli/project/LLM_reasoning/dataset/GSM8K/output/llama3-8b/white_box_baseline_llama31_8b.json"

    # load data
    label_data = load_json(label_file)
    gib_score_data = load_json(wo_mcs_gib_score_file)
    similarity_score_data = load_json(similarity_score_file)
    ptrue_score_data = load_pickle(ptrue_file)
    white_box_data = load_json(white_box_baseline)

    methods = [
        (Method.GIB, gib_score_data),
        (Method.COS_MAX, similarity_score_data),
        (Method.COS_MEAN, similarity_score_data),
        #(Method.COS_RAND, similarity_score_data),
        (Method.NLI_MAX, similarity_score_data),
        (Method.NLI_MEAN, similarity_score_data),
        #(Method.NLI_RAND, similarity_score_data),
        (Method.PTRUE, ptrue_score_data),
        (Method.SEQ, white_box_data),
        (Method.ENTROPY, white_box_data),
        (Method.LECO, white_box_data),
    ]

    results = []
    for method, score_data in methods:
        res = evaluate_method(label_data, score_data, method)
        if res:
            results.append(res)

    print("\nSummary:")
    #print(f"{'Method':<15} {'AUROC':<10} {'AUCPR':<10} {'ECE_uncal':<10} {'ECE':<10} {'Acc@80%':<10} {'AURC':<10} {'Brier':<10}")
    print(f"{'Method':<15} {'AUROC':<10} {'AUCPR':<10} {'Acc@80%':<10}")
    for r in results:
        auroc = f"{r['auroc']:.4f}" if r['auroc'] is not None else "N/A"
        aucpr = f"{r['aucpr']:.4f}" if r['aucpr'] is not None else "N/A"
        acc_cov80 = f"{r['acc_cov80']:.4f}" if r['acc_cov80'] is not None else "N/A"
        print(f"{r['method']:<15} {auroc:<10} {aucpr:<10} {acc_cov80:<10}")


if __name__ == "__main__":
    main()
