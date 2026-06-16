import argparse
import os
import pickle
import json
import numpy as np
import networkx as nx
import random
import torch
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class SimilarityCalculator:
    """Compute cosine and NLI similarities."""

    def __init__(self, nli_model_name: str = "microsoft/deberta-large-mnli", device: str = None):
        # Load NLI model
        self.nli_model_name = nli_model_name
        print(f"Loading NLI model: {self.nli_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_model.to(self.device)
        self.nli_model.eval()
        print(f"NLI model loaded on {self.device}")

    # ---- Cosine ----
    def cosine_distance(self, emb1, emb2):
        return cosine(emb1, emb2)  # smaller = more similar

    def cosine_similarity(self, emb1, emb2):
        return 1 - cosine(emb1, emb2)  # larger = more similar

    # ---- NLI ----
    @torch.no_grad()
    def _nli_entailment_probs_one_direction(self, pairs, batch_size=64, max_length=512):
        """Compute entailment probabilities for (premise, hypothesis)."""
        probs_all = []
        model = self.nli_model
        tok = self.tokenizer

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            mask_empty = [(not p.strip()) or (not h.strip()) for (p, h) in batch]
            non_empty_idx = [k for k, m in enumerate(mask_empty) if not m]

            if non_empty_idx:
                enc = tok(
                    [batch[k][0] for k in non_empty_idx],
                    [batch[k][1] for k in non_empty_idx],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length
                ).to(self.device)

                outputs = model(**enc)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                entail = probs[:, 2].detach().tolist()  # entailment
            else:
                entail = []

            ptr = 0
            for m in mask_empty:
                if m:
                    probs_all.append(0.0)
                else:
                    probs_all.append(entail[ptr])
                    ptr += 1

        return probs_all

    def nli_entailment_similarity_batch_symmetric(self, pairs, batch_size=64, max_length=512):
        """Symmetric entailment similarity for batch of (text1, text2)."""
        if not pairs:
            return []
        forward = self._nli_entailment_probs_one_direction(pairs, batch_size, max_length)
        reverse_pairs = [(b, a) for (a, b) in pairs]
        backward = self._nli_entailment_probs_one_direction(reverse_pairs, batch_size, max_length)
        return [0.5 * (f + b) for f, b in zip(forward, backward)]


class GraphSimilarityAnalyzer:
    """Analyzer with unified cosine + NLI methods (per-graph aggregation → cross-graph average)."""

    def __init__(self, label_file_path, embedding_file_path, nli_batch_size=64, nli_max_length=512, device=None):
        self.sim_calculator = SimilarityCalculator(device=device)
        self.nli_batch_size = nli_batch_size
        self.nli_max_length = nli_max_length

        self.correct_graph_ids, self.labeled_problems = self.load_labels_and_problems(label_file_path)

        with open(embedding_file_path, "rb") as f:
            all_embeddings = pickle.load(f)

        self.all_emb_dict: Dict[str, Dict[str, dict]] = {
            pid: emb_data for pid, emb_data in all_embeddings.items() if pid in self.labeled_problems
        }

        print(f"Loaded {len(self.correct_graph_ids)} correct graphs")
        print(f"Kept {len(self.all_emb_dict)} labeled problems")

    def load_labels_and_problems(self, label_file_path):
        with open(label_file_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        correct_graph_ids = set()
        labeled_problems = set()

        for problem_id, problem_data in label_data.items():
            labeled_problems.add(problem_id)
            for respond_id, respond_data in problem_data.items():
                for eval_item in respond_data.get('evaluations', []):
                    if eval_item.get('node_name') == 'NodeResult' and eval_item.get('is_correct') == 1:
                        correct_graph_ids.add(respond_id)
                        break
        return correct_graph_ids, labeled_problems

    def build_graph(self, structure, embeds, texts, graph_id=None):
        import re
        G = nx.DiGraph()
        try:
            for match in re.findall(r'\[([^,]+),\s*([^,]+),\s*([^\]]+)\]', structure):
                u, v, e = match[0].strip(), match[1].strip(), match[2].strip()
                for n in (u, v):
                    if n not in G:
                        G.add_node(n, embedding=embeds[n], text=texts.get(n, ""))
                G.add_edge(u, v, label=e, embedding=embeds[e], text=texts.get(e, ""))
            return G
        except KeyError as e:
            print(f"Error building graph {graph_id}: {e}")
            return None

    # -------- Cosine Methods (distance-based) --------
    def cosine_mean(self, ref_emb, other_graphs):
        sims = []
        for g in other_graphs.values():
            dists = [self.sim_calculator.cosine_distance(ref_emb, g.edges[e]["embedding"]) for e in g.edges()]
            if dists:
                sims.append(1.0 / (1.0 + np.mean(dists)))
        return np.mean(sims) if sims else 0.0

    def cosine_rand(self, ref_emb, other_graphs):
        sims = []
        for g in other_graphs.values():
            edges = list(g.edges())
            if edges:
                u, v = random.choice(edges)
                d = self.sim_calculator.cosine_distance(ref_emb, g.edges[(u, v)]["embedding"])
                sims.append(1.0 / (1.0 + d))
        return np.mean(sims) if sims else 0.0

    def cosine_max(self, ref_emb, other_graphs):
        sims = []
        for g in other_graphs.values():
            dists = [self.sim_calculator.cosine_distance(ref_emb, g.edges[e]["embedding"]) for e in g.edges()]
            if dists:
                sims.append(1.0 / (1.0 + min(dists)))
        return np.mean(sims) if sims else 0.0

    # -------- NLI Methods (similarity-based) --------
    def nli_mean(self, ref_edge_text, ref_target_text, other_graphs):
        sims = []
        for g in other_graphs.values():
            edge_pairs = [(ref_edge_text, g.edges[e].get("text", "")) for e in g.edges()]
            node_pairs = [(ref_target_text, g.nodes[v].get("text", "")) for _, v in g.edges()]
            edge_sims = self.sim_calculator.nli_entailment_similarity_batch_symmetric(edge_pairs, self.nli_batch_size, self.nli_max_length)
            node_sims = self.sim_calculator.nli_entailment_similarity_batch_symmetric(node_pairs, self.nli_batch_size, self.nli_max_length)
            per_graph = [0.5 * (e + n) for e, n in zip(edge_sims, node_sims)]
            if per_graph:
                sims.append(np.mean(per_graph))
        return np.mean(sims) if sims else 0.0

    def nli_rand(self, ref_edge_text, ref_target_text, other_graphs):
        sims = []
        for g in other_graphs.values():
            edges = list(g.edges())
            if edges:
                u, v = random.choice(edges)
                e_text = g.edges[(u, v)].get("text", "")
                n_text = g.nodes[v].get("text", "")
                edge_sim = self.sim_calculator.nli_entailment_similarity_batch_symmetric([(ref_edge_text, e_text)], 1, self.nli_max_length)[0]
                node_sim = self.sim_calculator.nli_entailment_similarity_batch_symmetric([(ref_target_text, n_text)], 1, self.nli_max_length)[0]
                sims.append(0.5 * (edge_sim + node_sim))
        return np.mean(sims) if sims else 0.0

    def nli_max(self, ref_edge_text, ref_target_text, other_graphs):
        sims = []
        for g in other_graphs.values():
            edge_pairs = [(ref_edge_text, g.edges[e].get("text", "")) for e in g.edges()]
            node_pairs = [(ref_target_text, g.nodes[v].get("text", "")) for _, v in g.edges()]
            edge_sims = self.sim_calculator.nli_entailment_similarity_batch_symmetric(edge_pairs, self.nli_batch_size, self.nli_max_length)
            node_sims = self.sim_calculator.nli_entailment_similarity_batch_symmetric(node_pairs, self.nli_batch_size, self.nli_max_length)
            per_graph = [0.5 * (e + n) for e, n in zip(edge_sims, node_sims)]
            if per_graph:
                sims.append(max(per_graph))
        return np.mean(sims) if sims else 0.0

    # -------- Graph Analysis --------
    def analyze_graph(self, problem_id, reference_id):
        emb_dict = self.all_emb_dict[problem_id]
        ref_data = emb_dict[reference_id]
        ref_embeds = ref_data["embeddings"]
        ref_struct = ref_data.get("sub_structure_str") or ref_data.get("structure_str")
        ref_texts = ref_data.get("reasoning_string", {})

        ref_graph = self.build_graph(ref_struct, ref_embeds, ref_texts, reference_id)
        if ref_graph is None:
            return None

        if reference_id in self.correct_graph_ids:
            return None  # skip correct graphs

        emb_dict_ids = list(emb_dict.keys())
        other_correct_ids = [gid for gid in emb_dict_ids if gid != reference_id and gid in self.correct_graph_ids]
        if not other_correct_ids:
            return None

        other_graphs = {}
        for other_id in other_correct_ids:
            other_data = emb_dict[other_id]
            other_embeds = other_data["embeddings"]
            other_struct = other_data.get("sub_structure_str") or other_data.get("structure_str")
            other_texts = other_data.get("reasoning_string", {})
            g = self.build_graph(other_struct, other_embeds, other_texts, other_id)
            if g is not None:
                other_graphs[other_id] = g

        edge_results = []
        for edge in ref_graph.edges():
            source, target = edge
            edge_label = ref_graph.edges[edge].get('label', f"Edge_{source}_{target}")
            ref_edge_emb = ref_graph.edges[edge]["embedding"]
            ref_edge_text = ref_graph.edges[edge].get("text", "")
            ref_target_text = ref_graph.nodes[target].get("text", "")

            scores = {
                "Cos-Mean": self.cosine_mean(ref_edge_emb, other_graphs),
                "Cos-Rand": self.cosine_rand(ref_edge_emb, other_graphs),
                "Cos-Max": self.cosine_max(ref_edge_emb, other_graphs),
                "NLI-Mean": self.nli_mean(ref_edge_text, ref_target_text, other_graphs),
                "NLI-Rand": self.nli_rand(ref_edge_text, ref_target_text, other_graphs),
                "NLI-Max": self.nli_max(ref_edge_text, ref_target_text, other_graphs),
            }

            edge_results.append({
                "edge_nodes": [source, target],
                "edge_label": edge_label,
                "scores": scores
            })

        return {
            "problem_id": problem_id,
            "graph_id": reference_id,
            "is_correct_graph": False,
            "num_correct_graphs_compared": len(other_graphs),
            "edge_details": edge_results
        }

    def run_full_analysis(self, output_path):
        all_results = []
        for problem_id in tqdm(self.all_emb_dict, desc="Processing problems"):
            emb_dict = self.all_emb_dict[problem_id]
            for reference_id in emb_dict.keys():
                result = self.analyze_graph(problem_id, reference_id)
                if result:
                    all_results.append(result)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"Analysis complete. Results saved to {output_path}")
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="NIBS similarity-based baseline analysis."
    )
    parser.add_argument(
        "--label-file",
        type=str,
        required=True,
        help="Path to evaluation.json (GPT correctness labels)."
    )
    parser.add_argument(
        "--embedding-file",
        type=str,
        required=True,
        help="Path to bert_embeddings_*.pkl."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to write NIBS_results.json."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the deberta-large-mnli model (e.g. 'cuda' or 'cpu')."
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    analyzer = GraphSimilarityAnalyzer(
        args.label_file,
        args.embedding_file,
        nli_batch_size=128,
        nli_max_length=512,
        device=args.device
    )
    analyzer.run_full_analysis(args.output_file)


if __name__ == "__main__":
    main()
