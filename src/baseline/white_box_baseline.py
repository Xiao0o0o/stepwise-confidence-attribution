import argparse
import os
import pickle
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    label_file: str
    embedding_file: str  # bert embeddings file
    transformed_file: str  # transformed data file for entropy calculation
    output_file: str
    morehopqa_context_file: Optional[str] = None  # for morehopqa dataset
    dataset_type: str = "morehopqa"  # {"math","gsm8k","morehopqa"}
    model_name: str = "llama3"
    batch_size: int = 8
    max_sequence_length: int = 1024
    device: str = None  

class ModelManager:
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_llm(self, model_name="llama3", device=None):
        if self._model is not None:
            return self._model, self._tokenizer
            
        model_configs = {
            "llama3": "meta-llama/Llama-3.1-8B-Instruct",
            "phi4": "microsoft/Phi-4-reasoning",
            "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_configs.keys())}")
        
        model_id = model_configs[model_name]
        logger.info(f"Loading model: {model_id}")
        
        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)

            if device == "cuda":
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
            
            logger.info(f"Model loaded successfully on {device}")
            return self._model, self._tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

class UncertaintyMetrics:
    @staticmethod
    def compute_sequence_likelihood(probs: List[float]) -> float:
        import math
        if not probs or len(probs) == 0:
            return 0.0
        epsilon = 1e-8
        log_likelihood = sum([math.log(max(p, epsilon)) for p in probs])
        try:
            return math.exp(log_likelihood / len(probs))
        except OverflowError:
            return 0.0
    
    @staticmethod
    def compute_step_entropies(
        model,
        tokenizer,
        question: str,
        steps: List[str],
        context: Optional[str] = None,
        dataset_type: str = "gsm8k",
        max_length: int = 1024,
    ) -> List[float]:
        """Compute entropy using the method from the second file"""
        if not steps:
            return []

        if dataset_type.lower() in {"math", "gsm8k"}:
            prefix = f"Question: {question}\nReasoning: "
        elif dataset_type.lower() == "morehopqa":
            ctx = context or ""
            prefix = f"Context: {ctx}\nQuestion: {question}\nReasoning: "
        else:
            prefix = f"Question: {question}\nReasoning: "

        reasoning_str = "; ".join(steps) + ";"
        full_input = prefix + reasoning_str

        # tokenize full input
        inputs = tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, :-1, :]  # [seq_len-1, vocab]

        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        cursor = len(prefix_ids)

        entropies = []
        for step in steps:
            step_ids = tokenizer(step, add_special_tokens=False).input_ids
            step_len = len(step_ids)
            if step_len == 0:
                entropies.append(0.0)
                continue

            start_idx = max(0, cursor - 1)
            end_idx = min(logits.shape[0], cursor + step_len - 1)

            if end_idx <= start_idx:
                entropy = 0.0
            else:
                step_logits = logits[start_idx:end_idx, :].to(torch.float32)
                log_probs = F.log_softmax(step_logits, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1).mean().item()
                if torch.isnan(torch.tensor(entropy)):
                    entropy = 0.0

            entropies.append(float(entropy))
            cursor += step_len

        return entropies
    
    @staticmethod
    def compute_leco_confidence(model, tokenizer, step_text: str, tau=0.3, K=3) -> float:
        if not step_text:
            return 0.0
        
        try:
            inputs = tokenizer(
                step_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, :-1, :]
                probs = F.softmax(logits, dim=-1)
                
                token_ids = inputs["input_ids"][:, 1:]
                if token_ids.size(1) == 0:
                    return 0.0
                
                token_probs = probs[0, torch.arange(token_ids.size(1)), token_ids[0]].cpu().numpy()
            
            if len(token_probs) == 0:
                return 0.0
            
            avg_score = token_probs.mean()
            trans_score = token_probs[:K].mean() if len(token_probs) >= K else token_probs.mean()
            
            epsilon = 1e-8
            P = token_probs / (token_probs.sum() + epsilon)
            U = np.ones_like(P) / len(P)
            kl_div = np.sum(P * np.log((P + epsilon) / (U + epsilon)))
            diver_score = np.log(kl_div * tau + 1.0)
            
            leco_score = avg_score + trans_score - diver_score
            return float(leco_score)
            
        except Exception as e:
            logger.warning(f"Error computing LECO: {e}")
            return 0.0

class GraphStepAnalyzer:
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.model_manager = ModelManager()
        
        # Load labels
        self.correct_graph_ids, self.labeled_problems = self._load_labels()
        
        # Load both bert embeddings and transformed data
        self.all_emb_dict = self._load_embeddings()
        self.transformed_data = self._load_transformed_data()
        
        # Load morehopqa context if needed
        self.morehopqa_contexts = {}
        if self.config.dataset_type.lower() == "morehopqa" and self.config.morehopqa_context_file:
            with open(self.config.morehopqa_context_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            self.morehopqa_contexts = {
                item["_id"]: " ".join([" ".join(c[1]) for c in item.get("context", [])])
                for item in dataset
            }
        
        # Load model
        self.model, self.tokenizer = self.model_manager.load_llm(
            config.model_name,
            config.device
        )
        
        self.metrics = UncertaintyMetrics()
        
        logger.info(f"Initialized analyzer with {len(self.correct_graph_ids)} correct graphs")
        logger.info(f"Processing {len(self.all_emb_dict)} labeled problems")
    
    def _load_labels(self) -> Tuple[set, set]:
        try:
            with open(self.config.label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            raise
        
        correct_graph_ids = set()
        labeled_problems = set()
        
        for problem_id, problem_data in label_data.items():
            labeled_problems.add(problem_id)
            for respond_id, respond_data in problem_data.items():
                for eval_item in respond_data.get('evaluations', []):
                    if (eval_item.get('node_name') == 'NodeResult' and 
                        eval_item.get('is_correct') == 1):
                        correct_graph_ids.add(respond_id)
                        break
        
        return correct_graph_ids, labeled_problems
    
    def _load_embeddings(self) -> Dict:
        """Load bert embeddings"""
        try:
            with open(self.config.embedding_file, "rb") as f:
                all_embeddings = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
        return {
            pid: emb_data 
            for pid, emb_data in all_embeddings.items() 
            if pid in self.labeled_problems
        }
    
    def _load_transformed_data(self) -> Dict:
        """Load transformed data for entropy calculation"""
        try:
            with open(self.config.transformed_file, "rb") as f:
                all_data = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load transformed data: {e}")
            raise

        flattened_data = {}
        for problem_id, responses in all_data.items():
            if problem_id not in self.labeled_problems:
                continue

            if not isinstance(responses, dict):
                logger.warning(
                    "Skipping problem %s because responses are not a dict",
                    problem_id
                )
                continue

            for response_id, item in responses.items():
                if not isinstance(item, dict):
                    logger.warning(
                        "Skipping response %s for problem %s because it is not a dict",
                        response_id,
                        problem_id
                    )
                    continue

                item["problem_id"] = problem_id
                flattened_data[response_id] = item

        return flattened_data
    
    # def build_graph(self, structure: str, embeds: dict, texts: dict, 
    #                probs: dict, graph_id: str = None) -> Optional[nx.DiGraph]:
    #     import re
    #     G = nx.DiGraph()
        
    #     try:
    #         pattern = r'\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]'
    #         matches = re.findall(pattern, structure)
            
    #         if not matches:
    #             logger.warning(f"No edges found in structure for graph {graph_id}")
    #             return None
            
    #         for match in matches:
    #             u, v, e = match[0].strip(), match[1].strip(), match[2].strip()
    #             for n in (u, v):
    #                 if n not in G:
    #                     node_text = texts.get(n, "")
    #                     node_probs = probs.get(n, [0.5])
    #                     G.add_node(n, text=node_text, probs=node_probs)
                
    #             edge_text = texts.get(e, "")
    #             edge_probs = probs.get(e, [0.5])
    #             G.add_edge(u, v, label=e, text=edge_text, probs=edge_probs)
            
    #         return G
            
    #     except Exception as e:
    #         logger.error(f"Error building graph {graph_id}: {e}")
    #         return None
    def build_graph(self, structure: str, embeds: dict, texts: dict, probs: dict, graph_id: str = None) -> Optional[nx.DiGraph]:
        import re
        G = nx.DiGraph()
        try:
            for match in re.findall(r'\[([^,]+),\s*([^,]+),\s*([^\]]+)\]', structure):
                u, v, e = match[0].strip(), match[1].strip(), match[2].strip()
                for n in (u, v):
                    if n not in G:
                        G.add_node(
                            n,
                            embedding=embeds.get(n),
                            text=texts.get(n, ""),
                            probs=probs.get(n, [])   
                        )
                G.add_edge(
                    u, v,
                    label=e,
                    embedding=embeds.get(e),
                    text=texts.get(e, ""),
                    probs=probs.get(e, [])        
                )
            return G
        except KeyError as e:
            print(f"Error building graph {graph_id}: {e}")
            return None
        
    def get_steps_from_transformed_data(self, reference_id: str) -> Tuple[str, List[str], Optional[str]]:
        """Extract question and steps from transformed data for entropy calculation"""
        if reference_id not in self.transformed_data:
            return "", [], None
        
        graph_data = self.transformed_data[reference_id]
        problem_id = graph_data.get("problem_id")
        reasoning_str = graph_data.get("string_q_a", "") or ""
        question = graph_data.get("question", "") or ""
        
        context = None
        if self.config.dataset_type.lower() == "morehopqa" and problem_id:
            context = self.morehopqa_contexts.get(problem_id, "")
        
        # Parse steps from reasoning string
        steps = []
        for raw_segment in reasoning_str.split(";"):
            segment = raw_segment.strip()
            if not segment:
                continue
            if ", Node" not in segment:
                continue
            
            edge_part, node_part = segment.split(", Node", 1)
            node_part = ("Node" + node_part).strip()
            
            if ":" in node_part:
                node_label, node_text = node_part.split(":", 1)
                node_text = node_text.strip()
                if node_text:
                    steps.append(node_text)
        
        return question, steps, context
    
    def analyze_single_graph(self, problem_id: str, reference_id: str) -> Optional[dict]:
        try:
            emb_dict = self.all_emb_dict[problem_id]
            ref_data = emb_dict[reference_id]
            
            embeds = ref_data["embeddings"]
            struct = ref_data.get("sub_structure_str") or ref_data.get("structure_str", "")
            texts = ref_data.get("reasoning_string", {})
            probs = ref_data.get("probs", {})
            
            ref_graph = self.build_graph(struct, embeds, texts, probs, reference_id)
            if ref_graph is None:
                return None
            
            # Get question and steps from transformed data for entropy calculation
            question, transformed_steps, context = self.get_steps_from_transformed_data(reference_id)
            
            # Compute entropies using the new method
            entropies = self.metrics.compute_step_entropies(
                self.model,
                self.tokenizer,
                question=question,
                steps=transformed_steps,
                context=context,
                dataset_type=self.config.dataset_type,
                max_length=self.config.max_sequence_length,
            ) if transformed_steps else []
            
            # Process edges and compute other metrics
            edge_results = []
            entropy_cursor = 0
            
            for edge in ref_graph.edges():
                src, tgt = edge
                edge_label = ref_graph.edges[edge].get('label', f"Edge_{src}_{tgt}")
                step_text = ref_graph.edges[edge].get("text", "")
                step_probs = ref_graph.edges[edge].get("probs", [0.5])
                
                if edge_label == "ResultEdge":
                    node_probs = ref_graph.nodes[tgt].get("probs", [])
                    node_text = ref_graph.nodes[tgt].get("text", "")
                    if node_probs:
                        step_probs = node_probs
                    if node_text and not step_text:
                        step_text = node_text
                
                # Compute sequence likelihood and LECO (unchanged)
                sl_norm = self.metrics.compute_sequence_likelihood(step_probs)
                leco = self.metrics.compute_leco_confidence(
                    self.model, self.tokenizer, step_text
                )
                
                # Get entropy from precomputed list
                entropy = entropies[entropy_cursor] if entropy_cursor < len(entropies) else 0.0
                if step_text:  # Only increment if we have actual step text
                    entropy_cursor += 1
                
                edge_results.append({
                    "edge_nodes": [src, tgt],
                    "edge_label": edge_label,
                    "scores": {
                        "sl_norm": sl_norm,
                        "entropy": entropy,
                        "leco": leco
                    }
                })
            
            return {
                "problem_id": problem_id,
                "graph_id": reference_id,
                "is_correct_graph": reference_id in self.correct_graph_ids,
                "edge_details": edge_results
            }
            
        except Exception as e:
            logger.error(f"Error analyzing graph {reference_id}: {e}")
            return None
    
    def analyze_graph_batch(self, problem_id: str, reference_ids: List[str]) -> List[dict]:
        results = []
        
        for ref_id in reference_ids:
            result = self.analyze_single_graph(problem_id, ref_id)
            if result:
                results.append(result)
        
        return results
    
    def run_full_analysis(self) -> List[dict]:
        all_results = []
        
        # Count statistics
        total_graphs_count = 0
        skipped_correct_count = 0
        
        for problem_id in tqdm(self.all_emb_dict, desc="Processing problems"):
            emb_dict = self.all_emb_dict[problem_id]
            
            # Skip correct graphs - only process incorrect graphs
            reference_ids = []
            for ref_id in emb_dict.keys():
                total_graphs_count += 1
                if ref_id in self.correct_graph_ids:
                    skipped_correct_count += 1
                    continue  # Skip this correct graph
                reference_ids.append(ref_id)
            
            if reference_ids:
                batch_results = self.analyze_graph_batch(problem_id, reference_ids)
                all_results.extend(batch_results)
        
        logger.info(f"Total graphs encountered: {total_graphs_count}")
        logger.info(f"Correct graphs skipped: {skipped_correct_count}")
        logger.info(f"Incorrect graphs processed: {len(all_results)}")

        try:
            output_path = Path(self.config.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis complete. Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
        
        return all_results

def main():
    parser = argparse.ArgumentParser(
        description="White-box baseline analysis over reasoning step graphs."
    )
    parser.add_argument("--label-file", required=True,
                        help="evaluation.json")
    parser.add_argument("--embedding-file", required=True,
                        help="bert_embeddings_*.pkl")
    parser.add_argument("--transformed-file", required=True,
                        help="transformed_*_with_probs.pkl")
    parser.add_argument("--output-file", required=True,
                        help="path to write white_box_baseline_<model>.json")
    parser.add_argument("--model-name", required=True,
                        choices=["llama3", "phi4", "deepseek"],
                        help="selects the HF model")
    parser.add_argument("--context-file", default=None,
                        help="optional morehopqa/morehopqa context json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    config = AnalysisConfig(
        label_file=args.label_file,
        embedding_file=args.embedding_file,
        transformed_file=args.transformed_file,
        output_file=args.output_file,
        morehopqa_context_file=args.context_file,  # for morehopqa dataset
        dataset_type="morehopqa",  # Set to "math", "gsm8k", or "morehopqa"
        model_name=args.model_name,
        batch_size=8,
        device=args.device,
    )

    try:
        analyzer = GraphStepAnalyzer(config)
        results = analyzer.run_full_analysis()
        
        # Final statistics
        total_graphs = len(results)
        logger.info(f"=== Final Results ===")
        logger.info(f"Total incorrect graphs analyzed: {total_graphs}")
        logger.info(f"Output saved to: {config.output_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()