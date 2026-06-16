import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import re
from itertools import combinations
import copy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import argparse
from find_mcs import MaximumCommonSubgraph




class MCSCache:
    """Cache for MCS computation results"""
    
    def __init__(self, cache_file="mcs_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """Load cache from file if exists"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_key(self, error_graph_id, correct_graph_id):
        """Generate cache key"""
        return f"{error_graph_id}___{correct_graph_id}"
    
    def get(self, error_graph_id, correct_graph_id):
        """Get cached MCS result"""
        key = self.get_key(error_graph_id, correct_graph_id)
        return self.cache.get(key)
    
    def set(self, error_graph_id, correct_graph_id, mcs_result):
        """Cache MCS result"""
        key = self.get_key(error_graph_id, correct_graph_id)
        self.cache[key] = {
            'node_mapping': mcs_result['node_mapping'],
            'edge_mapping': {str(k): v for k, v in mcs_result['edge_mapping'].items()},
            'mcs_size': mcs_result['mcs_size'],
            'mcs_score': mcs_result['mcs_score']
        }
        self._save_cache()

class GCN2Layer(nn.Module):
    """Simple GNN for encoding MCS features"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector [num_nodes]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Last layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
            
        return x

def build_graph(structure, embeds=None):
    """Build a directed graph from structure string (embeddings not needed for NLI)."""
    G = nx.DiGraph()
    
    # Parse the structure and store edge labels
    edge_labels = {}
    for u, v, e in re.findall(r'\[([^,]+),\s*([^,]+),\s*([^\]]+)\]', structure):
        u, v, e = u.strip(), v.strip(), e.strip()
        for n in (u, v):
            if n not in G:
                G.add_node(n)
        G.add_edge(u, v, label=e)  # Store the edge label
        edge_labels[(u, v)] = e
    
    # Store edge labels as graph attribute for later use
    G.graph['edge_labels'] = edge_labels
    
    return G

def load_evaluation_results(json_path: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def load_human_evaluation_data(excel_path: str) -> pd.DataFrame:
    """Load human evaluation data from Excel file."""
    _, ext = os.path.splitext(excel_path)

    if ext.lower() == '.csv':
        return pd.read_csv(excel_path)
    elif ext.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(excel_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# def is_correct_graph(graph_info: Dict, problem_id: str, graph_id: str, human_eval_df: pd.DataFrame) -> bool:
#     """
#     Determine if a graph is correct based on human evaluation of NodeResult.
    
#     Args:
#         graph_info: Graph data information
#         problem_id: Problem ID (e.g., 'grade school math_6075')
#         graph_id: Graph ID (e.g., 'grade school math_6075_0')
#         human_eval_df: Human evaluation DataFrame
    
#     Returns:
#         bool: True if correct graph, False if error graph
#     """
#     # Find corresponding NodeResult evaluation
#     node_result_rows = human_eval_df[
#         (human_eval_df['key'] == problem_id) & 
#         (human_eval_df['id'] == graph_id) & 
#         (human_eval_df['node'] == 'NodeResult')
#     ]
    
#     if len(node_result_rows) > 0:
#         # If NodeResult evaluation found, judge based on Human Evaluation
#         human_eval = node_result_rows.iloc[0]['correctness']
#         return human_eval == 1  # 1 means correct, 0 means error
#     else:
#         # If no NodeResult evaluation found, default to False
#         print(f"NodeResult not found for {problem_id} - {graph_id}")
#         return False

def is_correct_graph(graph_info: Dict, problem_id: str, graph_id: str, eval_results: Dict) -> bool:
    """
    Determine if a graph is correct based on NodeResult evaluation.
    
    Args:
        graph_info: Graph data information
        problem_id: Problem ID (e.g., '5ae789615542997ec2727695_12')
        graph_id: Graph ID (e.g., '5ae789615542997ec2727695_12_0')
        eval_results: Evaluation results dictionary from JSON
    
    Returns:
        bool: True if correct graph, False if error graph
    """
    # Check if problem exists in evaluation results
    if problem_id not in eval_results:
        print(f"Problem {problem_id} not found in evaluation results")
        return False
    
    # Check if graph_id exists for this problem
    if graph_id not in eval_results[problem_id]:
        print(f"Graph {graph_id} not found for problem {problem_id}")
        return False
    
    # Get the evaluation for this graph
    graph_eval = eval_results[problem_id][graph_id]
    
    # Find NodeResult evaluation
    for eval_item in graph_eval.get('evaluations', []):
        if eval_item.get('node_name') == 'NodeResult':
            is_correct = eval_item.get('is_correct', 0)
            if is_correct == -1:  # Invalid, skip
                print(f"NodeResult is invalid (-1) for {graph_id}")
                return False
            return is_correct == 1  # 1 means correct, 0 means error
    
    # If no NodeResult found, default to False
    print(f"NodeResult not found for {problem_id} - {graph_id}")
    return False


# def get_available_problems(graph_data: Dict, human_eval_df: pd.DataFrame) -> List[str]:
#     """Get list of problems that exist in both graph data and human evaluation data."""
#     graph_problems = set(graph_data.keys())
#     eval_problems = set(human_eval_df['key'].unique())
    
#     # Take intersection
#     available_problems = list(graph_problems.intersection(eval_problems))
#     print(f"Found {len(available_problems)} problems with both graph data and human evaluation:")
#     for prob in available_problems:
#         print(f"  - {prob}")
    
#     return available_problems

def get_available_problems(graph_data: Dict, eval_results: Dict) -> List[str]:
    """Get list of problems that exist in both graph data and evaluation results."""
    graph_problems = set(graph_data.keys())
    eval_problems = set(eval_results.keys())
    
    # Take intersection
    available_problems = list(graph_problems.intersection(eval_problems))
    print(f"Found {len(available_problems)} problems with both graph data and evaluation results:")
    for prob in available_problems[:5]:  # Show first 5 as example
        print(f"  - {prob}")
    
    return available_problems


# def separate_correct_error_graphs(problem_data: Dict, problem_id: str, human_eval_df: pd.DataFrame) -> Tuple[List[Tuple], List[Tuple]]:
#     """Separate correct graphs and error graphs based on human evaluation."""
#     correct_graphs = []
#     error_graphs = []
    
#     for graph_id, graph_info in problem_data.items():
#         if is_correct_graph(graph_info, problem_id, graph_id, human_eval_df):
#             correct_graphs.append((graph_id, graph_info))
#         else:
#             error_graphs.append((graph_id, graph_info))
    
#     return correct_graphs, error_graphs

def separate_correct_error_graphs(problem_data: Dict, problem_id: str, eval_results: Dict) -> Tuple[List[Tuple], List[Tuple]]:
    """Separate correct graphs and error graphs based on evaluation results."""
    correct_graphs = []
    error_graphs = []
    
    for graph_id, graph_info in problem_data.items():
        if is_correct_graph(graph_info, problem_id, graph_id, eval_results):
            correct_graphs.append((graph_id, graph_info))
        else:
            error_graphs.append((graph_id, graph_info))
    
    return correct_graphs, error_graphs


class GIBErrorDetectionModel(nn.Module):
    """
    Graph Information Bottleneck (GIB) based error detection model.
    Generates edge-level soft masks for error reasoning graphs to identify error steps.
    """
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128, mcs_finder: MaximumCommonSubgraph = None, mcs_cache: MCSCache = None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Initialize MCS finder
        if mcs_finder is None:
            self.mcs_finder = MaximumCommonSubgraph(node_threshold=0.7, edge_threshold=0.7)
        else:
            self.mcs_finder = mcs_finder

        # Initialize MCS cache
        if mcs_cache is None:
            self.mcs_cache = MCSCache()
        else:
            self.mcs_cache = mcs_cache
        
        # Node and edge feature encoders (using pre-trained BERT embeddings)
        self.node_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Graph-level encoder
        self.graph_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenate node and edge features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge-level soft mask predictor
        self.mask_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Edge features + graph features  # delete mcs features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output probability between 0-1
        )
        
        # MCS similarity predictor
        self.mcs_similarity_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.mcs_gnn = GCN2Layer(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2
        )

    def extract_graph_features(self, graph_data: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, nx.DiGraph, List[Tuple]]:
        """
        Extract features from graph data.
        
        Args:
            graph_data: Dictionary containing embeddings, reasoning_string, structure_str
            
        Returns:
            node_features: Node features [num_nodes, hidden_dim]
            edge_features: Edge features [num_edges, hidden_dim]
            graph_feature: Graph-level features [hidden_dim]
            G: NetworkX graph object
            edge_ids: List of edge IDs
        """
        embeddings = graph_data['embeddings']
        structure_str = graph_data['structure_str']
        device = next(self.parameters()).device
        
        # Parse graph structure
        G = build_graph(structure_str)
        
        # Extract node features
        node_embeddings = []
        node_ids = list(G.nodes())
        for node_id in node_ids:
            if node_id in embeddings:
                node_embeddings.append(embeddings[node_id])
            else:
                # If no corresponding embedding, use zero vector
                node_embeddings.append(np.zeros(self.embedding_dim))
        
        if node_embeddings:
            node_embeddings = torch.tensor(np.array(node_embeddings), dtype=torch.float32).to(device)
            node_features = self.node_encoder(node_embeddings)
        else:
            node_features = torch.zeros((0, self.hidden_dim))
        
        # Extract edge features
        edge_embeddings = []
        edge_ids = []
        for u, v in G.edges():
            edge_label = G.get_edge_data(u, v).get('label', f'{u}->{v}')
            edge_ids.append((u, v))
            if edge_label in embeddings:
                edge_embeddings.append(embeddings[edge_label])
            else:
                # If no corresponding embedding, use zero vector
                edge_embeddings.append(np.zeros(self.embedding_dim))
        
        if edge_embeddings:
            edge_embeddings = torch.tensor(np.array(edge_embeddings), dtype=torch.float32).to(device)
            edge_features = self.edge_encoder(edge_embeddings)
        else:
            edge_features = torch.zeros((0, self.hidden_dim))
        
        # Graph-level features (average of node and edge features)
        if len(node_features) > 0 and len(edge_features) > 0:
            graph_feature = self.graph_encoder(torch.cat([
                torch.mean(node_features, dim=0),
                torch.mean(edge_features, dim=0)
            ], dim=0))
        elif len(node_features) > 0:
            # Only node features available
            graph_feature = self.graph_encoder(torch.cat([
                torch.mean(node_features, dim=0),
                torch.zeros(self.hidden_dim)
            ], dim=0))
        elif len(edge_features) > 0:
            # Only edge features available
            graph_feature = self.graph_encoder(torch.cat([
                torch.zeros(self.hidden_dim),
                torch.mean(edge_features, dim=0)
            ], dim=0))
        else:
            # No features available
            graph_feature = torch.zeros(self.hidden_dim)
        
        return node_features, edge_features, graph_feature, G, edge_ids
    def _compute_fast_features(self, error_graph_data: Dict, correct_graphs_data: List[Dict]):
    
        error_G = build_graph(error_graph_data['structure_str'])
        
    
        node_similarities = []
        edge_similarities = []
        
        for correct_data in correct_graphs_data[:3]:  
            correct_G = build_graph(correct_data['structure_str'])
            
            
            node_sim = min(len(error_G.nodes()), len(correct_G.nodes())) / max(len(error_G.nodes()), len(correct_G.nodes()))
            node_similarities.append(node_sim)
            
            
            edge_sim = min(len(error_G.edges()), len(correct_G.edges())) / max(len(error_G.edges()), len(correct_G.edges())) if max(len(error_G.edges()), len(correct_G.edges())) > 0 else 0
            edge_similarities.append(edge_sim)
        
        
        features = [
            np.mean(node_similarities) if node_similarities else 0.0,
            np.max(node_similarities) if node_similarities else 0.0,
            np.mean(edge_similarities) if edge_similarities else 0.0,
            len(error_G.nodes()) / 20.0  
        ]
        
        mcs_feature_vector = torch.tensor(features, dtype=torch.float32)
        device = next(self.parameters()).device
        mcs_feature_vector = mcs_feature_vector.to(device)
        
        if not hasattr(self, 'mcs_feature_expander'):
            self.mcs_feature_expander = nn.Linear(4, self.hidden_dim)
        
        mcs_features = F.relu(self.mcs_feature_expander(mcs_feature_vector))
        
        return mcs_features, []

    def _get_default_mcs_features(self):
        """默认MCS特征"""
        default_features = torch.zeros(4, dtype=torch.float32)
        device = next(self.parameters()).device
        default_features = default_features.to(device)
        if not hasattr(self, 'mcs_feature_expander'):
            self.mcs_feature_expander = nn.Linear(4, self.hidden_dim)
        
        return F.relu(self.mcs_feature_expander(default_features))
    
    # def compute_mcs_features(self, error_graph_data: Dict, correct_graphs_data: List[Dict]) -> torch.Tensor:
    #     """
    #     Compute MCS features between error graph and correct graphs.
        
    #     Returns:
    #         mcs_features: MCS-related feature vector [hidden_dim]
    #     """
    #     error_reasoning = error_graph_data['reasoning_string']
    #     error_structure = error_graph_data['structure_str']
    #     error_G = build_graph(error_structure)

    #     # 图大小检查
    #     max_nodes = 10  # 节点数限制
    #     max_time = 300  # 5分钟时间限制
    #     if len(error_G.nodes()) > max_nodes:
    #         print(f"Large graph detected ({len(error_G.nodes())} nodes), using fast approximation")
    #         return self._compute_fast_features(error_graph_data, correct_graphs_data)

    #     mcs_scores = []
    #     mcs_sizes = []
    #     edge_mappings_list = []
        
    #     for correct_data in correct_graphs_data:
    #         correct_reasoning = correct_data['reasoning_string']
    #         correct_structure = correct_data['structure_str']
    #         correct_G = build_graph(correct_structure)
            
    #         try:
    #             # Use your MCS code to find maximum common subgraph
    #             common_subgraph, node_mapping, edge_mapping = self.mcs_finder.find_maximum_common_subgraph(
    #                 error_G, correct_G, error_reasoning, correct_reasoning
    #             )
                
    #             # Compute MCS-related metrics
    #             mcs_size = len(common_subgraph.nodes())
    #             total_nodes = len(error_G.nodes())
    #             mcs_score = mcs_size / total_nodes if total_nodes > 0 else 0.0
                
    #             mcs_scores.append(mcs_score)
    #             mcs_sizes.append(mcs_size)
    #             edge_mappings_list.append(edge_mapping)
                
    #         except Exception as e:
    #             print(f"MCS computation failed: {e}")
    #             mcs_scores.append(0.0)
    #             mcs_sizes.append(0)
    #             edge_mappings_list.append({})
        
    #     # Aggregate MCS features
    #     if mcs_scores:
    #         avg_mcs_score = np.mean(mcs_scores)
    #         max_mcs_score = np.max(mcs_scores)
    #         avg_mcs_size = np.mean(mcs_sizes)
    #         edge_coverage = len(set().union(*[em.keys() for em in edge_mappings_list])) / len(error_G.edges()) if len(error_G.edges()) > 0 else 0.0
    #     else:
    #         avg_mcs_score = max_mcs_score = avg_mcs_size = edge_coverage = 0.0
        
    #     # Convert MCS statistics to feature vector
    #     mcs_feature_vector = torch.tensor([
    #         avg_mcs_score, max_mcs_score, avg_mcs_size, edge_coverage
    #     ], dtype=torch.float32)
        
    #     device = next(self.parameters()).device
    #     mcs_feature_vector = mcs_feature_vector.to(device)

    #     # Expand to hidden_dim through linear layer
    #     if not hasattr(self, 'mcs_feature_expander'):
    #         self.mcs_feature_expander = nn.Linear(4, self.hidden_dim)
        
    #     mcs_features = F.relu(self.mcs_feature_expander(mcs_feature_vector))
        
    #     return mcs_features, edge_mappings_list
    def compute_mcs_features(self, error_graph_data: Dict, correct_graphs_data: List[Dict], 
                            error_graph_id: str = None, problem_id: str = None) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Compute MCS features using cached results and GNN encoding.
        """
        device = next(self.parameters()).device
        error_reasoning = error_graph_data['reasoning_string']
        error_structure = error_graph_data['structure_str']
        error_G = build_graph(error_structure)
        
        # Check graph size
        max_nodes = 15
        if len(error_G.nodes()) > max_nodes:
            print(f"Large graph detected ({len(error_G.nodes())} nodes), using fast approximation")
            return self._compute_fast_features(error_graph_data, correct_graphs_data)
        
        # Collect all MCS subgraphs
        all_mcs_graphs = []
        edge_mappings_list = []
        
        for correct_id, correct_data in correct_graphs_data:
            #correct_id = f"{problem_id}_{i}" if problem_id else f"correct_{i}"
            #print(f"Processing MCS for {error_graph_id} vs {correct_id}")
            # Check cache first
            cached_result = self.mcs_cache.get(error_graph_id, correct_id) if error_graph_id else None
            
            if cached_result:
                # Use cached result
                #print(f"Using cached MCS for {error_graph_id} vs {correct_id}")
                edge_mapping = {eval(k): v for k, v in cached_result['edge_mapping'].items()}
                edge_mappings_list.append(edge_mapping)
                
                # Reconstruct MCS graph from cached data
                mcs_graph = nx.DiGraph()
                for node1, node2 in cached_result['node_mapping'].items():
                    mcs_graph.add_node(node1)
                for edge in edge_mapping:
                    if edge[0] in mcs_graph and edge[1] in mcs_graph:
                        mcs_graph.add_edge(edge[0], edge[1])
                all_mcs_graphs.append(mcs_graph)
            else:
                # Compute MCS
                correct_reasoning = correct_data['reasoning_string']
                correct_structure = correct_data['structure_str']
                correct_G = build_graph(correct_structure)
                
                try:
                    common_subgraph, node_mapping, edge_mapping = self.mcs_finder.find_maximum_common_subgraph(
                        error_G, correct_G, error_reasoning, correct_reasoning
                    )
                    
                    # Cache the result
                    if error_graph_id:
                        mcs_result = {
                            'node_mapping': node_mapping,
                            'edge_mapping': edge_mapping,
                            'mcs_size': len(common_subgraph.nodes()),
                            'mcs_score': len(common_subgraph.nodes()) / len(error_G.nodes()) if len(error_G.nodes()) > 0 else 0.0
                        }
                        self.mcs_cache.set(error_graph_id, correct_id, mcs_result)
                    
                    edge_mappings_list.append(edge_mapping)
                    all_mcs_graphs.append(common_subgraph)
                    
                except Exception as e:
                    print(f"MCS computation failed: {e}")
                    edge_mappings_list.append({})
                    all_mcs_graphs.append(nx.DiGraph())
        
        # Encode MCS graphs using GNN
        if all_mcs_graphs:
            mcs_features = self._encode_mcs_with_gnn(all_mcs_graphs, error_graph_data)
            #print(f"Encoded MCS features shape: {mcs_features.shape}")
        else:
            mcs_features = torch.zeros(self.hidden_dim).to(device)
        
        return mcs_features, edge_mappings_list

    def _encode_mcs_with_gnn(self, mcs_graphs: List[nx.DiGraph], error_graph_data: Dict) -> torch.Tensor:
        """Encode multiple MCS graphs using GNN and aggregate"""
        device = next(self.parameters()).device
        
        embeddings = error_graph_data['embeddings']

        graph_features = []
        problem_graphs = []
        for idx, mcs_graph in enumerate(mcs_graphs):
            try:
                # if len(mcs_graph.nodes()) == 0:
                #     # Empty graph
                #     graph_features.append(torch.zeros(self.hidden_dim).to(device))
                #     continue
                if len(mcs_graph.nodes()) == 0 or not mcs_graph:
                    graph_features.append(torch.zeros(self.hidden_dim).to(device))
                    continue
                
                # Prepare node features
                node_list = list(mcs_graph.nodes())
                node_features = []
                
                for node in node_list:
                    if node in embeddings:
                        node_features.append(embeddings[node])
                    else:
                        node_features.append(np.zeros(self.embedding_dim))
                
                x = torch.tensor(np.array(node_features), dtype=torch.float32).to(device)
                
                # Prepare edge indices
                edge_list = list(mcs_graph.edges())
                if len(edge_list) > 0:
                    node_to_idx = {node: i for i, node in enumerate(node_list)}
                    # 检查每条边
                    for u, v in edge_list:
                        if u not in node_to_idx:
                            print(f"ERROR: Node {u} not in node_to_idx!")
                        if v not in node_to_idx:
                            print(f"ERROR: Node {v} not in node_to_idx!")

                    edge_index = torch.tensor(
                        [[node_to_idx[u], node_to_idx[v]] for u, v in edge_list],
                        dtype=torch.long
                    ).t().contiguous().to(device)
                    if edge_index.numel() > 0:
                        assert edge_index.max() < len(node_list), f"Edge index {edge_index.max()} >= num_nodes {len(node_list)}"
                        assert edge_index.min() >= 0, f"Edge index {edge_index.min()} < 0"
                        assert edge_index.shape[0] == 2, f"Edge index should have shape [2, E], but got {edge_index.shape}"
                    # Forward through GNN
                    with torch.no_grad():  # No need gradients for MCS encoding
                        graph_feat = self.mcs_gnn(x, edge_index).squeeze(0)
                else:
                    if x.size(0) > 0:
                        graph_feat = x.mean(dim=0)
                        # 确保维度正确
                        if graph_feat.dim() == 1:
                            graph_feat = graph_feat.unsqueeze(0)
                    else:
                        graph_feat = torch.zeros(0, self.hidden_dim).to(device)
                
                graph_features.append(graph_feat)
            except AssertionError as e:
                problem_graphs.append({
                    'graph_idx': idx,
                    'num_nodes': len(mcs_graph.nodes()),
                    'num_edges': len(mcs_graph.edges()),
                    'error': str(e)
                })
                print(f"Problem detected in graph {idx}: {e}")
            
        # Aggregate features from all MCS graphs
        if graph_features:
            # Mean pooling
            mcs_features = torch.stack(graph_features).mean(dim=0)
        else:
            mcs_features = torch.zeros(self.hidden_dim).to(device)
        
        return mcs_features
    
    def forward(self, error_graph_data: Dict, correct_graphs_data: List[Dict], 
                error_graph_id: str = None, problem_id: str = None) -> Dict:
        """
        Forward pass (prediction only, no loss, no MCS features).
        """
        error_node_features, error_edge_features, error_graph_feature, error_G, error_edge_ids = self.extract_graph_features(error_graph_data)
        if len(error_edge_ids) == 0:
            return {
                'edge_masks': torch.tensor([]),
                'edge_ids': []
            }
        
        device = next(self.parameters()).device
        edge_masks = []
        for i, (u, v) in enumerate(error_edge_ids):
            edge_input = torch.cat([
                error_edge_features[i].to(device),
                error_graph_feature.to(device)
            ], dim=0)
            mask_prob = self.mask_predictor(edge_input)
            edge_masks.append(mask_prob)
        
        edge_masks = torch.stack(edge_masks).squeeze(-1)
        return {
            'edge_masks': edge_masks,
            'edge_ids': error_edge_ids
        }
    
    # def compute_losses(self, error_graph_data: Dict, correct_graphs_data: List[Dict],
    #                   edge_masks: torch.Tensor, human_labels: Optional[torch.Tensor],
    #                   error_G: nx.DiGraph, error_edge_ids: List[Tuple], 
    #                   edge_mappings_list: List[Dict]) -> Dict:
    #     """Compute various loss functions."""
    #     losses = {}
        
        
    #     # 2. GIB information bottleneck loss - encourage sparsity
    #     entropy_loss = -torch.mean(
    #         edge_masks * torch.log(edge_masks + 1e-8) + 
    #         (1 - edge_masks) * torch.log(1 - edge_masks + 1e-8)
    #     )
    #     losses['entropy'] = entropy_loss
        
    #     # 3. MCS consistency loss
    #     mcs_loss = self.compute_mcs_consistency_loss(
    #         edge_masks, error_edge_ids, edge_mappings_list
    #     )
    #     losses['mcs_consistency'] = mcs_loss

    #     #losses['structure'] = torch.tensor(0.0)
        
    #     return losses
    def compute_losses(self, error_graph_data: Dict, correct_graphs_data: List[Dict],
                    edge_masks: torch.Tensor, human_labels: Optional[torch.Tensor],
                    error_G: nx.DiGraph, error_edge_ids: List[Tuple], 
                    edge_mappings_list: List[Dict]) -> Dict:
        """Compute stable losses with clamping and NaN guards."""
        losses = {}

        # ----- Stable entropy (binary entropy of mask) -----
        eps = 1e-6
        p = edge_masks.clamp(min=eps, max=1 - eps)
        # This is numerically stable and cannot produce NaN/Inf
        entropy_loss = -torch.mean(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        losses['entropy'] = entropy_loss

        # ----- MCS consistency loss (also clamp) -----
        mcs_loss = self.compute_mcs_consistency_loss(p, error_edge_ids, edge_mappings_list)
        losses['mcs_consistency'] = mcs_loss

        # ----- Final guard: replace any non-finite values -----
        for k, v in list(losses.items()):
            if not torch.isfinite(v):
                print(f"[WARN] {k} is non-finite ({v.item()}), replacing with 0.")
                losses[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            print(f"[WARN] total_loss is non-finite ({total_loss.item()}), replacing with 0.")
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return losses
    
    def compute_mcs_consistency_loss(self, edge_masks: torch.Tensor, error_edge_ids: List[Tuple], 
                                   edge_mappings_list: List[Dict]) -> torch.Tensor:
        """Compute MCS consistency loss."""
        if not edge_mappings_list:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        count = 0
        
        for edge_mapping in edge_mappings_list:
            if not edge_mapping:
                continue
                
            # Find edges that belong to MCS
            mcs_edge_indices = []
            non_mcs_edge_indices = []
            
            for i, edge in enumerate(error_edge_ids):
                if edge in edge_mapping:
                    mcs_edge_indices.append(i)
                else:
                    non_mcs_edge_indices.append(i)
            
            # MCS edges should have high mask values
            if mcs_edge_indices:
                mcs_masks = edge_masks[mcs_edge_indices]
                mcs_loss = -torch.mean(torch.log(mcs_masks + 1e-8))
                total_loss += mcs_loss
            
            # Non-MCS edges should have low mask values
            if non_mcs_edge_indices:
                non_mcs_masks = edge_masks[non_mcs_edge_indices]
                non_mcs_loss = -torch.mean(torch.log(1 - non_mcs_masks + 1e-8))
                total_loss += non_mcs_loss
            
            count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0)


class GIBTrainer:
    """Trainer class for GIB model."""
    
    def __init__(self, model: GIBErrorDetectionModel, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        self.model.to(self.device)
    
    def train_step(self, error_graph_data: Dict, correct_graphs_data: List[Dict], 
               error_graph_id: str, problem_id: str, eval_results: Dict) -> Dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Since we no longer have edge-level annotations, pass None
        # human_labels = None
        
        # # Forward pass
        # results = self.model(error_graph_data, correct_graphs_data, human_labels, 
        #                 error_graph_id, problem_id)
        
        # # Backward pass
        # total_loss = results['total_loss']
        # if total_loss.item() > 0:  # Only backpropagate when loss > 0
        #     total_loss.backward()
        #     self.optimizer.step()
        
        # return {
        #     'total_loss': total_loss.item(),
        #     'losses': {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in results['losses'].items()},
        #     'edge_masks': results['edge_masks'].detach().cpu(),
        #     'edge_ids': results['edge_ids']
        # }
        # 1. Forward prediction
        outputs = self.model(error_graph_data, correct_graphs_data, 
                             error_graph_id=error_graph_id, problem_id=problem_id)
        edge_masks = outputs['edge_masks']
        edge_ids = outputs['edge_ids']
        error_G = build_graph(error_graph_data['structure_str'])
        # Guard masks before loss
        if torch.isnan(edge_masks).any() or torch.isinf(edge_masks).any():
            print("[WARN] edge_masks contain non-finite values; fixing.")
            edge_masks = torch.nan_to_num(edge_masks, nan=0.5, posinf=1.0, neginf=0.0)

        # 2. Compute MCS edge mappings (supervision info)
        _, edge_mappings_list = self.model.compute_mcs_features(
            error_graph_data, correct_graphs_data, error_graph_id, problem_id
        )
        
        # 3. Compute losses
        losses = self.model.compute_losses(
            error_graph_data, correct_graphs_data,
            edge_masks, None, error_G, edge_ids, edge_mappings_list
        )
        #print(f"Losses: {losses}")
        total_loss = sum(losses.values())
        
        # 4. Backward + optimize
        if total_loss.item() > 0:  
            total_loss.backward()
            self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'losses': {k: v.item() if isinstance(v, torch.Tensor) else v 
                       for k, v in losses.items()},
            'edge_masks': edge_masks.detach().cpu(),
            'edge_ids': edge_ids
        }

class ConvergenceMonitor:
    """Monitor training convergence"""
    
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience  # Early stopping patience
        self.min_delta = min_delta  # Minimum improvement threshold
        
        # Record training metrics
        self.epoch_losses = []
        self.loss_components = defaultdict(list)
        self.edge_mask_stats = defaultdict(list)
        
        # Early stopping related
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        # Convergence detection
        self.loss_window = []
        self.window_size = 5
        
    def update(self, epoch, total_loss, loss_components, edge_masks_batch):
        """Update monitoring metrics"""
        # Record total loss
        self.epoch_losses.append(total_loss)
        
        # Record loss components
        for component, value in loss_components.items():
            self.loss_components[component].append(value)
        
        # Record edge mask statistics
        if len(edge_masks_batch) > 0:
            all_masks = np.concatenate([masks.numpy() for masks in edge_masks_batch])
            self.edge_mask_stats['mean'].append(np.mean(all_masks))
            self.edge_mask_stats['std'].append(np.std(all_masks))
            self.edge_mask_stats['min'].append(np.min(all_masks))
            self.edge_mask_stats['max'].append(np.max(all_masks))
            self.edge_mask_stats['sparsity'].append(np.sum(all_masks < 0.1) / len(all_masks))
        
        # Early stopping detection
        if total_loss < self.best_loss - self.min_delta:
            self.best_loss = total_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            self.should_stop = True
            
        # Convergence detection
        self.loss_window.append(total_loss)
        if len(self.loss_window) > self.window_size:
            self.loss_window.pop(0)
    
    def is_converged(self, convergence_threshold=1e-5):
        """Detect if training has converged"""
        if len(self.loss_window) < self.window_size:
            return False
            
        # Calculate loss change in recent epochs
        recent_losses = np.array(self.loss_window)
        loss_variance = np.var(recent_losses)
        loss_trend = np.abs(recent_losses[-1] - recent_losses[0])
        
        return loss_variance < convergence_threshold and loss_trend < convergence_threshold
    
    def print_status(self, epoch):
        """Print current training status"""
        if len(self.epoch_losses) == 0:
            return
            
        current_loss = self.epoch_losses[-1]
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Current Loss: {current_loss:.6f}")
        print(f"  Best Loss: {self.best_loss:.6f}")
        print(f"  Patience: {self.patience_counter}/{self.patience}")
        
        # Loss components
        if self.loss_components:
            print("  Loss Components:")
            for component, values in self.loss_components.items():
                if values:
                    print(f"    {component}: {values[-1]:.6f}")
        
        # Edge mask statistics
        if self.edge_mask_stats['mean']:
            print("  Edge Mask Stats:")
            print(f"    Mean: {self.edge_mask_stats['mean'][-1]:.4f}")
            print(f"    Std:  {self.edge_mask_stats['std'][-1]:.4f}")
            print(f"    Sparsity: {self.edge_mask_stats['sparsity'][-1]:.4f}")
        
        # Convergence status
        if self.is_converged():
            print("  CONVERGED")
        elif self.should_stop:
            print("  EARLY STOPPING")
        else:
            print("  Training......")
            
        print(f"{'='*60}")
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves"""
        if len(self.epoch_losses) < 2:
            print("Not enough data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.epoch_losses) + 1)
        
        # 1. Total loss curve
        axes[0, 0].plot(epochs, self.epoch_losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=self.best_loss, color='r', linestyle='--', alpha=0.7, label=f'Best: {self.best_loss:.4f}')
        axes[0, 0].legend()
        
        # 2. Loss components
        axes[0, 1].set_title('Loss Components')
        for component, values in self.loss_components.items():
            if values and len(values) == len(epochs):
                axes[0, 1].plot(epochs, values, label=component, linewidth=1.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Edge mask statistics
        axes[1, 0].set_title('Edge Mask Statistics')
        if self.edge_mask_stats['mean']:
            axes[1, 0].plot(epochs, self.edge_mask_stats['mean'], 'g-', label='Mean', linewidth=2)
            axes[1, 0].fill_between(epochs, 
                                  np.array(self.edge_mask_stats['mean']) - np.array(self.edge_mask_stats['std']),
                                  np.array(self.edge_mask_stats['mean']) + np.array(self.edge_mask_stats['std']),
                                  alpha=0.3, color='g')
            axes[1, 0].plot(epochs, self.edge_mask_stats['sparsity'], 'r--', label='Sparsity', linewidth=1.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Loss change rate
        if len(self.epoch_losses) > 1:
            loss_diff = np.diff(self.epoch_losses)
            axes[1, 1].plot(epochs[1:], loss_diff, 'purple', linewidth=1.5)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].set_title('Loss Change Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Δ Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        plt.show()
    
    def save_metrics(self, save_path):
        """Save training metrics"""
        # 转换所有numpy类型为Python原生类型
        def convert_to_native(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            return obj
        
        metrics = {
            'epoch_losses': convert_to_native(self.epoch_losses),
            'loss_components': convert_to_native(dict(self.loss_components)),
            'edge_mask_stats': convert_to_native(dict(self.edge_mask_stats)),
            'best_loss': float(self.best_loss),
            'final_converged': bool(self.is_converged())
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to: {save_path}")



# Key indicators for convergence assessment
def analyze_convergence(monitor):
    """Analyze convergence status"""
    print("\nConvergence Analysis:")
    
    if len(monitor.epoch_losses) < 10:
        print("Not enough training epochs for reliable analysis")
        return
    
    # 1. Loss trend analysis
    recent_losses = monitor.epoch_losses[-10:]
    loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
    
    print(f"1. Loss Trend (last 10 epochs): {loss_trend:.2e}")
    if abs(loss_trend) < 1e-5:
        print("Loss is stable")
    elif loss_trend < 0:
        print("Loss is decreasing")
    else:
        print("Loss is increasing")
    
    # 2. Loss variance analysis
    loss_variance = np.var(recent_losses)
    print(f"2. Loss Variance (last 10 epochs): {loss_variance:.2e}")
    if loss_variance < 1e-6:
        print("Very low variance - likely converged")
    elif loss_variance < 1e-4:
        print("Low variance - approaching convergence")
    else:
        print("High variance - still training")
    
    # 3. Edge mask stability
    if monitor.edge_mask_stats['mean']:
        recent_mask_means = monitor.edge_mask_stats['mean'][-10:]
        mask_variance = np.var(recent_mask_means)
        print(f"3. Edge Mask Stability: {mask_variance:.2e}")
        if mask_variance < 1e-4:
            print("Edge masks are stable")
        else:
            print("Edge masks still changing")
    
def save_model_and_results(model, monitor, save_dir):
    """
    Save trained model and training results
    
    Args:
        model: Trained GIBErrorDetectionModel
        monitor: ConvergenceMonitor with training metrics
        save_dir: Directory to save results
    """
    from datetime import datetime
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save model state dict
    model_path = os.path.join(save_dir, f"gib_model_{timestamp}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'node_threshold': model.mcs_finder.node_threshold,
            'edge_threshold': model.mcs_finder.edge_threshold
        },
        'training_info': {
            'total_epochs': len(monitor.epoch_losses),
            'best_loss': monitor.best_loss,
            'final_loss': monitor.epoch_losses[-1] if monitor.epoch_losses else None,
            'converged': monitor.is_converged(),
            'early_stopped': monitor.should_stop
        }
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # 2. Save complete model (including architecture)
    complete_model_path = os.path.join(save_dir, f"gib_complete_model_{timestamp}.pth")
    torch.save(model, complete_model_path)
    print(f"Complete model saved to: {complete_model_path}")
    
    # 3. Save training metrics
    metrics_path = os.path.join(save_dir, f"training_metrics_{timestamp}.json")
    monitor.save_metrics(metrics_path)
    
    # 4. Save training curves
    curves_path = os.path.join(save_dir, f"training_curves_{timestamp}.png")
    monitor.plot_training_curves(curves_path)
    
    # 5. Save model summary
    summary_path = os.path.join(save_dir, f"model_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"GIB Error Detection Model Summary\n")
        f.write(f"================================\n\n")
        f.write(f"Training Timestamp: {timestamp}\n")
        f.write(f"Total Epochs: {len(monitor.epoch_losses)}\n")
        f.write(f"Best Loss: {monitor.best_loss:.6f}\n")
        f.write(f"Final Loss: {monitor.epoch_losses[-1]:.6f}\n")
        f.write(f"Converged: {monitor.is_converged()}\n")
        f.write(f"Early Stopped: {monitor.should_stop}\n\n")
        
        f.write(f"Model Configuration:\n")
        f.write(f"- Embedding Dimension: {model.embedding_dim}\n")
        f.write(f"- Hidden Dimension: {model.hidden_dim}\n")
        f.write(f"- Node Threshold: {model.mcs_finder.node_threshold}\n")
        f.write(f"- Edge Threshold: {model.mcs_finder.edge_threshold}\n\n")
        
        if monitor.edge_mask_stats['mean']:
            f.write(f"Final Edge Mask Statistics:\n")
            f.write(f"- Mean: {monitor.edge_mask_stats['mean'][-1]:.4f}\n")
            f.write(f"- Std: {monitor.edge_mask_stats['std'][-1]:.4f}\n")
            f.write(f"- Sparsity: {monitor.edge_mask_stats['sparsity'][-1]:.4f}\n")
    
    print(f"Model summary saved to: {summary_path}")
    
    return {
        'model_path': model_path,
        'complete_model_path': complete_model_path,
        'metrics_path': metrics_path,
        'curves_path': curves_path,
        'summary_path': summary_path
    }

def enhanced_training_loop_with_saving(pkl_path, json_path, save_dir, epochs=1000, device='cuda'):
    """Enhanced training loop with model saving"""

    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    with open(pkl_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    eval_results = load_evaluation_results(json_path)
    all_problems = list(graph_data.keys())
    
    # Take first 100 problems that have evaluation results
    available_problems = []
    for prob in all_problems:
        if prob in eval_results:
            available_problems.append(prob)
        if len(available_problems) >= 100:
            break
    
    # Initialize MCS cache
    mcs_cache = MCSCache(os.path.join(save_dir, "mcs_cache.json"))
    
    if len(available_problems) == 0:
        print("No matching problems found!")
        return None, None, None
    
    print(f"Found {len(available_problems)} problems with evaluation data")
    
    # Initialize model and monitor
    model = GIBErrorDetectionModel(embedding_dim=768, hidden_dim=128, mcs_cache=mcs_cache)
    trainer = GIBTrainer(model, learning_rate=0.001)
    monitor = ConvergenceMonitor(patience=100, min_delta=1e-4)

    print(f"Starting training with {len(available_problems)} problems...")
    
    # Training loop
    max_epochs = epochs
    best_model_state = None
    for epoch in range(max_epochs):
        epoch_loss_sum = 0
        epoch_loss_components = defaultdict(list)
        epoch_edge_masks = []
        total_graphs = 0
        
        for problem_id in available_problems:
            problem_data = graph_data[problem_id]
            correct_graphs, error_graphs = separate_correct_error_graphs(
                problem_data, problem_id, eval_results
            )
            # Train on each error graph
            for error_graph_id, error_graph_data in error_graphs:
                if len(correct_graphs) > 0:
                    try:
                        results = trainer.train_step(
                            error_graph_data, correct_graphs, 
                            error_graph_id, problem_id, eval_results
                        )
                        
                        epoch_loss_sum += results['total_loss']
                        total_graphs += 1
                        
                        # Collect loss components
                        for component, value in results['losses'].items():
                            epoch_loss_components[component].append(value)
                        
                        # Collect edge masks
                        if len(results['edge_masks']) > 0:
                            epoch_edge_masks.append(results['edge_masks'])
                        
                    except Exception as e:
                        print(f"Error training graph {error_graph_id}: {e}")
                        continue
        
        # Calculate epoch averages
        if total_graphs > 0:
            avg_epoch_loss = epoch_loss_sum / total_graphs
            avg_loss_components = {
                component: np.mean(values) 
                for component, values in epoch_loss_components.items()
            }
            
            # Update monitor
            monitor.update(epoch, avg_epoch_loss, avg_loss_components, epoch_edge_masks)
            
            # Save best model state
            if avg_epoch_loss <= monitor.best_loss:
                best_model_state = model.state_dict().copy()
                print(f"New best model at epoch {epoch} with loss {avg_epoch_loss:.6f}")
            
            # Print status (every 5 epochs)
            if epoch % 5 == 0 or monitor.should_stop or monitor.is_converged():
                monitor.print_status(epoch)
            
            # Intermediate checkpoint saving (every 20 epochs)
            if epoch % 20 == 0 and epoch > 0:
                checkpoint_dir = os.path.join(save_dir, "intermediate_checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'monitor': monitor
                }, f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth")
                print(f"Intermediate checkpoint saved at epoch {epoch}")
            
            # Check if should stop
            if monitor.should_stop:
                print(f"\nEarly stopping at epoch {epoch}!")
                break
                
            if monitor.is_converged():
                print(f"\nConverged at epoch {epoch}!")
                break
        else:
            print(f"Epoch {epoch}: No valid training data")
    
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model state")
    
    # Post-training analysis
    print(f"\n{'='*80}")
    print("Training Completed!")
    print(f"Total epochs: {len(monitor.epoch_losses)}")
    print(f"Best loss: {monitor.best_loss:.6f}")
    print(f"Final loss: {monitor.epoch_losses[-1]:.6f}")
    print(f"Converged: {monitor.is_converged()}")
    
    # Save model and results
    print("\nSaving model and results...")

    saved_paths = save_model_and_results(model, monitor, save_dir=save_dir)

    return model, monitor, saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Train the GIBS error-detection model."
    )
    parser.add_argument(
        "--embedding-file",
        required=True,
        help="Path to bert_embeddings_*.pkl (graph data with embeddings).",
    )
    parser.add_argument(
        "--label-file",
        required=True,
        help="Path to evaluation.json (GPT correctness labels).",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Directory to save model checkpoints and training metrics.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run training on (e.g. 'cuda' or 'cpu').",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    enhanced_training_loop_with_saving(
        pkl_path=args.embedding_file,
        json_path=args.label_file,
        save_dir=args.save_dir,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
    model, monitor, saved_paths = enhanced_training_loop_with_saving()