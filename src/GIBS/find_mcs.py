import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import networkx as nx


class MaximumCommonSubgraph:
    """Find maximum common subgraph between two graphs with NLI entailment similarity thresholds."""
    
    def __init__(self, node_threshold=0.85, edge_threshold=0.8, model_name="microsoft/deberta-large-mnli", device='cuda:0'):
        """
        Initialize with similarity thresholds and NLI model.
        
        Args:
            node_threshold: Minimum entailment score for nodes to be considered identical (default: 0.85)
            edge_threshold: Minimum entailment score for edges to be considered identical (default: 0.8)
            model_name: Name of the NLI model to use (default: microsoft/deberta-large-mnli)
            device: Device to use for model (default: cuda:0)
        """
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold
        
        # Initialize NLI model and tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print(f"Initialized NLI model: {model_name} on {self.device}")
    
    def get_text_from_node(self, graph, node_id, reasoning_string):
        """Extract text content for a node from reasoning_string."""
        # Map node IDs to the keys in reasoning_string
        
        if node_id == "NodeResult":
            return reasoning_string.get("NodeResult", str(node_id))
        elif node_id == "NodeRaw":
            # NodeRaw might represent the problem statement or initial state
            # Check if there's a specific key for it, otherwise use a descriptive text
            return reasoning_string.get("NodeRaw", "Problem/Initial State")
        else:
            # For regular nodes like Node0, Node1, etc.
            if node_id in reasoning_string:
                return reasoning_string[node_id]
            
            # If node_id is like "Node2", directly look it up
            if node_id.startswith("Node") and node_id in reasoning_string:
                return reasoning_string[node_id]
            
            # If node_id is numeric like "2", try "Node2"
            if node_id.isdigit():
                node_key = f"Node{node_id}"
                if node_key in reasoning_string:
                    return reasoning_string[node_key]
            
            # Default fallback
            return str(node_id)
    
    def get_text_from_edge(self, graph, edge, reasoning_string):
        """Extract text content for an edge from reasoning_string."""
        u, v = edge
        
        # Get the edge label from the graph if available
        edge_label = None
        if hasattr(graph, 'graph') and 'edge_labels' in graph.graph:
            edge_label = graph.graph['edge_labels'].get((u, v))
        elif (u, v) in graph.edges:
            edge_data = graph.get_edge_data(u, v)
            if edge_data and 'label' in edge_data:
                edge_label = edge_data['label']
        
        # If we have an edge label (like Edge0, Edge1, etc.), use it to look up the text
        if edge_label and edge_label in reasoning_string:
            return reasoning_string[edge_label]
        
        # Fallback methods for special cases
        if v == "NodeResult" or v == "Result":
            return reasoning_string.get("ResultEdge", f"{u} -> {v}")
        
        # If no edge label found, try to guess based on target node
        # This is less reliable but provides a fallback
        target_num = None
        if v.startswith("Node") and v != "NodeRaw" and v != "NodeResult":
            target_num = v.replace("Node", "")
        elif v.isdigit():
            target_num = v
        
        if target_num is not None:
            # Try different edge key patterns
            for edge_key in [f"Edge{target_num}", f"Edge{int(target_num)}"]:
                if edge_key in reasoning_string:
                    return reasoning_string[edge_key]
        
        # Default fallback
        return f"{u} -> {v}"
    
    @torch.no_grad()
    def _pred(self, sen_1, sen_2):
        """
        Get NLI predictions for a pair of sentences.
        Returns logits: [Contradiction, Neutral, Entailment]
        """
        # Combine sentences with [SEP] token
        input_text = sen_1 + ' [SEP] ' + sen_2
        
        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        return logits
    
    def calculate_entailment(self, text1, text2):
        """
        Calculate entailment score between two texts using NLI.
        Returns the probability that text1 entails text2.
        """
        try:
            # Get predictions in both directions
            logits_1 = self._pred(text1, text2)
            logits_2 = self._pred(text2, text1)
            
            # Convert to probabilities
            probs_1 = torch.softmax(logits_1, dim=1)
            probs_2 = torch.softmax(logits_2, dim=1)
            
            # Extract entailment probabilities (index 2)
            # logits order: [Contradiction, Neutral, Entailment]
            entailment_1 = probs_1[0, 2].item()
            entailment_2 = probs_2[0, 2].item()
            
            # Return maximum of both directions for symmetric similarity
            return max(entailment_1, entailment_2)
            
        except Exception as e:
            print(f"Error calculating entailment: {e}")
            return 0.0
    
    def node_similarity(self, G1, G2, n1, n2, reasoning1, reasoning2):
        """Calculate entailment-based similarity between two nodes."""
        text1 = self.get_text_from_node(G1, n1, reasoning1)
        text2 = self.get_text_from_node(G2, n2, reasoning2)
        return self.calculate_entailment(text1, text2)
    
    def edge_similarity(self, G1, G2, e1, e2, reasoning1, reasoning2):
        """Calculate entailment-based similarity between two edges."""
        text1 = self.get_text_from_edge(G1, e1, reasoning1)
        text2 = self.get_text_from_edge(G2, e2, reasoning2)
        return self.calculate_entailment(text1, text2)
    
    def nodes_match(self, G1, G2, n1, n2, reasoning1, reasoning2):
        """Check if two nodes match based on entailment threshold."""
        return self.node_similarity(G1, G2, n1, n2, reasoning1, reasoning2) >= self.node_threshold
    
    def edges_match(self, G1, G2, e1, e2, reasoning1, reasoning2):
        """Check if two edges match based on entailment threshold."""
        return self.edge_similarity(G1, G2, e1, e2, reasoning1, reasoning2) >= self.edge_threshold
    
    def find_node_mappings(self, G1, G2, reasoning1, reasoning2):
        """Find all possible node mappings between two graphs based on entailment."""
        mappings = []
        nodes1 = list(G1.nodes())
        nodes2 = list(G2.nodes())
        
        # For each node in G1, find matching nodes in G2
        for n1 in nodes1:
            matching_nodes = []
            for n2 in nodes2:
                if self.nodes_match(G1, G2, n1, n2, reasoning1, reasoning2):
                    matching_nodes.append((n1, n2, self.node_similarity(G1, G2, n1, n2, reasoning1, reasoning2)))
            mappings.append(matching_nodes)
        
        return mappings
    
    def is_valid_mapping(self, G1, G2, mapping, reasoning1, reasoning2):
        """
        Check if a node mapping preserves edge structure with entailment constraints.
        
        Args:
            mapping: Dictionary {node_in_G1: node_in_G2}
        """
        # Check all edges in the induced subgraph of G1
        mapped_nodes_G1 = list(mapping.keys())
        subgraph_G1 = G1.subgraph(mapped_nodes_G1)
        
        for u, v in subgraph_G1.edges():
            # Check if corresponding edge exists in G2
            u_mapped = mapping[u]
            v_mapped = mapping[v]
            
            # For directed graphs
            if G2.has_edge(u_mapped, v_mapped):
                # Check edge similarity
                if not self.edges_match(G1, G2, (u, v), (u_mapped, v_mapped), reasoning1, reasoning2):
                    return False
            else:
                return False
        
        return True
    def find_maximum_common_subgraph(self, G1, G2, reasoning1, reasoning2):
        """
        Find the maximum common subgraph using edge-first approach with BFS expansion.

        Returns:
            tuple: (common_subgraph, node_mapping, edge_mapping)
        """
        from collections import deque

        # Step 1: Generate candidate edge pairs
        candidate_edge_pairs = []
        #print("Generating candidate edge pairs...")

        for e1 in G1.edges():
            u1, v1 = e1
            for e2 in G2.edges():
                u2, v2 = e2
                
                # Check if edges match and target nodes match
                if self.edges_match(G1, G2, e1, e2, reasoning1, reasoning2) and \
                    self.nodes_match(G1, G2, v1, v2, reasoning1, reasoning2):
                    
                    edge_sim = self.edge_similarity(G1, G2, e1, e2, reasoning1, reasoning2)
                    node_sim = self.node_similarity(G1, G2, v1, v2, reasoning1, reasoning2)
                    score = edge_sim + node_sim
                    
                    candidate_edge_pairs.append((e1, e2, score))

        # Sort by score (descending) and keep top half
        # candidate_edge_pairs.sort(key=lambda x: x[2], reverse=True)
        # max_candidates = max(len(candidate_edge_pairs) // 2, 1)
        # candidate_edge_pairs = candidate_edge_pairs[:max_candidates]
        candidate_edge_pairs.sort(key=lambda x: x[2], reverse=True)
        if len(candidate_edge_pairs) > 10:
            candidate_edge_pairs = candidate_edge_pairs[:10]

        #print(f"Found {len(candidate_edge_pairs)} candidate edge pairs")

        # Create a lookup for quick access
        edge_pair_dict = {e1: [] for e1, _, _ in candidate_edge_pairs}
        for e1, e2, score in candidate_edge_pairs:
            edge_pair_dict[e1].append((e2, score))

        # Step 2: Iterative growth of connected subgraphs
        best_mcs = nx.DiGraph()
        best_node_mapping = {}
        best_edge_mapping = {}
        best_size = 0

        # Early stopping threshold
        max_possible_size = min(len(G1.edges()), len(G2.edges()))

        for seed_e1, seed_e2, seed_score in candidate_edge_pairs:
            # if best_size >= max_possible_size:
            #     print(f"Found MCS with maximum possible size {best_size}, stopping early")
            #     break
                
            # Initialize current MCS with seed
            current_mcs = nx.DiGraph()
            current_node_mapping = {}
            current_edge_mapping = {}
            
            # Add seed edge
            u1, v1 = seed_e1
            u2, v2 = seed_e2
            
            current_mcs.add_edge(u1, v1)
            current_node_mapping[u1] = u2
            current_node_mapping[v1] = v2
            current_edge_mapping[seed_e1] = seed_e2
            
            # BFS expansion
            queue = deque()
            visited_edges = {seed_e1}
            
            # Add neighboring edges to queue
            # Edges sharing source node
            for _, out_v in G1.out_edges(u1):
                if (u1, out_v) not in visited_edges:
                    queue.append((u1, out_v))
                    visited_edges.add((u1, out_v))
            for _, out_v in G1.out_edges(v1):
                if (v1, out_v) not in visited_edges:
                    queue.append((v1, out_v))
                    visited_edges.add((v1, out_v))
                    
            # Edges sharing target node  
            for in_u, _ in G1.in_edges(u1):
                if (in_u, u1) not in visited_edges:
                    queue.append((in_u, u1))
                    visited_edges.add((in_u, u1))
            for in_u, _ in G1.in_edges(v1):
                if (in_u, v1) not in visited_edges:
                    queue.append((in_u, v1))
                    visited_edges.add((in_u, v1))
            
            # BFS expansion loop
            while queue:
                next_e1 = queue.popleft()
                x1, y1 = next_e1
                
                # Skip if not in candidates
                if next_e1 not in edge_pair_dict:
                    continue
                    
                # Find best compatible match
                best_match = None
                best_match_score = -1
                
                for next_e2, score in edge_pair_dict[next_e1]:
                    x2, y2 = next_e2
                    
                    # Compatibility check
                    compatible = True
                    
                    # Check source node mapping
                    if x1 in current_node_mapping:
                        if current_node_mapping[x1] != x2:
                            compatible = False
                    else:
                        # x1 not mapped yet, check if x2 is already a target
                        if x2 in current_node_mapping.values():
                            compatible = False
                            
                    # Check target node mapping
                    if compatible and y1 in current_node_mapping:
                        if current_node_mapping[y1] != y2:
                            compatible = False
                    elif compatible:
                        # y1 not mapped yet, check if y2 is already a target
                        if y2 in current_node_mapping.values():
                            compatible = False
                    
                    if compatible and score > best_match_score:
                        best_match = next_e2
                        best_match_score = score
                
                # If found compatible match, add to current MCS
                if best_match:
                    x2, y2 = best_match
                    
                    # Add edge to MCS
                    current_mcs.add_edge(x1, y1)
                    current_edge_mapping[next_e1] = best_match
                    
                    # Update node mappings
                    if x1 not in current_node_mapping:
                        current_node_mapping[x1] = x2
                    if y1 not in current_node_mapping:
                        current_node_mapping[y1] = y2
                    
                    # Add new neighboring edges to queue
                    # Edges from newly added nodes
                    for node in [x1, y1]:
                        if node in current_mcs:
                            # Outgoing edges
                            for _, out_v in G1.out_edges(node):
                                if (node, out_v) not in visited_edges:
                                    queue.append((node, out_v))
                                    visited_edges.add((node, out_v))
                            # Incoming edges
                            for in_u, _ in G1.in_edges(node):
                                if (in_u, node) not in visited_edges:
                                    queue.append((in_u, node))
                                    visited_edges.add((in_u, node))
            
            # Calculate current MCS size
            current_size = len(current_mcs.edges()) + len(current_mcs.nodes())
            
            if current_size > best_size:
                best_mcs = current_mcs.copy()
                best_node_mapping = current_node_mapping.copy()
                best_edge_mapping = current_edge_mapping.copy()
                best_size = current_size

        # Step 3: Build final common subgraph with similarity scores
        common_subgraph = nx.DiGraph()

        # Add nodes with similarity scores
        for n1, n2 in best_node_mapping.items():
            common_subgraph.add_node(n1,
                                    matched_in_G2=n2,
                                    similarity=self.node_similarity(G1, G2, n1, n2, reasoning1, reasoning2))

        # Add edges with similarity scores
        for e1, e2 in best_edge_mapping.items():
            u1, v1 = e1
            u2, v2 = e2
            common_subgraph.add_edge(u1, v1,
                                    matched_in_G2=e2,
                                    similarity=self.edge_similarity(G1, G2, e1, e2, reasoning1, reasoning2))

        #print(f"\nMCS found with {len(common_subgraph.nodes())} nodes and {len(common_subgraph.edges())} edges")

        return common_subgraph, best_node_mapping, best_edge_mapping
        