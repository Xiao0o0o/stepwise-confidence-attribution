"""
Stage 2 — Transform raw ReasoningGraph text into structured graphs.

Parses each sampled response into nodes/edges, aligns token-level logprobs to
each step, and emits a per-step ``string_q_a`` plus a ``structure_retrieve``
string. Dataset/model are parameters; paths follow ``dataset_config.py``.

Example:
    python tranformed_graph.py --dataset morehopqa --model llama
"""

import os
import sys
import pickle
import re
import math
import ast
import operator
import argparse
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_config import responses_path, transformed_path

def logprob_to_prob(logprob):
    if logprob is None:
        return None
    return math.exp(logprob)  

def safe_eval_math(expression_str):
    if not isinstance(expression_str, str):
        return expression_str
    if not re.search(r'\d', expression_str):
        return expression_str
    
    keywords_to_remove = ['lambda:', 'int', 'float', 'str', 'round']
    
    for keyword in keywords_to_remove:
        expression_str = expression_str.replace(keyword, '')
    
    expression_str = ''.join(expression_str.split())
    
    try:
        return float(expression_str)
    except ValueError:
        pass

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  
            return node.n
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            return operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            return operators[type(node.op)](operand)
        elif isinstance(node, ast.Expression):
            return eval_node(node.body)
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
    
    try:
        tree = ast.parse(expression_str, mode='eval')
        result = eval_node(tree)
        return result
    except Exception as e:
        try:
            allowed_names = {
                "__builtins__": {},
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
            }
            result = eval(expression_str, allowed_names, {})
            return result
        except:
            return expression_str

def build_token_position_map(logprobs):
    if not logprobs:
        return {}, []
    
    
    full_text = ""
    char_to_token = {} 
    token_texts = []
    
    for i, logprob_obj in enumerate(logprobs):
        if hasattr(logprob_obj, 'decoded_token'):
            token_text = logprob_obj.decoded_token
        else:
            continue
            
        start_pos = len(full_text)
        full_text += token_text
        end_pos = len(full_text)
    
        for pos in range(start_pos, end_pos):
            char_to_token[pos] = i
        
        token_texts.append(token_text)
    
    return char_to_token, token_texts, full_text

def extract_token_info_v2(text, logprobs, start_pos, end_pos):
    #print(f'extract_token_info: text[{start_pos}:{end_pos}] = "{text[start_pos:end_pos] if start_pos and end_pos else "None"}"')
    
    if start_pos is None or end_pos is None:
        return [], []
    
    if not logprobs:
        return [], []
    
    char_to_token, token_texts, full_text = build_token_position_map(logprobs)
    
    if full_text != text:
        print(f"Warning: Reconstructed text length {len(full_text)} does not match original text length {len(text)}")
        target_substr = text[start_pos:end_pos]
        if target_substr in full_text:
            new_start = full_text.find(target_substr)
            if new_start != -1:
                start_pos = new_start
                end_pos = new_start + len(target_substr)
                #print(f"Found substring in reconstructed text at position {new_start}")
    
    token_indices = set()
    for pos in range(start_pos, min(end_pos, len(full_text))):
        if pos in char_to_token:
            token_indices.add(char_to_token[pos])

    token_indices = sorted(list(token_indices))
    
    token_logprobs = []
    token_probs = []
    
    for idx in token_indices:
        if idx < len(logprobs):
            logprob_obj = logprobs[idx]
            if hasattr(logprob_obj, 'logprob'):
                lp = logprob_obj.logprob
                token_logprobs.append(lp)
                token_probs.append(logprob_to_prob(lp))
            else:
                token_logprobs.append(None)
                token_probs.append(None)
    
    #print(f'Found {len(token_indices)} tokens, probs: {token_probs[:5]}...' if token_probs else 'No tokens found')
    
    return token_logprobs, token_probs

def find_all_positions(text, pattern):
    positions = []
    for match in re.finditer(pattern, text):
        positions.append((match.start(), match.end(), match.group(0)))
    return positions

def parse_reasoning_graph_bb(text, logprobs):
    nodes = []
    final_answer = None
    logprobs_info = {}
    probs_info = {}
    
    graph_positions = [m.start() for m in re.finditer(r'ReasoningGraph\(', text)]
    start_pos = graph_positions[0]
    end_pos = graph_positions[1] if len(graph_positions) > 1 else len(text)

    processing_text = text[start_pos:end_pos]

    node_pattern = r"ReasoningNode\(\s*id=(\d+),\s*description=([^,]+),\s*output=([^,]+?),\s*depends_on=\[(.*?)\]\s*\)"
    
    for node_match in re.finditer(node_pattern, text, re.DOTALL):
        node_id = int(node_match.group(1))
        description = node_match.group(2).strip().strip('\'"')
        output_str = node_match.group(3).strip().strip('\'"')

        calculated_output = safe_eval_math(output_str)
        
        depends_on_str = node_match.group(4).strip()
        depends_on = []
        if depends_on_str:
            depends_on = [int(x.strip()) for x in depends_on_str.split(',')]
        
        node_start = node_match.start()
        node_end = node_match.end()
        
        desc_pattern = rf"description=([^,]+)"
        desc_positions = find_all_positions(text[node_start:node_end], desc_pattern)

        output_pattern = rf"output=({re.escape(output_str)})"
        output_positions = find_all_positions(text[node_start:node_end], output_pattern)
        
        desc_content_start = None
        desc_content_end = None
        
        if desc_positions:
            desc_pos = desc_positions[0]
            full_desc = desc_pos[2]
            quotes_match = re.search(r'[\'"]([^\'"]*)[\'"]', full_desc)
            if quotes_match:
                quote_start = quotes_match.start(1)
                quote_end = quotes_match.end(1)
                desc_content_start = node_start + desc_pos[0] + quote_start
                desc_content_end = node_start + desc_pos[0] + quote_end
            else:
                desc_content_start = node_start + desc_pos[0] + len("description=")
                desc_content_end = node_start + desc_pos[0] + len("description=") + len(description)
            
        output_content_start = None
        output_content_end = None
        
        if output_positions:
            output_pos = output_positions[0]
            full_output = output_pos[2]
            quotes_match = re.search(r'[\'"]([^\'"]*)[\'"]', full_output)
            if quotes_match:
                quote_start = quotes_match.start(1)
                quote_end = quotes_match.end(1)
                output_content_start = node_start + output_pos[0] + quote_start
                output_content_end = node_start + output_pos[0] + quote_end
            else:
                output_content_start = node_start + output_pos[0] + len("output=")
                output_content_end = node_start + output_pos[1]
        
        desc_logprobs, desc_probs = extract_token_info_v2(
            text, logprobs, desc_content_start, desc_content_end
        )
        output_logprobs, output_probs = extract_token_info_v2(
            text, logprobs, output_content_start, output_content_end
        )
        
        # print(f'Node {node_id} - desc_probs: {desc_probs[:3] if desc_probs else "empty"}')
        # print(f'Node {node_id} - output_probs: {output_probs[:3] if output_probs else "empty"}')
        
        node_key = f"Node{node_id}"
        logprobs_info[node_key] = output_logprobs  
        probs_info[node_key] = output_probs
        logprobs_info[f"_temp_desc_{node_id}"] = desc_logprobs
        probs_info[f"_temp_desc_{node_id}"] = desc_probs
        
        nodes.append({
            'id': node_id,
            'description': description,
            'output': calculated_output,  
            'output_raw': output_str,   
            'depends_on': depends_on
        })
    
    #final_answer_pattern = r"final_answer=(.+?)(?=\s*\)\s*$)"
    final_answer_pattern = r"final_answer=(.+?)(?=\s*\))"
    final_answer_match = re.search(final_answer_pattern, text)
    
    if final_answer_match:
        final_answer_str = final_answer_match.group(1).strip().strip('\'"')
        final_answer = safe_eval_math(final_answer_str)
        
        final_start = final_answer_match.start(1)
        final_end = final_answer_match.end(1)
        
        final_logprobs, final_probs = extract_token_info_v2(
            text, logprobs, final_start, final_end
        )
        
        logprobs_info["NodeResult"] = final_logprobs
        probs_info["NodeResult"] = final_probs
    
    return {
        'nodes': nodes,
        'final_answer': final_answer,
        'logprobs_info': logprobs_info,
        'probs_info': probs_info
    }

def parse_reasoning_graph(text, logprobs):
    nodes = []
    final_answer = None
    logprobs_info = {}
    probs_info = {}
    
    graph_positions = [m.start() for m in re.finditer(r'ReasoningGraph\(', text)]
    
    if not graph_positions:
        return None
        
    start_pos = graph_positions[0]
    end_pos = graph_positions[1] if len(graph_positions) > 1 else len(text)

    processing_text = text[start_pos:end_pos]
    node_pattern = r"ReasoningNode\(\s*id=(\d+),\s*description=([^,]+),\s*output=([^,]+?),\s*depends_on=\[(.*?)\]\s*\)"
    
    for node_match in re.finditer(node_pattern, processing_text, re.DOTALL):
        node_id = int(node_match.group(1))
        description = node_match.group(2).strip().strip('\'"')
        
        output_str = node_match.group(3).strip().strip('\'"')
        
        calculated_output = safe_eval_math(output_str)
        
        depends_on_str = node_match.group(4).strip()
        depends_on = []
        if '#' in depends_on_str:
            depends_on_str = depends_on_str.split('#')[0]
        
        if '?' in depends_on_str:
            depends_on_str = depends_on_str.split('?')[0]
            
        if depends_on_str:
            depends_on = [int(x.strip()) for x in depends_on_str.split(',')]
        
        node_start = start_pos + node_match.start()  
        node_end = start_pos + node_match.end()     
        
        desc_pattern = rf"description=([^,]+)"
        desc_positions = find_all_positions(text[node_start:node_end], desc_pattern)

        output_pattern = rf"output=({re.escape(output_str)})"
        output_positions = find_all_positions(text[node_start:node_end], output_pattern)
        
        desc_content_start = None
        desc_content_end = None
        
        if desc_positions:
            desc_pos = desc_positions[0]
            full_desc = desc_pos[2]
            
            quotes_match = re.search(r'[\'"]([^\'"]*)[\'"]', full_desc)
            if quotes_match:
                quote_start = quotes_match.start(1)
                quote_end = quotes_match.end(1)
                desc_content_start = node_start + desc_pos[0] + quote_start
                desc_content_end = node_start + desc_pos[0] + quote_end
            else:
                desc_content_start = node_start + desc_pos[0] + len("description=")
                desc_content_end = node_start + desc_pos[0] + len("description=") + len(description)
                
        output_content_start = None
        output_content_end = None
        
        if output_positions:
            output_pos = output_positions[0]
            full_output = output_pos[2]
            quotes_match = re.search(r'[\'"]([^\'"]*)[\'"]', full_output)
            if quotes_match:
                quote_start = quotes_match.start(1)
                quote_end = quotes_match.end(1)
                output_content_start = node_start + output_pos[0] + quote_start
                output_content_end = node_start + output_pos[0] + quote_end
            else:
                output_content_start = node_start + output_pos[0] + len("output=")
                output_content_end = node_start + output_pos[1]

        desc_logprobs, desc_probs = extract_token_info_v2(
            text, logprobs, desc_content_start, desc_content_end
        )
        output_logprobs, output_probs = extract_token_info_v2(
            text, logprobs, output_content_start, output_content_end
        )
        
        node_key = f"Node{node_id}"
        logprobs_info[node_key] = output_logprobs  
        probs_info[node_key] = output_probs
        
        logprobs_info[f"_temp_desc_{node_id}"] = desc_logprobs
        probs_info[f"_temp_desc_{node_id}"] = desc_probs
        
        nodes.append({
            'id': node_id,
            'description': description,
            'output': calculated_output,  
            'output_raw': output_str,     
            'depends_on': depends_on
        })

    final_answer_pattern = r"final_answer=(.+?)(?=\s*\))"
    final_answer_match = re.search(final_answer_pattern, processing_text)  
    
    if final_answer_match:
        final_answer_str = final_answer_match.group(1).strip().strip('\'"')
        final_answer = safe_eval_math(final_answer_str)
        

        final_start = start_pos + final_answer_match.start(1)  
        final_end = start_pos + final_answer_match.end(1)     
        
        final_logprobs, final_probs = extract_token_info_v2(
            text, logprobs, final_start, final_end
        )
        
        logprobs_info["NodeResult"] = final_logprobs
        probs_info["NodeResult"] = final_probs
    
    return {
        'nodes': nodes,
        'final_answer': final_answer,
        'logprobs_info': logprobs_info,
        'probs_info': probs_info
    }


def process_reasoning_graph(parsed_graph):
    
    if not parsed_graph or 'nodes' not in parsed_graph or not parsed_graph['nodes']:
        return "Structure:{}", "", None, {}, {}
    
    nodes = parsed_graph['nodes']
    final_answer = parsed_graph.get('final_answer')
    logprobs_info = parsed_graph.get('logprobs_info', {}).copy()  
    probs_info = parsed_graph.get('probs_info', {}).copy()
    
    temp_desc_logprobs = {}
    temp_desc_probs = {}
    
    keys_to_remove = []
    for key in logprobs_info:
        if key.startswith('_temp_desc_'):
            node_id = key.split('_')[-1]
            temp_desc_logprobs[node_id] = logprobs_info[key]
            keys_to_remove.append(key)
    
    for key in probs_info:
        if key.startswith('_temp_desc_'):
            node_id = key.split('_')[-1]
            temp_desc_probs[node_id] = probs_info[key]
            keys_to_remove.append(key)

    for key in keys_to_remove:
        if key in logprobs_info:
            del logprobs_info[key]
        if key in probs_info:
            del probs_info[key]
    structure_retrieve_items = []
    string_q_a = []
    edge_counter = nodes[0]['id'] if nodes else 1
    
    for node in nodes:
        node_id = node['id']
        description = node['description']
        output = node['output']  
        depends_on = node['depends_on']
        
        
        if not depends_on:
            edge_id = f"Edge{edge_counter}"
            structure_retrieve_items.append(f"[NodeRaw, Node{node_id}, {edge_id}]")
            
            qa_entry = f"{edge_id}: {description}?, Node{node_id}: {output};"
            string_q_a.append(qa_entry)
            
            logprobs_info[edge_id] = temp_desc_logprobs.get(str(node_id), [])
            probs_info[edge_id] = temp_desc_probs.get(str(node_id), [])
            
            edge_counter += 1
        else:
            for dep_id in depends_on:
                edge_id = f"Edge{edge_counter}"
                structure_retrieve_items.append(f"[Node{dep_id}, Node{node_id}, {edge_id}]")
                
                qa_entry = f"{edge_id}: {description}?, Node{node_id}: {output};"
                string_q_a.append(qa_entry)
                
                logprobs_info[edge_id] = temp_desc_logprobs.get(str(node_id), [])
                probs_info[edge_id] = temp_desc_probs.get(str(node_id), [])
                
                edge_counter += 1
    
    if nodes:
        last_node = nodes[-1]
        
        structure_retrieve_items.append(f"[Node{last_node['id']}, NodeResult, ResultEdge]")
        
        qa_entry = f"ResultEdge: What is the final answer?, NodeResult: {final_answer};"
        string_q_a.append(qa_entry)
        
        logprobs_info["ResultEdge"] = [0.0]  
        probs_info["ResultEdge"] = [1.0]  

    combined_logprobs = logprobs_info
    combined_probs = probs_info
    
    # structure_retrieve
    structure_retrieve = "Structure:{" + ", ".join(structure_retrieve_items) + "}"
    
    return structure_retrieve, "\n".join(string_q_a), final_answer, combined_logprobs, combined_probs

def main():
    parser = argparse.ArgumentParser(description='Transform raw reasoning traces into graphs.')
    parser.add_argument('--dataset', required=True, choices=['morehopqa', 'gsm8k', 'math'])
    parser.add_argument('--model', required=True, choices=['llama', 'deepseek', 'phi4'])
    parser.add_argument('--output-root', default='output',
                        help='Root directory for artifacts (default: output).')
    parser.add_argument('--input', default=None,
                        help='Override input responses .pkl (default: derived from dataset/model).')
    parser.add_argument('--output', default=None,
                        help='Override output transformed .pkl (default: derived from dataset/model).')
    args = parser.parse_args()

    path_source = args.input or responses_path(args.output_root, args.dataset, args.model)
    output_path = args.output or transformed_path(args.output_root, args.dataset, args.model)

    # Load the source data
    with open(path_source, 'rb') as f:
        source_data = pickle.load(f)
    
    print('Collecting the dataset...')
    transformed_data = {}

    #for problem_key, problem_data in source_data.items():
    for problem_key, problem_data in tqdm(source_data.items()):
        question = problem_data['question']
        ans = problem_data['answer']
        responses = problem_data['responses']
        response_ids = problem_data['response_ids']
        logprobs_list = problem_data.get('logprobs', [None] * len(responses))
        
        
        transformed_data[problem_key] = []
        
        
        for i, (response_id, response) in enumerate(zip(response_ids, responses)):
            
            logprobs = logprobs_list[i] if i < len(logprobs_list) else None
            
            
            new_id = f"{problem_key}_{response_id}"
            
            
            entry = {
                'id': new_id,
                'structure_raw': response,  
                'question': question,
                'answer': ans,  
                'faith': None  
            }
            
            
            try:
                
                parsed_graph = parse_reasoning_graph(response, logprobs)
                
                
                structure_retrieve, string_q_a, final_answer, combined_logprobs, combined_probs = process_reasoning_graph(parsed_graph)
                entry['structure_retrieve'] = structure_retrieve
                entry['string_q_a'] = string_q_a
                
                entry['logprobs'] = combined_logprobs
                entry['probs'] = combined_probs
                
                
                entry['response_str'] = f"final_answer= {final_answer}"
            except Exception as e:
                
                print(f"Error processing response {new_id}: {e}")
                import traceback
                traceback.print_exc()
                entry['structure_retrieve'] = "Structure:{}"
                entry['string_q_a'] = ""
                entry['response_str'] = str(response)
                entry['logprobs'] = {}
                entry['probs'] = {}
            
            
            transformed_data[problem_key].append(entry)

    print("Converting data format...")
    transformed_data_dict_format = {}
    for problem_key, response_list in transformed_data.items():
        problem_dict = {}
        for response in response_list:
            graph_id = response['id']
            problem_dict[graph_id] = response
        transformed_data_dict_format[problem_key] = problem_dict

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(transformed_data_dict_format, f)

    print(f"Transformation complete. Data saved to {output_path}")

    
    total_problems = len(transformed_data)
    total_responses = sum(len(responses) for responses in transformed_data.values())
    print(f"Processed {total_problems} problems with {total_responses} total responses")

    
    if transformed_data:
        first_problem = next(iter(transformed_data.values()))
        if first_problem:
            print("\nExample entry:")
            example = first_problem[0]
            print(f"ID: {example['id']}")
            print(f"Question: {example['question'][:100]}...")
            print(f"Structure retrieve: {example['structure_retrieve'][:200]}...")
            print(f"Response: {example['response_str']}")
        
            
            if example.get('probs'):
                print(f"Probs keys: {list(example['probs'].keys())[:5]}")
                for key in list(example['probs'].keys())[:2]:
                    probs = example['probs'][key]
                    print(f"  {key}: {probs[:3] if len(probs) > 3 else probs}")

if __name__ == "__main__":
    main()