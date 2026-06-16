"""
Stage 1 — Generate LLM reasoning traces (ReasoningGraph objects) with vLLM.

For every question in the source dataset this script samples ``--num-responses``
reasoning graphs from the chosen base model and stores them (with token-level
logprobs) in a pickle file.

The dataset and the model are passed as parameters, so a single file covers
morehopqa / gsm8k / math and llama / deepseek / phi4. Output paths are relative
to ``--output-root`` (default: ``output``) following the layout in
``dataset_config.py``.

Examples
--------
    # HotpotQA (MoreHopQA) with Llama-3.1-8B, 10 questions, 5 samples each
    python generate_responds_vllm.py --dataset morehopqa --model llama \
        --num-responses 5 --max-questions 10

    # GSM8K with Phi-4
    python generate_responds_vllm.py --dataset gsm8k --model phi4

Requires a CUDA GPU and the vLLM package.
"""

import os
import sys
import json
import pickle
import time
import argparse
from threading import Lock
from typing import Dict, List, Union

import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_config import (
    get_dataset_config, get_model_id, responses_path, run_dir, load_source_dataset,
)

# Default generation configuration (overridable on the command line)
NUM_RESPONSES = 20
BATCH_SIZE_QUESTIONS = 2

# Save lock for thread safety
save_lock = Lock()


# ============================================================================
# Models
# ============================================================================
class BaseModel:
    """Base class for all models"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        print("This may take a few minutes...")

    def generate_with_logprobs(self, prompts, temperature=0.8, max_tokens=1024, top_p=0.95):
        raise NotImplementedError("Must be implemented by subclass")


def _extract_results(outputs):
    results = []
    for output in outputs:
        output_obj = output.outputs[0]
        logprobs = []
        if hasattr(output_obj, 'logprobs') and output_obj.logprobs:
            for logprob_data in output_obj.logprobs:
                if isinstance(logprob_data, dict):
                    token_logprob = list(logprob_data.values())[0] if logprob_data else 0.0
                    logprobs.append(token_logprob)
                else:
                    logprobs.append(0.0)
        results.append({
            "response": output_obj.text,
            "token_ids": output_obj.token_ids if hasattr(output_obj, 'token_ids') else [],
            "logprobs": logprobs,
        })
    return results


class LlamaModel(BaseModel):
    """Llama 3.1 8B Model"""
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        super().__init__(model_name)
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=2048,
            gpu_memory_utilization=0.95,
            disable_log_stats=True,
        )
        print("Model loaded successfully!")

    def generate_with_logprobs(self, prompts, temperature=0.8, max_tokens=1024, top_p=0.95):
        if isinstance(prompts, str):
            prompts = [prompts]
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, top_k=50, max_tokens=max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>"], logprobs=1,
        )
        return _extract_results(self.llm.generate(prompts, sampling_params))


class DeepSeekModel(BaseModel):
    """DeepSeek R1 Distill Model"""
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"):
        super().__init__(model_name)
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=2048,
            gpu_memory_utilization=0.98,
            disable_log_stats=True,
        )
        print("Model loaded successfully!")

    def generate_with_logprobs(self, prompts, temperature=0.8, max_tokens=1024, top_p=0.95):
        if isinstance(prompts, str):
            prompts = [prompts]
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, top_k=50, max_tokens=max_tokens,
            stop=["</s>", "Human:", "User:"], logprobs=1,
        )
        return _extract_results(self.llm.generate(prompts, sampling_params))


class Phi4Model(BaseModel):
    """Phi-4 Reasoning Model"""
    def __init__(self, model_name="microsoft/Phi-4-reasoning"):
        super().__init__(model_name)
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=2048,
            gpu_memory_utilization=0.98,
        )
        print("Model loaded successfully!")

    def generate_with_logprobs(self, prompts, temperature=0.8, max_tokens=2048, top_p=0.95):
        if isinstance(prompts, str):
            prompts = [prompts]
        # Phi-4 uses ChatML format with a special system prompt
        system_prompt = ("<|im_start|>system<|im_sep|> You are Phi, a language model trained by "
                         "Microsoft to help users. Your role as an assistant involves thoroughly "
                         "exploring questions through a systematic thinking process before providing "
                         "the final precise and accurate solutions.<|im_end|>")
        formatted_prompts = [
            f"{system_prompt}\n<|im_start|>user<|im_sep|>{p}<|im_end|>\n<|im_start|>assistant<|im_sep|>"
            for p in prompts
        ]
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, top_k=50, max_tokens=max_tokens,
            stop=["<|im_end|>"], logprobs=1,
        )
        return _extract_results(self.llm.generate(formatted_prompts, sampling_params))


MODEL_CLASSES = {
    "llama": LlamaModel,
    "deepseek": DeepSeekModel,
    "phi4": Phi4Model,
}


# ============================================================================
# Helpers
# ============================================================================
def check_gpu_availability():
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")


def save_results_sync(results: Dict, path: str):
    with save_lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(results, f)


def save_progress_tracker(processed_ids: set, path: str):
    with save_lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(processed_ids, f)


def load_progress_tracker(path: str) -> set:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return set()


def load_existing_results(temp_dir: str, model: str, dataset: str):
    """Load the latest checkpoint (for resuming) from ``temp_dir`` if present."""
    progress_file = os.path.join(temp_dir, f'{model}_{dataset}_processed_ids.pkl')
    processed_ids = load_progress_tracker(progress_file)

    if not os.path.exists(temp_dir):
        return {}, processed_ids

    prefix = f'{model}_{dataset}_responses_checkpoint_'
    checkpoint_files = [f for f in os.listdir(temp_dir)
                        if f.startswith(prefix) and f.endswith('.pkl')]
    if not checkpoint_files:
        return {}, processed_ids

    checkpoint_numbers = []
    for f in checkpoint_files:
        try:
            checkpoint_numbers.append(int(f.split('_checkpoint_')[1].split('.pkl')[0]))
        except Exception:
            continue
    if not checkpoint_numbers:
        return {}, processed_ids

    latest = max(checkpoint_numbers)
    checkpoint_path = os.path.join(temp_dir, f'{prefix}{latest}.pkl')
    print(f"Loading checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        results = pickle.load(f)
    processed_ids = set(results.keys())
    print(f"Found existing results: {len(processed_ids)} questions completed")
    return results, processed_ids


def process_dataset(data: List[Dict], model: BaseModel, model_key: str, dataset: str,
                    output_root: str, num_responses: int, batch_size: int,
                    prompt_template: str, use_context: bool):
    """Sample ``num_responses`` reasoning graphs per question, with checkpointing."""
    out_dir = run_dir(output_root, dataset, model_key)
    temp_dir = os.path.join(out_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    existing_results, processed_ids = load_existing_results(temp_dir, model_key, dataset)
    results = existing_results.copy()

    remaining_data = [item for item in data if item['_id'] not in processed_ids]
    print(f"Total questions: {len(data)}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining to process: {len(remaining_data)}")
    if len(remaining_data) == 0:
        print("All questions have been processed!")
        return results

    total_batches = (len(remaining_data) + batch_size - 1) // batch_size
    total_processed = len(processed_ids)

    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(remaining_data))
        batch_data = remaining_data[start_idx:end_idx]

        batch_prompts, batch_info = [], []
        for example in batch_data:
            data_id = example['_id']
            if use_context:
                context_str = (str(example['context'])
                               if isinstance(example.get('context'), list)
                               else example.get('context', ''))
                prompt = prompt_template.format(question=example['question'], context=context_str)
            else:
                prompt = prompt_template.format(question=example['question'])
            for resp_idx in range(num_responses):
                batch_prompts.append(prompt)
                batch_info.append((data_id, example, resp_idx))

        try:
            batch_outputs = model.generate_with_logprobs(batch_prompts)
            for i, (data_id, example, resp_idx) in enumerate(batch_info):
                key = data_id
                if key not in results:
                    results[key] = {
                        'id': data_id,
                        'question': example['question'],
                        'context': example.get('context', ''),
                        'answer': example.get('answer', 'N/A'),
                        'response_ids': [],
                        'responses': [],
                        'token_ids': [],
                        'logprobs': [],
                    }
                    processed_ids.add(data_id)
                output = batch_outputs[i]
                results[key]['responses'].append(output['response'])
                results[key]['token_ids'].append(output['token_ids'])
                results[key]['logprobs'].append(output['logprobs'])
                results[key]['response_ids'].append(resp_idx)

            total_processed += len(batch_data)

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                checkpoint_path = os.path.join(
                    temp_dir, f'{model_key}_{dataset}_responses_checkpoint_{total_processed}.pkl')
                save_results_sync(results, checkpoint_path)
                save_progress_tracker(
                    processed_ids, os.path.join(temp_dir, f'{model_key}_{dataset}_processed_ids.pkl'))
                print(f"\nSaved checkpoint: {total_processed} total questions processed")
        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate reasoning traces with vLLM.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['morehopqa', 'gsm8k', 'math'],
                        help='Which dataset to use.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['llama', 'deepseek', 'phi4'],
                        help='Which base model to sample from.')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Source JSON file. Defaults to the per-dataset path in dataset_config.py.')
    parser.add_argument('--output-root', type=str, default='output',
                        help='Root directory for all generated artifacts (default: output).')
    parser.add_argument('--num-responses', type=int, default=NUM_RESPONSES,
                        help='Number of sampled responses per question (default: 20).')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE_QUESTIONS,
                        help='Questions processed per vLLM batch (default: 2).')
    parser.add_argument('--max-questions', type=int, default=None,
                        help='Only process the first N questions (handy for demos).')
    args = parser.parse_args()

    cfg = get_dataset_config(args.dataset)
    model_id = get_model_id(args.model)
    source_path = args.dataset_path or cfg['default_source']

    check_gpu_availability()

    print(f"Loading {args.dataset} dataset from: {source_path}")
    source_data = load_source_dataset(source_path, args.dataset)
    print(f"Loaded {len(source_data)} questions")

    if args.max_questions is not None:
        source_data = source_data[:args.max_questions]
        print(f"Limited to first {len(source_data)} questions")

    model = MODEL_CLASSES[args.model](model_name=model_id)

    print(f"Generating {args.num_responses} responses/question with model '{args.model}'")
    start_time = time.time()
    results = process_dataset(
        source_data, model, args.model, args.dataset, args.output_root,
        args.num_responses, args.batch_size, cfg['prompt'], cfg['use_context'],
    )

    final_output_path = responses_path(args.output_root, args.dataset, args.model)
    save_results_sync(results, final_output_path)

    total_time = time.time() - start_time
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    print(f"Saved all results to {final_output_path}")
    total_responses = sum(len(r['responses']) for r in results.values())
    print(f"Total responses generated: {total_responses}; questions: {len(results)}")


if __name__ == "__main__":
    main()
