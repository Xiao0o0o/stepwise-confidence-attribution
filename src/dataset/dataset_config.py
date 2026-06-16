"""
Single source of truth for dataset / model configuration and the canonical
file layout used by the whole pipeline.

Every stage (generation -> transform -> bert -> labeling -> methods ->
evaluation) derives its input/output file paths from the helpers below, so the
naming convention only ever lives in one place.

Directory layout (all relative to the repository root by default):

    <output_root>/<dataset>/<model>/
        <model>_<dataset>_responses.pkl                  # stage 1 (generate)
        transformed_<model>_<dataset>_with_probs.pkl     # stage 2 (transform)
        bert_embeddings_<model>_<dataset>_with_probs.pkl # stage 3 (bert)
        evaluation.json                                  # stage 4 (gpt labels)
        NIBS_results.json                                # NIBS method
        GIBS_results.json                                # GIBS inference
        white_box_baseline_<model>.json                  # white-box baseline
        ptrue_<model>_<dataset>.pkl                      # P(True) baseline
        gib_checkpoints/                                 # GIBS training output

`dataset` is one of: morehopqa, gsm8k, math.
`model`   is one of: llama, deepseek, phi4.
"""

import os
import re
import json


# ----------------------------------------------------------------------------
# Models: short name -> HuggingFace model id used by vLLM / the baselines
# ----------------------------------------------------------------------------
MODELS = {
    "llama":    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "phi4":     "microsoft/Phi-4-reasoning",
}

# white_box_baseline.py / p_true.py use these short keys for --model-name
WHITEBOX_MODEL_KEY = {
    "llama":    "llama3",
    "deepseek": "deepseek",
    "phi4":     "phi4",
}


# ----------------------------------------------------------------------------
# Prompts. All three prompts must instruct the model to emit a ``ReasoningGraph``
# object, because every downstream stage parses that exact structure. The only
# real difference between datasets is whether a CONTEXT block is supplied.
# ----------------------------------------------------------------------------
_REASONING_GRAPH_TYPE = """OUTPUT_TYPE:
ReasoningGraph

```python
class ReasoningNode:
  id: int
  description: str
  output: Union[int, float, str]
  depends_on: list[int]

class ReasoningGraph:
  nodes: list[ReasoningNode]
  final_answer: Union[int, float, str]

OUTPUT_OBJECT:
"""

_INSTRUCTIONS = """Please structure your respond to the last INPUT_OBJECT based on the CONTEXT with OUTPUT_OBJECT according to OUTPUT_TYPE.

INSTRUCTIONS:
- Only output an object like `ReasoningGraph(...)`. No reasoning, explanation or thinking.
- Do NOT define or repeat any class or function.
- ONLY produce an OUTPUT_OBJECT that instantiates the OUTPUT_TYPE.
- The output must be valid Python using the given type names.
- Do NOT generate code, explanation, or helper variables.

INPUT_OBJECT:
  1 + 1 =

OUTPUT_TYPE:
  Answer

  ```python
  class Answer:
    final_answer: int
  ```

OUTPUT_OBJECT:
  ```python
  Answer(
    final_answer=2
  )
  ```
"""

# morehopqa / MoreHopQA: multi-hop QA that needs the supporting CONTEXT passages.
PROMPT_WITH_CONTEXT = (
    _INSTRUCTIONS
    + """
INPUT_OBJECT:
{question}

CONTEXT:
{context}

"""
    + _REASONING_GRAPH_TYPE
)

# gsm8k / math: self-contained word / math problems, no external context.
PROMPT_NO_CONTEXT = (
    _INSTRUCTIONS
    + """
INPUT_OBJECT:
{question}

"""
    + _REASONING_GRAPH_TYPE
)


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
DATASETS = {
    "morehopqa": {
        # MoreHopQA (multi-hop QA built on HotpotQA passages).
        "default_source": "data/source/morehopqa_sample.json",
        "format": "json",          # a JSON list of objects with _id/question/answer/context
        "use_context": True,
        "prompt": PROMPT_WITH_CONTEXT,
        "id_prefix": "morehopqa",
        "answer_extract": None,
    },
    "gsm8k": {
        # GSM8K test.jsonl: one JSON per line, answer ends with "#### <number>".
        "default_source": "data/source/gsm8k_sample.jsonl",
        "format": "jsonl",
        "use_context": False,
        "prompt": PROMPT_NO_CONTEXT,
        "id_prefix": "gsm8k",
        "answer_extract": "gsm8k",  # pull the number after "####"
    },
    "math": {
        # Placeholder: drop the MATH source json in and (optionally) give it its
        # own prompt here. Defaults to the no-context prompt for now.
        "default_source": "data/source/math_sample.json",
        "format": "json",
        "use_context": False,
        "prompt": PROMPT_NO_CONTEXT,
        "id_prefix": "math",
        "answer_extract": None,
    },
}


_GSM8K_ANS_RE = re.compile(r"####\s*(-?[0-9\.,]+)")


def _extract_answer(raw_answer, mode):
    """Normalize the gold answer. gsm8k stores the full solution ending in '#### N'."""
    if mode == "gsm8k" and isinstance(raw_answer, str):
        m = _GSM8K_ANS_RE.search(raw_answer)
        if m:
            return m.group(1).strip().replace(",", "")
    return raw_answer


def load_source_dataset(source_path, dataset):
    """Load a source dataset (json or jsonl) and normalize every item to
    {_id, question, answer, context}. Handles missing _id and gsm8k answer
    extraction, so the generation stage stays dataset-agnostic."""
    cfg = get_dataset_config(dataset)
    fmt = cfg.get("format", "json")

    if fmt == "jsonl" or source_path.endswith(".jsonl"):
        with open(source_path, "r") as f:
            items = [json.loads(line) for line in f if line.strip()]
    else:
        with open(source_path, "r") as f:
            items = json.load(f)

    normalized = []
    for idx, it in enumerate(items):
        normalized.append({
            "_id": it.get("_id", f"{cfg['id_prefix']}_{idx}"),
            "question": it["question"],
            "answer": _extract_answer(it.get("answer", "N/A"), cfg.get("answer_extract")),
            "context": it.get("context", ""),
        })
    return normalized


def get_dataset_config(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASETS)}.")
    return DATASETS[dataset]


def get_model_id(model: str) -> str:
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(MODELS)}.")
    return MODELS[model]


# ----------------------------------------------------------------------------
# Canonical file paths
# ----------------------------------------------------------------------------
def run_dir(output_root: str, dataset: str, model: str) -> str:
    return os.path.join(output_root, dataset, model)


def responses_path(output_root: str, dataset: str, model: str) -> str:
    return os.path.join(run_dir(output_root, dataset, model),
                        f"{model}_{dataset}_responses.pkl")


def transformed_path(output_root: str, dataset: str, model: str) -> str:
    return os.path.join(run_dir(output_root, dataset, model),
                        f"transformed_{model}_{dataset}_with_probs.pkl")


def bert_path(output_root: str, dataset: str, model: str) -> str:
    return os.path.join(run_dir(output_root, dataset, model),
                        f"bert_embeddings_{model}_{dataset}_with_probs.pkl")


def evaluation_path(output_root: str, dataset: str, model: str) -> str:
    return os.path.join(run_dir(output_root, dataset, model), "evaluation.json")


def nibs_path(output_root: str, dataset: str, model: str) -> str:
    return os.path.join(run_dir(output_root, dataset, model), "NIBS_results.json")


def gibs_path(output_root: str, dataset: str, model: str) -> str:
    return os.path.join(run_dir(output_root, dataset, model), "GIBS_results.json")


def gib_checkpoint_dir(output_root: str, dataset: str, model: str) -> str:
    return os.path.join(run_dir(output_root, dataset, model), "gib_checkpoints")


def white_box_path(output_root: str, dataset: str, model: str) -> str:
    return os.path.join(run_dir(output_root, dataset, model),
                        f"white_box_baseline_{model}.json")


def ptrue_path(output_root: str, dataset: str, model: str) -> str:
    return os.path.join(run_dir(output_root, dataset, model),
                        f"ptrue_{model}_{dataset}.pkl")
