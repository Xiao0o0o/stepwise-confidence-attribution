# Step-wise Confidence Estimation with NIBS and GIBS

A framework for generating, labeling, and **attributing step-wise confidence
scores** to the reasoning traces of Large Language Models (LLMs). Each LLM
response is parsed into a **reasoning graph** (nodes = intermediate results,
edges = reasoning steps), and every step is scored by how confident we should
be that it is correct.

Two Information-Bottleneck (IB) methods are implemented, plus several baselines:

| Method | Type | Idea |
| --- | --- | --- |
| **NIBS** (Non-parametric IB Selection) | training-free | scores each step by its consensus/similarity against the set of *correct* reasoning graphs |
| **GIBS** (Graph IB Selection) | trainable | learns an edge-level mask over the reasoning graph that keeps the informative steps |
| P(True) | baseline | LLM self-evaluation probability of each step being true |
| White-box | baseline | sequence-likelihood / entropy / LECO from model internals |

---

## 1. Repository layout

```
stepwise/
├── README.md                  ← you are here
├── requirements.txt
├── data/
│   ├── source/                ← raw input datasets
│   │   ├── morehopqa_sample.json   (10 cases, used by the demo)
│   │   ├── morehopqa_full.json     (full MoreHopQA, 1118 cases)
│   │   ├── gsm8k_sample.jsonl      (10 cases)
│   │   └── gsm8k_full.jsonl        (full GSM8K test, 1319 cases)
│   ├── MorehopQA/  GSM8K/  Math/   ← pre-computed paper results (per model)
│   └── ...
├── output/                    ← all artifacts the pipeline generates (git-ignored)
├── notebooks/
│   └── demo_morehopqa.ipynb    ← end-to-end walk-through on 10 MoreHopQA cases
└── src/
    ├── dataset/
    │   ├── dataset_config.py          ← single source of truth: datasets, models, paths, prompts
    │   ├── generate_responds_vllm.py  ← Stage 1: sample reasoning traces (vLLM, GPU)
    │   ├── tranformed_graph.py        ← Stage 2: text → structured graph + aligned probs (CPU)
    │   └── bert_graph.py              ← Stage 3: BERT embeddings per step (GPU/CPU)
    ├── evaluation/
    │   ├── gpt_eval.py                ← Stage 4: LLM-judge step correctness labels (OpenAI key)
    │   └── evaluation_metrics.py      ← Stage 6: AUROC / AUPRC / Acc@coverage for every method
    ├── NIBS/
    │   └── similarity_based_baseline.py   ← NIBS scores  → NIBS_results.json (GPU)
    ├── GIBS/
    │   ├── GIB_MCS_wo_mcs_feature.py      ← GIBS training → checkpoint (GPU)
    │   ├── evaluation_GIB_wo_mcs.py       ← GIBS inference → GIBS_results.json (GPU)
    │   └── find_mcs.py                    ← Maximum-Common-Subgraph helper (DeBERTa-MNLI)
    └── baseline/
        ├── p_true.py                      ← P(True) baseline → ptrue_*.pkl (GPU)
        └── white_box_baseline.py          ← white-box baseline → white_box_baseline_*.json (GPU)
```

### File-naming convention (the contract)

Every stage reads/writes under `output/<dataset>/<model>/` and the names are
generated in **one** place — [`src/dataset/dataset_config.py`](src/dataset/dataset_config.py).
`dataset ∈ {morehopqa, gsm8k, math}`, `model ∈ {llama, deepseek, phi4}`.

```
output/<dataset>/<model>/
  <model>_<dataset>_responses.pkl                  # Stage 1
  transformed_<model>_<dataset>_with_probs.pkl     # Stage 2
  bert_embeddings_<model>_<dataset>_with_probs.pkl # Stage 3
  evaluation.json                                  # Stage 4 (ground-truth labels)
  NIBS_results.json                                # NIBS
  GIBS_results.json                                # GIBS inference
  white_box_baseline_<model>.json                  # white-box baseline
  ptrue_<model>_<dataset>.pkl                      # P(True) baseline
  gib_checkpoints/                                 # GIBS training output
```

All scripts are meant to be **run from the repository root** so that the
relative `output/...` paths resolve correctly.

> **Raw responses.** The full set of raw LLM reasoning traces (the Stage 1
> `*_responses.pkl` outputs, too large to bundle here) is available on Google
> Drive: <https://drive.google.com/drive/folders/1QrIiW1nHXY3YMaXwIQhA-ER7HEvw_8kR?usp=sharing>.
> Download them into `output/<dataset>/<model>/` to re-run Stages 2–6 without
> regenerating, or generate your own with Stage 1.

---

## 2. Installation

Tested with **Python 3.10** and the pinned versions in
[`requirements.txt`](requirements.txt) (torch 2.5.1, transformers 4.51.3,
vllm 0.6.6.post1).

```bash
git clone <this-repo>
cd stepwise

# On the cluster (mamba/conda):
module load mamba
mamba create -n reasoning python=3.10 -y
mamba activate reasoning

# ...or a plain venv:  python3.10 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

> `torch` / `vllm` are CUDA builds (target CUDA 12.x) — install them on a GPU node.
> The CPU-only stages (2, 6, and 3 with `--device cpu`) need just numpy / scikit-learn /
> torch / transformers, so a CPU box is enough to try those.

Hardware / access notes:

- **Stage 1 (generation)** needs a CUDA GPU and `vllm`, plus access to the base
  model on HuggingFace (e.g. `meta-llama/Llama-3.1-8B-Instruct`, which is gated —
  run `huggingface-cli login` first).
- **Stages 3, 5 (BERT / NIBS / GIBS / baselines)** use GPUs but can fall back to
  CPU for tiny demos (`--device cpu`). NIBS/MCS load `microsoft/deberta-large-mnli`.
- **Stage 4 (labeling)** needs an OpenAI API key (`export OPENAI_API_KEY=sk-...`).
- **Stages 2, 6 (transform / final metrics)** are pure CPU (numpy/sklearn).

---

## 3. Quick start — the demo notebook

The fastest way to see the whole pipeline is
[`notebooks/demo_morehopqa.ipynb`](notebooks/demo_morehopqa.ipynb). It runs all six
stages on **10 MoreHopQA cases**, writing every intermediate artifact to
`output/`, and uses **relative paths only** so it works straight after cloning.

Stages that need a GPU or an API key are clearly marked and **gated** — if the
requirement is missing, the cell explains what to do and the notebook continues.
The final evaluation cell always produces real numbers because it can fall back
to the pre-computed results bundled in `data/MorehopQA/Phi4/`.

```bash
pip install jupyter
jupyter notebook notebooks/demo_morehopqa.ipynb
```

---

## 4. Full pipeline, step by step

Below, `DATASET=morehopqa` and `MODEL=llama` are used as the running example.
Swap in `gsm8k`/`math` and `deepseek`/`phi4` as needed.

### Stage 1 — Generate reasoning traces  *(GPU + vLLM)*

Samples `--num-responses` reasoning graphs per question from the base model.

```bash
python src/dataset/generate_responds_vllm.py \
    --dataset morehopqa --model llama \
    --dataset-path data/source/morehopqa_sample.json \
    --num-responses 5 --max-questions 10
# → output/morehopqa/llama/llama_morehopqa_responses.pkl
```

Key flags: `--num-responses` (samples per question, default 20),
`--max-questions` (limit for demos), `--batch-size`, `--dataset-path`
(defaults to the per-dataset source in `dataset_config.py`),
`--output-root` (default `output`).

### Stage 2 — Transform traces into graphs  *(CPU)*

Parses each raw `ReasoningGraph(...)` into nodes/edges, aligns token logprobs to
each step, and emits `string_q_a` + `structure_retrieve`.

```bash
python src/dataset/tranformed_graph.py --dataset morehopqa --model llama
# → output/morehopqa/llama/transformed_llama_morehopqa_with_probs.pkl
```

### Stage 3 — BERT embeddings  *(GPU or CPU)*

Embeds every step / node / edge / response with `bert-base-uncased`.

```bash
python src/dataset/bert_graph.py --dataset morehopqa --model llama --device cuda
# → output/morehopqa/llama/bert_embeddings_llama_morehopqa_with_probs.pkl
```

### Stage 4 — Label step correctness with an LLM judge  *(OpenAI key)*

GPT-4o-mini marks each edge-node pair correct (1) / incorrect (0). These labels
are the ground truth for the final metrics.

```bash
export OPENAI_API_KEY=sk-...
python src/evaluation/gpt_eval.py \
    --dataset morehopqa --model llama \
    --source data/source/morehopqa_sample.json
# → output/morehopqa/llama/evaluation.json
```

> Without a key the script runs a **dry run**: it still writes `evaluation.json`
> with the right schema, but every label is `-1` (unknown). Replace with a real
> key (or copy a pre-computed `evaluation.json` from `data/MorehopQA/...`) to get
> meaningful metrics.

### Stage 5 — Run the confidence-estimation methods

**NIBS** *(GPU, DeBERTa-MNLI)* — needs the labels + embeddings:

```bash
python src/NIBS/similarity_based_baseline.py \
    --label-file     output/morehopqa/llama/evaluation.json \
    --embedding-file output/morehopqa/llama/bert_embeddings_llama_morehopqa_with_probs.pkl \
    --output-file    output/morehopqa/llama/NIBS_results.json
```

**GIBS** *(GPU)* — train, then run inference:

```bash
# (1) train
python src/GIBS/GIB_MCS_wo_mcs_feature.py \
    --embedding-file output/morehopqa/llama/bert_embeddings_llama_morehopqa_with_probs.pkl \
    --label-file     output/morehopqa/llama/evaluation.json \
    --save-dir       output/morehopqa/llama/gib_checkpoints \
    --epochs 1000

# (2) inference with the trained checkpoint
python src/GIBS/evaluation_GIB_wo_mcs.py \
    --model-path     output/morehopqa/llama/gib_checkpoints/<gib_model_*.pth> \
    --embedding-file output/morehopqa/llama/bert_embeddings_llama_morehopqa_with_probs.pkl \
    --label-file     output/morehopqa/llama/evaluation.json \
    --output-file    output/morehopqa/llama/GIBS_results.json
```

**Baselines** *(GPU)*:

```bash
# P(True) — note: this script's --dataset uses {morehopqa, gsm8k, math}
python src/baseline/p_true.py \
    --model llama3 --dataset morehopqa \
    --data_path output/morehopqa/llama/transformed_llama_morehopqa_with_probs.pkl \
    --save_path output/morehopqa/llama/ptrue_llama_morehopqa.pkl

# White-box — --model-name uses {llama3, phi4, deepseek}
python src/baseline/white_box_baseline.py \
    --label-file       output/morehopqa/llama/evaluation.json \
    --embedding-file   output/morehopqa/llama/bert_embeddings_llama_morehopqa_with_probs.pkl \
    --transformed-file output/morehopqa/llama/transformed_llama_morehopqa_with_probs.pkl \
    --output-file      output/morehopqa/llama/white_box_baseline_llama.json \
    --model-name llama3 \
    --context-file data/source/morehopqa_sample.json
```

### Stage 6 — Evaluate all methods  *(CPU)*

Computes **AUROC / AUPRC / Accuracy@80%-coverage** for every method by comparing
its step scores against the `evaluation.json` labels.

```bash
python src/evaluation/evaluation_metrics.py --dataset morehopqa --model llama
```

`--dataset`/`--model` auto-derive every input path. You can instead pass them
explicitly (handy for the bundled paper results):

```bash
python src/evaluation/evaluation_metrics.py \
    --label-file     data/MorehopQA/Phi4/evaluation.json \
    --gib-file       data/MorehopQA/Phi4/GIBS_results.json \
    --nibs-file      data/MorehopQA/Phi4/NIBS_results.json \
    --ptrue-file     data/MorehopQA/Phi4/ptrue_phi4_morehopqa_.pkl \
    --white-box-file data/MorehopQA/Phi4/white_box_baseline_phi4.json
```

---

## 5. Datasets and prompts

Three datasets are supported. Each has its own entry (source path + prompt) in
[`dataset_config.py`](src/dataset/dataset_config.py); all prompts emit the same
`ReasoningGraph` structure so the downstream stages are dataset-agnostic.

| `--dataset` | data | source format | prompt | context? |
| --- | --- | --- | --- | --- |
| `morehopqa` | MoreHopQA (multi-hop QA over HotpotQA passages) | `.json` list | `PROMPT_WITH_CONTEXT` | yes — supporting passages |
| `gsm8k`    | GSM8K grade-school math | `.jsonl` (answer auto-extracted from `#### N`) | `PROMPT_NO_CONTEXT` | no |
| `math`     | MATH (drop your source in `data/source/math_sample.json`) | `.json` list | `PROMPT_NO_CONTEXT` (customize) | no |

Source loading is centralized in `dataset_config.load_source_dataset()`, which
normalizes every dataset to `{_id, question, answer, context}` (auto-generating
`_id`s and extracting the GSM8K `#### N` answer). So **the same three dataset
scripts generate GSM8K with no changes** — just:

```bash
python src/dataset/generate_responds_vllm.py --dataset gsm8k --model phi4 \
    --dataset-path data/source/gsm8k_full.jsonl
python src/dataset/tranformed_graph.py --dataset gsm8k --model phi4
python src/dataset/bert_graph.py       --dataset gsm8k --model phi4
```

To add/adjust a dataset (prompt, source format, answer extraction), edit
`DATASETS` / the `PROMPT_*` constants in `dataset_config.py` — nothing else needs
to change.

---

## 6. Argument conventions cheat-sheet

| Stage / script | dataset flag values | model flag values |
| --- | --- | --- |
| dataset stages, gpt_eval, evaluation_metrics | `morehopqa`, `gsm8k`, `math` | `llama`, `deepseek`, `phi4` |
| `baseline/p_true.py` (`--dataset`/`--model`) | `morehopqa`, `gsm8k`, `math` | `llama3`, `phi4`, `deepseek` |
| `baseline/white_box_baseline.py` (`--model-name`) | — | `llama3`, `phi4`, `deepseek` |

(The baseline scripts predate the unified naming; the table above is the exact
mapping for the HotpotQA/MoreHopQA data.)

---

## 7. Troubleshooting

- **`FileNotFoundError: output/...`** — run scripts from the repository root, or
  pass `--output-root /abs/path`.
- **`import dataset_config` fails** — the dataset scripts add their own folder to
  `sys.path`, so this should work from any cwd; if you copied a script elsewhere,
  keep `dataset_config.py` next to it.
- **Gated HuggingFace model** — `huggingface-cli login` and accept the model
  license before Stage 1.
- **No GPU** — you can still run Stages 2 and 6, plus Stage 3 with `--device cpu`.
  Use the bundled `data/<dataset>/<model>/` results to exercise Stage 6.

---

## 8. Citation

If you use this code, please cite:

```bibtex
@article{liu2026diagnosing,
  title={Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution},
  author={Liu, Xiaoou and Chen, Tiejin and Zhang, Dengjia and Wang, Yaqing and Cheng, Lu and Wei, Hua},
  journal={arXiv preprint arXiv:2605.19228},
  year={2026}
}
```
