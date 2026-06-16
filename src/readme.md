# src/

Source code for the step-wise confidence-estimation pipeline.

See the top-level [`README.md`](../README.md) for the full, step-by-step
instructions (installation, the end-to-end demo notebook, and how to run every
stage), and [`dataset/dataset_config.py`](dataset/dataset_config.py) for the
dataset/model/path conventions.

Pipeline at a glance:

1. `dataset/generate_responds_vllm.py` — sample reasoning traces (vLLM, GPU)
2. `dataset/tranformed_graph.py` — text → structured reasoning graph (CPU)
3. `dataset/bert_graph.py` — BERT embeddings per step (GPU/CPU)
4. `evaluation/gpt_eval.py` — LLM-judge step-correctness labels (OpenAI key)
5. methods — `NIBS/`, `GIBS/`, `baseline/`
6. `evaluation/evaluation_metrics.py` — AUROC / AUPRC / Acc@coverage
