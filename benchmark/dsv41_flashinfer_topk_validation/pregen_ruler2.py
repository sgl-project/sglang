"""Generate the RULER2 dataset sgl-eval reuses for --ruler2-seq-len 65536,
--ruler2-dataset-size 20 and the checkpoint tokenizer. Run with the eval venv's
python and its bin first on PATH (the generators spawn `python -m ...`)."""

from sgl_eval.evals._ruler2 import ALL_TASKS, Ruler2Config, _ensure_task_data

cfg = Ruler2Config(
    max_seq_length=65536,
    tokenizer_path="/data/models/DeepSeek-V4.1-Flash-1017",
    dataset_size=20,
)
print("cache dir:", cfg.cache_dir, flush=True)
for task in ALL_TASKS:
    print(task, _ensure_task_data(cfg, task), flush=True)
print("done", flush=True)
