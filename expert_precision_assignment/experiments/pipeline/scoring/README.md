# scoring/ — offline accuracy scoring for openai-mode sweeps

One script per task, each taking a `data/results/<task>/mc*_*.jsonl` trace + its `pipeline/prompt/<task>.meta.jsonl` sidecar, and emitting a sibling `mc*_*.scores.json` that `pipeline/collect_result/collect_results.py` will pick up.

| task | script | vendoring required? |
|---|---|---|
| SuperGPQA | `score_traces_supergpqa.py` | no — pure regex letter extraction |
| IFBench (AllenAI) | `score_traces_ifbench.py` | **yes** — `vendored/ifbench/` |
| LiveCodeBench v6 | `score_traces_lcb_v6.py` | **yes** — `vendored/lcb_runner/` |

## Vendoring

Both IFBench and LCB depend on upstream verifier / executor code. Pinned SHAs ensure reproducibility. Run from this `pipeline/scoring/` directory:

### IFBench — 3 Python files + pip deps

```bash
mkdir -p vendored
cd vendored
git clone https://github.com/allenai/IFBench.git ifbench_repo
cd ifbench_repo && git checkout cb932e352a505306ad0115272211df14bb8f628f && cd ..
mkdir -p ifbench
ln -sfn ../ifbench_repo/instructions.py             ifbench/instructions.py
ln -sfn ../ifbench_repo/instructions_registry.py    ifbench/instructions_registry.py
ln -sfn ../ifbench_repo/instructions_util.py        ifbench/instructions_util.py
ln -sfn ../ifbench_repo/evaluation_lib.py           ifbench/evaluation_lib.py
cd ..

pip install absl-py langdetect nltk immutabledict spacy emoji syllapy
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm
```

### LiveCodeBench runner — one package + pip deps

```bash
mkdir -p vendored
cd vendored
git clone https://github.com/LiveCodeBench/LiveCodeBench.git lcb_runner_repo
cd lcb_runner_repo && git checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 && cd ..
ln -sfn lcb_runner_repo/lcb_runner lcb_runner
cd ..

pip install pebble datasets
```

## Usage

All commands run from `experiments/`:

```bash
# SuperGPQA — runs directly, no vendoring
for t in data/results/supergpqa/mc*_*.jsonl; do
    python pipeline/scoring/score_traces_supergpqa.py \
        --trace "$t" --meta pipeline/prompt/supergpqa.meta.jsonl
done

# IFBench — needs pipeline/scoring/vendored/ifbench/
for t in data/results/ifbench/mc*_*.jsonl; do
    python pipeline/scoring/score_traces_ifbench.py \
        --trace "$t" --meta pipeline/prompt/ifbench.meta.jsonl
done

# LCB v6 — needs pipeline/scoring/vendored/lcb_runner/.
# EXECUTES MODEL-GENERATED CODE; see security note.
for t in data/results/livecodebench_v6/mc*_*.jsonl; do
    python pipeline/scoring/score_traces_lcb_v6.py \
        --trace "$t" --meta pipeline/prompt/livecodebench_v6.meta.jsonl
done

# Summary CSV (auto-picks up .scores.json sidecars when present).
python pipeline/collect_result/collect_results.py \
    --results_dir data/results/supergpqa \
    --out_csv data/results/supergpqa/summary.csv
```

## Security note for LCB scoring

`score_traces_lcb_v6.py` runs model-generated Python code via `lcb_runner.evaluation.testing_util.run_test`, which uses subprocess-with-CPU-time-limit. **This is not a security sandbox.** The user has opted for same-machine scoring; if you later run against untrusted models, move this script into an isolated container / VM.
