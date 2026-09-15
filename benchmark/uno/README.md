# UNO full-dataset math evaluation

`run_math_eval.py` evaluates AR, UNO, DFLASH, EAGLE, or EAGLE3 with identical
datasets, prompts, sampling parameters, and grading. It creates an in-process
`sgl.Engine`; there is no separate server process. Engine startup is excluded
from the timed interval, and no additional request warmup is run.

The offline runner downloads a pinned revision of MATH-500. It applies the same boxed-answer instruction and Qwen
reasoning chat template to every engine, then grades with `math_verify`.

Install SGLang with its evaluation dependencies:

```bash
pip install -e "python[test]"
```

## Reproduce the H200 table

Run from the SGLang repository root. Each invocation below produces one row of
the MATH-500 portion of the PR table, using one sample per problem.

```bash
export MODEL_PATH=Qwen/Qwen3-8B
export TOKENIZER_PATH=Qwen/Qwen3-8B
export UNO_LORA_PATH=s-sahoo/uno-qwen3-8B
export DATA_ROOT=/path/to/math-eval-data
export RESULT_ROOT=/path/to/math-eval-results

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --tokenizer-path "$TOKENIZER_PATH"
  --data-root "$DATA_ROOT"
  --context-length 40960
  --max-tokens 32768
  --temperature 1
  --top-k 50
  --top-p 0.95
  --random-seed 42
)

run_ar() {
  local benchmark=$1 samples=$2 requests=$3 output_name=$4
  PYTHONPATH=python python -m benchmark.uno.run_math_eval \
    "${COMMON_ARGS[@]}" \
    --benchmark "$benchmark" \
    --num-samples "$samples" \
    --max-running-requests "$requests" \
    --output-dir "$RESULT_ROOT/$output_name"
}

run_linear_uno() {
  local benchmark=$1 samples=$2 requests=$3 output_name=$4
  PYTHONPATH=python python -m benchmark.uno.run_math_eval \
    "${COMMON_ARGS[@]}" \
    --benchmark "$benchmark" \
    --num-samples "$samples" \
    --max-running-requests "$requests" \
    --output-dir "$RESULT_ROOT/$output_name" \
    --speculative-algorithm UNO \
    --uno-lora-path "$UNO_LORA_PATH" \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 8
}

run_tree_uno() {
  local benchmark=$1 samples=$2 requests=$3 output_name=$4
  PYTHONPATH=python python -m benchmark.uno.run_math_eval \
    "${COMMON_ARGS[@]}" \
    --benchmark "$benchmark" \
    --num-samples "$samples" \
    --max-running-requests "$requests" \
    --output-dir "$RESULT_ROOT/$output_name" \
    --speculative-algorithm UNO \
    --uno-lora-path "$UNO_LORA_PATH" \
    --speculative-num-steps 15 \
    --speculative-eagle-topk 32 \
    --speculative-num-draft-tokens 32
}
```

Run the two batch-64 AR and linear `B/K/V = 8/1/8` rows:

```bash
run_ar         math500 1  64 ar-math500-c64
run_linear_uno math500 1  64 uno-linear-b8-k1-v8-math500-c64
```

Run the two batch-1 AR and tree `B/K/V = 16/32/32` rows:

```bash
run_ar       math500 1  1 ar-math500-c1
run_tree_uno math500 1  1 uno-tree-b16-k32-v32-math500-c1
```

Each output directory contains raw generations, per-answer grades, and
`summary.json` and `summary.md`. AR TPF is one. UNO TPF counts both full
target-model forwards in each cycle: the diffusion-pathway draft and
AR-pathway verification forwards.

## Other speculative decoders

The runner uses the same public option names as `sglang serve`. For example,
DFLASH can be evaluated with:

```bash
PYTHONPATH=python python -m benchmark.uno.run_math_eval \
  "${COMMON_ARGS[@]}" \
  --benchmark math500 \
  --num-samples 1 \
  --output-dir "$RESULT_ROOT/dflash-b8-math500-c64" \
  --max-running-requests 64 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
  --speculative-dflash-block-size 8 \
  --speculative-draft-attention-backend fa3
```

EAGLE or EAGLE3 can be evaluated with the corresponding draft model:

```bash
export EAGLE_DRAFT_MODEL=/path/to/compatible-eagle-draft-model

PYTHONPATH=python python -m benchmark.uno.run_math_eval \
  "${COMMON_ARGS[@]}" \
  --benchmark math500 \
  --num-samples 1 \
  --output-dir "$RESULT_ROOT/eagle3-b8-math500-c64" \
  --max-running-requests 64 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "$EAGLE_DRAFT_MODEL" \
  --speculative-num-steps 7 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 8
```

For EAGLE and DFLASH, TPF follows SGLang's acceptance-length convention and
counts generated tokens per target verification forward.

## GSM8K and AIME

For GSM8K and AIME, launch the model with `sglang serve` and the desired
speculative-decoding options, then evaluate it with `sgl-eval`:

```bash
sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1
sgl-eval run aime25 --base-url http://127.0.0.1:30000/v1 \
  --n-repeats 10 --num-threads 64 --max-tokens 32768 \
  --temperature 1 --top-p 0.95 --seed 42
```

Use `aime24` or `aime26` for the other AIME editions. These commands use
sgl-eval's dataset, prompt, and grader; historical scores from the offline
runner must be remeasured with this protocol.
