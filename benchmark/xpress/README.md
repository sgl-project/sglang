# Offline XPress benchmark

`data/gsm8k_test_40.json` contains 40 GSM8K test questions selected with
`random.Random(42).sample(range(1319), 40)` from the
[official GSM8K test split](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl).
The answer field is retained for provenance; the serving benchmark does not
score answers.
`data/gsm8k_train_5.json` contains the first five examples from the
[official GSM8K training split](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/train.jsonl).
`eval_gsm8k.py` combines them with the 40 test questions in the same
`Question: ...\nAnswer:` completion format as the
[vLLM XPress evaluation](https://github.com/vllm-project/vllm/pull/54448).
`data/gsm8k_5shot_40.jsonl` contains those 40 prepared prompts for a
PR-style serving benchmark. The five training examples are distinct from the
test questions.
`data/math500_40.jsonl` contains 40 MATH-500 test problems selected with
`random.Random(42).sample(range(500), 40)` from
[HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/blob/main/test.jsonl).
The prompt contains only the problem; the reference solution is retained in
the second turn required by SGLang's local custom dataset loader.

Start the server with a local target model and draft checkpoint, then run:

```sh
TARGET_MODEL=/nas/disk1/Qwen3-8B \
XPRESS_DRAFT=/nas/disk2/xk/model/Qwen3-8B-XPress-b16 \
PORT=8898 sh benchmark/xpress/serve.sh xpress

TARGET_MODEL=/nas/disk1/Qwen3-8B PORT=8898 \
OUTPUT_DIR=/tmp/xpress-gsm8k-results sh benchmark/xpress/bench.sh xpress
```

For the DFlash baseline, start the local checkpoint with `DFLASH_DRAFT`:

```sh
TARGET_MODEL=/nas/disk1/Qwen3-8B \
DFLASH_DRAFT=/nas/disk1/Qwen3-8B-DFlash \
PORT=8898 sh benchmark/xpress/serve.sh dflash
```

The server script also accepts `XPRESS_DRAFT` as a fallback for `dflash` mode
to support existing commands. Run `bench.sh dflash` with the same
`TARGET_MODEL`, `PORT`, `OUTPUT_DIR`, and benchmark settings to compare
throughput. Both model paths must be local directories containing `config.json`.

Optionally, use a local DSpark checkpoint as the third PR baseline:

```sh
TARGET_MODEL=/nas/disk1/Qwen3-8B \
DSPARK_DRAFT=/path/to/Qwen3-8B-Markov-b16 \
PORT=8898 sh benchmark/xpress/serve.sh dspark

TARGET_MODEL=/nas/disk1/Qwen3-8B PORT=8898 \
OUTPUT_DIR=/tmp/xpress-gsm8k-results sh benchmark/xpress/bench.sh dspark
```

The benchmark samples only the local 40-question file. SGLang's `random`
dataset loader repeats or truncates each question's token IDs to exactly 1024
input tokens. `--tokenize-prompt` sends those IDs directly, and the output
length is 100 tokens. This is a controlled throughput test using GSM8K text;
it does not reproduce the PR's full GSM8K accuracy evaluation. The benchmark
sets Hugging Face offline environment variables and requires a local tokenizer.
Use `NUM_PROMPTS` (up to 40), `INPUT_LEN`, `OUTPUT_LEN`, `CONCURRENCIES`, and
`DATASET_PATH` to override the defaults.

After starting each mode, run the five-shot accuracy check separately:

```sh
PORT=8898 python3 benchmark/xpress/eval_gsm8k.py \
  --mode xpress --output /tmp/xpress-gsm8k-accuracy.jsonl
```

The accuracy check uses temperature 0 and at most 256 new tokens, as in the
vLLM XPress end-to-end test. It reports the exact score on this 40-question
subset and aggregates per-request speculative acceptance length and rate when
SGLang provides those counters. No threshold from the full GSM8K test applies
to such a small sample.

For a serving benchmark closer to the PR's GSM8K conditions, run this after
starting each mode:

```sh
TARGET_MODEL=/nas/disk1/Qwen3-8B PORT=8898 \
OUTPUT_DIR=/tmp/xpress-pr-gsm8k-results \
sh benchmark/xpress/bench_pr_sample.sh xpress

TARGET_MODEL=/nas/disk1/Qwen3-8B PORT=8898 DATASET=math500 \
OUTPUT_DIR=/tmp/xpress-pr-gsm8k-results \
sh benchmark/xpress/bench_pr_sample.sh xpress
```

Then restart the DFlash server and run `bench_pr_sample.sh dflash` for each
dataset with the same `OUTPUT_DIR`; optionally repeat for DSpark. GSM8K uses
five-shot completion prompts, while MATH-500 uses its natural problem text.
Both use temperature 0, EOS stopping, and a 2048-token output cap. They run
1/4/16/32 concurrency, two rounds for XPress and DSpark, and one for DFlash.
The script prints mean throughput, mean TPOT, acceptance length if the server
reports it, and paired throughput ratios.

The PR's throughput experiments used one H200, FA3, CUDA graphs, prefix
caching, 8192 context, and up to 2048 output tokens, with 1/4/16/32/64
concurrency and up to 512 prompts. `serve.sh` explicitly selects FA3 and 8192
context; SGLang enables CUDA graphs and radix prefix caching by default.
`bench.sh` defaults to 40 prompts, 1024 input tokens, 100 output tokens, and
1/4/16/32 concurrency. These small-sample results are useful for a paired
XPress-versus-baseline check on the same server, but their absolute throughput
is not directly comparable to the PR's table. `bench_pr_sample.sh` uses
only 40 GSM8K or MATH-500 questions; it does not reproduce the PR's
seven-dataset sweep or its 512/500-prompt GSM8K/MATH-500 runs. Concurrency 64 would exceed the sample size, so it
is omitted. Compare ratios only between modes on the same SGLang revision and
hardware.
