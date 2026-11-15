# DP-attention routing under mixed long/short traffic

Benchmark for sgl-project/sglang#50088, which makes `--load-balance-method total_tokens`
count the unprefilled part of a chunked-prefill request. It replays one fixed, open-loop
trace against a running server and records the DP rank of every request and the arrival
time of every streamed token.

| file | what it does |
|---|---|
| `driver.py` | Sends the trace to one server; writes one JSON per trial with every request's routing and token timings. |
| `run_bench.sh` | For each `LABEL\|SGLANG_CHECKOUT\|EXTRA_ARGS` spec: starts a server from that checkout, runs `driver.py`, stops the server. |
| `summarize.py` | Pools the runs per build and config, and prints the markdown tables used in the PR. |
| `results/*.tar.gz` | Raw per-request results of the runs reported in the PR. |
| `env.txt` | Hardware and software of those runs. |

## Trace

Each trial sends 32 requests, with prompts made of random token ids, so neither the tokenizer nor the prefix cache is involved:

- 30 short requests, 1024 prompt tokens and 96 output tokens each, one every 0.4 s;
- one 32K and one 64K prompt, 1 output token each. The first arrives at 1.05 s and their order alternates between trials. In the **burst** profile the second arrives 0.2 s later; in the **staggered** profile, 4 s later.

Each server run sends one warm-up trial, then 3 trials of each profile. Clients never pin a DP rank.

## Server configuration

Every run uses the arguments below. `run_bench.sh` adds `--revision` and each spec's
`EXTRA_ARGS`, which come last and therefore override an earlier value:

```
--model-path Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
--tp 2 --attn-dp-size 2 --context-length 73728
--attention-backend fa3 --mm-attention-backend sdpa --kv-cache-dtype fp8_e4m3
--chunked-prefill-size 8192 --disable-radix-cache --enable-metrics
--mem-fraction-static 0.80 --load-balance-method total_tokens
```

This setup differs from the Qwen3.8-27B cookbook recipe (single H200, 32768-token chunks)
on purpose. Two GPUs with `--attn-dp-size 2` are needed to have DP ranks to route between.
The chunk is smaller because H20 prefill is much slower than H200's, and the server logs
report 4096 tokens per rank. The radix cache is off because the prompts are random.

## Reproduce

The PR numbers compare sglang `9f6d998` (main at the time) with `9f6d998` plus the PR's
`python/` change. The PR head now sits on a newer main, but neither touched file changed
in between.

```bash
# Two checkouts: baseline and fix.
git clone https://github.com/sgl-project/sglang.git sglang-main
git -C sglang-main checkout 9f6d9981b8a5538372c29c988794f08d761a695b
git clone https://github.com/sgl-project/sglang.git sglang-fix
git -C sglang-fix checkout 9f6d9981b8a5538372c29c988794f08d761a695b
curl -sL https://github.com/sgl-project/sglang/pull/50088.diff | git -C sglang-fix apply

# One environment with sglang's dependencies (versions used: env.txt). run_bench.sh puts
# each checkout's python/ first on PYTHONPATH, so it takes precedence over the installed copy.
pip install -e "sglang-main/python"

MAIN=$PWD/sglang-main FIX=$PWD/sglang-fix
export CUDA_VISIBLE_DEVICES=0,1   # two GPUs
# Used for every recorded run: skip the startup DeepGEMM pre-compilation (tens of minutes
# for this model); kernels are JIT-compiled on first use, during the warm-up trial.
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
```

The model is downloaded from Hugging Face on the first start. Set `MODEL=/path/to/snapshot MODEL_REVISION=""`
to use a local copy instead. One server run takes about 8.5 minutes on 2×H20: about 5 minutes
to start and about 3.5 minutes for the trace.

**A/B with the default scheduler (main table of the PR).** Two fresh servers per build, alternating:

```bash
bash run_bench.sh results/ab "fix1|$FIX|" "main1|$MAIN|" "fix2|$FIX|" "main2|$MAIN|"
python3 summarize.py results/ab
```

**Scheduler settings (Streaming smoothness table).**

```bash
bash run_bench.sh results/grid \
  "fix-mixed|$FIX|--enable-mixed-chunk"            "main-mixed|$MAIN|--enable-mixed-chunk" \
  "main-small|$MAIN|--chunked-prefill-size 4096"   "fix-small|$FIX|--chunked-prefill-size 4096" \
  "fix-mixsmall|$FIX|--enable-mixed-chunk --chunked-prefill-size 4096" \
  "main-mixsmall|$MAIN|--enable-mixed-chunk --chunked-prefill-size 4096" \
  "main-base|$MAIN|" "fix-base|$FIX|"
python3 summarize.py results/grid
```

`run_bench.sh` stops at the first server or client failure with a non-zero exit code, and it
always shuts the server down. Each run writes `LABEL-server.log`, `LABEL.commit`, and `LABEL/`
containing the trial JSONs plus `/server_info` and `/model_info`.

## Recorded results

```bash
tar xzf results/ab-20261112.tar.gz && python3 summarize.py ab-20261112
tar xzf results/grid-20261113.tar.gz && python3 summarize.py grid-20261113
tar xzf results/validation-20261115.tar.gz && python3 summarize.py validation-20261115
```

`ab-20261112` and `grid-20261113` are the runs behind the PR tables. `validation-20261115` is
an independent rerun (one server per build) with the exact `run_bench.sh` / `driver.py` in this
directory, done to check that these scripts reproduce the result. Its `fix1.commit` is
`3cd2424`, a local commit of `9f6d998` plus the PR's `python/` change.

All three were taken on the same 2×H20 machine (see `env.txt`), with a local snapshot of
`Qwen/Qwen3.8-27B-FP8` at revision `017b9c7`. In the saved `server_info`, `model_path` was
replaced with the Hugging Face id. `ab-20261112` and `grid-20261113` were produced by an
earlier copy of `driver.py` that differs only in its command-line options and in the per-trial
summary fields it prints. The trace and the per-request fields are the same, and the old per-run `summary.json` files are not included,
because `summarize.py` recomputes everything from the per-request data.

Expect different absolute numbers on other GPUs or software versions. The comparison is
only meaningful between builds measured on the same machine in the same session.

## Metrics

Short = the 1024-token requests; long = the 32K/64K prompts.

- **Busier rank's share**: the larger rank's fraction of all prompt tokens in a trial (50% is even).
- **Output throughput**: output tokens / (last token time - first send time) per trial.
- **TTFT** and **end-to-end latency**: measured per request from send to first and last token.
- **Worst stall**: a short request's longest gap between two streamed tokens. It shows decode
  starvation that a pooled inter-token p99 hides, because such a stall happens at most once per request.
- **Inter-token p99** and **>500 ms gaps per request**: over all short-request token gaps in a trial.
  With `--enable-mixed-chunk`, one long stall becomes many short gaps, so the gap count is not
  comparable across scheduler settings.

Per-trial values (share, throughput, long TTFT, inter-token p99, gap count) are reported as the
median over trials; per-request values are pooled across trials. See the `summarize.py` docstring.
