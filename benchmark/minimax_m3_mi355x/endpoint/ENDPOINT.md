# MiniMax-M3 external endpoint on 4x MI350X/MI355X: image + serving command

Everything below is on branch `M3-perf` of `github.com/kevin-mii/sglang` (this directory is `benchmark/minimax_m3_mi355x/endpoint/`).

## Image

`Dockerfile` overlays this branch's `python/sglang` tree, the aiter FlyDSL XCD-swizzle fix and the M3 tuned MoE table on
`lmsysorg/sglang-rocm:v0.5.19-rocm724-mi35x-20260911` (digest `sha256:e62c69db892add25bb6a82f50a61726895b0814014c360e8acdd61fabc93a5c6`;
sglang 822e73ccdd, an ancestor of this branch; aiter 4ad99832). The branch does not change `python/sglang/kernels/aot`, so the
base image's compiled sgl-kernel, aiter and Triton are exactly the stack these numbers were measured on.

```bash
bash make_context.sh /tmp/m3ctx            # build context from this checkout (13 MB)
bash /tmp/m3ctx/build_on_cpu_vm.sh /tmp/m3ctx   # rx CPU VM build + push to 976589843892.dkr.ecr.us-west-2.amazonaws.com/radixark/sglang-ext:<tag>
```

`build_on_cpu_vm.sh` follows the radixark `build-sglang-images-cpu-vm` skill: it needs `rx login --google` and the AWS `radixark`
profile on the machine that runs it (the GPU devbox has neither). The tag is `m3-mi355x-<date>-<sglang commit>`.

## Serving command

Models: `amd/MiniMax-M3-MXFP4` (quark MXFP4 checkpoint) and `Inferact/MiniMax-M3-EAGLE3-GQA` under `$MODEL_ROOT` (default `/models`).

```bash
docker run --rm -it --network host --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 64g \
  -v /models:/models -e MODEL_ROOT=/models -e SGLANG_API_KEY=<bearer key> -e PORT=30000 \
  <image> bash /sgl-workspace/sglang/benchmark/minimax_m3_mi355x/endpoint/serve_endpoint.sh
```

`serve_endpoint.sh` expands to (env vars listed in the script):

```
python3 -m sglang.launch_server --model-path $MODEL_ROOT/MiniMax-M3-MXFP4 --served-model-name MiniMax-M3 --trust-remote-code \
  --tp 4 --host 0.0.0.0 --port 30000 --api-key $SGLANG_API_KEY --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 8192 \
  --mem-fraction-static 0.9 --speculative-algorithm EAGLE3 --speculative-draft-model-path $MODEL_ROOT/MiniMax-M3-EAGLE3-GQA \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-attention-mode decode \
  --triton-attention-num-kv-splits 64 --cuda-graph-backend-prefill breakable --reasoning-parser minimax-m3 --tool-call-parser minimax-m3 \
  --enable-metrics --enable-cache-report --max-running-requests 48 --max-queued-requests 64 --watchdog-timeout 3600
```

Knobs: `SPEC=none` turns speculative decoding off; `PTPC_FP8=1` adds the online per-token FP8 dense path; `INDEX_TOPK_FREQ` (default 4)
is the sparse-index top-k share frequency; `MAXRUN` / `MAXQUEUE` size admission (a full queue answers HTTP 429).

## Vendor quality check (MiniMax provider handbook 2026-08-15)

All runs: TP4 on 4x MI350X (gfx950) in this box, MXFP4 quark checkpoint, fp8 KV, the serving command above (mem 0.9), on the
branch head after the eval fixes from zcnrex/sglang `m3-deploy`. Full numbers and per-sample records: `/scratch/results/` on the box.

| Check | Result | Reference |
|---|---|---|
| m3_format_check text suite (161 cases) | 156 passed, 4 skipped, 1 xfailed, 0 failed (production config with an API key) | vendor suite; first run failed 5 (streamed `[SILENCE]` lost its `]`, no trace id on 4xx, unknown model accepted, two auth cases) -> fixed in this branch |
| Provider-Verifier pass@10 (102 prompts x 10) | Query-Success 100%, ToolCalls-Match 84.8%, Schema-Accuracy 98.0%, Error-Only-Reasoning 0%, Language-Following 100%, Scenario-Check 100% | official MiniMax-M3: 100 / 98.8 / 98.9 / 0 / 100 / 100 |
| aime25 pass@1 avg-of-16 (MiniMax harness prompt, T=1.0, top_p 0.95, 98304 max tokens) | EAGLE3: 85.42% +/- 4.19 (SEM 1.05), non-stop 1.25%, avg 16.5K tokens. No speculative decoding: 87.50% +/- 5.09 (SEM 1.27), non-stop 0.21%, avg 18.0K tokens | official endpoint 91.46% +/- 3.84 (SEM 0.96), non-stop 0.21%; handbook asks non-stop < 0.5% |
| GSM8K-500 (5-shot) | 0.862 (EAGLE3 and no-spec alike) | branch reference 0.85-0.89 |
| Overload / auth / model name | 200 concurrent 4K-token requests: 112 x 200 + 88 x 429; no key -> 401; unknown model -> 404 with `X-Request-Id` | handbook: failures at 120% load must be 429 |
| Steady decode, 24 streams over ~70K-token contexts | 2,475 tok/s aggregate (103 tok/s per stream), accept length 2.5 | handbook: P50 TPS > 60 at some load tier |

### What the ToolCalls-Match gap is

In the 15% of tool prompts that came back as plain `stop`, the model had emitted the call, but with the `name="` glue of the invoke tag
dropped or doubled (`<invoke name Read">`, `<invoke nameRead">`, `<invoke name name="Read">`, or a namespace token spliced into the
tag) after 60-80K tokens of context, and the strict parser dropped it. The parser now accepts those forms (commit `db61ee4589`; 38 of 40
raw invokes sampled from the failing prompts parse, the other two are unrecoverable). The handbook asks providers for exactly this
leniency. Whether the malformed tags themselves come from the MXFP4 checkpoint or from the speculative verify path is what the
per-config probe below measures.

### What the aime25 gap is

Speculative decoding costs at most ~2 points on the full set (within one SEM) but 14/112 on the seven hardest problems, and it triples the
non-stop rate (1.25% vs 0.21%, the handbook wants < 0.5%). Turning off the index top-k sharing across layers did not help (54/112 vs
50/112 on the hard set), so that optimization is not the cause. The engine without speculation still sits ~4 points under the official
endpoint, which points at the MXFP4 checkpoint; the MXFP8 checkpoint is probed below.

PROBE_PLACEHOLDER
