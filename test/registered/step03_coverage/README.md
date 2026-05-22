# step03_coverage — attention backend × graph runner × spec decoding smoke

These are self-contained bash scripts that launch a sglang server with
`--load-format dummy` (random weights), send one `/generate` request,
and check the JSON shape. Each test prints `PASS: <name>` or `FAIL: <reason>`
on the last line and tears down cleanly.

The suite was built to validate the step03 attention-init API
unification (PR #26019). Each test pins a specific combination of:

- **attention backend** (`--attention-backend ...`)
- **graph runner** (full cuda graph / piecewise / breakable / eager)
- **spec decoding** (none / EAGLE / EAGLE3)
- **misc axes** (TBO, DP-attention with idle ranks, hybrid mamba, SWA)

## Layout

Each test = one `<test_name>.sh` script. They all source `_common.sh`,
which provides:

- `step03_preamble`     — runs `pip install sglang-kernel`, sets env vars
- `wait_server_ready`   — polls the server log for "fired up" banner
- `generate_and_check`  — curls `/generate` and validates JSON
- `shutdown_server`     — kills the background server cleanly
- `run_server_smoke`    — the end-to-end driver

A test script sets `TEST_NAME`, `MODEL_PATH`, `LAUNCH_ARGS=(...)`, then
calls `run_server_smoke`. Optional overrides: `PORT`, `READY_TIMEOUT`,
`EXTRA_ENV`.

## Running on the cluster

Standard template (substitute `<TEST>` and `<MODEL>`):

```bash
sudo srun --partition=gb300 --nodes=1 --gpus-per-node=4 --mem=0 --time=01:00:00 --mpi=pmix \
  --container-image '/mnt/vast/squash/lmsysorg+sglang+dev-cu13.sqsh' \
  --container-mounts '/mnt/vast/models:/mnt/vast/models,/mnt/home/cheng.wan/project/sglang:/sgl-workspace/sglang,/mnt/home/cheng.wan/project/sglang:/workspace/sglang,/mnt/vast/cheng.wan/.cache:/root/.cache,/mnt/vast/yangminl/models/DeepSeek-V4-Flash:/flash_model,/mnt/home/cheng.wan:/nfs_home' \
  bash /sgl-workspace/sglang/test/registered/step03_coverage/<TEST>.sh
```

`/root/.cache` mount points the container at the shared HF cache so
Qwen3-0.6B, Llama-2-7b-chat-hf, DeepSeek-V3.2 etc. resolve locally.
`/flash_model` is needed only for `dsv4_*` tests.

For parallel runs across the matrix, launch each `srun ... &` in the
background, capture each `srun`'s log to a separate file, and wait.
The cluster has many GB300 nodes; cap at ~10 in flight.

## Test matrix

| Script                           | Backend         | Graph runner   | Spec   | Model                            | Note                                              |
|----------------------------------|-----------------|----------------|--------|----------------------------------|---------------------------------------------------|
| `triton_eager.sh`                | triton          | eager          | none   | Qwen/Qwen3-0.6B                  | no-cuda-graph baseline                            |
| `triton_cudagraph.sh`            | triton          | full CG        | none   | Qwen/Qwen3-0.6B                  |                                                   |
| `triton_pcg.sh`                  | triton          | PCG            | none   | Qwen/Qwen3-0.6B                  |                                                   |
| `flashinfer_cudagraph.sh`        | flashinfer      | full CG        | none   | Qwen/Qwen3-0.6B                  |                                                   |
| `flashinfer_pcg.sh`              | flashinfer      | PCG            | none   | Qwen/Qwen3-0.6B                  | exercises EXTEND-mode PCG capture                 |
| `fa3_cudagraph.sh`               | fa3             | full CG        | none   | Qwen/Qwen3-0.6B                  | force fa3 on B200 (default would be trtllm_mha)   |
| `fa3_pcg.sh`                     | fa3             | PCG            | none   | Qwen/Qwen3-0.6B                  |                                                   |
| `flashinfer_mla_cudagraph.sh`    | flashinfer (MLA)| full CG        | none   | lmsys/sglang-ci-dsv3-test        | tiny DSV3 MLA model                               |
| `flashmla_cudagraph.sh`          | flashmla        | full CG        | none   | lmsys/sglang-ci-dsv3-test        |                                                   |
| `cutlass_mla_cudagraph.sh`       | cutlass_mla     | full CG        | none   | lmsys/sglang-ci-dsv3-test        | Blackwell-only                                    |
| `trtllm_mla_cudagraph.sh`        | trtllm_mla      | full CG        | none   | lmsys/sglang-ci-dsv3-test        |                                                   |
| `trtllm_mha_cudagraph.sh`        | trtllm_mha      | full CG        | none   | Qwen/Qwen3-0.6B                  | default MHA backend on B200                       |
| `trtllm_mha_pcg.sh`              | trtllm_mha      | PCG            | none   | Qwen/Qwen3-0.6B                  | regression guard for step03 prefill bug           |
| `triton_eagle.sh`                | triton          | full CG        | EAGLE  | meta-llama/Llama-2-7b-chat-hf    | EAGLE2 (spec v1 default)                          |
| `fa3_eagle3.sh`                  | fa3             | full CG        | EAGLE3 | meta-llama/Llama-3.1-8B-Instruct | multi-layer draft                                 |
| `dsv4_cudagraph.sh`              | dsv4            | full CG        | none   | DeepSeek-V4-Flash (`/flash_model`)| needs `flashinfer_mxfp4` + #26024                  |
| `dsv4_eagle.sh`                  | dsv4            | full CG        | EAGLE  | DeepSeek-V4-Flash                | motivating workload                               |
| `dsa_cudagraph.sh`               | dsa             | full CG        | none   | deepseek-ai/DeepSeek-V3.2        | sparse attention                                  |
| `hybrid_mamba.sh`                | auto (mamba+attn)| full CG       | none   | nvidia/Nemotron-H-8B-Base-8K     | hybrid SSM model                                  |
| `gptoss_swa.sh`                  | auto            | full CG        | none   | openai/gpt-oss-20b               | sliding-window attention                          |
| `breakable_cuda_graph.sh`        | triton          | breakable CG   | none   | Qwen/Qwen3-0.6B                  | BCG runner                                        |
| `tbo_cudagraph.sh`               | auto + tbo wrap | full CG        | none   | Qwen/Qwen3-30B-A3B               | two-batch overlap + deepep + dp-attn              |
| `dp_idle.sh`                     | triton          | full CG        | none   | Qwen/Qwen3-30B-A3B               | DP=4 TP=4 with idle DP rank path                  |

## Intentional skips

- **aiter / wave** — AMD/HIP-only backends. Cluster is NVIDIA GB300, so
  these can't be exercised here.
- **fa4, dual_chunk_flash_attn, intel_amx, ascend, intel_xpu,
  torch_native, flex_attention, tokenspeed_mla** — niche / platform-specific
  backends not in the production matrix for GB300.
- **cpu graph runner** — `--device cpu`; out of scope here (GPU cluster).
- **ngram spec decoding** — orthogonal to attention init; covered by
  the existing `test/registered/spec/test_spec_ngram.py` suite.

## DSV4 specifics

`dsv4_*` tests need PR #26024 (`fp8.py: route to DEEP_GEMM when
is_fp4_expert and DeepGEMM available`, commit `be24b5c4b1`).

When validating on the cluster's current main HEAD, cherry-pick #26024
onto your test branch before running:

```bash
git fetch origin be24b5c4b1
git cherry-pick be24b5c4b1
```

If #26024 is not yet merged to main and you cannot cherry-pick, skip
the `dsv4_*` scripts — the failure will be the unrelated FP8/FP4
routing bug, not anything step03 touched.

## What "PASS" means

A pass means:
- The server reached the "fired up and ready to roll" banner within
  the timeout (default 30 min; 40 min for the big-model tests).
- A `/generate` POST with a real prompt and 16-token decode returned
  HTTP 200 with JSON containing a non-empty `text` field.
- No NaN markers, no Python tracebacks, no `"error"` keys in the body.

A pass does **not** assert text quality — `--load-format dummy` makes
output gibberish. The contract is: forward pass runs end-to-end through
the configured backend without crashing.
