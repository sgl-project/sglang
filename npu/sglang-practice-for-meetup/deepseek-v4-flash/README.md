# DeepSeek-V4-Flash on Ascend NPU — runbook

Goal: serve `DeepSeek-V4-Flash` with SGLang on an Ascend NPU slice, verify it answers correctly, and produce
a defensible performance table (TTFT / TPOT / throughput / goodput) that can be compared against another
hardware or configuration.

Every command below is grounded in the Ascend NPU docs shipped in this repository:
`docs/docs/hardware-platforms/ascend-npus/` — primarily
`model-deployment/tutorials/deepseek_v4_flash.mdx`,
`model-deployment/best-practices/deepseek_v4_flash.mdx`,
`getting-started/installation.mdx`, and `reference/glossary.mdx`.

---

## 0. Before anything: does it fit?

DeepSeek-V4-Flash is a 284B-parameter MoE (13B active). Two checkpoint families exist:

| Checkpoint | Size | Precision | Ascend support |
| --- | --- | --- | --- |
| `deepseek-ai/DeepSeek-V4-Flash` (native) | 159.6 GB | FP4 experts + FP8 rest | MXFP8/MXFP4 paths are **950PR/DT only** |
| `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp` (validated for Ascend) | ~300 GB | W8A8 INT8 + MTP head | A2 / A3 / 950PR-DT, via `--quantization modelslim` |

Rule of thumb: the W8A8 checkpoint needs **≥ ~360 GB of total device HBM** to leave room for KV cache and
activations. That means:

- **950PR/DT × 4** — intended target. `--tp-size 4` (on 950PR/DT, one card = one die, so card count =
  device count = tp-size, per `reference/glossary.mdx`).
- **A2 × 8** or **A3 × 4 cards (8 dies)** — `--tp-size 8`.
- **A2 × 4** (4 × 64 GB = 256 GB) — **does not fit**. Do not burn the lab slot on it; use
  `env/00_common.sh` output as the capacity request, or benchmark a model that fits (e.g. `Qwen3-32B`)
  and record the constraint.

`env/00_common.sh` performs this check automatically and refuses to continue on an infeasible slice.

---

## 1. Container

The NPU runtime (CANN, torch_npu, kernels) is baked into the image. The docs publish validated tags:

```bash
# 950PR/DT series
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-950-B105
# A3 series
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-a3-B105
```

The meetup environment may hand you a pre-built image under a local name such as `sglang-950_26.2` or
`sglang-A2_26.2`; that is a rename of one of the above, so set:

```bash
export IMAGE=sglang-950_26.2     # or the registry tag above
```

`deploy/01_container.sh` builds the `docker run` from the documented pattern (`--privileged --net=host`,
driver/firmware mounts, `/dev/davinci*`, `/dev/davinci_manager`, `/dev/hisi_hdc`, `--shm-size 16g`, weights
mounted read-only at `/models`), but only passes the devices that actually exist on the slice.

Always verify inside the container before launching anything:

```bash
npu-smi info                 # devices, HBM, health
npu-smi info -l              # chip count: 16 = A3, 8 = A2 or 950
ls /dev/davinci*             # 950PR/DT: davinci0..7 ; A2: davinci0..7 ; A3: davinci0..15
python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
python3 -c "import sglang; print(sglang.__version__)"
```

## 2. Weights

The Ascend-validated W8A8 + MTP checkpoint is on ModelScope:

```text
Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp   (~300 GB)
```

On the meetup box the checkpoint is **already provisioned** at
`/workspace/shared_assets/OssSGLangMeetup/model/DeepSeek-V4-Flash-0731` (156 GB, 48 shards, native FP4+FP8 —
confirmed on 2026-11-14), so nothing needs downloading. For another environment,
`deploy/02_weights.sh` first looks for an already-provisioned copy (`/models`, `/data`, `~/models`,
`$MODEL_DIR`) and only downloads if missing. Download options, in order of preference in this environment:

```bash
modelscope download --model Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp --local_dir /models/DeepSeek-V4-Flash-w8a8-mtp
# or, if only Hugging Face tooling is available (direct HF is often blocked; use the mirror):
export HF_ENDPOINT=https://hf-mirror.com
hf download Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp --local-dir /models/DeepSeek-V4-Flash-w8a8-mtp
```

Disk: leave ≥ 400 GB free for weights + image + logs. On a 500 GB volume, check before pulling.

## 3. Launch (single node, PD-mixed)

Base command, adapted from the A3 half-node example in the tutorial to **4 devices**:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export FORCE_DRAFT_MODEL_NON_QUANT=1
export HCCL_OP_EXPANSION_MODE=AIV
export INF_NAN_MODE_FORCE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

python3 -m sglang.launch_server \
    --model-path "$MODEL_DIR" \
    --host 127.0.0.1 --port 6688 \
    --trust-remote-code --device npu --attention-backend dsv4 \
    --page-size 128 --tp-size 4 --dp-size 4 --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --quantization modelslim --kv-cache-dtype bfloat16 \
    --mem-fraction-static 0.75 \
    --prefill-max-requests 80 --max-prefill-tokens 40000 --chunked-prefill-size 65536 \
    --max-running-requests 80 \
    --cuda-graph-bs-decode 1 2 4 8 16 \
    --speculative-algorithm EAGLE --speculative-num-steps 2 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
    --watchdog-timeout 9000
```

Notes, in the order they will bite:

- `--tp-size` must equal the number of visible devices on 950PR/DT (1 die/card). On A3 it is **twice** the
  card count. Getting this wrong is the most common first failure.
- `--attention-backend dsv4` is required; the default backend does not implement the hybrid CSA/HCA
  attention.
- **Before launching on the meetup image**, run `deploy/00_patch_npu_mxfp8.py` once (fixes the grouped-`wo_a`
  scale layout) and make sure the two DSv4 fallback switches are set:
  `SGLANG_OPT_USE_TILELANG_MHC_PRE=False`, `SGLANG_OPT_USE_TILELANG_MHC_POST=False`,
  `SGLANG_OPT_USE_FUSED_HASH_TOPK=False`. The first two avoid a missing `tilelang` dependency; the third
  avoids a JIT build of the hash top-k kernel that the image cannot compile.
- On the meetup image the DSv4 sparse-attention op lives in the `custom_transformer` CANN vendor. Source
  `opp/vendors/custom_transformer/bin/set_env.bash` (and the `customize` one) and put those vendor dirs on
  `ASCEND_CUSTOM_OPP_PATH`; without it you get
  `aclnnKvQuantSparseAttnSharedkvMetadata … not in libopapi.so`.
- DeepEP needs a larger HCCL window than the 200 MB default: `export HCCL_BUFFSIZE=1200` and
  `export DEEPEP_HCCL_BUFFSIZE=2048`, otherwise `aclnnCamMoeDispatchNormal` fails with a tiling error.
- DSv4 decode graph capture is not usable in this build: add `--disable-cuda-graph` (eager decode) and label
  numbers accordingly.
- `--quantization modelslim` is required for the W8A8 checkpoint. `bfloat16` KV is the validated setting;
  FP8 KV is not available on A3/A2 and needs the 950 MX path.
- Start at `--mem-fraction-static 0.75` on a 4-device slice (the docs' 0.8 assumes 8 dies). If startup OOMs,
  drop to 0.70, then 0.65, before touching anything else.
- EAGLE numbers above are the documented speculative configuration; the W8A8 checkpoint name includes the
  bundled `mtp` head, which is what makes speculation work without a separate draft model.
- ZBAL (`SGLANG_ZBAL_*`, `ZBAL_*`) appears in the validated 8-die config. **Leave it off for the 4-device
  first run**; enable it only after a baseline works, and scale `SGLANG_ZBAL_LOCAL_MEM_SIZE` with the number
  of dies rather than copying `62084` verbatim.

Readiness: the log must print `The server is fired up and ready to roll!`; `/health` returns 200.

## 4. Smoke test

```bash
curl -s http://127.0.0.1:6688/health
curl -s http://127.0.0.1:6688/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64}'
```

The docs' acceptance criterion is that the response contains `Paris`. V4 ships **no Jinja chat template**
(the model repository uses an `encoding/` Python API instead), so if `/v1/chat/completions` returns a
template error, use the native `/generate` endpoint — SGLang's dsv4 path handles the encoding. Record which
endpoint you used: it changes what a later benchmark measures.

## 5. Benchmark

Use `deploy/03_launch.sh` → `bench/run_matrix.sh` (5 cells, ~20 min) → `bench/summarize.py`. The matrix and
its interpretation are documented in [`../docs/measurements.md`](docs/measurements.md).

Rules that make the numbers defensible:

- `--warmup-requests` > 0 always; discard the first cell run after server start.
- flush prefix cache between cells (`--flush-cache`) — otherwise the second cell's TTFT is a cache hit, not a
  prefill measurement.
- same seed and prompt lengths across configurations you intend to compare.
- record the full tuple with every result: chip type/count, CANN, torch_npu, SGLang version, image tag,
  tp/dp, quantization, context cap, command line.
- report median and p90/p99, not the mean alone.

## 6. Known traps

See `docs/troubleshooting.md`. The four that end most lab attempts early: wrong `--tp-size`;
`source set_env.sh` not run (CANN "not found"); startup OOM from an over-large `--mem-fraction-static` or
context cap; and 4×A2 allocation, where the checkpoint simply does not fit.
