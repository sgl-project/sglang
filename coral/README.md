# CORAL Qwen3.6 H100 overlay for SGLang v0.5.22

This directory contains the experimental kernel and speculative-decoding hooks used for the
MLCommons Agentic Inference submission on one H100 SXM per server. The base is the public
`lmsysorg/sglang:v0.5.22-cu130` image and the `v0.5.22` source tag
(`baa9befa381511cd4d56f83980951e5a45fffc6d`).

The overlay is loaded with `PYTHONPATH` and `sitecustomize.py`. Every switch is opt-in.
It does not change the model checkpoint or the target LM head.

## Components

- `q6_penalties.patch` — presence, frequency and repetition penalties under EAGLE/NEXTN
  speculation. The patch applies earlier draft tokens per verify row and feeds every newly
  committed token back into the penalizers. This is required for token-for-token equivalence
  with the non-speculative target.
- `sitecustomize.py` — post-import hooks for the `CORAL_*` switches.
- `coral_kernels.py` — Triton kernels for H100 FP8 decode, MoE, GDN and fused bookkeeping.
- `coral_mma.py` — W8A16 MMA for M=1..16 decode and verify rows.
- `coral_spec_fuse.py` — fused draft-extend post-processing for NEXTN top-k 1.
- The remaining `coral_*.py` files are optional DFlash2 experiments and are not enabled by
  the Track A environment below.

## H100 FP8 + MTP-4 environment

```bash
PYTHONPATH=/opt/coral
CORAL_MOE=1 CORAL_MOE_ROUTE=v9 CORAL_NORMQUANT=1 CORAL_GEMV=1
CORAL_FP8GEMV=1 CORAL_FP8GEMV_NS=1 CORAL_GDN_BV=4
CORAL_GDN_SPLIT_VIEWS=1 CORAL_GDN_VERIFY_BV=4 CORAL_FP8GEMV_MMA=1
CORAL_MAX_M=8 CORAL_MOE_MAX_M=8 CORAL_SPEC_MROPE_FAST=1
CORAL_NORM_NOQUANT=1 CORAL_NORM_ROUTER=1 CORAL_GDN_VERIFY_VIEWS=1
CORAL_GDN_ZVIEW=1 CORAL_FA3_DRAFT_EXTEND_GRAPH=1 CORAL_SPEC_FUSE=1
```

Server arguments:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
  --served-model-name qwen3.6-35b-a3b-fp8 \
  --host 0.0.0.0 --port 30000 --tp-size 1 \
  --mem-fraction-static 0.85 --context-length 262144 \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --speculative-token-map /data/tokmap/top49152.pt \
  --enable-linear-replayssm-spec
```

## Build once

```bash
docker build -f coral/Dockerfile -t coral/sglang:v0.5.22-q6-coral .
```

The image applies `q6_penalties.patch` to the SGLang checkout in the base image. If the
checkout in a future base image has moved, the build fails instead of serving an unpatched
source tree.

## Test

```bash
python -m pytest coral/test_dispatch.py coral/test_spec_fuse_patterns.py -q
python -m pytest test/registered/unit/spec/test_eagle_penalties.py -q
```

`test_dispatch.py` stubs Triton and checks the kernel routing. Kernel numerics are checked on
the GPU by the benchmark scripts next to this overlay in the private evaluation tree.
