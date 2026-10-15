# DeepSeek-V4.1 mono kernels (gfx950)

The fused decode-layer kernels (whole-layer K1 + K2, DSpark draft stages) and the K2i indexer scorer.
Design and scope: `README.md`. Host wiring (spec, weight views, step metadata, the attention backends' indexer hook)
lives in `sglang.kernels.ops.mono.deepseek_v41_gfx950`.

## Origin

- `layer.py`, `runner.py`, `sources.py`, `stages/`, `attention/`: vendored from
  [vllm-project/vllm](https://github.com/vllm-project/vllm) PR #60397 @ 193922d6ff,
  `vllm/models/deepseek_v41/amd/mono/` (Apache-2.0, SPDX header in every file). The seam and MoE stages are adapted
  from ROCm/ATOM `atom/models/deepseek_v41/mono`; the attention stages are new in vLLM; `attention/device.py` adapts
  the FlyDSL GLM-5 layer kernel's primitives.
- `indexer/`: K2i, vendored from [ROCm/ATOM](https://github.com/ROCm/ATOM) @ 68506823,
  `atom/models/deepseek_v41/mono/kernels/` (MIT), adapted to SGLang's FP4 index plane.

## Shared code

Every device helper comes from `../common` (ATOM @ 454d0e3023). vLLM's private copy of it (`common/`, adapted from
ATOM @ 97359d6) was removed: its definitions were identical to `../common` apart from docstrings, except `Mailbox`
(epoch tags: now `Mailbox(tag=...)`), `cdiv` (traced values: now `plan.arith.cdiv`), `stamp` (gained `@traced`) and
`PeerBuffer` (now `sglang.kernels.ops.mono.common_hip.peer`).
`attention/device.py` keeps only the attention-specific primitives; two names stay local on purpose:
`kernel_symbol` (symbols in the `aiter` namespace) and `stamp(c, point)` (the attention timeline, a different helper
from `common.device.stamps.stamp`). `test/registered/mono/test_mono_specs.py` fails on any other redefinition.

The JIT cache key (`sources.SOURCES`) digests this directory and `../common`, so an edit to either rebuilds.
