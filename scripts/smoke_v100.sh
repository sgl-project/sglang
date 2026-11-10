#!/usr/bin/env bash
# Validate an already-built SGLang-V100 environment and warm lazy sampling JIT.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  printf '[smoke_v100] Do not source this file; run: bash %q\n' \
    "${BASH_SOURCE[0]}" >&2
  return 2
fi

set -Eeuo pipefail

if [[ -n "${SGLANG_V100_PYTHON:-}" ]] && [[ -x "$SGLANG_V100_PYTHON" ]]; then
  PYTHON="$SGLANG_V100_PYTHON"
elif [[ -x "$HOME/miniconda3/envs/sglang-v100/bin/python" ]]; then
  PYTHON="$HOME/miniconda3/envs/sglang-v100/bin/python"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "sglang-v100" ]] && \
  [[ -x "${CONDA_PREFIX:-}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  printf '[smoke_v100] sglang-v100 Python was not found. Activate the environment first.\n' >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$PYTHON" - <<'PY'
import glob
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import flashinfer.sampling as flashinfer_sampling
import sgl_kernel
from flashinfer.sampling import top_k_top_p_sampling_from_probs
from sglang.srt.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from sglang.srt.function_call.base_format_detector import get_model_structural_tag
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.srt.layers.quantization.marlin_utils import (
    _sm70_marlin_v100_repack_ops,
)
from sglang.srt.layers.quantization.sm70_fp16_moe import _load_sm70_ops
from sglang.srt.layers.quantization.sm70_turbomind_fp8 import (
    _load_sm70_turbomind_fp8_ops,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

expected = os.environ.get("SGLANG_V100_FLASHINFER_DIR")
sampling_path = Path(flashinfer_sampling.__file__).resolve()
if expected:
    expected_path = Path(expected).resolve()
    assert sampling_path.is_relative_to(expected_path), (
        f"wrong FlashInfer source: {sampling_path}; expected {expected_path}"
    )

assert torch.__version__.startswith("2.9.1"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (7, 0)
assert "/sm70/" in sgl_kernel.common_ops.__file__.replace("\\", "/")
assert NCCLLibrary().ncclGetRawVersion() == 22705
gptq_repack, awq_repack = _sm70_marlin_v100_repack_ops()
assert gptq_repack is not None, (
    "marlin_v100 GPTQ repack is missing; the stock SM70 repack silently "
    "produces zero weights"
)
assert awq_repack is not None, "marlin_v100 AWQ repack is missing"
assert _load_sm70_turbomind_fp8_ops(), "TurboMind SM70 FP8 ops are missing"
assert _load_sm70_ops(), "TurboMind SM70 FP16 MoE ops are missing"
assert hasattr(torch.ops.sglang_sm70_turbomind, "awq_dequantize_out"), (
    "TurboMind SM70 exact AWQ dequantizer is missing; rebuild against the "
    "pinned, attributed TurboMind source subset"
)

# Qwen3.8 MTP target verification and draft extension enter the paged QSA
# path through forward_extend(). They must select the native SM70 decoder;
# otherwise graph capture falls through to an unavailable Ampere FA2 wheel.
qsa_rows = 4
qsa_q = torch.empty(
    (qsa_rows, 6, 256), device="cuda", dtype=torch.float16
)
qsa_k = torch.empty((16, 1, 256), device="cuda", dtype=torch.float8_e5m2)
qsa_v = torch.empty_like(qsa_k)
qsa_indices = torch.zeros((qsa_rows, 1), device="cuda", dtype=torch.int32)
qsa_metadata = SimpleNamespace(
    sequence_lengths=torch.ones(qsa_rows, device="cuda", dtype=torch.int32),
)
for qsa_mode in (
    ForwardMode.DECODE,
    ForwardMode.TARGET_VERIFY,
    ForwardMode.DRAFT_EXTEND_V2,
):
    assert QwenSparseAttnBackend._can_use_sm70_sparse_decode(
        qsa_q,
        qsa_k,
        qsa_v,
        SimpleNamespace(forward_mode=qsa_mode),
        qsa_metadata,
        qsa_indices,
    ), f"native SM70 QSA routing is disabled for {qsa_mode.name}"

# Cold chunked prefill has no radix prefix on its first 8K chunk. Exercise
# the self-contained TileLang dense route that handles it; the runtime image
# intentionally has no external flash_attn_v100 package to fall back to.
qsa_prefill_rows = 64
qsa_prefill_q = torch.randn(
    (qsa_prefill_rows, 6, 256), device="cuda", dtype=torch.float16
) * 0.02
qsa_prefill_k = torch.randn(
    (qsa_prefill_rows, 1, 256), device="cuda", dtype=torch.float16
) * 0.02
qsa_prefill_v = torch.randn_like(qsa_prefill_k) * 0.02
qsa_prefill_output = QwenSparseAttnBackend._forward_sm70_dense_prefill(
    qsa_prefill_q,
    qsa_prefill_k,
    qsa_prefill_v,
    [qsa_prefill_rows],
    1.0 / 16.0,
)
assert qsa_prefill_output.shape == qsa_prefill_q.shape
assert torch.isfinite(qsa_prefill_output).all()

# Exercise the exact W8A16 block-FP8 operator used by Qwen3.6-27B-FP8.
torch.manual_seed(7)
weight = (
    torch.randn((128, 128), device="cuda", dtype=torch.float16) * 0.1
).to(torch.float8_e4m3fn)
scales = torch.ones((1, 1), device="cuda", dtype=torch.float32)
packed, packed_scales, meta = torch.ops.sglang_sm70_turbomind.fp8_prepare(
    weight, scales, 128, False
)
activation = torch.randn((3, 128), device="cuda", dtype=torch.float16)
actual = torch.empty((3, 128), device="cuda", dtype=torch.float16)
torch.ops.sglang_sm70_turbomind.fp8_gemm(
    actual,
    activation,
    packed,
    packed_scales,
    128,
    int(meta[0]),
    int(meta[1]),
    False,
)
expected = activation @ weight.to(torch.float16).T
assert torch.equal(actual, expected), (actual - expected).abs().max()

# Exercise the routed FP16 expert projection used by Qwen3.6-35B-A3B.  This
# also verifies that the unified extension contains the weight converter,
# strided expert pointers, and fused gated-SiLU epilogue.
num_experts, rows, hidden_size, output_size = 4, 32, 128, 128
expert_weight = (
    torch.randn(
        (num_experts, output_size, hidden_size),
        device="cuda",
        dtype=torch.float16,
    )
    * 0.01
)
prepared_weight, prepared_meta = torch.ops.sglang_sm70_turbomind.f16_prepare(
    expert_weight.reshape(num_experts * output_size, hidden_size).contiguous()
)
prepared_weight = prepared_weight.reshape(
    num_experts, output_size, hidden_size
)
expert_ptrs = torch.ops.sglang_sm70_turbomind.f16_moe_build_ptrs(
    prepared_weight, int(prepared_meta[0]), num_experts
)[0]
expert_input = (
    torch.randn((rows, hidden_size), device="cuda", dtype=torch.float16) * 0.1
)
expert_output = torch.empty(
    (rows, output_size // 2), device="cuda", dtype=torch.float16
)
expert_offsets = torch.arange(
    0,
    rows + 1,
    rows // num_experts,
    device="cuda",
    dtype=torch.int32,
)
torch.ops.sglang_sm70_turbomind.f16_moe_gemm(
    expert_output,
    expert_input,
    expert_offsets,
    expert_ptrs,
    num_experts,
    hidden_size,
    output_size,
    True,
)
expert_reference = torch.empty_like(expert_output)
rows_per_expert = rows // num_experts
for expert in range(num_experts):
    begin = expert * rows_per_expert
    end = begin + rows_per_expert
    projected = torch.nn.functional.linear(
        expert_input[begin:end], expert_weight[expert]
    )
    expert_reference[begin:end] = (
        torch.nn.functional.silu(projected[:, 0::2]) * projected[:, 1::2]
    )
expert_error = (expert_output - expert_reference).abs().max()
assert expert_error <= 5e-4, expert_error

# SGLang uses distinct CUDA streams for graph capture and overlap scheduling.
# TurboMind workspaces must therefore be isolated per stream: a single shared
# scratch buffer can pass the check above while corrupting concurrent GEMMs.
wide_weight = (
    torch.randn((1024, 1024), device="cuda", dtype=torch.float16) * 0.05
).to(torch.float8_e4m3fn)
wide_scales = torch.ones((8, 8), device="cuda", dtype=torch.float32)
wide_packed, wide_packed_scales, wide_meta = (
    torch.ops.sglang_sm70_turbomind.fp8_prepare(
        wide_weight, wide_scales, 128, False
    )
)
wide_k_ld, wide_q_ld = int(wide_meta[0]), int(wide_meta[1])
wide_inputs = [
    torch.randn((64, 1024), device="cuda", dtype=torch.float16)
    for _ in range(2)
]
wide_baselines = []
for wide_input in wide_inputs:
    wide_output = torch.empty((64, 1024), device="cuda", dtype=torch.float16)
    torch.ops.sglang_sm70_turbomind.fp8_gemm(
        wide_output,
        wide_input,
        wide_packed,
        wide_packed_scales,
        128,
        wide_k_ld,
        wide_q_ld,
        False,
    )
    wide_baselines.append(wide_output)
torch.cuda.synchronize()
streams = [torch.cuda.Stream(), torch.cuda.Stream()]
stream_outputs = [
    [torch.empty((64, 1024), device="cuda", dtype=torch.float16) for _ in range(4)]
    for _ in streams
]
for stream, wide_input, outputs in zip(streams, wide_inputs, stream_outputs):
    with torch.cuda.stream(stream):
        for output in outputs:
            torch.ops.sglang_sm70_turbomind.fp8_gemm(
                output,
                wide_input,
                wide_packed,
                wide_packed_scales,
                128,
                wide_k_ld,
                wide_q_ld,
                False,
            )
torch.cuda.synchronize()
for baseline, outputs in zip(wide_baselines, stream_outputs):
    for output in outputs:
        assert torch.equal(output, baseline), (output - baseline).abs().max()

# The Docker dependency set pins XGrammar 0.1.32, whose builtin helper uses a
# different Qwen Coder model key and API than earlier releases.
assert get_model_structural_tag is not None
structural_tag = get_model_structural_tag(
    model="qwen_3_coder",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "execute",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ],
    tool_choice="auto",
    reasoning=False,
)
assert structural_tag is not None, "Qwen Coder structural tag is unavailable"

loaded_common_ops = Path(sgl_kernel.common_ops.__file__).resolve()
assert loaded_common_ops.name == "common_ops.abi3.so", loaded_common_ops
common_ops = [
    Path(path).resolve()
    for path in glob.glob(
        str(Path(sgl_kernel.__file__).parent / "sm70" / "common_ops*.so")
    )
]
for stale_artifact in common_ops:
    if stale_artifact != loaded_common_ops:
        stale_artifact.unlink()
        print("Removed stale kernel artifact:", stale_artifact)
common_ops = list(loaded_common_ops.parent.glob("common_ops*.so"))
assert common_ops == [loaded_common_ops], f"stale common_ops variants remain: {common_ops}"

# FlashInfer builds this module lazily. Doing it here prevents the first
# non-greedy chat request from paying the cold compilation cost.
probs = torch.full((1, 128), 1.0 / 128, device="cuda", dtype=torch.float32)
top_k = torch.tensor([20], device="cuda", dtype=torch.int32)
top_p = torch.tensor([0.8], device="cuda", dtype=torch.float32)
top_k_top_p_sampling_from_probs(
    probs, top_k, top_p, filter_apply_order="joint"
)
torch.cuda.synchronize()

print("SGLang V100 environment is ready:", torch.__version__)
print("FlashInfer SM70 sampling:", sampling_path)
print("Attention: SGLang TileLang SM70 package")
print("SM70 kernel:", sgl_kernel.common_ops.__file__)
print("SM70 Marlin repack: registered")
print("SM70 TurboMind FP8: registered")
print("SM70 TurboMind FP16 MoE: registered")
print("SM70 TurboMind exact AWQ dequantizer: registered")
print("NCCL:", NCCLLibrary().ncclGetVersion())
PY
