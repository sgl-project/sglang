"""Two-stream MergedColumnParallelLinear LoRA forward (O9).

Monkey-patched onto :class:`MergedColumnParallelLinearWithLoRA` by
:func:`sglang.srt.lora.trtllm_lora_temp.install_two_stream_overrides` when
``SGLANG_LORA_TWO_STREAM=1``. Covers the merged-column LoRA modules not handled
by O7 (QKV) / O8 (o_proj) / O1 (MoE experts):

  * Qwen3.5 mamba ``in_proj_qkvz`` (a MergedColumnParallelLinear, every mamba layer)
  * dense ``gate_up_proj`` MLP layers (e.g. Qwen3-VL non-expert MLP)

Same shape as O7: the LoRA-A shrink reads ``input_`` (same input as the base
GEMM, no write conflict) and runs on the side stream concurrent with the base
merged-column GEMM on the main stream; the LoRA-B expand needs both the shrink
output and base_output, so it runs after the rejoin on the main stream. The
expand mirrors ``MergedColumnParallelLinearWithLoRA.apply_lora`` — gate_up vs
general n-slice.
"""

import torch

from sglang.srt.distributed import tensor_model_parallel_all_gather
from sglang.srt.lora.trtllm_lora_temp import (
    get_lora_side_stream,
    get_original_merged_column_forward,
    is_two_stream_active,
    lora_overlap_alloc_stream,
)


def merged_column_lora_forward(self, input_: torch.Tensor):
    """O9 — side-stream LoRA-A shrink ‖ base merged-column GEMM."""
    if not self.set_lora or not is_two_stream_active(input_):
        return get_original_merged_column_forward()(self, input_)

    from sglang.srt.lora.trtllm_lora_temp.triton_ops import (
        gate_up_lora_b_fwd,
        qkv_lora_b_fwd,
        sgemm_lora_a_fwd,
    )

    bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
    side_stream = get_lora_side_stream()
    # sgemm_info is host-side (LoRABatchInfo); compute once, share both calls.
    sgemm_info = self.lora_backend._sgemm_info()
    lora_n_slices = self._get_lora_n_slices()
    use_gate_up = lora_n_slices == 2 and self.use_gate_up_lora

    # Shrink on side stream, concurrent with the base merged-column GEMM on main.
    _alloc = lora_overlap_alloc_stream()  # capture MAIN stream here (before the fork)
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        shrink_intermediate = sgemm_lora_a_fwd(
            input_,
            self.A_buffer,
            sgemm_info,
            stack_num=lora_n_slices,
            out_alloc_stream=_alloc,
        )

    output_parallel = self.base_layer.quant_method.apply(self.base_layer, input_, bias)

    # Rejoin: expand reads both side-produced shrink_intermediate and base_output.
    torch.cuda.current_stream().wait_stream(side_stream)
    if use_gate_up:
        output_dim = self.B_buffer.shape[-2] // 2
        output_parallel = gate_up_lora_b_fwd(
            shrink_intermediate,
            self.B_buffer,
            sgemm_info,
            output_dim,
            output_parallel,
        )
    else:
        output_parallel = qkv_lora_b_fwd(
            shrink_intermediate,
            self.B_buffer,
            sgemm_info,
            self.output_offset,
            self.max_out_dim,
            output_parallel,
            n_slices=lora_n_slices,
        )

    if self.base_layer.gather_output:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    return output, output_bias
