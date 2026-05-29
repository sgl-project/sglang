"""Two-stream attention LoRA forward implementations (O7 + O8).

These are monkey-patched onto :class:`QKVParallelLinearWithLoRA` and
:class:`RowParallelLinearWithLoRA` by
:func:`sglang.srt.lora.trtllm_moe.install_two_stream_overrides` when
``SGLANG_LORA_TWO_STREAM=1``. The saved-original forward methods are
preserved and called for batches where two-stream isn't active.
"""
import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.lora.trtllm_moe import (
    get_lora_side_stream,
    get_original_qkv_forward,
    get_original_row_forward,
    is_two_stream_active,
)


def qkv_proj_lora_forward(self, input_: torch.Tensor):
    """O7 — side-stream LoRA-A shrink ‖ base qkv_proj GEMM.

    The shrink reads ``input_`` and the LoRA-A weights — same input as the
    base GEMM, no write conflict. The expand needs the shrink intermediate
    AND base_output, so it runs after the rejoin on the main stream.
    """
    if not self.set_lora or not is_two_stream_active(input_):
        return get_original_qkv_forward()(self, input_)

    from sglang.srt.lora.triton_ops import qkv_lora_b_fwd, sgemm_lora_a_fwd

    bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
    side_stream = get_lora_side_stream()
    # sgemm_info is host-side (LoRABatchInfo); compute once, share both calls.
    sgemm_info = self.lora_backend._sgemm_info()

    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        shrink_intermediate = sgemm_lora_a_fwd(
            input_, self.A_buffer_qkv, sgemm_info, stack_num=3
        )

    # Base qkv_proj GEMM on main, concurrent with the side-stream shrink.
    output_parallel = self.base_layer.quant_method.apply(
        self.base_layer, input_, bias
    )

    # Rejoin: expand reads both side-produced shrink_intermediate and base_output.
    torch.cuda.current_stream().wait_stream(side_stream)
    output_parallel = qkv_lora_b_fwd(
        shrink_intermediate,
        self.B_buffer_qkv,
        sgemm_info,
        self.output_offset,
        self.max_qkv_out_dim,
        output_parallel,
        n_slices=3,
    )

    if self.base_layer.gather_output:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    return output, output_bias


def row_parallel_lora_forward(
    self, input_: torch.Tensor, skip_all_reduce: bool = False, forward_batch=None
):
    """O8 — side-stream LoRA-A shrink ‖ base row-parallel (o_proj) GEMM.

    Mirrors O7 but the row-parallel context adds: input split per TP rank
    (when not already parallel), bias on rank 0 only, optional cross-rank
    all-reduce on both base output and lora_a intermediate when reducing.

    Falls back to the saved-original :meth:`forward` for non-decode batches
    or when LoRA isn't set on this layer.
    """
    # We need ``input_parallel`` to gate the per-batch decode check (its
    # token-count drives the threshold, not the unsplit ``input_``).
    if self.base_layer.input_is_parallel:
        input_parallel = input_
    else:
        tp_rank = get_tensor_model_parallel_rank()
        splitted_input = split_tensor_along_last_dim(
            input_, num_partitions=self.base_layer.tp_size
        )
        input_parallel = splitted_input[tp_rank].contiguous()

    if not self.set_lora or not is_two_stream_active(input_parallel):
        return get_original_row_forward()(self, input_, skip_all_reduce, forward_batch)

    bias_ = (
        None
        if (self.base_layer.tp_rank > 0 or self.base_layer.skip_bias_add)
        else self.base_layer.bias
    )

    side_stream = get_lora_side_stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        lora_a_output = self.lora_backend.run_lora_a_sgemm(
            input_parallel, self.A_buffer
        )

    # Base row-parallel GEMM on main, concurrent with the side-stream shrink.
    output_parallel = self.base_layer.quant_method.apply(
        self.base_layer, input_parallel, bias=bias_
    )

    torch.cuda.current_stream().wait_stream(side_stream)

    should_reduce = (
        self.base_layer.reduce_results
        and self.base_layer.tp_size > 1
        and not skip_all_reduce
    )

    if should_reduce:
        output_ = tensor_model_parallel_all_reduce(output_parallel)
        lora_a_output = tensor_model_parallel_all_reduce(lora_a_output)
        output_ = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            output_offset_cpu=self.output_offset_cpu,
            base_output=output_,
        )
    else:
        # Two-stream already produced lora_a_output on the side stream; finish
        # the LoRA with just the expand atomic-add against output_parallel.
        output_parallel = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            output_offset_cpu=self.output_offset_cpu,
            base_output=output_parallel,
        )
        output_ = output_parallel

    output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
    return output_, output_bias
