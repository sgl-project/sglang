"""The shared row-projection forward program for matched trainer/rollout TP1."""


def matched_tp1_row_linear(x, weight, bias=None):
    """Weight is [output, input]; both engines call this exact implementation."""
    from sglang.srt.batch_invariant_ops import matmul_persistent

    return matmul_persistent(x.contiguous(), weight.t(), bias=bias)


def top_row_parallel_forward(layer, input_parallel, *, bias, output_tensor):
    """TOP row program for RowParallelLinear; None when the stock path applies."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.true_on_policy.config import should_use_matched_tp1_row_linear

    if not should_use_matched_tp1_row_linear():
        return None
    if output_tensor is not None:
        raise ValueError("Matched TP1 row linear writes no caller-owned output buffer")
    if isinstance(layer.quant_method, Fp8LinearMethod):
        from sglang.srt.true_on_policy import fp8

        # The checkpoint's block format selects the program (fp8.maybe_linear's dispatch).
        out = fp8.maybe_linear(layer.quant_method, layer, input_parallel, bias)
        if out is None:
            raise NotImplementedError(
                "[top] an FP8 row projection needs TOP's block-FP8 program"
            )
        return out
    return matched_tp1_row_linear(input_parallel, layer.weight, bias)
