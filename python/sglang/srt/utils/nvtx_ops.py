import torch

# Add for nvtx profiling
try:
    from torch.cuda.nvtx import range_pop, range_push

    NVTX_OP = True
except ImportError:
    NVTX_OP = False

if NVTX_OP:

    @torch.library.custom_op(
        "sglang::nvtx_push_range_for_gemm", mutates_args=("input_tensor",)
    )
    def nvtx_push_range_for_gemm(
        op_name: str,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        weight_shape = getattr(weight, "shape", None)
        weight_scale_shape = getattr(weight_scale, "shape", None)
        input_scale_shape = getattr(input_scale, "shape", None)
        bias_shape = getattr(bias, "shape", None)
        if torch.cuda.is_current_stream_capturing():
            nvtx_message = f"[FW_GEMM] op:{op_name},type:G,input:{input_tensor.shape},weight:{weight_shape},weight_scale:{weight_scale_shape},input_scale:{input_scale_shape},bias:{bias_shape}"
        else:
            nvtx_message = f"[FW_GEMM] op:{op_name},type:E,input:{input_tensor.shape},weight:{weight_shape},weight_scale:{weight_scale_shape},input_scale:{input_scale_shape},bias:{bias_shape}"
        range_push(nvtx_message)

    @nvtx_push_range_for_gemm.register_fake
    def nvtx_push_range_for_gemm_fake(
        op_name: str,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        pass

    @torch.library.custom_op(
        "sglang::nvtx_pop_range_for_gemm", mutates_args=("input_tensor",)
    )
    def nvtx_pop_range_for_gemm(input_tensor: torch.Tensor) -> None:
        range_pop()

    @nvtx_pop_range_for_gemm.register_fake
    def nvtx_pop_range_for_gemm_fake(input_tensor: torch.Tensor) -> None:
        pass

else:

    @torch.library.custom_op(
        "sglang::nvtx_push_range_for_gemm", mutates_args=("input_tensor",)
    )
    def nvtx_push_range_for_gemm(
        op_name: str,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        pass

    @nvtx_push_range_for_gemm.register_fake
    def nvtx_push_range_for_gemm_fake(
        op_name: str,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        pass

    @torch.library.custom_op(
        "sglang::nvtx_pop_range_for_gemm", mutates_args=("input_tensor",)
    )
    def nvtx_pop_range_for_gemm(input_tensor: torch.Tensor) -> None:
        pass

    @nvtx_pop_range_for_gemm.register_fake
    def nvtx_pop_range_for_gemm_fake(input_tensor: torch.Tensor) -> None:
        pass
