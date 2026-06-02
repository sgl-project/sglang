from __future__ import annotations

from typing import Optional

import torch

import sglang.srt.layers.quantization.fp8_kernel as _fp8_kernel

try:
    from aiter.fused_moe import moe_sorting as _aiter_moe_sorting
except ImportError:
    _aiter_moe_sorting = None

try:
    from aiter import get_hip_quant as _aiter_get_hip_quant
    from aiter import partial_transpose as _aiter_partial_transpose
    from aiter.ops.enum import QuantType as _AiterQuantType

    _aiter_per_group_quant = _aiter_get_hip_quant(_AiterQuantType.per_1x128)
except ImportError:
    _aiter_partial_transpose = None
    _aiter_per_group_quant = None

try:
    from petit_kernel import mul_nvfp4_a16, process_nvfp4_scales, repack_nvfp4

    _PETIT_NVFP4_IMPORT_ERROR = None
except ImportError as e:
    mul_nvfp4_a16 = None
    process_nvfp4_scales = None
    repack_nvfp4 = None
    _PETIT_NVFP4_IMPORT_ERROR = e

    def _raise_petit_import_error() -> None:
        raise ValueError(
            "Petit is not installed. Please install it with `pip install petit-kernel`."
        )

    def _check_petit_nvfp4_supported(
        quant_method: str, group_size: Optional[int]
    ) -> tuple[bool, Optional[str]]:
        return (
            False,
            "Petit is not installed. Please install it with `pip install petit-kernel`.",
        )

    def prepare_nvfp4_layer_for_petit(layer: torch.nn.Module) -> None:
        _raise_petit_import_error()

    def apply_petit_nvfp4_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        size_n: int,
        size_k: int,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _raise_petit_import_error()

    def prepare_mxfp4_layer_for_petit(layer: torch.nn.Module) -> None:
        _raise_petit_import_error()

    def apply_petit_mxfp4_dense(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        size_n: int,
        size_k: int,
        bias: Optional[torch.Tensor] = None,
        global_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _raise_petit_import_error()

    def materialize_petit_mxfp4_weight(
        layer: torch.nn.Module,
        out_dtype: torch.dtype = torch.bfloat16,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        _raise_petit_import_error()

    def prepare_mxfp4_moe_layer_for_petit(layer: torch.nn.Module) -> None:
        _raise_petit_import_error()

    def apply_petit_mxfp4_moe(
        hidden_states: torch.Tensor,
        hidden_states_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w13_size_n: int,
        w13_size_k: int,
        w2_size_n: int,
        w2_size_k: int,
        num_local_experts: int,
        activation: str,
        expert_mask: Optional[torch.Tensor] = None,
        num_local_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _raise_petit_import_error()


fp8_dtype = _fp8_kernel.fp8_dtype

_PETIT_MOE_BLOCK_SIZE = 32


try:
    from petit_kernel import mul_mxfp4_a16, repack_mxfp4
except ImportError:
    mul_mxfp4_a16 = None
    repack_mxfp4 = None

try:
    from petit_kernel import (
        fused_moe_fp8_blockscale_g1u1_mxfp4 as _raw_fused_moe_fp8_blockscale_g1u1_mxfp4,
    )
    from petit_kernel import (
        process_mxfp4_scales,
        repack_moe_mxfp4_kernel_layout,
    )

    from sglang.srt.utils.custom_op import register_custom_op

    try:
        from petit_kernel import repack_moe_w13_mxfp4_kernel_layout
    except ImportError:
        repack_moe_w13_mxfp4_kernel_layout = repack_moe_mxfp4_kernel_layout
except ImportError:
    _raw_fused_moe_fp8_blockscale_g1u1_mxfp4 = None
    process_mxfp4_scales = None
    repack_moe_mxfp4_kernel_layout = None
    repack_moe_w13_mxfp4_kernel_layout = None
    register_custom_op = None


def _check_petit_nvfp4_supported(
    quant_method: str, group_size: Optional[int]
) -> tuple[bool, Optional[str]]:
    if quant_method != "NVFP4":
        return (
            False,
            "Petit currently only supports: NVFP4"
            " quantizations in sglang. Please check the "
            "`hf_quant_config.json` file for your model's "
            "quant configuration.",
        )
    if group_size is not None and group_size != 16:
        return (
            False,
            "Petit currently only supports: group_size=16 quantizations.",
        )
    return (True, None)


def verify_petit_nvfp4_supported(quant_method: str, group_size: Optional[int]) -> None:
    supported, error_msg = _check_petit_nvfp4_supported(quant_method, group_size)
    if not supported:
        raise ValueError(error_msg)


def _require_petit_nvfp4() -> None:
    if (
        mul_nvfp4_a16 is not None
        and process_nvfp4_scales is not None
        and repack_nvfp4 is not None
    ):
        return
    error = (
        f" Import failed with: {_PETIT_NVFP4_IMPORT_ERROR}"
        if _PETIT_NVFP4_IMPORT_ERROR is not None
        else ""
    )
    raise ValueError(
        "Petit NVFP4 kernels are not available. Please install/rebuild "
        f"petit-kernel with NVFP4 support.{error}"
    )


def _default_global_scale(device: torch.device) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.ones((1,), dtype=torch.float32, device=device),
        requires_grad=False,
    )


def _quantize_fp8_block128(
    x: torch.Tensor,
    num_local_tokens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, model_dim = x.shape
    if model_dim % 128 != 0:
        raise ValueError(f"hidden size must be divisible by 128, got {model_dim}")

    if _aiter_per_group_quant is None:
        raise RuntimeError(
            "Petit MXFP4 MoE requires AITER per_1x128 quantization on ROCm."
        )
    input_q, input_scale = _aiter_per_group_quant(
        x,
        group_size=128,
        quant_dtype=fp8_dtype,
        num_rows=num_local_tokens,
    )

    if num_local_tokens is not None and _aiter_partial_transpose is not None:
        input_scale_layout = torch.empty(
            (model_dim // 128, tokens), dtype=torch.float32, device=x.device
        )
        _aiter_partial_transpose(
            input_scale_layout,
            input_scale,
            num_rows=num_local_tokens,
        )
    else:
        input_scale_layout = input_scale.transpose(0, 1).contiguous()

    return input_q.contiguous(), input_scale_layout.contiguous()


def _normalize_num_local_tokens(
    num_local_tokens: Optional[torch.Tensor],
    *,
    max_tokens: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if num_local_tokens is None:
        return None

    num_local_tokens_i32 = num_local_tokens.to(device=device, dtype=torch.int32)
    num_local_tokens_i32 = torch.clamp(
        num_local_tokens_i32.reshape(-1)[:1],
        min=0,
        max=max_tokens,
    ).contiguous()
    return num_local_tokens_i32


def _prepare_petit_aiter_sorting_inputs(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorting_expert_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Prepare Petit inputs for AITER sorting without changing global IDs.

    AITER accepts valid global expert IDs and uses expert_mask to skip non-local
    experts. For invalid IDs, reuse the DeepEP-style sentinel: append one masked
    expert slot and remap invalid IDs to that slot before sorting.
    """

    num_base_experts = int(sorting_expert_mask.numel())
    num_sorting_experts = num_base_experts + 1
    sorting_expert_mask_prepared = torch.empty(
        (num_sorting_experts,),
        dtype=torch.int32,
        device=sorting_expert_mask.device,
    )
    sorting_expert_mask_prepared[:num_base_experts].copy_(sorting_expert_mask)
    sorting_expert_mask_prepared[num_base_experts:].zero_()

    topk_ids_i32 = topk_ids.to(torch.int32).contiguous()
    sentinel_expert = torch.full(
        (),
        num_base_experts,
        dtype=topk_ids_i32.dtype,
        device=topk_ids_i32.device,
    )
    topk_ids_i32 = torch.where(
        (topk_ids_i32 >= 0) & (topk_ids_i32 < num_base_experts),
        topk_ids_i32,
        sentinel_expert,
    )
    topk_weights_f32 = topk_weights.to(torch.float32).contiguous()

    return (
        topk_ids_i32,
        topk_weights_f32,
        sorting_expert_mask_prepared,
        num_sorting_experts,
    )


def prepare_nvfp4_layer_for_petit(layer: torch.nn.Module) -> None:
    _require_petit_nvfp4()
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    qweight = layer.weight.view(torch.int32).contiguous()
    petit_qweight = repack_nvfp4(qweight, size_n=part_size_n, size_k=part_size_k)
    layer.weight = torch.nn.Parameter(petit_qweight, requires_grad=False)

    weight_scale = process_nvfp4_scales(
        scales=layer.weight_scale, size_k=part_size_k, size_n=part_size_n
    )
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)


def _require_petit_mxfp4_dense() -> None:
    if mul_mxfp4_a16 is None or repack_mxfp4 is None or process_mxfp4_scales is None:
        raise ValueError(
            "Petit MXFP4 dense kernels are not available. "
            "Please rebuild/install petit-kernel with MXFP4 dense support."
        )


def prepare_mxfp4_layer_for_petit(layer: torch.nn.Module) -> None:
    _require_petit_mxfp4_dense()
    if hasattr(layer, "weight_scale_2"):
        return

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    qweight = layer.weight.view(torch.int32).contiguous()
    petit_qweight = repack_mxfp4(qweight, size_n=part_size_n, size_k=part_size_k)
    layer.weight = torch.nn.Parameter(petit_qweight, requires_grad=False)

    weight_scale = process_mxfp4_scales(
        scales=layer.weight_scale, size_k=part_size_k, size_n=part_size_n
    )
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.weight_scale_2 = _default_global_scale(layer.weight.device)


def apply_petit_mxfp4_dense(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _require_petit_mxfp4_dense()

    if weight.dtype == torch.uint8:
        weight = weight.view(torch.int32)

    reshaped_x = input.reshape(-1, input.shape[-1])
    if reshaped_x.shape[1] != size_k:
        raise ValueError(
            f"Petit MXFP4 dense expected input K={size_k}, got {reshaped_x.shape[1]}."
        )
    if weight.shape != (size_n // 16, size_k * 2):
        raise ValueError(
            "Petit MXFP4 dense weight has incompatible shape: "
            f"expected {(size_n // 16, size_k * 2)}, got {tuple(weight.shape)}."
        )
    if weight_scale.shape != (size_n // 32, size_k):
        raise ValueError(
            "Petit MXFP4 dense weight_scale has incompatible shape: "
            f"expected {(size_n // 32, size_k)}, got {tuple(weight_scale.shape)}."
        )
    if not reshaped_x.is_contiguous():
        reshaped_x = reshaped_x.contiguous()

    out_shape = input.shape[:-1] + (size_n,)
    if global_scale is None:
        global_scale = torch.ones(1, device=reshaped_x.device, dtype=torch.float32)

    output = mul_mxfp4_a16(
        a=reshaped_x,
        b=weight,
        s=weight_scale,
        global_scale=global_scale,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        solution_id=-1,
    )
    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)


def materialize_petit_mxfp4_weight(
    layer: torch.nn.Module,
    out_dtype: torch.dtype = torch.bfloat16,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Materialize a Petit MXFP4 linear weight as dense [N, K] via Petit GEMM.

    This is used only by MLA setup code that must split kv_b_proj into cached
    dense w_kc/w_vc tensors. The runtime dense projection path should call
    apply_petit_mxfp4_dense directly instead of materializing the full weight.
    """
    _require_petit_mxfp4_dense()

    if not hasattr(layer, "weight_scale_2"):
        prepare_mxfp4_layer_for_petit(layer)

    size_n = layer.output_size_per_partition
    size_k = layer.input_size_per_partition
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    device = layer.weight.device
    output = torch.empty((size_n, size_k), device=device, dtype=out_dtype)

    for start in range(0, size_k, chunk_size):
        end = min(start + chunk_size, size_k)
        rows = end - start
        basis = torch.zeros((rows, size_k), device=device, dtype=torch.bfloat16)
        diag = torch.arange(rows, device=device)
        basis[diag, start + diag] = 1
        projected = apply_petit_mxfp4_dense(
            input=basis,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            global_scale=layer.weight_scale_2,
            size_n=size_n,
            size_k=size_k,
        )
        output[:, start:end].copy_(projected.to(out_dtype).t())

    return output


def prepare_mxfp4_moe_layer_for_petit(layer: torch.nn.Module) -> None:
    w13_size_n = layer.w13_weight.shape[1]
    w13_size_k = layer.w13_weight.shape[2] * 2
    w2_size_n = layer.w2_weight.shape[1]
    w2_size_k = layer.w2_weight.shape[2] * 2

    w13_weight, w13_weight_scale = repack_moe_w13_mxfp4_kernel_layout(
        layer.w13_weight.contiguous(),
        layer.w13_weight_scale.contiguous(),
    )
    w2_weight, w2_weight_scale = repack_moe_mxfp4_kernel_layout(
        layer.w2_weight.contiguous(),
        layer.w2_weight_scale.contiguous(),
    )

    layer.w13_weight = torch.nn.Parameter(w13_weight.contiguous(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_weight.contiguous(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        w13_weight_scale.contiguous(), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        w2_weight_scale.contiguous(), requires_grad=False
    )
    layer.petit_moe_w13_size_n = w13_size_n
    layer.petit_moe_w13_size_k = w13_size_k
    layer.petit_moe_w2_size_n = w2_size_n
    layer.petit_moe_w2_size_k = w2_size_k

    if hasattr(layer, "dispatcher") and hasattr(torch, "float4_e2m1fn_x2"):
        layer.dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})


def apply_petit_mxfp4_moe_layer(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return apply_petit_mxfp4_moe(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_weight_scale=layer.w13_weight_scale,
        w2_weight_scale=layer.w2_weight_scale,
        w13_size_n=layer.petit_moe_w13_size_n,
        w13_size_k=layer.petit_moe_w13_size_k,
        w2_size_n=layer.petit_moe_w2_size_n,
        w2_size_k=layer.petit_moe_w2_size_k,
        num_local_experts=layer.num_local_experts,
        activation=activation,
        expert_mask=expert_mask,
        num_local_tokens=num_local_tokens,
    )


def apply_petit_nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _require_petit_nvfp4()
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    output = mul_nvfp4_a16(
        a=reshaped_x,
        b=weight,
        s=weight_scale,
        global_scale=weight_scale_2,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        solution_id=-1,
    )
    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)


def _apply_petit_mxfp4_moe_impl(
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_size_n: int,
    w13_size_k: int,
    w2_size_n: int,
    w2_size_k: int,
    num_local_experts: int,
    activation: str,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if activation != "silu":
        raise NotImplementedError("Petit MXFP4 MoE only supports silu activation.")
    if hidden_states.dim() != 2:
        raise ValueError("hidden_states must be rank-2")
    if topk_ids.dim() != 2 or topk_weights.dim() != 2:
        raise ValueError("topk_ids/topk_weights must be rank-2")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids/topk_weights shape mismatch")
    if topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "Petit MXFP4 MoE hidden_states/topk row mismatch: "
            f"hidden_states has {hidden_states.shape[0]} rows, "
            f"topk has {topk_ids.shape[0]} rows."
        )

    if expert_mask is None:
        sorting_expert_mask = torch.ones(
            (num_local_experts,),
            device=topk_ids.device,
            dtype=torch.int32,
        )
    else:
        sorting_expert_mask = expert_mask.to(
            device=topk_ids.device, dtype=torch.int32
        ).contiguous()

    if hidden_states.numel() == 0:
        return torch.empty(
            (hidden_states.shape[0], w2_weight.shape[1]),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )

    if hidden_states_scale is not None:
        raise TypeError(
            "Petit MXFP4 MoE on gfx94x expects floating-point hidden_states; "
            "pre-quantized MXFP4 dispatch activations are not supported."
        )
    if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "Petit MXFP4 MoE expected floating-point hidden_states when "
            "hidden_states_scale is absent, got "
            f"{hidden_states.dtype}."
        )

    # AITER quantization and sorting both use num_local_tokens to ignore padded
    # rows.
    sorting_num_local_tokens = _normalize_num_local_tokens(
        num_local_tokens,
        max_tokens=hidden_states.shape[0],
        device=hidden_states.device,
    )

    input_q, input_scale = _quantize_fp8_block128(
        hidden_states,
        num_local_tokens=sorting_num_local_tokens,
    )
    if _aiter_moe_sorting is None:
        raise RuntimeError("Petit MXFP4 MoE requires AITER moe_sorting on ROCm.")

    (
        topk_ids_i32,
        topk_weights_f32,
        sorting_expert_mask,
        num_sorting_experts,
    ) = _prepare_petit_aiter_sorting_inputs(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        sorting_expert_mask=sorting_expert_mask,
    )

    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = (
        _aiter_moe_sorting(
            topk_ids_i32,
            topk_weights_f32,
            num_sorting_experts,
            hidden_states.shape[1],
            torch.bfloat16,
            _PETIT_MOE_BLOCK_SIZE,
            expert_mask=sorting_expert_mask,
            num_local_tokens=sorting_num_local_tokens,
        )
    )

    if sorted_token_ids.device != hidden_states.device:
        sorted_token_ids = sorted_token_ids.to(device=hidden_states.device)
        sorted_weights = sorted_weights.to(device=hidden_states.device)
        sorted_expert_ids = sorted_expert_ids.to(device=hidden_states.device)
        num_valid_ids = num_valid_ids.to(device=hidden_states.device)

    sorted_token_ids = sorted_token_ids.contiguous()
    sorted_weights = sorted_weights.contiguous()
    sorted_expert_ids = sorted_expert_ids.contiguous()
    num_valid_ids = num_valid_ids.contiguous()

    if _raw_fused_moe_fp8_blockscale_g1u1_mxfp4 is None:
        raise RuntimeError("Petit MXFP4 MoE kernel is not available.")

    return _raw_fused_moe_fp8_blockscale_g1u1_mxfp4(
        input_q=input_q,
        w13_q=w13_weight,
        w2_q=w2_weight,
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk_ids.shape[1],
        input_scale=input_scale,
        fc1_scale=w13_weight_scale,
        fc2_scale=w2_weight_scale,
        out=None,
    )


def _fake_apply_petit_mxfp4_moe(
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_size_n: int,
    w13_size_k: int,
    w2_size_n: int,
    w2_size_k: int,
    num_local_experts: int,
    activation: str,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return hidden_states.new_empty(
        (hidden_states.shape[0], w2_weight.shape[1]), dtype=torch.bfloat16
    )


_petit_mxfp4_moe_op = None
if register_custom_op is not None:
    _petit_mxfp4_moe_op = register_custom_op(
        op_name="petit_mxfp4_routed_moe",
        fake_impl=_fake_apply_petit_mxfp4_moe,
    )(_apply_petit_mxfp4_moe_impl)


def apply_petit_mxfp4_moe(
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_size_n: int,
    w13_size_k: int,
    w2_size_n: int,
    w2_size_k: int,
    num_local_experts: int,
    activation: str,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    impl = (
        _petit_mxfp4_moe_op
        if _petit_mxfp4_moe_op is not None and hidden_states.device.type == "cuda"
        else _apply_petit_mxfp4_moe_impl
    )
    return impl(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight_scale=w2_weight_scale,
        w13_size_n=w13_size_n,
        w13_size_k=w13_size_k,
        w2_size_n=w2_size_n,
        w2_size_k=w2_size_k,
        num_local_experts=num_local_experts,
        activation=activation,
        expert_mask=expert_mask,
        num_local_tokens=num_local_tokens,
    )
