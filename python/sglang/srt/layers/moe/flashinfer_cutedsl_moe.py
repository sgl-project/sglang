from typing import Optional

import torch
from flashinfer import (
    scaled_fp4_grouped_quantize,
    silu_and_mul_scaled_nvfp4_experts_quantize,
)
from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

from sglang.srt.distributed.parallel_state import get_tp_group

_SYMMETRIC_BARRIER_TENSORS: dict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, object]
] = {}


def _get_symmetric_barrier_tensors(
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    import cutlass.torch as cutlass_torch
    import torch.distributed._symmetric_memory as torch_symmetric_memory

    group = get_tp_group().device_group
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    cache_key = (id(group), device_index, num_sms)
    if cache_key not in _SYMMETRIC_BARRIER_TENSORS:
        barrier_flag_local = torch_symmetric_memory.empty(
            (num_sms,), device=device, dtype=torch.int32
        )
        barrier_flag_local.zero_()
        barrier_handle = torch_symmetric_memory.rendezvous(
            barrier_flag_local, group=group
        )
        barrier_flag_multicast = cutlass_torch.as_tensor(
            barrier_handle.multicast_ptr,
            barrier_flag_local.shape,
            barrier_flag_local.dtype,
        )
        _SYMMETRIC_BARRIER_TENSORS[cache_key] = (
            barrier_flag_local,
            barrier_flag_multicast,
            barrier_handle,
        )
    barrier_flag_local, barrier_flag_multicast, _barrier_handle = (
        _SYMMETRIC_BARRIER_TENSORS[cache_key]
    )
    return group.size(), barrier_flag_local, barrier_flag_multicast


def get_cute_dtype(input: torch.Tensor) -> str:
    if input.dtype == torch.bfloat16:
        return "bfloat16"
    elif input.dtype == torch.float16:
        return "float16"
    elif input.dtype == torch.float32:
        return "float32"
    else:
        raise ValueError(f"Unsupported cute dtype {input.dtype}")


def flashinfer_cutedsl_moe_masked(
    hidden_states: tuple[torch.Tensor, Optional[torch.Tensor]],
    input_global_scale: torch.Tensor,
    w1: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alpha,
    w2: torch.Tensor,
    a2_global_scale: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alpha,
    masked_m: torch.Tensor,
    topk_weights: Optional[torch.Tensor] = None,
    recv_rank_info: Optional[torch.Tensor] = None,
    recv_idx_info: Optional[torch.Tensor] = None,
    combine_out: Optional[torch.Tensor] = None,
    combine_out_ptrs: Optional[torch.Tensor] = None,
    down_sm_count: Optional[int] = None,
    down_signals: Optional[torch.Tensor] = None,
    down_start_event: Optional[torch.cuda.Event] = None,
):
    """
    Perform masked Mixture-of-Experts computation with FlashInfer's CuteDSL
    kernels.

    Args:
        hidden_states: Either of the following case
            * tuple[torch.Tensor, None]: [num_experts, m, k], bf16, None means no quant
            * tuple[torch.Tensor, torch.Tensor]: [num_experts, m, k // 2], uint8, [num_experts, m, k // 16], float8_e4m3fn
        input_global_scale (torch.Tensor): (l,)
        w1 (torch.Tensor): fp4 weights, [l, 2 * n, k // 2], uint8
        w1_blockscale (torch.Tensor): blockscale factors, e4m3,
        w1_alpha (torch.Tensor): (l,)
        w2 (torch.Tensor): fp4 weights, [l, k, n // 2], uint8
        a2_global_scale (torch.Tensor): (l,)
        w2_blockscale (torch.Tensor): blockscale factors, e4m3,
        w2_alpha (torch.Tensor): (l,)
        masked_m (torch.Tensor): Masked dimension indices

    Notes:
        - Assumes max(masked_m) == m.
    """

    # === Assertions on dtypes ===
    assert w1.dtype == torch.uint8, f"w1 must be uint8 (fp4 packed), got {w1.dtype}"
    assert (
        w1_blockscale.dtype == torch.float8_e4m3fn
    ), f"w1_blockscale must be float8_e4m3fn, got {w1_blockscale.dtype}"
    assert (
        w1_alpha.dtype == torch.float32
    ), f"w1_alpha must be float32, got {w1_alpha.dtype}"
    assert w2.dtype == torch.uint8, f"w2 must be uint8 (fp4 packed), got {w2.dtype}"
    assert (
        a2_global_scale.dtype == torch.float32
    ), f"a2_global_scale must be float32, got {a2_global_scale.dtype}"
    assert (
        w2_blockscale.dtype == torch.float8_e4m3fn
    ), f"w2_blockscale must be float8_e4m3fn, got {w2_blockscale.dtype}"
    assert (
        w2_alpha.dtype == torch.float32
    ), f"w2_alpha must be float32, got {w2_alpha.dtype}"
    assert (
        len(hidden_states) == 2
    ), f"hidden_states must be a tuple of length 2, got {len(hidden_states)}"

    # === Assertions on shapes ===
    n = w2.shape[-1] * 2  # intermediate dimension

    if hidden_states[1] is not None:

        a_q = hidden_states[0].view(torch.uint8)
        a_q_sf = hidden_states[1].view(torch.float8_e4m3fn)
        m, k_by_2, num_experts = a_q.shape
        k = k_by_2 * 2
    else:
        num_experts, m, k = hidden_states[0].shape

        assert (
            input_global_scale.dtype == torch.float32
        ), f"input_global_scale must be float32, got {input_global_scale.dtype}"
        assert input_global_scale.shape == (
            num_experts,
        ), f"input_global_scale must be (l,), got {input_global_scale.shape}"

        a_q, a_q_sf = scaled_fp4_grouped_quantize(
            hidden_states[0],
            masked_m,
            input_global_scale,
        )

    assert w1.shape[-2] == 2 * n, f"w1 last-2 dim must be 2*n, got {w1.shape}"
    assert (
        w1.shape[-1] * 2 == k
    ), f"w1 last dim * 2 must equal k, got {w1.shape[-1]} vs k={k}"
    assert w2.shape[-2:] == (
        k,
        n // 2,
    ), f"w2 shape mismatch, got {w2.shape[-2:]}, expected {(k, n//2)}"
    assert w1_alpha.shape == (
        num_experts,
    ), f"w1_alpha must be (l,), got {w1_alpha.shape}"
    assert a2_global_scale.shape == (
        num_experts,
    ), f"a2_global_scale must be (l,), got {a2_global_scale.shape}"
    assert w2_alpha.shape == (
        num_experts,
    ), f"w2_alpha must be (l,), got {w2_alpha.shape}"

    # TODO(kaixih@nvidia): dtype should be based on inputs.
    gateup_output = torch.empty(
        (num_experts, m, n * 2), dtype=torch.bfloat16, device=a_q.device
    )
    gateup_output = gateup_output.permute(1, 2, 0)  # requirement of kernel
    sf_vec_size = 16
    assert a_q_sf.dtype == torch.float8_e4m3fn
    assert a_q.dtype == torch.uint8
    ab_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    c_dtype = "bfloat16"

    # Gemm1
    grouped_gemm_nt_masked(
        (a_q, a_q_sf),
        (w1.permute(1, 2, 0), w1_blockscale),
        gateup_output,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w1_alpha.view(1, 1, num_experts),
        alpha_dtype=get_cute_dtype(w1_alpha),
    )  # in logical [m, n, l]

    # SILU and quantization
    diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
        gateup_output.permute(2, 0, 1),
        masked_m,
        a2_global_scale,
    )

    if down_start_event is not None:
        down_start_event.record()

    # Gemm2
    is_combine_fusion = combine_out is not None
    if is_combine_fusion:
        missing = [
            name
            for name, value in (
                ("topk_weights", topk_weights),
                ("recv_rank_info", recv_rank_info),
                ("recv_idx_info", recv_idx_info),
                ("combine_out_ptrs", combine_out_ptrs),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "fused grouped GEMM combine is missing DeepEP metadata: "
                + ", ".join(missing)
            )
        num_ranks, barrier_flag_local, barrier_flag_multicast = (
            _get_symmetric_barrier_tensors(a_q.device)
        )
        c_dtype = get_cute_dtype(combine_out)
        out = combine_out
        combine_fusion_kwargs = dict(
            is_swap_ab=True,
            is_combine_fusion=True,
            topk_weights=topk_weights,
            idx_src_info=recv_idx_info,
            rank_src_info=recv_rank_info,
            out_ptrs=combine_out_ptrs,
            num_ranks=num_ranks,
            barrier_flag_local=barrier_flag_local,
            barrier_flag_multicast=barrier_flag_multicast,
        )
    else:
        out = torch.empty((num_experts, m, k), dtype=torch.bfloat16, device=a_q.device)
        out = out.permute(1, 2, 0)  # requirement of kernel
        combine_fusion_kwargs = {}

    grouped_gemm_nt_masked(
        (diq, diq_sf),
        (w2.permute(1, 2, 0), w2_blockscale),
        out,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        alpha=w2_alpha.view(1, 1, num_experts),
        alpha_dtype=get_cute_dtype(w2_alpha),
        **(
            dict(
                sm_count=down_sm_count,
                dst_signals=down_signals,
            )
            if down_sm_count is not None or down_signals is not None
            else {}
        ),
        **combine_fusion_kwargs,
    )

    if is_combine_fusion:
        return out

    # in logical [m, k, l]
    return out.permute(2, 0, 1)
