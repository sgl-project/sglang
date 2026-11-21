"""CuTe DSL Fused Sigmoid Gating Delta Rule Kernel for GDN Decode."""

import logging
import math
from typing import Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

logger = logging.getLogger(__name__)

_compiled_kernels: Dict[Tuple, object] = {}
_SUPPORTED_STATE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)

TILE_K = 128
TILE_V = 16
WARPS_PER_BLOCK = 4
NUM_THREADS = WARPS_PER_BLOCK * 32


def _define_kernels():
    """Define the register-resident CuTe DSL kernel for GDN decode."""

    @cute.kernel
    def gdn_decode_kernel(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        """Zero-smem, 128-bit vectorized GDN decode kernel.

        Each thread owns ``elems = 128 // state_dtype.width`` contiguous K
        elements (8 for BF16/FP16 -> 16 lanes per V-row, 2 V-rows per warp;
        4 for FP32 -> 32 lanes per V-row, 1 V-row per warp). All reductions
        stay within warp shuffles with no shared memory and no block barriers.
        """
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()

        num_v_blocks = V // TILE_V
        v_block = bidx % num_v_blocks
        batch_idx = bidx // num_v_blocks
        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)

        pool_idx = h0_indices[i_n]
        if pool_idx >= 0:
            state_dtype = h0_source.element_type
            elems = 128 // state_dtype.width
            k_lanes = TILE_K // elems
            v_groups = 32 // k_lanes
            k_lane = lane % k_lanes
            v_sub = lane // k_lanes

            # Vectorized load of Q and K slices into registers.
            gQ_tile = cute.local_tile(q[(i_n, i_h, None)], (elems,), (k_lane,))
            gK_tile = cute.local_tile(k[(i_n, i_h, None)], (elems,), (k_lane,))
            rQ_raw = cute.make_rmem_tensor((elems,), q.element_type)
            rK_raw = cute.make_rmem_tensor((elems,), k.element_type)
            cute.autovec_copy(gQ_tile, rQ_raw)
            cute.autovec_copy(gK_tile, rK_raw)

            rQ = cute.make_rmem_tensor((elems,), cutlass.Float32)
            rK = cute.make_rmem_tensor((elems,), cutlass.Float32)
            sum_q = cutlass.Float32(0.0)
            sum_k = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(elems):
                q_val = cutlass.Float32(rQ_raw[j])
                k_val = cutlass.Float32(rK_raw[j])
                rQ[j] = q_val
                rK[j] = k_val
                sum_q += q_val * q_val
                sum_k += k_val * k_val

            if use_qk_l2norm:
                if k_lanes == 32:
                    for offset in [16, 8, 4, 2, 1]:
                        sum_q += cute.arch.shuffle_sync_bfly(
                            sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        sum_k += cute.arch.shuffle_sync_bfly(
                            sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                else:
                    for offset in [8, 4, 2, 1]:
                        sum_q += cute.arch.shuffle_sync_bfly(
                            sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        sum_k += cute.arch.shuffle_sync_bfly(
                            sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )

                inv_norm_q = cute.rsqrt(sum_q + 1e-6) * scale
                inv_norm_k = cute.rsqrt(sum_k + 1e-6)
                for j in cutlass.range_constexpr(elems):
                    rQ[j] = rQ[j] * inv_norm_q
                    rK[j] = rK[j] * inv_norm_k
            else:
                for j in cutlass.range_constexpr(elems):
                    rQ[j] = rQ[j] * scale

            # Compute scalar decay gate g and update rate beta once per warp.
            r_g = cutlass.Float32(0.0)
            r_beta = cutlass.Float32(0.0)
            if lane == 0:
                x = cutlass.Float32(a[i_n, i_hv]) + cutlass.Float32(dt_bias[i_hv])
                beta_x = softplus_beta * x
                softplus_x = cutlass.Float32(0.0)
                if beta_x <= softplus_threshold:
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta)
                        * cute.log(cutlass.Float32(1.0) + cute.exp(beta_x))
                    )
                else:
                    softplus_x = x
                r_g = cute.exp(-cute.exp(cutlass.Float32(A_log[i_hv])) * softplus_x)
                r_beta = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-cutlass.Float32(b[i_n, i_hv]))
                )
            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)

            v_per_warp = TILE_V // WARPS_PER_BLOCK
            v_iters = v_per_warp // v_groups
            v_start = v_block * TILE_V + warp_idx * v_per_warp

            rH_raw = cute.make_rmem_tensor((elems,), state_dtype)
            rH = cute.make_rmem_tensor((elems,), cutlass.Float32)

            for vi in cutlass.range_constexpr(v_iters):
                v_global = v_start + vi * v_groups + v_sub
                r_v = cutlass.Float32(v[i_n, i_hv, v_global])

                gH_tile = cute.local_tile(
                    h0_source[(pool_idx, i_hv, v_global, None)], (elems,), (k_lane,)
                )
                cute.autovec_copy(gH_tile, rH_raw)

                sum_hk = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(elems):
                    h_val = cutlass.Float32(rH_raw[j]) * r_g
                    rH[j] = h_val
                    sum_hk += h_val * rK[j]

                if k_lanes == 32:
                    for offset in [16, 8, 4, 2, 1]:
                        sum_hk += cute.arch.shuffle_sync_bfly(
                            sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                        )
                else:
                    for offset in [8, 4, 2, 1]:
                        sum_hk += cute.arch.shuffle_sync_bfly(
                            sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                        )

                v_new = (r_v - sum_hk) * r_beta

                sum_hq = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(elems):
                    h_new = rH[j] + rK[j] * v_new
                    rH_raw[j] = state_dtype(h_new)
                    sum_hq += h_new * rQ[j]

                cute.autovec_copy(rH_raw, gH_tile)

                if k_lanes == 32:
                    for offset in [16, 8, 4, 2, 1]:
                        sum_hq += cute.arch.shuffle_sync_bfly(
                            sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                        )
                else:
                    for offset in [8, 4, 2, 1]:
                        sum_hq += cute.arch.shuffle_sync_bfly(
                            sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                        )

                if k_lane == 0:
                    o[i_n, i_hv, v_global] = o.element_type(sum_hq)

    return gdn_decode_kernel


def _create_jit_function():
    """Create JIT-compiled launcher function for GDN decode."""
    gdn_decode = _define_kernels()

    @cute.jit
    def run_gdn_decode(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        n_indices = h0_indices.layout.shape[0]
        grid_x = n_indices * HV * (V // TILE_V)

        gdn_decode(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            h0_source,
            h0_indices,
            o,
            softplus_beta,
            softplus_threshold,
            scale,
            H,
            HV,
            V,
            use_qk_l2norm,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=0,
            stream=stream,
        )

    return run_gdn_decode


_jit_function = None


def _get_jit_function():
    global _jit_function
    if _jit_function is None:
        _jit_function = _create_jit_function()
    return _jit_function


def _get_compiled_kernel(N, H, HV, K, V, h0_source, tensor_dtypes):
    """Get or compile the kernel for given dimensions and tensor dtypes.

    ``h0_source`` is the caller's real state pool (``[pool_size, HV, V, K]``):
    ``from_dlpack`` bakes its exact dtype and strides into the compiled kernel,
    so non-FP32 state pools (BF16/FP16) and envelope-strided pools compile
    against their true layout.
    """
    global _compiled_kernels

    pool_size = h0_source.shape[0]
    state_dtype = h0_source.dtype
    device = h0_source.device
    (
        q_dtype,
        k_dtype,
        v_dtype,
        a_dtype,
        b_dtype,
        A_log_dtype,
        dt_bias_dtype,
        indices_dtype,
    ) = tensor_dtypes
    key = (
        N,
        H,
        HV,
        K,
        V,
        pool_size,
        tuple(h0_source.stride()),
        state_dtype,
        tensor_dtypes,
        device,
    )
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    # CuTe launches skip dtype checks; compiling with mismatched dtypes causes
    # silent bit reinterpretation (or OOB writes for state_dtype / indices_dtype).
    q = torch.zeros(N, H, K, dtype=q_dtype, device=device)
    k = torch.zeros(N, H, K, dtype=k_dtype, device=device)
    v = torch.zeros(N, HV, V, dtype=v_dtype, device=device)
    a = torch.zeros(N, HV, dtype=a_dtype, device=device)
    b = torch.zeros(N, HV, dtype=b_dtype, device=device)
    o = torch.zeros(N, HV, V, dtype=q_dtype, device=device)
    A_log = torch.zeros(HV, dtype=A_log_dtype, device=device)
    dt_bias = torch.zeros(HV, dtype=dt_bias_dtype, device=device)
    h0_indices = torch.zeros(N, dtype=indices_dtype, device=device)

    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    kernel_func = _get_jit_function()

    compiled_kernel = cute.compile(
        kernel_func,
        from_dlpack(q, assumed_align=16),
        from_dlpack(k, assumed_align=16),
        from_dlpack(v, assumed_align=16),
        from_dlpack(a, assumed_align=16),
        from_dlpack(b, assumed_align=16),
        from_dlpack(A_log, assumed_align=16),
        from_dlpack(dt_bias, assumed_align=16),
        from_dlpack(h0_source.detach(), assumed_align=16),
        from_dlpack(h0_indices, assumed_align=16),
        from_dlpack(o, assumed_align=16),
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=K**-0.5,
        H=H,
        HV=HV,
        V=V,
        use_qk_l2norm=True,
        stream=stream,
    )

    _compiled_kernels[key] = compiled_kernel
    logger.info(
        f"CuTe DSL GDN kernel compiled: N={N}, H={H}, HV={HV}, K={K}, V={V}, "
        f"pool_size={pool_size}, pool_strides={tuple(h0_source.stride())}, "
        f"state_dtype={state_dtype}, dt_bias_dtype={dt_bias_dtype}, q_dtype={q_dtype}"
    )

    return compiled_kernel


def _check_compiled_constants(
    *, K, scale, use_qk_l2norm_in_kernel, softplus_beta, softplus_threshold
):
    # _get_compiled_kernel bakes these values in; reject any it would ignore.
    if (
        (scale is not None and not math.isclose(scale, K**-0.5, rel_tol=1e-6))
        or not use_qk_l2norm_in_kernel
        or softplus_beta != 1.0
        or softplus_threshold != 20.0
    ):
        raise ValueError(
            "CuTe DSL GDN decode only supports scale=K**-0.5, "
            "use_qk_l2norm_in_kernel=True, softplus_beta=1.0 and "
            f"softplus_threshold=20.0; got {scale=}, {use_qk_l2norm_in_kernel=}, "
            f"{softplus_beta=}, {softplus_threshold=}"
        )


def cutedsl_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """CuTe DSL implementation of fused sigmoid gating delta rule update.

    ``initial_state_source`` is the ``[pool_size, HV, V, K]`` state pool
    (or its flattened 1D view). Envelope-strided pools with compact per-slot
    ``[HV, V, K]`` blocks are updated in place.
    """
    del cu_seqlens

    B_q, T_q, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    N = initial_state_indices.shape[0]

    is_varlen_decode = B_q == 1 and T_q == N and N > 1
    _check_compiled_constants(
        K=K,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )

    if K != TILE_K or V % TILE_V != 0 or HV % H != 0:
        raise ValueError(
            f"CuTe DSL GDN decode requires K={TILE_K}, V % {TILE_V} == 0, "
            f"and HV % H == 0; got H={H}, HV={HV}, K={K}, V={V}"
        )

    if B_q * T_q != N:
        raise ValueError(
            f"CuTe DSL GDN decode expects one token per sequence (B_q * T_q == N); "
            f"got q.shape={tuple(q.shape)}, N={N}"
        )

    if initial_state_source.dtype not in _SUPPORTED_STATE_DTYPES:
        raise ValueError(
            "CuTe DSL GDN decode only supports initial_state_source dtypes "
            f"{_SUPPORTED_STATE_DTYPES}; got {initial_state_source.dtype}"
        )

    if initial_state_indices.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise ValueError(
            "CuTe DSL GDN decode only supports initial_state_indices dtypes "
            f"{_SUPPORTED_INDEX_DTYPES}; got {initial_state_indices.dtype}"
        )

    o_shape = (1, N, HV, V) if is_varlen_decode else (N, 1, HV, V)
    if N == 0:
        return q.new_empty(o_shape)

    slot_elems = HV * V * K
    if initial_state_source.dim() == 1:
        if initial_state_source.numel() % slot_elems != 0:
            raise ValueError(
                f"1D initial_state_source numel ({initial_state_source.numel()}) "
                f"must be divisible by HV * V * K ({slot_elems})"
            )
        pool_size = initial_state_source.numel() // slot_elems
        h0_source = initial_state_source.view(pool_size, HV, V, K)
    elif initial_state_source.dim() == 4:
        h0_source = initial_state_source
    else:
        raise ValueError(
            f"Unexpected initial_state_source shape: {initial_state_source.shape}"
        )

    if (
        h0_source.stride()[1:] != (V * K, K, 1)
        or (h0_source.stride(0) * h0_source.element_size() % 16 != 0)
        or (h0_source.data_ptr() % 16 != 0)
    ):
        raise ValueError(
            "CuTe DSL GDN decode requires a compact 16B-aligned per-slot "
            f"[HV, V, K] state block; got strides {tuple(h0_source.stride())} "
            f"for shape {tuple(h0_source.shape)} and dtype {h0_source.dtype}"
        )

    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"Expected A_log and dt_bias with numel={HV}; "
            f"got A_log.shape={tuple(A_log.shape)}, dt_bias.shape={tuple(dt_bias.shape)}"
        )

    q_flat = q.contiguous().reshape(N, H, K)
    k_flat = k.contiguous().reshape(N, H, K)
    v_flat = v.contiguous().reshape(N, HV, V)
    a_flat = a.contiguous().reshape(N, HV)
    b_flat = b.contiguous().reshape(N, HV)
    A_log_flat = A_log.contiguous().reshape(HV)
    dt_bias_flat = dt_bias.contiguous().reshape(HV)
    indices_flat = initial_state_indices.contiguous()

    o = q.new_empty(o_shape)
    o_flat = o.view(N, HV, V)

    tensor_dtypes = (
        q_flat.dtype,
        k_flat.dtype,
        v_flat.dtype,
        a_flat.dtype,
        b_flat.dtype,
        A_log_flat.dtype,
        dt_bias_flat.dtype,
        indices_flat.dtype,
    )
    compiled_kernel = _get_compiled_kernel(N, H, HV, K, V, h0_source, tensor_dtypes)

    stream = cuda.CUstream(torch.cuda.current_stream(h0_source.device).cuda_stream)
    compiled_kernel(
        from_dlpack(q_flat.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=2
        ),
        from_dlpack(k_flat.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=2
        ),
        from_dlpack(v_flat.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=2
        ),
        from_dlpack(a_flat.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(b_flat.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(A_log_flat.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=0
        ),
        from_dlpack(dt_bias_flat.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=0
        ),
        from_dlpack(h0_source.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=3
        ),
        from_dlpack(indices_flat.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=0
        ),
        from_dlpack(o_flat.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=2
        ),
        stream,
    )

    return o
