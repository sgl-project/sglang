"""Context-parallel GDN prefill kernels for Blackwell SM100."""

# Vendored from flashinfer-ai/flashinfer main at 76704c4 (SM100 GDN CP
# prefill closure, incl. #4436 pooled state / checkpointing / dtype parity);
# pending a FlashInfer release that ships it.

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from flashinfer.gdn_kernels.blackwell.gdn_prefill import (
    _cutlass_state_dtype,
    _mark_state_layout,
)
from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 import (
    _FullyFusedDeltaRuleSm120,
)
from flashinfer.utils import (
    _get_cache_buf,
    get_compute_capability,
    get_device_sm_count,
)

from sglang.kernels.ops.attention.gdn_cp_prefill.custom_compile_cache import (
    cached_compile,
    get_cached_compile,
)
from sglang.kernels.ops.attention.gdn_cp_prefill.delta_rule_cp_sm120 import (
    CPDeltaRuleFixupHmmaSm120,
    CPDeltaRuleFixupSimtSm120,
    CPDeltaRuleTPrecomputeSm120,
)
from sglang.kernels.ops.attention.gdn_cp_prefill.flashinfer_compat import (
    get_device_name,
)
from sglang.kernels.ops.attention.gdn_cp_prefill.gated_delta_net_cp import (
    CPDeltaRuleFixupUtcmmaSm100,
    CPDeltaRuleMNPrecomputeUtcmma1Sm100,
)
from sglang.kernels.ops.attention.gdn_cp_prefill.gated_delta_net_cp_prefill import (
    CPDeltaRulePrefillTcgen05Sm100,
)
from sglang.kernels.ops.attention.gdn_cp_prefill.varlen_helper import (
    CP_CHUNK_LEN_GRANULARITY,
    choose_cp_chunk_len_host,
    integer_dtype_to_cutlass,
    is_integer_dtype,
    max_num_chunks_host,
    workspace_num_chunks_host,
)


@functools.cache
def _blackwell_compile_options(device: torch.device):
    major, minor = get_compute_capability(device)
    if major != 10:
        raise RuntimeError(
            f"SM100 CP delta rule requires a compute 10.x device, got {major}.{minor}"
        )
    return cute.EnableTVMFFI(True), cute.GPUArch(f"sm_{major}{minor}a")


def _get_cp_workspace(name, shape, dtype, device):
    nbytes = math.prod(shape) * dtype.itemsize
    return _get_cache_buf(name, nbytes, device)[:nbytes].view(dtype).view(shape)


class CPDeltaRuleTPrecomputeSm100(CPDeltaRuleTPrecomputeSm120):
    """SM100 specialization of the architecture-portable T precompute body."""


def _cu_seqlens_dtype(dtype):
    return integer_dtype_to_cutlass(dtype)


@functools.cache
def _get_t_precompute_kernel(kernel_dtype, cu_seqlens_dtype):
    return CPDeltaRuleTPrecomputeSm100(kernel_dtype, cu_seqlens_dtype=cu_seqlens_dtype)


def cp_delta_rule_t_precompute_dsl_sm100(
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    max_seqlen: int | None = None,
    *,
    _skip_check: bool = False,
    _device=None,
    _stream=None,
):
    """Precompute signed, beta-folded CP T tiles on SM100."""
    import cuda.bindings.driver as cuda_driver

    device = k.device if _device is None else _device
    if not _skip_check:
        if k.ndim != 3:
            raise RuntimeError(
                f"k must have shape (total_seqlen, num_k_heads, D), got {tuple(k.shape)}"
            )
        if beta.ndim != 2 or beta.shape[0] != k.shape[0]:
            raise RuntimeError(
                f"beta must have shape (total_seqlen, num_sab_heads), got {tuple(beta.shape)}"
            )
        if total_seqlen != k.shape[0]:
            raise RuntimeError(
                f"total_seqlen must match k.shape[0], got {total_seqlen} and {k.shape[0]}"
            )
        _cu_seqlens_dtype(cu_seqlens.dtype)
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided")
    if not _skip_check and max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    _, num_k_heads, d = k.shape
    num_sab_heads = beta.shape[1]
    if not _skip_check:
        if num_sab_heads < num_k_heads or num_sab_heads % num_k_heads != 0:
            raise RuntimeError(
                "beta heads must be a positive multiple of k heads, "
                f"got beta heads={num_sab_heads} and k heads={num_k_heads}"
            )
        if d != 128:
            raise RuntimeError(
                f"CPDeltaRuleTPrecomputeSm100 only supports D=128, got {d}"
            )
        if k.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"CPDeltaRuleTPrecomputeSm100 only supports fp16/bf16 inputs, got {k.dtype}"
            )
        if beta.dtype != torch.float32:
            raise RuntimeError(f"beta must have dtype torch.float32, got {beta.dtype}")
        for name, tensor in (("k", k), ("beta", beta)):
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")

    total_t_blocks = workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
    max_t_blocks_per_seq = max_num_chunks_host(max_seqlen, 64)
    t = _get_cp_workspace(
        "gdn_cp_sm100_t",
        (total_t_blocks, num_sab_heads, 64, 64),
        k.dtype,
        device,
    )
    if total_t_blocks == 0:
        return t
    k_tma = k.as_strided((d, total_seqlen, num_k_heads), (1, num_k_heads * d, d))
    kernel_dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[
        k.dtype
    ]
    stream = (
        _stream
        if _stream is not None
        else cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    )

    kernel = _get_t_precompute_kernel(kernel_dtype, _cu_seqlens_dtype(cu_seqlens.dtype))
    compile_options = _blackwell_compile_options(device)
    compiled = get_cached_compile(kernel, compile_options)
    if compiled is None:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )
        kernel_args = (
            from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(beta.reshape(-1), assumed_align=16).mark_layout_dynamic(),
            from_dlpack(t.view(-1), assumed_align=128).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            cutlass.Int32(num_k_heads),
            cutlass.Int32(num_sab_heads),
            cutlass.Int32(total_t_blocks),
            cutlass.Int32(max_t_blocks_per_seq),
            cutlass.Int32(num_seqs),
            stream,
        )
        compiled = cached_compile(kernel, *kernel_args, compile_options=compile_options)
    compiled(
        k_tma,
        beta.reshape(-1),
        t.view(-1),
        cu_seqlens,
        num_k_heads,
        num_sab_heads,
        total_t_blocks,
        max_t_blocks_per_seq,
        num_seqs,
        stream,
    )
    return t


@functools.cache
def _get_mn_precompute_kernel(kernel_dtype, cu_seqlens_dtype):
    return CPDeltaRuleMNPrecomputeUtcmma1Sm100(kernel_dtype, cu_seqlens_dtype)


def cp_delta_rule_mn_precompute_dsl_sm100(
    k: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    alpha: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int = 4096,
    max_seqlen: int | None = None,
    *,
    _skip_check: bool = False,
    _device=None,
    _stream=None,
):
    """Precompute local transfer and state workspaces on SM100."""
    import cuda.bindings.driver as cuda_driver

    device = k.device if _device is None else _device
    if not _skip_check:
        if k.ndim != 3:
            raise RuntimeError(
                f"k must have shape (total_seqlen, num_k_heads, D), got {tuple(k.shape)}"
            )
        if v.ndim != 3 or v.shape[0] != k.shape[0] or v.shape[2] != k.shape[2]:
            raise RuntimeError(
                f"v must have shape (total_seqlen, num_v_heads, D), got {tuple(v.shape)}"
            )
        if alpha.ndim != 2 or alpha.shape[0] != k.shape[0]:
            raise RuntimeError(
                f"alpha must have shape (total_seqlen, num_sab_heads), got {tuple(alpha.shape)}"
            )
        if total_seqlen != k.shape[0]:
            raise RuntimeError(
                f"total_seqlen must match k.shape[0], got {total_seqlen} and {k.shape[0]}"
            )
        if cp_chunk_len % 64 != 0:
            raise RuntimeError(
                f"cp_chunk_len must be a multiple of 64, got {cp_chunk_len}"
            )
        _cu_seqlens_dtype(cu_seqlens.dtype)
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided")
    if not _skip_check and max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    _, num_k_heads, d = k.shape
    num_v_heads = v.shape[1]
    num_sab_heads = alpha.shape[1]
    if not _skip_check:
        if num_sab_heads < num_k_heads or num_sab_heads % num_k_heads != 0:
            raise RuntimeError(
                "alpha heads must be a positive multiple of k heads, "
                f"got alpha heads={num_sab_heads} and k heads={num_k_heads}"
            )
        if num_sab_heads < num_v_heads or num_sab_heads % num_v_heads != 0:
            raise RuntimeError(
                "alpha heads must be a positive multiple of v heads, "
                f"got alpha heads={num_sab_heads} and v heads={num_v_heads}"
            )
    total_t_blocks = workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
    if not _skip_check and t.shape != (total_t_blocks, num_sab_heads, 64, 64):
        raise RuntimeError(
            f"t must have shape {(total_t_blocks, num_sab_heads, 64, 64)}, got {tuple(t.shape)}"
        )
    total_cp_chunks = workspace_num_chunks_host(cu_seqlens, cp_chunk_len, total_seqlen)
    if not _skip_check:
        if d != 128:
            raise RuntimeError(
                f"CPDeltaRuleMNPrecomputeSm100 only supports D=128, got {d}"
            )
        if k.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"CPDeltaRuleMNPrecomputeSm100 only supports fp16/bf16 inputs, got {k.dtype}"
            )
        if v.dtype != k.dtype or t.dtype != k.dtype:
            raise RuntimeError(
                f"v/t dtypes must match k dtype, got k={k.dtype}, v={v.dtype}, t={t.dtype}"
            )
        if alpha.dtype != torch.float32:
            raise RuntimeError(
                f"alpha must have dtype torch.float32, got {alpha.dtype}"
            )
        for name, tensor in (("k", k), ("v", v), ("t", t), ("alpha", alpha)):
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")

    max_cp_chunks_per_seq = max_num_chunks_host(max_seqlen, cp_chunk_len)
    k_tma = k.as_strided((d, total_seqlen, num_k_heads), (1, num_k_heads * d, d))
    v_tma = v.as_strided((d, total_seqlen, num_v_heads), (1, num_v_heads * d, d))
    t_tma = t.as_strided(
        (64, 64, num_sab_heads, total_t_blocks),
        (64, 1, 64 * 64, num_sab_heads * 64 * 64),
    )
    workspace_shape = (total_cp_chunks, num_sab_heads, d, d)
    transfer_t = _get_cp_workspace(
        "gdn_cp_sm100_local_transfer", workspace_shape, torch.float32, device
    )
    state_t = _get_cp_workspace(
        "gdn_cp_sm100_local_state", workspace_shape, torch.float32, device
    )
    if total_cp_chunks == 0:
        return transfer_t, state_t

    kernel_dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[
        k.dtype
    ]
    stream = (
        _stream
        if _stream is not None
        else cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    )
    kernel = _get_mn_precompute_kernel(
        kernel_dtype, _cu_seqlens_dtype(cu_seqlens.dtype)
    )
    compile_options = _blackwell_compile_options(device)
    compiled = get_cached_compile(kernel, compile_options)
    if compiled is None:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )
        kernel_args = (
            from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(v_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(t_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(alpha.view(-1), assumed_align=16).mark_layout_dynamic(),
            from_dlpack(transfer_t.view(-1), assumed_align=16).mark_layout_dynamic(),
            from_dlpack(state_t.view(-1), assumed_align=16).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            cutlass.Int32(cp_chunk_len),
            cutlass.Int32(num_k_heads),
            cutlass.Int32(num_v_heads),
            cutlass.Int32(num_sab_heads),
            cutlass.Int32(total_cp_chunks),
            cutlass.Int32(max_cp_chunks_per_seq),
            cutlass.Int32(num_seqs),
            stream,
        )
        compiled = cached_compile(kernel, *kernel_args, compile_options=compile_options)
    compiled(
        k_tma,
        v_tma,
        t_tma,
        alpha.view(-1),
        transfer_t.view(-1),
        state_t.view(-1),
        cu_seqlens,
        cp_chunk_len,
        num_k_heads,
        num_v_heads,
        num_sab_heads,
        total_cp_chunks,
        max_cp_chunks_per_seq,
        num_seqs,
        stream,
    )
    return transfer_t, state_t


class CPDeltaRuleFixupSm100(CPDeltaRuleFixupSimtSm120):
    """SM100 specialization of the register-prefetched SIMT fixup body."""


class CPDeltaRuleFixupHmmaSm100(CPDeltaRuleFixupHmmaSm120):
    """SM100 specialization of the HMMA fixup fallback."""


@functools.cache
def _get_fixup_kernel(
    needs_initial_state,
    store_final_state,
    initial_state_dtype,
    output_state_dtype,
    use_state_indices,
    state_indices_dtype,
    initial_state_inner_strides,
    output_state_inner_strides,
    kernel_kind,
    cu_seqlens_dtype,
):
    if kernel_kind == "simt_row4":
        return CPDeltaRuleFixupSm100(
            needs_initial_state=needs_initial_state,
            rows_per_cta=4,
            store_final_state=store_final_state,
            initial_state_dtype=initial_state_dtype,
            state_dtype=output_state_dtype,
            cu_seqlens_dtype=cu_seqlens_dtype,
            use_state_indices=use_state_indices,
            state_indices_dtype=state_indices_dtype,
            initial_state_inner_strides=initial_state_inner_strides,
            output_state_inner_strides=output_state_inner_strides,
            store_initial_state=needs_initial_state,
        )
    if kernel_kind == "simt_row8":
        return CPDeltaRuleFixupSm100(
            needs_initial_state=needs_initial_state,
            rows_per_cta=8,
            store_final_state=store_final_state,
            initial_state_dtype=initial_state_dtype,
            state_dtype=output_state_dtype,
            cu_seqlens_dtype=cu_seqlens_dtype,
            use_state_indices=use_state_indices,
            state_indices_dtype=state_indices_dtype,
            initial_state_inner_strides=initial_state_inner_strides,
            output_state_inner_strides=output_state_inner_strides,
            store_initial_state=needs_initial_state,
        )
    if kernel_kind == "hmma":
        return CPDeltaRuleFixupHmmaSm100(
            needs_initial_state=needs_initial_state,
            store_final_state=store_final_state,
            initial_state_dtype=initial_state_dtype,
            state_dtype=output_state_dtype,
            cu_seqlens_dtype=cu_seqlens_dtype,
            use_state_indices=use_state_indices,
            state_indices_dtype=state_indices_dtype,
            initial_state_inner_strides=initial_state_inner_strides,
            output_state_inner_strides=output_state_inner_strides,
            store_initial_state=needs_initial_state,
        )
    if kernel_kind in ("utcmma", "utcmma64", "utcmma128"):
        return CPDeltaRuleFixupUtcmmaSm100(
            rows_per_cta=128 if kernel_kind == "utcmma128" else 64,
            needs_initial_state=needs_initial_state,
            store_final_state=store_final_state,
            initial_state_dtype=initial_state_dtype,
            state_dtype=output_state_dtype,
            cu_seqlens_dtype=cu_seqlens_dtype,
            use_state_indices=use_state_indices,
            state_indices_dtype=state_indices_dtype,
            initial_state_inner_strides=initial_state_inner_strides,
            output_state_inner_strides=output_state_inner_strides,
            store_initial_state=needs_initial_state,
        )
    raise ValueError(f"Unsupported fixup kernel kind: {kernel_kind}")


def cp_delta_rule_fixup_dsl_sm100(
    local_transfer: torch.Tensor,
    local_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int = 4096,
    initial_state: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    initial_state_workspace: torch.Tensor | None = None,
    *,
    _skip_check: bool = False,
    _kernel_kind: str | None = None,
    _device=None,
    _stream=None,
):
    """Fix SM100 CP chunk states into global chunk-boundary states."""
    import cuda.bindings.driver as cuda_driver

    device = local_transfer.device if _device is None else _device
    if not _skip_check:
        if local_transfer.ndim != 4:
            raise RuntimeError(
                "local_transfer must have shape (num_chunks, num_heads, DimV, DimK), "
                f"got {tuple(local_transfer.shape)}"
            )
        if local_state.shape != local_transfer.shape:
            raise RuntimeError(
                "local_state shape must match local_transfer shape, "
                f"got {tuple(local_state.shape)} and {tuple(local_transfer.shape)}"
            )
        if local_transfer.shape[-2:] != (128, 128):
            raise RuntimeError(
                f"CPDeltaRuleFixupSm100 only supports D=128, got {tuple(local_transfer.shape[-2:])}"
            )
        if local_transfer.dtype != torch.float32 or local_state.dtype != torch.float32:
            raise RuntimeError(
                "CPDeltaRuleFixupSm100 only supports float32 inputs, "
                f"got {local_transfer.dtype} and {local_state.dtype}"
            )
        use_state_indices = state_indices is not None
        for name, tensor in (
            ("initial_state", initial_state),
            ("output_state", output_state),
        ):
            if tensor is not None:
                expected_tail = (local_transfer.shape[1], 128, 128)
                if tuple(tensor.shape[1:]) != expected_tail or (
                    not use_state_indices and tensor.shape[0] != cu_seqlens.shape[0] - 1
                ):
                    raise RuntimeError(
                        f"{name} must have shape "
                        f"{('[N_pool]' if use_state_indices else f'[{cu_seqlens.shape[0] - 1}]')}"
                        f" + {expected_tail}, got {tuple(tensor.shape)}"
                    )
                _cutlass_state_dtype(tensor.dtype)
        if use_state_indices:
            if not is_integer_dtype(state_indices.dtype) or state_indices.shape != (
                cu_seqlens.shape[0] - 1,
            ):
                raise RuntimeError(
                    f"state_indices must have shape {(cu_seqlens.shape[0] - 1,)} and an integer dtype"
                )
        for name, tensor in (
            ("local_transfer", local_transfer),
            ("local_state", local_state),
            ("state_indices", state_indices),
        ):
            if tensor is not None and not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
        for name, tensor in (
            ("initial_state", initial_state),
            ("output_state", output_state),
        ):
            if (
                tensor is not None
                and not use_state_indices
                and not tensor.is_contiguous()
            ):
                raise RuntimeError(f"{name} must be contiguous")

    total_cp_chunks, num_heads, _, d = local_transfer.shape
    if not _skip_check:
        if cp_chunk_len <= 0:
            raise RuntimeError(f"cp_chunk_len must be positive, got {cp_chunk_len}")
        _cu_seqlens_dtype(cu_seqlens.dtype)
        expected_chunks = workspace_num_chunks_host(
            cu_seqlens, cp_chunk_len, total_seqlen
        )
        if expected_chunks != total_cp_chunks:
            raise RuntimeError(
                "local_transfer/local_state first dim must equal "
                f"chunk_bound(num_seqs, total_seqlen, cp_chunk_len)={expected_chunks}, got {total_cp_chunks}"
            )
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1

    fixed_state = _get_cp_workspace(
        "gdn_cp_sm100_fixed_state", local_state.shape, local_state.dtype, device
    )
    if initial_state is not None and initial_state_workspace is None:
        initial_state_workspace = _get_cp_workspace(
            "gdn_cp_sm100_initial_state",
            (num_seqs, num_heads, d, d),
            torch.float32,
            device,
        )
    if total_cp_chunks == 0:
        return fixed_state

    local_transfer_tma = local_transfer.as_strided(
        (d, d, num_heads, total_cp_chunks), (1, d, d * d, num_heads * d * d)
    )
    local_state_tma = local_state.as_strided(
        (d, d, num_heads, total_cp_chunks), (d, 1, d * d, num_heads * d * d)
    )
    stream = (
        _stream
        if _stream is not None
        else cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    )
    needs_initial_state = initial_state is not None
    store_final_state = output_state is not None
    use_state_indices = state_indices is not None
    initial_state_dtype = _cutlass_state_dtype(
        initial_state.dtype if needs_initial_state else torch.float32
    )
    output_state_dtype = _cutlass_state_dtype(
        output_state.dtype if store_final_state else torch.float32
    )
    if _kernel_kind is None:
        num_parallel_states = num_seqs * num_heads
        num_sms = get_device_sm_count(device)
        # SIMT4 sustains two CTAs/SM. UTC64 and UTC128 launch D/rows_per_cta
        # CTAs per state, so UTC64 switches out when its two-CTA grid exceeds one wave.
        simt_row4_one_wave_states = num_sms * 2 // (d // 4)
        utcmma64_one_wave_states = num_sms // (d // 64)
        if num_parallel_states <= simt_row4_one_wave_states:
            _kernel_kind = "simt_row4"
        elif num_parallel_states <= utcmma64_one_wave_states:
            _kernel_kind = "utcmma64"
        else:
            _kernel_kind = "utcmma128"
    kernel = _get_fixup_kernel(
        needs_initial_state,
        store_final_state,
        initial_state_dtype,
        output_state_dtype,
        use_state_indices,
        _cu_seqlens_dtype(state_indices.dtype) if use_state_indices else None,
        (
            tuple(initial_state.stride()[1:])
            if use_state_indices and needs_initial_state
            else None
        ),
        (
            tuple(output_state.stride()[1:])
            if use_state_indices and store_final_state
            else None
        ),
        _kernel_kind,
        _cu_seqlens_dtype(cu_seqlens.dtype),
    )
    compile_options = _blackwell_compile_options(device)
    compiled = get_cached_compile(kernel, compile_options)
    if compiled is None:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )
        initial_state_cute = None
        if needs_initial_state:
            initial_state_cute = from_dlpack(initial_state, assumed_align=16)
            _mark_state_layout(initial_state_cute, use_state_indices, d)
        initial_state_workspace_cute = (
            from_dlpack(
                initial_state_workspace.reshape(-1), assumed_align=16
            ).mark_layout_dynamic()
            if needs_initial_state
            else None
        )
        output_state_cute = None
        if store_final_state:
            output_state_cute = from_dlpack(output_state, assumed_align=16)
            _mark_state_layout(output_state_cute, use_state_indices, d)
        kernel_args = (
            from_dlpack(local_transfer_tma, assumed_align=128).mark_layout_dynamic(
                leading_dim=0
            ),
            from_dlpack(local_state_tma, assumed_align=128).mark_layout_dynamic(
                leading_dim=1
            ),
            initial_state_cute,
            initial_state_workspace_cute,
            from_dlpack(
                fixed_state.reshape(-1), assumed_align=128
            ).mark_layout_dynamic(),
            output_state_cute,
            (
                from_dlpack(state_indices, assumed_align=4).mark_layout_dynamic()
                if use_state_indices
                else None
            ),
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            cutlass.Int32(cp_chunk_len),
            cutlass.Int32(total_cp_chunks),
            cutlass.Int32(num_seqs),
            cutlass.Int32(num_heads),
            stream,
        )
        compiled = cached_compile(kernel, *kernel_args, compile_options=compile_options)
    compiled(
        local_transfer_tma,
        local_state_tma,
        initial_state if needs_initial_state else None,
        initial_state_workspace.reshape(-1) if needs_initial_state else None,
        fixed_state.reshape(-1),
        output_state if store_final_state else None,
        state_indices if use_state_indices else None,
        cu_seqlens,
        cp_chunk_len,
        total_cp_chunks,
        num_seqs,
        num_heads,
        stream,
    )
    return fixed_state


@functools.cache
def _get_prefill_kernel(
    kernel_dtype,
    state_dtype,
    checkpoint_dtype,
    num_sm,
    is_gqa,
    head_ratio,
    needs_initial_state,
    store_final_state,
    enable_checkpoints,
    cu_seqlens_dtype,
    checkpoint_cu_starts_dtype,
):
    return CPDeltaRulePrefillTcgen05Sm100(
        io_dtype=kernel_dtype,
        acc_dtype=cutlass.Float32,
        state_dtype=state_dtype,
        checkpoint_dtype=checkpoint_dtype,
        mma_tiler_qk=(64, 64, 128),
        mma_tiler_qs=(128, 64, 128),
        mma_tiler_qkv=(128, 64, 64),
        mma_tiler_kv=(128, 128, 64),
        max_active_clusters=num_sm,
        num_sm=num_sm,
        is_GQA=is_gqa,
        head_ratio=head_ratio,
        use_initial_state=needs_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        is_persistent=False,
        cu_seqlens_dtype=cu_seqlens_dtype,
        checkpoint_cu_starts_dtype=checkpoint_cu_starts_dtype,
    )


def cp_delta_rule_prefill_dsl_sm100(
    o: torch.Tensor,
    state: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    fixed_state: torch.Tensor,
    alpha: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int = 4096,
    max_seqlen: int | None = None,
    initial_state: torch.Tensor | None = None,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    *,
    _skip_check: bool = False,
    _device=None,
    _stream=None,
):
    """Run CP main prefill with precomputed T and fixed-up chunk states on SM100."""
    import cuda.bindings.driver as cuda_driver

    device = q.device if _device is None else _device
    if not _skip_check:
        if q.ndim != 3:
            raise RuntimeError(
                f"q must have shape (total_seqlen, num_q_heads, D), got {tuple(q.shape)}"
            )
        if k.ndim != 3 or v.ndim != 3 or o.ndim != 3:
            raise RuntimeError(
                "k, v, and o must have shape (total_seqlen, num_heads, D), "
                f"got k={tuple(k.shape)}, v={tuple(v.shape)}, o={tuple(o.shape)}"
            )
        if (
            k.shape[0] != q.shape[0]
            or v.shape[0] != q.shape[0]
            or o.shape[0] != q.shape[0]
        ):
            raise RuntimeError("q, k, v, and o must have the same total_seqlen")
        if (
            k.shape[2] != q.shape[2]
            or v.shape[2] != q.shape[2]
            or o.shape[2] != q.shape[2]
        ):
            raise RuntimeError("q, k, v, and o must have the same D")
        if total_seqlen != q.shape[0]:
            raise RuntimeError(
                f"total_seqlen must match q.shape[0], got {total_seqlen} and {q.shape[0]}"
            )
        if cp_chunk_len % 64 != 0:
            raise RuntimeError(
                f"cp_chunk_len must be a multiple of 64, got {cp_chunk_len}"
            )
        _cu_seqlens_dtype(cu_seqlens.dtype)
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided")
    _, num_q_heads, d = q.shape
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    enable_checkpoints = checkpoint_every_n_tokens > 0
    total_t_blocks = workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
    total_cp_chunks = workspace_num_chunks_host(cu_seqlens, cp_chunk_len, total_seqlen)
    if not _skip_check:
        if d != 128:
            raise RuntimeError(f"CPDeltaRulePrefillSm100 only supports D=128, got {d}")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"CPDeltaRulePrefillSm100 only supports fp16/bf16 inputs, got {q.dtype}"
            )
        if (
            k.dtype != q.dtype
            or v.dtype != q.dtype
            or o.dtype != q.dtype
            or t.dtype != q.dtype
        ):
            raise RuntimeError("q/k/v/o/t dtypes must match")
        if alpha.dtype != torch.float32:
            raise RuntimeError(
                f"alpha must have dtype torch.float32, got {alpha.dtype}"
            )
        if o.shape != (total_seqlen, num_sab_heads, d):
            raise RuntimeError(
                f"o must have shape {(total_seqlen, num_sab_heads, d)}, got {tuple(o.shape)}"
            )
        if t.shape != (total_t_blocks, num_sab_heads, 64, 64):
            raise RuntimeError(
                f"t must have shape {(total_t_blocks, num_sab_heads, 64, 64)}, got {tuple(t.shape)}"
            )
        expected_workspace_shape = (total_cp_chunks, num_sab_heads, d, d)
        expected_state_shape = (num_seqs, num_sab_heads, d, d)
        if fixed_state.shape != expected_workspace_shape:
            raise RuntimeError(
                f"fixed_state must have shape {expected_workspace_shape}, got {tuple(fixed_state.shape)}"
            )
        if state is not None:
            if state.shape != expected_state_shape:
                raise RuntimeError(
                    f"state must have shape {expected_state_shape}, got {tuple(state.shape)}"
                )
            _cutlass_state_dtype(state.dtype)
        if initial_state is not None:
            if initial_state.shape != expected_state_shape:
                raise RuntimeError(
                    f"initial_state must have shape {expected_state_shape}, got {tuple(initial_state.shape)}"
                )
        if checkpoint_every_n_tokens < 0 or checkpoint_every_n_tokens % 64 != 0:
            raise RuntimeError(
                "checkpoint_every_n_tokens must be a non-negative multiple of 64, "
                f"got {checkpoint_every_n_tokens}"
            )
        if enable_checkpoints:
            if state_checkpoints is None or checkpoint_cu_starts is None:
                raise RuntimeError(
                    "state_checkpoints and checkpoint_cu_starts are required when "
                    "checkpointing is enabled"
                )
            if tuple(state_checkpoints.shape[1:]) != (num_sab_heads, d, d):
                raise RuntimeError(
                    "state_checkpoints must have shape "
                    f"[*, {num_sab_heads}, {d}, {d}], got {tuple(state_checkpoints.shape)}"
                )
            if checkpoint_cu_starts.shape != (num_seqs + 1,):
                raise RuntimeError(
                    "checkpoint_cu_starts must have shape "
                    f"{(num_seqs + 1,)}, got {tuple(checkpoint_cu_starts.shape)}"
                )
            if not is_integer_dtype(checkpoint_cu_starts.dtype):
                raise RuntimeError(
                    "checkpoint_cu_starts must have an integer dtype, "
                    f"got {checkpoint_cu_starts.dtype}"
                )
        elif state_checkpoints is not None or checkpoint_cu_starts is not None:
            raise RuntimeError(
                "state_checkpoints and checkpoint_cu_starts must be None when "
                "checkpointing is disabled"
            )
        if not _FullyFusedDeltaRuleSm120.can_implement(
            num_q_heads, num_k_heads, num_v_heads, d, q.element_size()
        ):
            raise RuntimeError(
                "q/v heads must be positive multiples of k heads, "
                f"got q={num_q_heads}, k={num_k_heads}, v={num_v_heads}"
            )
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("t", t),
            ("fixed_state", fixed_state),
            ("alpha", alpha),
            ("o", o),
            ("state_checkpoints", state_checkpoints),
            ("checkpoint_cu_starts", checkpoint_cu_starts),
        ):
            if tensor is not None and not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
        if state is not None and state.stride()[1:] != (d * d, d, 1):
            raise RuntimeError("state must be contiguous in its inner [H, V, K] modes")
        if initial_state is not None and initial_state.stride()[1:] != (d * d, d, 1):
            raise RuntimeError(
                "initial_state must be contiguous in its inner [H, V, K] modes"
            )
        if (
            state is not None
            and initial_state is not None
            and state.dtype != initial_state.dtype
        ):
            raise RuntimeError(
                f"state and initial_state dtypes must match, got {state.dtype} and {initial_state.dtype}"
            )
    if total_cp_chunks == 0:
        return

    max_cp_chunks_per_seq = max_num_chunks_host(max_seqlen, cp_chunk_len)
    t_tma = t.as_strided(
        (64, 64, num_sab_heads, total_t_blocks),
        (64, 1, 64 * 64, num_sab_heads * 64 * 64),
    )
    kernel_dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[
        q.dtype
    ]
    stream = (
        _stream
        if _stream is not None
        else cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    )
    needs_initial_state = initial_state is not None
    store_final_state = state is not None
    num_sm = get_device_sm_count(device)
    is_gqa = num_q_heads >= num_v_heads
    head_ratio = num_q_heads // num_v_heads if is_gqa else num_v_heads // num_q_heads
    state_dtype = (
        state.dtype
        if store_final_state
        else initial_state.dtype if needs_initial_state else torch.float32
    )
    checkpoint_dtype = state_checkpoints.dtype if enable_checkpoints else state_dtype
    kernel = _get_prefill_kernel(
        kernel_dtype,
        _cutlass_state_dtype(state_dtype),
        _cutlass_state_dtype(checkpoint_dtype),
        num_sm,
        is_gqa,
        head_ratio,
        needs_initial_state,
        store_final_state,
        enable_checkpoints,
        _cu_seqlens_dtype(cu_seqlens.dtype),
        (_cu_seqlens_dtype(checkpoint_cu_starts.dtype) if enable_checkpoints else None),
    )
    compile_options = _blackwell_compile_options(device)
    workspace_size = (
        num_seqs
        * num_sab_heads
        * max_cp_chunks_per_seq
        * CPDeltaRulePrefillTcgen05Sm100.num_tensormaps
        * CPDeltaRulePrefillTcgen05Sm100.bytes_per_tensormap
    )
    tensormaps_t = _get_cache_buf(
        "gdn_cp_sm100_prefill_tensormaps", workspace_size, device
    )[:workspace_size]
    compiled = get_cached_compile(kernel, compile_options)
    if compiled is None:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )
        initial_state_cute = None
        if needs_initial_state:
            initial_state_cute = from_dlpack(initial_state, assumed_align=16)
            _mark_state_layout(initial_state_cute, False, d)
        state_cute = None
        if store_final_state:
            state_cute = from_dlpack(state, assumed_align=16)
            _mark_state_layout(state_cute, False, d)
        state_checkpoints_cute = None
        checkpoint_cu_starts_cute = None
        if enable_checkpoints:
            state_checkpoints_cute = from_dlpack(state_checkpoints, assumed_align=16)
            _mark_state_layout(state_checkpoints_cute, False, d)
            checkpoint_cu_starts_cute = from_dlpack(
                checkpoint_cu_starts, assumed_align=8
            ).mark_layout_dynamic()
        kernel_args = (
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(alpha, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(t_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(o, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            from_dlpack(fixed_state, assumed_align=16).mark_layout_dynamic(),
            initial_state_cute,
            state_cute,
            state_checkpoints_cute,
            checkpoint_cu_starts_cute,
            cutlass.Int32(checkpoint_every_n_tokens),
            cutlass.Int32(cp_chunk_len),
            cutlass.Int32(total_cp_chunks),
            cutlass.Int32(max_cp_chunks_per_seq),
            cutlass.Int32(num_seqs),
            cutlass.Float32(scale),
            from_dlpack(tensormaps_t, assumed_align=128).mark_layout_dynamic(),
            stream,
        )
        compiled = cached_compile(kernel, *kernel_args, compile_options=compile_options)
    compiled(
        q,
        k,
        v,
        alpha,
        t_tma,
        o,
        cu_seqlens,
        fixed_state,
        initial_state if needs_initial_state else None,
        state if store_final_state else None,
        state_checkpoints if enable_checkpoints else None,
        checkpoint_cu_starts if enable_checkpoints else None,
        checkpoint_every_n_tokens,
        cp_chunk_len,
        total_cp_chunks,
        max_cp_chunks_per_seq,
        num_seqs,
        scale,
        tensormaps_t,
        stream,
    )


def cp_delta_rule_dsl_sm100(
    o: torch.Tensor,
    state: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    *,
    initial_state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    max_seqlen: int | None = None,
    cp_chunk_len: int | None = None,
    cp_chunk_len_granularity: int = CP_CHUNK_LEN_GRANULARITY,
):
    """Run the context-parallel delta-rule prefill pipeline on SM100."""
    import cuda.bindings.driver as cuda_driver

    # Fetch launch metadata once and reuse it across the four kernel wrappers.
    device = q.device
    device_name = get_device_name(device)
    device_capability = get_compute_capability(device)
    num_sms = get_device_sm_count(device)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)

    total_seqlen = q.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None and num_seqs == 1:
        max_seqlen = total_seqlen
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided when num_seqs != 1")
    if max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    if cp_chunk_len is None:
        num_heads = max(q.shape[1], v.shape[1])
        cp_chunk_len = choose_cp_chunk_len_host(
            max_seqlen,
            num_heads,
            num_sms,
            chunk_len_granularity=cp_chunk_len_granularity,
            device_capability=device_capability,
            total_seqlen=total_seqlen,
            num_seqs=num_seqs,
            device_name=device_name,
        )
    if q.ndim != 3:
        raise RuntimeError(
            f"q must have shape (total_seqlen, num_q_heads, D), got {tuple(q.shape)}"
        )
    if k.ndim != 3 or v.ndim != 3 or o.ndim != 3:
        raise RuntimeError(
            "k, v, and o must have shape (total_seqlen, num_heads, D), "
            f"got k={tuple(k.shape)}, v={tuple(v.shape)}, o={tuple(o.shape)}"
        )
    if (
        k.shape[0] != total_seqlen
        or v.shape[0] != total_seqlen
        or o.shape[0] != total_seqlen
    ):
        raise RuntimeError("q, k, v, and o must have the same total_seqlen")
    if k.shape[2] != q.shape[2] or v.shape[2] != q.shape[2] or o.shape[2] != q.shape[2]:
        raise RuntimeError("q, k, v, and o must have the same D")
    if alpha.shape != beta.shape or alpha.shape[0] != total_seqlen:
        raise RuntimeError(
            "alpha and beta must have shape (total_seqlen, num_sab_heads), "
            f"got alpha={tuple(alpha.shape)} and beta={tuple(beta.shape)}"
        )
    if cp_chunk_len % 64 != 0:
        raise RuntimeError(f"cp_chunk_len must be a multiple of 64, got {cp_chunk_len}")
    _cu_seqlens_dtype(cu_seqlens.dtype)
    if not cu_seqlens.is_contiguous():
        raise RuntimeError("cu_seqlens must be contiguous")

    _, num_q_heads, d = q.shape
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    enable_checkpoints = checkpoint_every_n_tokens > 0
    if o.shape[1] != num_sab_heads or alpha.shape[1] != num_sab_heads:
        raise RuntimeError(
            f"o/alpha/beta heads must equal max(q heads, v heads)={num_sab_heads}"
        )
    if not _FullyFusedDeltaRuleSm120.can_implement(
        num_q_heads, num_k_heads, num_v_heads, d, q.element_size()
    ):
        raise RuntimeError(
            "CPDeltaRuleSm100 only supports GQA/GVA head counts, "
            f"got q={num_q_heads}, k={num_k_heads}, v={num_v_heads}"
        )
    expected_state_shape = (num_seqs, num_sab_heads, d, d)
    use_state_indices = state_indices is not None
    if state is not None and (
        (not use_state_indices and state.shape != expected_state_shape)
        or (use_state_indices and tuple(state.shape[1:]) != expected_state_shape[1:])
    ):
        raise RuntimeError(
            f"state must have shape "
            f"{('[N_pool]' if use_state_indices else f'[{num_seqs}]')} + {expected_state_shape[1:]}, "
            f"got {tuple(state.shape)}"
        )
    if initial_state is not None and (
        (not use_state_indices and initial_state.shape != expected_state_shape)
        or (
            use_state_indices
            and tuple(initial_state.shape[1:]) != expected_state_shape[1:]
        )
    ):
        raise RuntimeError(
            f"initial_state must have shape "
            f"{('[N_pool]' if use_state_indices else f'[{num_seqs}]')} + {expected_state_shape[1:]}, "
            f"got {tuple(initial_state.shape)}"
        )
    if use_state_indices and (
        not is_integer_dtype(state_indices.dtype) or state_indices.shape != (num_seqs,)
    ):
        raise RuntimeError(
            f"state_indices must have shape {(num_seqs,)} and an integer dtype"
        )
    if checkpoint_every_n_tokens < 0 or checkpoint_every_n_tokens % 64 != 0:
        raise RuntimeError(
            "checkpoint_every_n_tokens must be a non-negative multiple of 64, "
            f"got {checkpoint_every_n_tokens}"
        )
    if enable_checkpoints:
        if state_checkpoints is None or checkpoint_cu_starts is None:
            raise RuntimeError(
                "state_checkpoints and checkpoint_cu_starts are required when "
                "checkpointing is enabled"
            )
        if tuple(state_checkpoints.shape[1:]) != (num_sab_heads, d, d):
            raise RuntimeError(
                "state_checkpoints must have shape "
                f"[*, {num_sab_heads}, {d}, {d}], got {tuple(state_checkpoints.shape)}"
            )
        if checkpoint_cu_starts.shape != (num_seqs + 1,):
            raise RuntimeError(
                "checkpoint_cu_starts must have shape "
                f"{(num_seqs + 1,)}, got {tuple(checkpoint_cu_starts.shape)}"
            )
        if not is_integer_dtype(checkpoint_cu_starts.dtype):
            raise RuntimeError(
                "checkpoint_cu_starts must have an integer dtype, "
                f"got {checkpoint_cu_starts.dtype}"
            )
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        raise RuntimeError(
            "state_checkpoints and checkpoint_cu_starts must be None when "
            "checkpointing is disabled"
        )
    if d != 128:
        raise RuntimeError(f"CPDeltaRuleSm100 only supports D=128, got {d}")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"CPDeltaRuleSm100 only supports fp16/bf16 inputs, got {q.dtype}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype or o.dtype != q.dtype:
        raise RuntimeError("q/k/v/o dtypes must match")
    if alpha.dtype != torch.float32 or beta.dtype != torch.float32:
        raise RuntimeError("alpha and beta must have dtype torch.float32")
    for tensor in (state, initial_state):
        if tensor is not None:
            _cutlass_state_dtype(tensor.dtype)
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("alpha", alpha),
        ("beta", beta),
        ("o", o),
        ("state_indices", state_indices),
        ("state_checkpoints", state_checkpoints),
        ("checkpoint_cu_starts", checkpoint_cu_starts),
    ):
        if tensor is not None and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    for name, tensor in (("state", state), ("initial_state", initial_state)):
        if tensor is not None and not use_state_indices and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    with torch.cuda.nvtx.range("cp_delta_rule_sm100"):
        t = cp_delta_rule_t_precompute_dsl_sm100(
            k,
            beta,
            cu_seqlens,
            total_seqlen,
            max_seqlen=max_seqlen,
            _skip_check=True,
            _device=device,
            _stream=stream,
        )
        local_transfer, local_state = cp_delta_rule_mn_precompute_dsl_sm100(
            k,
            v,
            t,
            alpha,
            cu_seqlens,
            total_seqlen,
            cp_chunk_len=cp_chunk_len,
            max_seqlen=max_seqlen,
            _skip_check=True,
            _device=device,
            _stream=stream,
        )
        initial_state_workspace = None
        if initial_state is not None:
            initial_state_workspace = _get_cp_workspace(
                "gdn_cp_sm100_initial_state",
                (num_seqs, num_sab_heads, d, d),
                torch.float32,
                device,
            )
        fixed_state = cp_delta_rule_fixup_dsl_sm100(
            local_transfer,
            local_state,
            cu_seqlens,
            total_seqlen,
            cp_chunk_len=cp_chunk_len,
            initial_state=initial_state,
            output_state=state,
            state_indices=state_indices,
            initial_state_workspace=initial_state_workspace,
            _skip_check=True,
            _device=device,
            _stream=stream,
        )
        cp_delta_rule_prefill_dsl_sm100(
            o,
            None,
            q,
            k,
            v,
            t,
            fixed_state,
            alpha,
            scale,
            cu_seqlens,
            total_seqlen,
            cp_chunk_len=cp_chunk_len,
            max_seqlen=max_seqlen,
            initial_state=initial_state_workspace,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            _skip_check=True,
            _device=device,
            _stream=stream,
        )
