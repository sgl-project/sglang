# SPDX-License-Identifier: Apache-2.0
"""NVIDIA split KDA prefill backend (K1-K4 CuTe/Triton/cuTile pipeline).

Wraps the vendored ``kda_nvidia_prefill`` package through its FLA-compatible
``chunk_kda_fwd`` interface. SGLang owns the boundary conversion from its
V-major ``[B,H,V,K]`` cache to the vendor's K-major ``[B,H,K,V]`` state and
back; the vendored split kernels keep their original internal layouts.

The serving wrapper repacks ordinary packed prefill batches into bounded
equal-length groups (B <= 8) and pads every sequence to one of
2k/4k/8k/16k. This makes short multi-sequence prefill use the same NVIDIA KDA
pipeline while bounding compile variants and transient workspaces. Everything
that is not an ordinary state-committing prefill stays on its dedicated path:

- single-sequence prefill, including chunks of one long request, where Triton
  retains the neutral BS=1 performance;
- track batches needing an *interior* chunk snapshot (mamba extra_buffer
  with a non-boundary track point): the pipeline never materializes
  per-chunk states. Boundary-aligned track batches reuse the final state;
- spec-decode extends, which must stay rollback-able.

Inputs are staged into per-bucket persistent buffers (2k/4k/8k/16k):
stable pointers keep the vendored pipeline's pointer-keyed wrapper and
fast-path caches hot, and bound both the workspace set and the compile
variants. Pad rows are state-neutral (k/v/beta zero => no rank-1 update;
g sentinel -1000 => decay exactly 1).

Any unexpected kernel failure falls back to Triton for that batch with a
loud log (attempt-and-verify, same stance as the fused decode kernel).
"""

import logging
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

logger = logging.getLogger(__name__)

_BUCKETS = (2048, 4096, 8192, 16384)
_MAX_NVIDIA_KDA_BATCH = 8


def _to_nvidia_kda_state_layout(
    state: torch.Tensor, *, head_k_dim: int, head_v_dim: int
) -> torch.Tensor:
    """Materialize SGLang [B,H,V,K] state as vendor [B,H,K,V]."""
    expected = (head_v_dim, head_k_dim)
    if state.ndim != 4 or tuple(state.shape[-2:]) != expected:
        raise ValueError(
            "SGLang KDA state must be [B,H,V,K] with "
            f"(V,K)={expected}, got {tuple(state.shape)}"
        )
    return state.transpose(-1, -2).float().contiguous()


def _from_nvidia_kda_state_layout(
    state: torch.Tensor,
    *,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize vendor [B,H,K,V] state as SGLang [B,H,V,K]."""
    expected = (head_k_dim, head_v_dim)
    if state.ndim != 4 or tuple(state.shape[-2:]) != expected:
        raise ValueError(
            "NVIDIA KDA state must be [B,H,K,V] with "
            f"(K,V)={expected}, got {tuple(state.shape)}"
        )
    return state.transpose(-1, -2).to(dtype=dtype).contiguous()


class NvidiaKDAKernel(LinearAttnKernelBase):
    def __init__(self):
        # This kernel uses tcgen05 + TMEM, which are available on datacenter
        # Blackwell (SM100/SM103, reported as capability major 10), but not on
        # Blackwell desktop SM120 even though its capability number is larger.
        self.supports_prefill = torch.cuda.is_available() and (
            torch.cuda.get_device_capability()[0] == 10
        )
        self._fwd = None
        self._l2norm = None
        self._triton = TritonKDAKernel()
        # Stable detached fp32 views of the (frozen) gate params: nn.Parameters
        # trip cute's from_dlpack (grad-tracking) and per-call views defeat
        # the pointer-keyed wrapper caches. Keyed per source tensor -- this
        # kernel instance is shared by every KDA layer (one dispatcher per
        # attention backend) and each layer has its own A_log/dt_bias.
        self._param_flat = {}
        # One-shot engagement log per (batch size, bucket) (observability).
        self._engaged_logged = set()
        self._unsupported_logged = set()
        # (batch size, bucket, shape, device) -> staging dict.
        self._staging = {}

    def _ensure_loaded(self):
        if self._fwd is None:
            from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
            from sglang.kernels.ops.attention.linear.kda_nvidia_prefill import (
                chunk_kda_fwd,
            )

            self._fwd = chunk_kda_fwd
            self._l2norm = l2norm_fwd
            logger.info("Using NVIDIA chunked KDA prefill (Blackwell)")

    def decode(self, *args, **kwargs):
        raise NotImplementedError("NvidiaKDAKernel is prefill-only")

    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("NvidiaKDAKernel does not support target_verify")

    def _triton_extend(
        self,
        q,
        k,
        v,
        g,
        beta,
        ssm_states,
        cache_indices,
        query_start_loc,
        A_log,
        dt_bias,
        lower_bound,
        return_intermediate_states,
        kwargs,
    ):
        return self._triton.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            return_intermediate_states=return_intermediate_states,
            **kwargs,
        )

    def _flat_param(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        key = (t.data_ptr(), t.dtype, tuple(t.shape))
        v = self._param_flat.get(key)
        if v is None:
            v = t.detach().reshape(-1).float().contiguous()
            self._param_flat[key] = v
        return v

    def _get_staging(self, batch_size, bucket, num_heads, head_k_dim, head_v_dim, dev):
        key = (
            batch_size,
            bucket,
            num_heads,
            head_k_dim,
            head_v_dim,
            dev.index,
        )
        st = self._staging.get(key)
        if st is None:
            st = {
                "q": torch.zeros(
                    batch_size,
                    bucket,
                    num_heads,
                    head_k_dim,
                    dtype=torch.bfloat16,
                    device=dev,
                ),
                "k": torch.zeros(
                    batch_size,
                    bucket,
                    num_heads,
                    head_k_dim,
                    dtype=torch.bfloat16,
                    device=dev,
                ),
                "v": torch.zeros(
                    batch_size,
                    bucket,
                    num_heads,
                    head_v_dim,
                    dtype=torch.bfloat16,
                    device=dev,
                ),
                "g": torch.full(
                    (batch_size, bucket, num_heads, head_k_dim),
                    -1000.0,
                    dtype=torch.bfloat16,
                    device=dev,
                ),
                "beta": torch.zeros(
                    batch_size,
                    bucket,
                    num_heads,
                    dtype=torch.bfloat16,
                    device=dev,
                ),
                "s0": torch.zeros(
                    batch_size,
                    num_heads,
                    head_k_dim,
                    head_v_dim,
                    dtype=torch.float32,
                    device=dev,
                ),
            }
            self._staging[key] = st
        return st

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        num_tokens = q.shape[1]
        seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
        track_h_src = kwargs.get("track_ssm_h_src")
        supported_shape = (
            q.ndim == 4
            and k.ndim == 4
            and v.ndim == 4
            and q.shape[-1] == 128
            and k.shape[-1] == 128
            and v.shape[-1] == 128
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and g.dtype == torch.bfloat16
            and beta.dtype == torch.float32
            and A_log is not None
        )
        needs_interior_snapshot = (
            return_intermediate_states
            and track_h_src is not None
            and track_h_src.numel() > 0
        )
        eligible = (
            not needs_interior_snapshot
            and not kwargs.get("is_spec_decode")
            and seq_lens_cpu is not None
            and len(seq_lens_cpu) > 1
            and supported_shape
        )
        if not eligible:
            if (
                seq_lens_cpu is not None
                and len(seq_lens_cpu) > 1
                and not kwargs.get("is_spec_decode")
                and not needs_interior_snapshot
                and not supported_shape
            ):
                unsupported_key = (
                    q.dtype,
                    k.dtype,
                    v.dtype,
                    g.dtype,
                    beta.dtype,
                    q.shape[-1],
                    k.shape[-1],
                    v.shape[-1],
                    A_log is not None,
                )
                if unsupported_key not in self._unsupported_logged:
                    self._unsupported_logged.add(unsupported_key)
                    logger.warning(
                        "NVIDIA KDA prefill only supports BF16 q/k/v/g, FP32 beta, "
                        "K=V=128 with A_log; "
                        "got q/k/v/g/beta=%s/%s/%s/%s/%s, Kq/Kk/V=%d/%d/%d, "
                        "A_log=%s. Falling back to Triton.",
                        q.dtype,
                        k.dtype,
                        v.dtype,
                        g.dtype,
                        beta.dtype,
                        q.shape[-1],
                        k.shape[-1],
                        v.shape[-1],
                        A_log is not None,
                    )
            return self._triton_extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log,
                dt_bias,
                lower_bound,
                return_intermediate_states,
                kwargs,
            )

        self._ensure_loaded()

        seq_lens = [int(length) for length in seq_lens_cpu]
        if (
            any(length <= 0 or length > _BUCKETS[-1] for length in seq_lens)
            or sum(seq_lens) != num_tokens
            or len(seq_lens) != cache_indices.numel()
        ):
            logger.warning(
                "NVIDIA KDA cannot repack prefill shape T=%d seq_lens=%s; falling "
                "back to Triton",
                num_tokens,
                seq_lens,
            )
            return self._triton_extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log,
                dt_bias,
                lower_bound,
                return_intermediate_states,
                kwargs,
            )

        num_heads = q.shape[2]
        head_k_dim = q.shape[-1]
        head_v_dim = v.shape[-1]

        alog_flat = self._flat_param(A_log)
        dtb_flat = self._flat_param(dt_bias)

        all_slot_indices = torch.where(
            cache_indices >= 0, cache_indices, ssm_states.shape[0] - 1
        ).to(torch.int64)
        state_backup = ssm_states.index_select(0, all_slot_indices).clone()
        packed_output = torch.empty_like(v)

        try:
            seq_start = 0
            token_start = 0
            while seq_start < len(seq_lens):
                group_lens = seq_lens[seq_start : seq_start + _MAX_NVIDIA_KDA_BATCH]
                group_size = len(group_lens)
                group_tokens = sum(group_lens)
                bucket = next(b for b in _BUCKETS if b >= max(group_lens))
                engage_key = (group_size, bucket)
                if engage_key not in self._engaged_logged:
                    self._engaged_logged.add(engage_key)
                    logger.info(
                        "NVIDIA KDA prefill engaged: sequences=%d max_T=%d -> B=%d "
                        "bucket=%d",
                        len(seq_lens),
                        max(group_lens),
                        group_size,
                        bucket,
                    )

                st = self._get_staging(
                    group_size,
                    bucket,
                    num_heads,
                    head_k_dim,
                    head_v_dim,
                    q.device,
                )
                st["q"].zero_()
                st["k"].zero_()
                st["v"].zero_()
                st["g"].fill_(-1000.0)
                st["beta"].zero_()

                row_token_start = token_start
                for row, length in enumerate(group_lens):
                    row_token_end = row_token_start + length
                    st["q"][row, :length].copy_(
                        self._l2norm(q[0, row_token_start:row_token_end].contiguous())
                    )
                    st["k"][row, :length].copy_(
                        self._l2norm(k[0, row_token_start:row_token_end].contiguous())
                    )
                    st["v"][row, :length].copy_(v[0, row_token_start:row_token_end])
                    g_in = g[0, row_token_start:row_token_end]
                    if g_in.dim() == 2:
                        g_in = g_in.view(length, num_heads, head_k_dim)
                    st["g"][row, :length].copy_(g_in)
                    st["beta"][row, :length].copy_(
                        beta[0, row_token_start:row_token_end]
                    )
                    row_token_start = row_token_end

                group_slots = all_slot_indices[seq_start : seq_start + group_size]
                st["s0"].copy_(
                    _to_nvidia_kda_state_layout(
                        ssm_states.index_select(0, group_slots),
                        head_k_dim=head_k_dim,
                        head_v_dim=head_v_dim,
                    )
                )
                res = self._fwd(
                    st["q"],
                    st["k"],
                    st["v"],
                    st["g"],
                    st["beta"],
                    scale=head_k_dim**-0.5,
                    initial_state=st["s0"],
                    output_final_state=True,
                    cu_seqlens=None,
                    safe_gate=lower_bound is not None,
                    lower_bound=lower_bound,
                    use_gate_in_kernel=True,
                    A_log=alog_flat,
                    dt_bias=dtb_flat,
                )
                group_output, final_state = res[0], res[1]

                row_token_start = token_start
                for row, length in enumerate(group_lens):
                    row_token_end = row_token_start + length
                    packed_output[0, row_token_start:row_token_end].copy_(
                        group_output[row, :length]
                    )
                    row_token_start = row_token_end

                ssm_states.index_copy_(
                    0,
                    group_slots,
                    _from_nvidia_kda_state_layout(
                        final_state,
                        head_k_dim=head_k_dim,
                        head_v_dim=head_v_dim,
                        dtype=ssm_states.dtype,
                    ),
                )
                seq_start += group_size
                token_start += group_tokens
        except Exception:
            ssm_states.index_copy_(0, all_slot_indices, state_backup)
            logger.warning(
                "NVIDIA KDA prefill failed (T=%d sequences=%d); restored states "
                "and fell back to Triton for this batch",
                num_tokens,
                len(seq_lens),
                exc_info=True,
            )
            return self._triton_extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log,
                dt_bias,
                lower_bound,
                return_intermediate_states,
                kwargs,
            )

        if return_intermediate_states:
            # Boundary-aligned track: the snapshot comes from the final state
            # already scattered into the pool (track_ssm_final_src); h is
            # never indexed (track_ssm_h_src is empty), a zero-row stand-in
            # keeps the (out, h) contract.
            h_empty = q.new_empty(
                (1, 0) + tuple(ssm_states.shape[1:]), dtype=torch.float32
            )
            return packed_output, h_empty
        return packed_output
