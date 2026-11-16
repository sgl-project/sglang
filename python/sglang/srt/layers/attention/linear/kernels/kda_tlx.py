import logging
from itertools import accumulate
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

logger = logging.getLogger(__name__)


def _dense_heads(*tensors: torch.Tensor) -> bool:
    return all(t.stride(-1) == 1 and t.stride(-2) == t.shape[-1] for t in tensors)


def _dense_slots(pool: torch.Tensor, heads: int, value_dim: int, key_dim: int) -> bool:
    return (
        pool.ndim == 4
        and tuple(pool.shape[1:]) == (heads, value_dim, key_dim)
        and pool.stride()[1:] == (value_dim * key_dim, key_dim, 1)
        and pool.stride(0) >= heads * value_dim * key_dim
    )


class TlxKDAKernel(TritonKDAKernel):
    """gfx950 KDA decode and chunked prefill from ``sglang.kernels.ops.attention.tlx``.

    Both kernels take the raw conv output, gate and beta with ``A_log`` /
    ``dt_bias`` and update the fp32 state pool in place. As in the other KDA
    kernel backends, batches that need what these kernels cannot provide
    (per-chunk or tracked states, a rollback-able draft extend, prefill CUDA
    graph tables) run on the inherited Triton kernel, and inputs they do not
    support fall back to Triton with a one-time warning.
    """

    # decode() takes split q/k/v; the packed path would bypass the TLX decode.
    supports_packed_decode: bool = False

    def __init__(self):
        super().__init__()
        self._unsupported_logged: set = set()

    def _warn_unsupported(self, phase, q, k, v, ssm_states, A_log, dt_bias) -> None:
        key = (
            phase,
            q.dtype,
            k.dtype,
            v.dtype,
            ssm_states.dtype,
            tuple(ssm_states.shape[1:]),
            ssm_states.stride(),
            tuple(q.shape[-2:]),
            tuple(v.shape[-2:]),
            None if A_log is None else A_log.numel(),
            None if dt_bias is None else dt_bias.numel(),
        )
        if key in self._unsupported_logged:
            return
        self._unsupported_logged.add(key)
        logger.warning(
            "TLX KDA %s supports BF16 q/k/v with dense heads, an FP32 [slots, H, V, "
            "K] state dense within each slot, as many v heads as q heads, K and V "
            "up to 128 (prefill: K=V=128 and A_log); got q/k/v=%s/%s/%s, "
            "state=%s %s with strides %s, q heads x K=%s, v heads x V=%s, "
            "A_log/dt_bias elements=%s/%s. Falling back to Triton.",
            *key,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        heads, key_dim = q.shape[-2:]
        batch = cache_indices.shape[0]
        needs_triton = kwargs.get("replayssm_d") is not None
        if not needs_triton and not (
            q.ndim == 4
            and q.shape[:2] == (1, batch)
            and k.shape == q.shape
            and v.shape[:3] == q.shape[:3]
            and key_dim <= 128
            and v.shape[-1] <= 128
            and all(x.dtype == torch.bfloat16 for x in (q, k, v))
            and ssm_states.dtype == torch.float32
            and _dense_slots(ssm_states, heads, v.shape[-1], key_dim)
            and _dense_heads(q, k, v)
            and a.numel() == batch * heads * key_dim
            and b.numel() == batch * heads
            and A_log.numel() == heads
            and dt_bias.numel() == heads * key_dim
        ):
            self._warn_unsupported("decode", q, k, v, ssm_states, A_log, dt_bias)
            needs_triton = True
        if needs_triton:
            return super().decode(
                q,
                k,
                v,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                lower_bound=lower_bound,
                **kwargs,
            )
        from sglang.kernels.ops.attention.tlx.kda.gfx950_decode import (
            kda_recurrent_decode,
        )

        return kda_recurrent_decode(
            q,
            k,
            v,
            a.reshape(1, batch, heads, key_dim),
            b.reshape(1, batch, heads),
            scale=key_dim**-0.5,
            state_pool=ssm_states,
            read_indices=cache_indices,
            write_indices=cache_indices,
            cu_seqlens=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=True,
        )

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
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        needs_triton = (
            return_intermediate_states
            or kwargs.get("track_state") is not None
            or kwargs.get("is_spec_decode", False)
            or kwargs.get("chunk_indices") is not None
        )
        heads = q.shape[-2]
        if not needs_triton and not (
            A_log is not None
            and dt_bias is not None
            and q.ndim == 4
            and q.shape[0] == 1
            and q.shape == k.shape == g.shape
            and v.shape[:3] == q.shape[:3]
            and q.shape[-1] == v.shape[-1] == 128
            and beta.shape == q.shape[:3]
            and A_log.numel() == heads
            and dt_bias.numel() == heads * q.shape[-1]
            and all(x.dtype == torch.bfloat16 for x in (q, k, v))
            and ssm_states.dtype == torch.float32
            and _dense_slots(ssm_states, heads, v.shape[-1], q.shape[-1])
            and _dense_heads(q, k, v, g)
        ):
            self._warn_unsupported("prefill", q, k, v, ssm_states, A_log, dt_bias)
            needs_triton = True
        if needs_triton:
            return super().extend(
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
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
                **kwargs,
            )
        from sglang.kernels.ops.attention.tlx.kda.gfx950_prefill import (
            kda_paged_prefill,
        )

        seq_lens = kwargs.get("extend_seq_lens_cpu")
        cu_seqlens_cpu = None
        if (
            seq_lens is not None
            and len(seq_lens) == query_start_loc.numel() - 1
            and sum(seq_lens) == q.shape[1]
        ):
            cu_seqlens_cpu = [0, *accumulate(seq_lens)]
        output, _ = kda_paged_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=q.shape[-1] ** -0.5,
            cu_seqlens=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
            state_pool=ssm_states,
            state_indices=cache_indices,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        return output
