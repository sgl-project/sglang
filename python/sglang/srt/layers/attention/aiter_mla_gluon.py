"""aiter Gluon MLA decode wrappers for the aiter attention backend.

Two independent kernels, both turned off by ``SGLANG_AITER_MLA_GLUON=0``:

- gfx950: ``aiter.ops.triton.gluon.mla_gluon`` for low head-count fp8 MLA
  (e.g. Kimi K3 TP8, 12 heads/GPU) over token-level kv_indices. Needs aiter with
  ROCm/aiter #4480 (batch>1 ``bh16bn128``) and #4555 (decode CUDA graph KV splits)
  and Triton >= 3.7 (Gluon ``cga_layout``); otherwise the caller zero-pads heads
  for ``mla_decode_fwd``.
- gfx1250: ``aiter.ops.triton.attention.mla.mla_decode_fwd`` over paged block
  tables with a bf16 query. Needs aiter with ROCm/aiter#5935.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx1250_supported

if TYPE_CHECKING:
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


# ===== gfx950: token-indexed mla_gluon =====


@functools.lru_cache(maxsize=1)
def _gluon_fn():
    """aiter mla gluon entry point return None if disabled"""
    if not envs.SGLANG_AITER_MLA_GLUON.get():
        logger.info("aiter mla gluon is disabled manually.")
        return None
    if is_gfx1250_supported():
        logger.info("aiter mla gluon is disabled on gfx1250, which uses paged Gluon.")
        return None
    try:
        import triton
        import triton.experimental.gluon.language as gl
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon
    except ImportError as exc:
        logger.info("aiter mla gluon import error message: %s", exc)
        return None
    # mla_gluon builds its shared layouts with cga_layout, added in Triton 3.7;
    # older Triton fails at compile time with an opaque error instead.
    if "cga_layout" not in inspect.signature(gl.PaddedSharedLayout).parameters:
        logger.info(
            "aiter mla gluon is disabled due to triton %s has no Gluon cga_layout (need >= 3.7)",
            getattr(triton, "__version__", "unknown"),
        )
        return None
    return mla_gluon


def log_mla_gluon_capability(log: logging.Logger | None = None) -> None:
    """Report whether Gluon MLA decode is valid; the reason is logged by _gluon_fn."""
    ready = _gluon_fn() is not None
    (log or logger).info(
        "aiter mla gluon is %s",
        "enabled" if ready else "disabled; falling back to zero-pad mla_decode_fwd",
    )


def prefer_mla_gluon_decode(
    *, head_pad_mode: str, num_head: int, kv_cache_dtype: torch.dtype
) -> bool:
    return (
        head_pad_mode == "zero"
        and num_head == 12
        and kv_cache_dtype == fp8_dtype
        and _gluon_fn() is not None
    )


def mla_gluon_decode(
    *,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    layer: RadixAttention,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: float,
    min_kv_seq_len: int,
    kv_scale: float = 1.0,
    qlen: int = 1,
    use_2d_view: bool = False,
    return_lse: bool = False,
):
    """Run Gluon MLA decode for fused Q [num_tokens, H, 576] and MLA KV pool.
    Returns [num_tokens, H, v_head_dim] (or ``(out, lse)`` when ``return_lse``),
    or None when Gluon is unavailable.
    """
    mla_gluon = _gluon_fn()
    if mla_gluon is None:
        return None

    num_head = layer.tp_q_head_num
    kv_lora_rank = layer.v_head_dim
    qk_rope_head_dim = layer.qk_head_dim - kv_lora_rank
    batch_size = q.shape[0] // qlen

    q_nope, q_pe = torch.split(q, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    if qlen > 1:
        # Splitting the leading dim is a stride change, so these stay views of
        # the non-contiguous torch.split outputs.
        q_nope = q_nope.view(batch_size, qlen, num_head, kv_lora_rank)
        q_pe = q_pe.view(batch_size, qlen, num_head, qk_rope_head_dim)
        o = q.new_empty((batch_size, qlen, num_head, kv_lora_rank))
    else:
        o = q.new_empty((batch_size, num_head, kv_lora_rank))

    extra_kwargs = {}
    if return_lse:
        extra_kwargs["return_lse"] = True

    result = mla_gluon(
        q_nope,
        q_pe,
        k_buffer.view(-1, layer.qk_head_dim),
        o,
        kv_indices,
        kv_indptr,
        sm_scale,
        k_pe=None,
        kv_pe_offset=kv_lora_rank,
        use_2d_view=use_2d_view,
        kv_scale=kv_scale,
        min_kv_seq_len=min_kv_seq_len,
        **extra_kwargs,
    )
    # Hand back the caller's flat [num_tokens, H, v] layout either way.
    out = o.flatten(0, 1) if qlen > 1 else o
    if not return_lse:
        return out
    # mla_gluon writes the output into `o` and returns it alongside the lse.
    _, lse = result
    return out, lse


# ===== gfx1250: paged mla_decode_fwd =====


@functools.lru_cache(maxsize=1)
def _paged_mla_decode_fwd() -> Optional[Callable]:
    if not envs.SGLANG_AITER_MLA_GLUON.get():
        return None
    from aiter.ops.triton.attention.mla import mla_decode_fwd

    return mla_decode_fwd


class PagedGluonMlaDecode:
    """gfx1250 paged Gluon MLA decode state of one attention backend."""

    @classmethod
    def create(
        cls, backend: AiterAttnBackend, *, kv_lora_rank: int, qk_rope_head_dim: int
    ) -> Optional[PagedGluonMlaDecode]:
        """None unless the backend settings fit the kernel."""
        if not (
            backend.input_dtype == torch.bfloat16
            and qk_rope_head_dim > 0
            # The only page size aiter's test_mla_decode_fwd covers for bf16/fp8 KV.
            and backend.page_size == 64
            and backend.dcp_world_size == 1
        ):
            return None
        fwd = _paged_mla_decode_fwd()
        if fwd is None:
            return None
        return cls(
            backend,
            mla_decode_fwd=fwd,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )

    def __init__(
        self,
        backend: AiterAttnBackend,
        *,
        mla_decode_fwd: Callable,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
    ):
        self.mla_decode_fwd = mla_decode_fwd
        # Held for kv_layers: a frozen-KV MTP draft swaps the backend's pool.
        self.backend = backend
        self.req_to_token = backend.req_to_token
        self.page_size = backend.page_size
        # First layer that runs this kernel; accepts() depends only on the batch,
        # so it never moves.
        self.first_layer_id: Optional[int] = None
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.page_starts = torch.arange(
            0, self.req_to_token.shape[1], self.page_size, device=backend.device
        )
        self.block_tables: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.cu_seqlens_q: Optional[torch.Tensor] = None

    def accepts(self, forward_batch: ForwardBatch) -> bool:
        """One query token per sequence; the caller runs the rest on ASM."""
        return (
            forward_batch.spec_info is None and forward_batch.forward_mode.is_decode()
        )

    def forward(
        self,
        *,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        kv_descale: Optional[torch.Tensor],
    ):
        # Page table and lengths are per forward: rebuilt at the first kernel layer
        # from graph-static batch tensors, so CUDA graph replay refreshes them.
        if self.first_layer_id is None:
            self.first_layer_id = layer.layer_id
        if layer.layer_id == self.first_layer_id:
            bs = forward_batch.batch_size
            self.block_tables = (
                self.req_to_token[
                    forward_batch.req_pool_indices[:bs, None],
                    self.page_starts[None, :],
                ]
                // self.page_size
            )
            self.seq_lens = forward_batch.seq_lens[:bs].to(torch.int32)
            self.cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device=q.device)
        kv_buffer = self.backend.kv_layers.read(layer.layer_id).key()
        kv_buffer = kv_buffer.view(
            -1, self.page_size, kv_buffer.shape[-2], kv_buffer.shape[-1]
        )
        self.mla_decode_fwd(
            q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            kv_buffer=kv_buffer,
            out=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            cu_seqlens_q=self.cu_seqlens_q,
            seqused_k=self.seq_lens,
            # 0: the kernel sizes its KV split from a fixed guess, which keeps
            # CUDA graph replay valid for any sequence length.
            max_seqlen_kv=0,
            block_tables=self.block_tables,
            softmax_scale=layer.scaling,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            causal=True,
            q_descale=None,
            kv_descale=kv_descale,
        )
