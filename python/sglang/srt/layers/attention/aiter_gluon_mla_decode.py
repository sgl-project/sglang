"""Paged Gluon MLA decode for the aiter backend on gfx1250.

The kernel is ``aiter.ops.triton.attention.mla.mla_decode_fwd``.

Needs an aiter with ROCm/aiter#5935 (in the Docker pin since 2bb549102). Before
it the kernel writes its segment softmax state over the output whenever a batch
runs one segment per sequence (batch 97 and above on 96 CUs), without an error,
so an aiter without it keeps ``decode_attention_fwd``.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx1250_supported, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def _has_single_segment_fix(mla_module) -> bool:
    """Whether the Gluon decode kernel has ROCm/aiter#5935.

    aiter has no version to check, and the fix only adds a constexpr guard
    around the segment softmax stores, so look for that guard in the source.
    """
    kernel = getattr(mla_module, "gluon_mla_decode_fwd_kernel_non_pipelined", None)
    src = getattr(kernel, "src", None)
    return isinstance(src, str) and "NUM_SEGMENTS_PER_SEQ > 1" in src


@functools.lru_cache(maxsize=1)
def _mla_decode_fwd() -> Optional[Callable]:
    if not (is_hip() and envs.SGLANG_USE_AITER.get() and is_gfx1250_supported()):
        return None
    try:
        from aiter.ops.triton.attention import mla
    except ImportError:
        return None
    if not _has_single_segment_fix(mla):
        logger.warning(
            "aiter's gfx1250 Gluon MLA decode predates ROCm/aiter#5935 and "
            "corrupts single-segment batches; keeping decode_attention_fwd"
        )
        return None
    return mla.mla_decode_fwd


class GluonMlaDecode:
    """Paged Gluon MLA decode state of one attention backend."""

    @classmethod
    def create(
        cls, backend: AttentionBackend, kv_lora_rank: int, qk_rope_head_dim: int
    ) -> Optional[GluonMlaDecode]:
        """None unless the KV pool and backend settings fit the kernel.

        The kernel reads the KV pool as pages, so it needs a paged pool
        (power-of-two page size >= 16) and the plain causal MLA case.
        ``dcp_size`` and ``dcp_world_size`` are the same split; aiter names it
        the latter.
        """
        page_size = backend.page_size
        dcp_size = getattr(backend, "dcp_size", None)
        if dcp_size is None:
            dcp_size = getattr(backend, "dcp_world_size", 1)
        if not (
            backend.use_mla
            and page_size >= 16
            and page_size & (page_size - 1) == 0
            and dcp_size == 1
            and not getattr(backend, "enable_deterministic", False)
            and getattr(backend, "sliding_window_size", None) is None
            and not backend.use_sliding_window_kv_pool
        ):
            return None
        pool = backend.token_to_kv_pool
        kv_buffer = pool.get_key_buffer(pool.start_layer)
        if not (
            kv_buffer.dim() == 3
            and kv_buffer.shape[1] == 1
            and kv_buffer.shape[0] % page_size == 0
            and kv_buffer.is_contiguous()
        ):
            return None
        fwd = _mla_decode_fwd()
        if fwd is None:
            return None
        return cls(backend, fwd, kv_lora_rank, qk_rope_head_dim)

    def __init__(
        self,
        backend: AttentionBackend,
        mla_decode_fwd: Callable,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
    ):
        self.mla_decode_fwd = mla_decode_fwd
        self.backend = backend
        self.req_to_token = backend.req_to_token
        self.page_size = backend.page_size
        # The first layer that runs this kernel; set on the first forward.
        self.first_layer_id: Optional[int] = None
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.page_starts = torch.arange(
            0, self.req_to_token.shape[1], self.page_size, device=backend.device
        )
        self.kv_descale_one = torch.ones(1, dtype=torch.float32, device=backend.device)
        self.block_tables: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.cu_seqlens_q: Optional[torch.Tensor] = None
        self._layer_ok: Dict[int, bool] = {}

    def supports(self, layer: RadixAttention, logits_soft_cap: float) -> bool:
        """Whether this layer's static attention settings fit the kernel."""
        ok = self._layer_ok.get(layer.layer_id)
        if ok is None:
            ok = (
                logits_soft_cap == 0
                and (
                    layer.xai_temperature_len is None or layer.xai_temperature_len <= 0
                )
                and (layer.sliding_window_size is None or layer.sliding_window_size < 0)
                and layer.qk_head_dim == self.kv_lora_rank + self.qk_rope_head_dim
                and layer.v_head_dim == self.kv_lora_rank
            )
            self._layer_ok[layer.layer_id] = ok
        return ok

    def accepts(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        logits_soft_cap: float,
        sinks,
    ) -> bool:
        """One query token, plain causal MLA. Speculative decode, sinks and a
        soft cap stay on decode_attention_fwd."""
        if sinks is not None or forward_batch.spec_info is not None:
            return False
        if not forward_batch.forward_mode.is_decode_or_idle():
            return False
        return self.supports(layer, logits_soft_cap)

    def forward(
        self,
        *,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        # Page table and lengths are per forward, not per layer: build them at
        # the first layer that runs this kernel from the same (graph-static)
        # batch tensors the Triton path reads, so CUDA graph replay refreshes
        # them too. supports() is static per layer, so that layer is the same
        # in every forward.
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
        # K is divided by k_scale before it is written to the cache, whatever
        # the cache dtype, so the kernel has to multiply it back.
        if layer.k_scale is not None and layer.k_scale_float != 1.0:
            kv_descale = layer.k_scale
        elif kv_buffer.dtype == torch.bfloat16:
            kv_descale = None
        else:
            kv_descale = self.kv_descale_one
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
