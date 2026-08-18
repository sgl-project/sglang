"""Moonmath MLA attention backend for CDNA3 (gfx942).

Subclasses AiterAttnBackend and takes over absorbed MLA decode with the
moonmath_attention A16W8 kernel (bf16 Q / fp8 KV), which reads sglang's existing
fused-576 MLATokenToKVPool key buffer directly (page_size=1, device-driven,
cuda-graph safe). H <= 16:

    decode        q_len 1     -> mla_decode_a16w8_paged_dev

Everything else -- prefill, spec-verify, bf16 KV, unsupported geometry -- falls
back to AiterAttnBackend.

aiter's MLA kernels also require num_head in {4, 8} or a multiple of 16 in
[16, 128], which excludes Kimi-K3's 12 heads at TP8. The moonmath kernels take H
as a runtime parameter and mask the trailing rows of the 16-row MFMA tile, so
H=12 runs natively; the inherited fallback paths keep aiter's limit, so Q is
zero-padded to 16 heads for them (`_mla_decode_fwd_with_head_pad`).
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

try:  # pragma: no cover - AMD-only dependency, mirrors aiter_backend's guard
    from aiter.mla import mla_decode_fwd
except ImportError:  # pragma: no cover
    mla_decode_fwd = None

logger = logging.getLogger(__name__)

KV_LORA_RANK = 512
KV_CACHE_DIM = 576  # 512 latent + 64 rope

# Both A16W8 kernels serve one 16-head TP shard (H is a runtime parameter that
# masks the trailing rows of the single 16-row MFMA query tile).
_MAX_HEADS = 16
# Narrowest head count aiter's asm MLA has a kernel for: its qh16 kernels bake
# gqa=16 into the ISA, so a 12-head call has nothing to dispatch to.
_AITER_MLA_MIN_HEADS = 16

_MAX_BATCH = 8192  # size of the staged int32 seq_lens buffer


class MoonmathMLABackend(AiterAttnBackend):
    """MLA decode via moonmath_attention; aiter for everything else."""

    # This backend has its own MLA kernels, so aiter's head-count assert must not
    # reject it at construction.
    skip_mla_head_count_assert = True

    def __init__(self, model_runner):
        super().__init__(model_runner)
        import moonmath_attention.mla as mla  # fail fast if not installed

        self._mla = mla
        # The kernels take the fused-576 pool as fp8 e4m3fnuz at a per-tensor
        # descale; a bf16 KV cache has no A16W8 arm and falls back to aiter.
        self._enabled = (
            bool(self.use_mla) and self.kv_cache_dtype == torch.float8_e4m3fnuz
        )

        # `parts` must not vary inside a captured graph, so the kv-split is
        # frozen per (arm, bs, H, q_len) at first call / capture and reused.
        self._parts: dict[tuple, int] = {}
        self._max_ctx = model_runner.model_config.context_len
        # sum(seq_lens) can never exceed the pool, so slots // bs bounds the mean
        # sequence length at that batch. See _plan_seq_len.
        self._kv_pool_slots = int(self.token_to_kv_pool.size)
        # int32 staging for device seq_lens (sglang carries int64 in eager mode).
        self._seq_lens_i32 = torch.zeros(
            _MAX_BATCH, dtype=torch.int32, device=model_runner.device
        )
        self._logged_decode = False
        logger.info(
            "moonmath_mla: enabled=%s num_head=%s kv_cache_dtype=%s",
            self._enabled,
            self.num_head,
            self.kv_cache_dtype,
        )

    # ── metadata ─────────────────────────────────────────────────────────────
    # The kernels want int32 device seq_lens; sglang carries int64. Stage it in
    # the metadata hooks -- once per forward rather than once per MLA layer, and
    # out-of-graph before every replay, which is where a device buffer that a
    # captured kernel reads must be refreshed.
    def _stage_seq_lens_i32(self, forward_batch: ForwardBatch) -> None:
        if not self._enabled or not forward_batch.forward_mode.is_decode():
            return
        bs = forward_batch.batch_size
        if bs <= _MAX_BATCH:
            self._seq_lens_i32[:bs].copy_(forward_batch.seq_lens)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self._stage_seq_lens_i32(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        super().init_forward_metadata_out_graph(forward_batch, in_capture)
        self._stage_seq_lens_i32(forward_batch)

    # ── kv-split planning ────────────────────────────────────────────────────
    def _plan_seq_len(self, bs: int) -> int:
        """Sequence length the frozen kv-split is planned from.

        `parts` must be constant inside a captured graph, but the optimal split
        tracks the live sequence length, and planning from the context window
        over-splits badly at batch. All requests' KV shares one pool, so
        `slots // bs` is the mean length at that batch -- a better input, and a
        safe one: every `parts >= 1` is numerically correct.
        """
        return max(1, min(self._max_ctx, self._kv_pool_slots // max(bs, 1)))

    def _cached_parts(self, key: tuple, plan) -> int:
        parts = self._parts.get(key)
        if parts is None:
            parts = plan()
            self._parts[key] = parts
        return parts

    # ── shared eligibility ───────────────────────────────────────────────────
    def _shape_eligible(self, q, layer: RadixAttention, fb: ForwardBatch) -> bool:
        """Absorbed-MLA geometry the A16W8 kernels are compiled for."""
        return (
            self._enabled
            and q.dtype == torch.bfloat16
            and 1 <= layer.tp_q_head_num <= _MAX_HEADS
            and layer.qk_head_dim == KV_CACHE_DIM
            and layer.v_head_dim == KV_LORA_RANK
            and layer.tp_k_head_num == 1
            and layer.logit_cap == 0
            and 0 < fb.batch_size <= _MAX_BATCH
            and self.forward_metadata is not None
            and self.forward_metadata.kv_indices is not None
            and self.forward_metadata.kv_indptr is not None
        )

    def _kv_indices_int32(self):
        # Both are int32 already, so `.to` returns the argument. It must STAY
        # free: a real cast allocates a fresh tensor per layer, and a captured
        # graph holds the address it saw at capture time.
        meta = self.forward_metadata
        return meta.kv_indices.to(torch.int32), meta.kv_indptr.to(torch.int32)

    def _split_q(self, q, *shape):
        """`q` as the contiguous (latent, rope) pair the kernel ABI takes."""
        q = q.reshape(*shape, KV_CACHE_DIM)
        return q[..., :KV_LORA_RANK].contiguous(), q[..., KV_LORA_RANK:].contiguous()

    # ── decode ───────────────────────────────────────────────────────────────
    def _decode_eligible(self, q, layer: RadixAttention, fb: ForwardBatch) -> bool:
        return (
            fb.forward_mode.is_decode()
            and fb.spec_info is None
            and self._shape_eligible(q, layer, fb)
        )

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None
    ):
        """Absorbed MLA decode.

        `q` is the 576-wide `cat([q_nope_out, q_pe])` and `k` the 576-wide
        `cat([k_nope, k_pe])`; both are split back into the 512-wide latent and
        the 64-wide rope halves the kernel's ABI takes as separate pointers.
        """
        if sinks is not None or not self._decode_eligible(q, layer, forward_batch):
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, sinks
            )

        fb = forward_batch
        B, H = fb.batch_size, layer.tp_q_head_num
        if save_kv_cache and k is not None:
            self.token_to_kv_pool.set_kv_buffer(layer, fb.out_cache_loc, k, v)

        parts = self._cached_parts(
            ("decode", B, H),
            lambda: self._mla.mla_decode_a16w8_plan_parts_capped(
                B, H, self._plan_seq_len(B), KV_LORA_RANK
            ),
        )
        if not self._logged_decode:
            self._logged_decode = True
            logger.info("moonmath_mla: decode bs=%d H=%d parts=%d", B, H, parts)

        q_lat, q_pe = self._split_q(q, B, H)
        out = torch.empty(B, H, KV_LORA_RANK, dtype=torch.bfloat16, device=q.device)
        kv_indices, kv_indptr = self._kv_indices_int32()
        self._mla.mla_decode_a16w8_paged_dev(
            q_lat,
            q_pe,
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            out,
            self._seq_lens_i32[:B],
            None,
            kv_indices,
            kv_indptr,
            parts,
            layer.scaling,
            1.0 if layer.k_scale is None else float(layer.k_scale),
        )
        return out.reshape(B, H * KV_LORA_RANK)

    # ── prefill fallback: aiter asm-MLA head padding ─────────────────────────
    @staticmethod
    def _aiter_mla_needs_head_pad(num_head: int) -> bool:
        """Whether the base class's repeat-interleave cannot reach gqa 16.

        `head_repeat_factor = 16 // num_head` only lands on 16 when
        `16 % num_head == 0`. At 12 the factor is 1, so nothing is repeated and
        the kernel sees gqa=12; same for 3, 5, 6, 7 and 9..15.
        """
        return 0 < num_head < _AITER_MLA_MIN_HEADS and (
            _AITER_MLA_MIN_HEADS % num_head != 0
        )

    def _mla_decode_fwd_with_head_pad(self, q, k_buffer_flat, layer, **kwargs):
        """`mla_decode_fwd` with Q zero-padded up to aiter's 16-head minimum.

        Every aiter asm-MLA call this backend makes goes through here, so one
        override covers prefill, TARGET_VERIFY, DRAFT_EXTEND_V2 and the shapes
        the decode arm declines. Unconditional at the head counts
        `_aiter_mla_needs_head_pad` selects: there aiter aborts the process, so
        there is no prior behaviour to preserve.

        The padded rows do compute something meaningless -- a zero query still
        attends over the real K/V -- but it never reaches head h < H, because
        MLA attention has no reduction across the head axis: QK, softmax and PV
        are per (head, query position); the split-KV combine is launched on
        `grid = (bs, nhead)` and indexes its partials by `cur_head`; and the
        output write is one disjoint row per head. So `o[:, :H]` is
        bit-identical to an unpadded call.

        The pad is on the query side only. K/V are the shared 576-wide latent
        rows (`tp_k_head_num == 1`), so there is nothing per-query-head to
        widen -- and padding them would be wrong, injecting zero-score keys into
        every real head's softmax denominator.
        """
        num_head = q.shape[1]
        if not self._aiter_mla_needs_head_pad(num_head):
            return super()._mla_decode_fwd_with_head_pad(
                q, k_buffer_flat, layer, **kwargs
            )
        assert mla_decode_fwd is not None, "aiter.mla.mla_decode_fwd did not import"
        assert num_head == layer.tp_q_head_num, (
            f"head-pad: q has {num_head} heads but layer declares "
            f"{layer.tp_q_head_num}; the output slice would be wrong."
        )

        # (0, 0) leaves qk_head_dim alone; (0, n) appends n zero heads. The pad
        # is captured as kernels, so a replay re-zeroes it.
        q_padded = torch.nn.functional.pad(
            q, (0, 0, 0, _AITER_MLA_MIN_HEADS - num_head)
        )
        o = q.new_empty(
            (q.shape[0], _AITER_MLA_MIN_HEADS, layer.v_head_dim),
            dtype=self.input_dtype,
        )
        mla_decode_fwd(q_padded, k_buffer_flat, o, **kwargs)
        # Not a strided view of the 16-head buffer: downstream is a transpose +
        # BMM, and a kernel assuming contiguity would read the padded heads.
        return o[:, :num_head, :].contiguous()
