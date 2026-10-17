"""Moonmath MLA attention backend for CDNA3 (gfx942).

Subclasses AiterAttnBackend and takes over absorbed MLA with the
moonmath_amd A16W8 kernel (bf16 Q / fp8 KV), which reads sglang's existing
fused-576 MLATokenToKVPool key buffer directly (page_size=1, device-driven,
cuda-graph safe). One op, `mla_decode_a16w8`, serves both shapes for H <= 128:

    decode        q_len 1   -> mla_decode_a16w8
    TARGET_VERIFY q_len > 1 -> mla_decode_a16w8

Everything else -- prefill, bf16 KV, unsupported geometry -- falls back to
AiterAttnBackend.

Under decode context parallelism each rank attends its own KV shard with the
causal limit taken from the global lengths, and returns its partial with the LSE
for the cross-rank merge, in aiter's natural-log base.

The verify arm is what makes speculative decoding work here at all: aiter's
asm MLA has no kernel past qseqlen 4, so a larger draft window aborts the
process during cuda-graph capture. Q stays bf16 in both arms, so there is no
query scale to calibrate and the verify logits are not perturbed.

aiter's MLA kernels also require num_head in {4, 8} or a multiple of 16 in
[16, 128], which excludes Kimi-K3's 12 heads at TP8. The moonmath kernel takes H
as a runtime parameter, so H=12 runs natively; the inherited fallback paths keep
aiter's limit, so Q is zero-padded to 16 heads for them
(`_mla_decode_fwd_with_head_pad`).
"""

from __future__ import annotations

import logging
import math

import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.environ import envs
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel

try:  # pragma: no cover - AMD-only dependency, mirrors aiter_backend's guard
    from aiter.mla import mla_decode_fwd
except ImportError:  # pragma: no cover
    mla_decode_fwd = None

logger = logging.getLogger(__name__)

KV_LORA_RANK = 512
KV_CACHE_DIM = 576  # 512 latent + 64 rope

# H is a runtime parameter of the kernel, up to 128.
_MAX_HEADS = 128
# Narrowest head count aiter's asm MLA has a kernel for: its qh16 kernels bake
# gqa=16 into the ISA, so a 12-head call has nothing to dispatch to.
_AITER_MLA_MIN_HEADS = 16
# The kernel launches B * ceil(q_len * H / 96) query row slices and accepts at
# most 304 of them; a larger batch falls back to aiter.
_KERNEL_SLICE_ROWS = 96
_KERNEL_MAX_ROW_SLICES = 304

_MAX_BATCH = 8192  # size of the staged int32 seq_lens buffer
_LN2 = math.log(2.0)


class MoonmathMLABackend(AiterAttnBackend):
    """MLA decode and spec-verify via moonmath_amd; aiter for the rest."""

    # This backend has its own MLA kernels, so aiter's head-count assert must not
    # reject it at construction.
    skip_mla_head_count_assert = True

    def __init__(self, model_runner):
        super().__init__(model_runner)
        import moonmath_amd.mla as mla  # fail fast if not installed

        # The unified op reuses the old q_len-1 kernel's name; only the package
        # that ships it also exports the DCP merge.
        if not hasattr(mla, "mla_dcp_lse_merge_ranks"):
            raise ImportError(
                "moonmath_mla needs moonmath_amd with the unified mla_decode_a16w8 op"
            )
        self._mla = mla
        # The kernels take the fused-576 pool as fp8 e4m3fnuz at a per-tensor
        # descale; a bf16 KV cache has no A16W8 arm and falls back to aiter.
        self._enabled = (
            bool(self.use_mla) and self.kv_cache_dtype == torch.float8_e4m3fnuz
        )
        self._multiq = self._enabled and envs.SGLANG_MOONMATH_MLA_MULTIQ_VERIFY.get()

        # int32 staging for device seq_lens (sglang carries int64 in eager mode).
        self._seq_lens_i32 = torch.zeros(
            _MAX_BATCH, dtype=torch.int32, device=model_runner.device
        )
        # Under DCP, `_seq_lens_i32` holds rank-local lengths and these the global
        # ones; the kv view is the shard the kernel reads for this forward.
        self._dcp_rank = get_parallel().attn_dcp_rank if self.dcp_world_size > 1 else 0
        self._glen_i32 = torch.zeros_like(self._seq_lens_i32)
        self._dcp_kv_indptr: torch.Tensor | None = None
        self._dcp_kv_indices: torch.Tensor | None = None
        # Capture-stable shard over prefix + draft window, set by
        # init_cuda_graph_state when DCP verify runs through the kernel.
        self._dcp_graph_verify_kv_indptr: torch.Tensor | None = None
        self._dcp_graph_verify_kv_indices: torch.Tensor | None = None
        self._dcp_graph_verify_local_lens_cpu: torch.Tensor | None = None
        self._logged_decode = False
        self._logged_verify = False
        logger.info(
            "moonmath_mla: enabled=%s multiq_verify=%s num_head=%s kv_cache_dtype=%s",
            self._enabled,
            self._multiq,
            self.num_head,
            self.kv_cache_dtype,
        )

    # ── metadata ─────────────────────────────────────────────────────────────
    # The kernels want int32 device seq_lens; sglang carries int64. Stage it in
    # the metadata hooks -- once per forward rather than once per MLA layer, and
    # out-of-graph before every replay, which is where a device buffer that a
    # captured kernel reads must be refreshed.
    def _stage_seq_lens_i32(self, forward_batch: ForwardBatch, in_graph: bool) -> None:
        mode = forward_batch.forward_mode
        # TARGET_VERIFY is an extend mode, not is_decode(), so it needs staging
        # of its own or the window kernel reads the previous forward's values.
        is_verify = self._multiq and mode.is_target_verify()
        if not self._enabled or not (mode.is_decode() or is_verify):
            return
        bs = forward_batch.batch_size
        if bs > _MAX_BATCH:
            return
        if self.dcp_world_size > 1:
            self._stage_dcp_shard(forward_batch, is_verify, in_graph)
        elif is_verify:
            # `fb.seq_lens` at TARGET_VERIFY EXCLUDES the draft tokens, but their
            # KV is already written and the window kernel wants the TOTAL span:
            # its position t sees the first `seq_lens[b] - q_len + t + 1` slots.
            # Take the span from `kv_indptr`, which is what `kv_indices` was
            # built against, so it is by construction the number of slots this
            # request owns -- no assumption that every request drafted the same
            # number of tokens. Device-side, so it stays graph-capture safe.
            indptr = self.forward_metadata.kv_indptr
            torch.sub(indptr[1 : bs + 1], indptr[:bs], out=self._seq_lens_i32[:bs])
        else:
            self._seq_lens_i32[:bs].copy_(forward_batch.seq_lens)

    def _stage_dcp_shard(
        self, forward_batch: ForwardBatch, is_verify: bool, in_graph: bool
    ) -> None:
        """This rank's shard: rank-local lengths and indices, global lengths.

        aiter shards only the committed prefix for verify and attends the window
        separately; the kernel reads the window from the pool, so verify keeps a
        shard over prefix + window of its own.
        """
        bs = forward_batch.batch_size
        glen = self._glen_i32[:bs]
        glen.copy_(forward_batch.seq_lens[:bs])
        if is_verify:
            glen.add_(self.num_draft_tokens)
            kv_indptr, kv_indices = self._plan_dcp_verify_shard(
                forward_batch, glen, in_graph
            )
        else:
            kv_indptr, kv_indices = (
                self.forward_metadata.kv_indptr,
                self.forward_metadata.kv_indices,
            )
        torch.sub(kv_indptr[1 : bs + 1], kv_indptr[:bs], out=self._seq_lens_i32[:bs])
        self._dcp_kv_indptr, self._dcp_kv_indices = kv_indptr, kv_indices

    def _plan_dcp_verify_shard(
        self, forward_batch: ForwardBatch, kv_lens: torch.Tensor, in_graph: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = forward_batch.batch_size
        if in_graph:
            kv_indptr = self._dcp_graph_verify_kv_indptr[: bs + 1]
            kv_indices = self._dcp_graph_verify_kv_indices
            seq_lens_cpu = None
            static_local_lens_cpu = self._dcp_graph_verify_local_lens_cpu[:bs]
        else:
            kv_indptr = kv_lens.new_zeros(bs + 1)
            kv_indices = kv_lens.new_empty(
                forward_batch.seq_lens_sum + bs * self.num_draft_tokens
            )
            seq_lens_cpu = forward_batch.seq_lens_cpu[:bs] + self.num_draft_tokens
            static_local_lens_cpu = None
        kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
        num_token_blocks = self._kv_index_blocks(bs)
        create_flashinfer_kv_indices_triton[(bs, num_token_blocks)](
            self.req_to_token,
            forward_batch.req_pool_indices,
            kv_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token.stride(0),
            TOKEN_BLOCK_PARALLEL=num_token_blocks > 1,
        )
        self._plan_dcp_decode_metadata(
            kv_indptr,
            kv_indices,
            kv_lens.clone(),
            seq_lens_cpu,
            bs,
            static_local_kv_lens_cpu=static_local_lens_cpu,
        )
        return kv_indptr, kv_indices

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: torch.Tensor | None = None,
    ):
        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)
        if self.dcp_world_size <= 1 or not self._multiq:
            return
        max_kv_len = self.max_context_len + self.num_draft_tokens
        self._dcp_graph_verify_kv_indptr = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self._dcp_graph_verify_kv_indices = torch.zeros(
            max_bs * max_kv_len, dtype=torch.int32, device=self.device
        )
        # Sizes the shard without a device sync, as aiter's own verify plan does.
        self._dcp_graph_verify_local_lens_cpu = torch.full(
            (max_bs,), -(-max_kv_len // self.dcp_world_size), dtype=torch.int32
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self._stage_seq_lens_i32(forward_batch, in_graph=False)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        super().init_forward_metadata_out_graph(forward_batch, in_capture)
        self._stage_seq_lens_i32(forward_batch, in_graph=True)

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

    @staticmethod
    def _within_kernel_domain(bs: int, q_len: int, num_heads: int) -> bool:
        """Whether one launch fits the kernel's query row-slice budget."""
        row_slices = -(-(q_len * num_heads) // _KERNEL_SLICE_ROWS)
        return bs * row_slices <= _KERNEL_MAX_ROW_SLICES

    def _kv_indices_int32(self, bs: int):
        # Both are int32 already, so `.to` returns the argument. It must STAY
        # free: a real cast allocates a fresh tensor per layer, and a captured
        # graph holds the address it saw at capture time. The kernel takes B
        # from kv_indptr's length, so hand it exactly bs + 1 entries (a view).
        if self.dcp_world_size > 1:
            kv_indices, kv_indptr = self._dcp_kv_indices, self._dcp_kv_indptr
        else:
            kv_indices = self.forward_metadata.kv_indices
            kv_indptr = self.forward_metadata.kv_indptr
        return kv_indices.to(torch.int32), kv_indptr[: bs + 1].to(torch.int32)

    def _split_q(self, q, *shape):
        """`q` as the contiguous (latent, rope) pair the kernel ABI takes."""
        q = q.reshape(*shape, KV_CACHE_DIM)
        return q[..., :KV_LORA_RANK].contiguous(), q[..., KV_LORA_RANK:].contiguous()

    def _run_kernel(self, q, layer: RadixAttention, fb: ForwardBatch, q_len: int):
        """`B * q_len` query rows -> out, or `(out, lse)` under DCP."""
        B, H = fb.batch_size, layer.tp_q_head_num
        T = B * q_len
        dcp = self.dcp_world_size > 1
        q_lat, q_pe = self._split_q(q, T, H)
        out = torch.empty(T, H, KV_LORA_RANK, dtype=torch.bfloat16, device=q.device)
        lse = torch.empty(T, H, dtype=torch.float32, device=q.device) if dcp else None
        kv_indices, kv_indptr = self._kv_indices_int32(B)
        self._mla.mla_decode_a16w8(
            q_lat,
            q_pe,
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            out,
            self._seq_lens_i32[:B],
            kv_indices,
            kv_indptr,
            layer.scaling,
            1.0 if layer.k_scale is None else float(layer.k_scale),
            lse=lse,
            glen=self._glen_i32[:B] if dcp else None,
            cp_rank=self._dcp_rank,
            cp_world=self.dcp_world_size,
        )
        out = out.reshape(T, H * KV_LORA_RANK)
        if lse is None:
            return out
        # The kernel's LSE is base 2; the DCP merge reads aiter's natural log.
        return out, lse.mul_(_LN2)

    # ── decode ───────────────────────────────────────────────────────────────
    def _decode_eligible(self, q, layer: RadixAttention, fb: ForwardBatch) -> bool:
        return (
            fb.forward_mode.is_decode()
            and fb.spec_info is None
            and self._shape_eligible(q, layer, fb)
            and self._within_kernel_domain(fb.batch_size, 1, layer.tp_q_head_num)
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

        if not self._logged_decode:
            self._logged_decode = True
            logger.info("moonmath_mla: decode bs=%d H=%d", B, H)

        return self._run_kernel(q, layer, fb, q_len=1)

    # ── TARGET_VERIFY: the multi-query draft window ──────────────────────────
    def _verify_eligible(self, q, layer: RadixAttention, fb: ForwardBatch) -> bool:
        """Falls back to aiter rather than raising: a mishandled shape here
        surfaces as mis-accepted tokens, not as an exception."""
        return (
            self._multiq
            and fb.forward_mode.is_target_verify()
            and fb.spec_info is not None
            and self._shape_eligible(q, layer, fb)
            and self._within_kernel_domain(
                fb.batch_size, fb.spec_info.num_tokens_per_req, layer.tp_q_head_num
            )
        )

    def _forward_verify(self, q, k, v, layer, fb, save_kv_cache):
        """TARGET_VERIFY through the A16W8 multi-query window, or None to fall back.

        `seq_lens` is the TOTAL span including the draft tokens (staged in
        `_stage_seq_lens_i32`) and the kernel applies end-aligned causal masking
        inside the window, which is bitwise equal to per-position q_len=1 calls.
        """
        B, H = fb.batch_size, layer.tp_q_head_num
        q_len = fb.spec_info.num_tokens_per_req
        # The DCP shard was planned for num_draft_tokens per request.
        if q.shape[0] != B * q_len or (
            self.dcp_world_size > 1 and q_len != self.num_draft_tokens
        ):
            return None

        if save_kv_cache and k is not None:
            self.token_to_kv_pool.set_kv_buffer(layer, fb.out_cache_loc, k, v)

        if not self._logged_verify:
            self._logged_verify = True
            logger.info(
                "moonmath_mla: multi-query verify bs=%d q_len=%d H=%d",
                B,
                q_len,
                H,
            )

        return self._run_kernel(q, layer, fb, q_len=q_len)

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None
    ):
        if sinks is None and self._verify_eligible(q, layer, forward_batch):
            out = self._forward_verify(q, k, v, layer, forward_batch, save_kv_cache)
            if out is not None:
                return out
        return super().forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, sinks
        )

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
        override covers prefill, DRAFT_EXTEND_V2 and the shapes the two moonmath
        arms decline. Unconditional at the head counts
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
