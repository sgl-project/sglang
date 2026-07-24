from __future__ import annotations

import logging
import os
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from sglang.kernels.ops.attention.deepseek_v4_rope import (
    apply_rotary_emb_triton,
    fused_norm_rope_inplace_triton,
    fused_softmax_pool_triton,
)
from sglang.kernels.ops.attention.dsv4.fused_compress_triton import (
    fused_ape_pool_norm_rope,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation
from sglang.srt.layers.attention.dsv4.compressor import Compressor as _CompressorBase
from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation

try:
    from sglang.kernels.ops.attention.deepseek_v4_rope import fused_softmax_pool_triton
except ImportError:
    fused_softmax_pool_triton = None
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang.kernels.ops.attention.dsv4.rms_normalize_hip import rms_normalize_triton

logger = logging.getLogger(__name__)


def capture_c4_state_windows_unified(
    *,
    backend,
    state_pool,
    kv_score_input: torch.Tensor,
    forward_batch,
    is_indexer: bool,
    layer_id: int,
    ratio: int,
) -> None:
    """Strict-mode c4 / c4-indexer state capture for the unified-kv prefill path
    (compressor_v2.forward_unified).

    Under unified-kv the model calls backend.forward_core_compressor, not the
    compressor's own forward, so the legacy per-request capture never runs and
    the state staging stayed empty (every (rid, B) bind missed and strict reuse
    was rejected). This mirrors _capture_compress_state_windows exactly (same
    per-request buffer, page boundaries, slot and D2H copy) so a reusing request
    restores the boundary bit-exact. Must run before the in-place compress
    transform mutates kv_score_input; the caller invokes it at the top of
    forward_unified for the extend path. No-op unless the host state pool is
    wired and ratio == 4.
    """
    if ratio != 4:
        return
    token_to_kv_pool = backend.token_to_kv_pool
    attr = "_c4_indexer_state_host_pool" if is_indexer else "_c4_state_host_pool"
    hp = getattr(token_to_kv_pool, attr, None)
    if hp is None:
        return
    layer_index = getattr(token_to_kv_pool, "_c4_state_layer_index", None)
    if layer_index is None:
        return
    li = layer_index.get(layer_id)
    if li is None:
        return
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    extend_lens = forward_batch.extend_seq_lens_cpu
    if extend_lens is None or prefix_lens is None:
        return

    req_pool_indices = forward_batch.req_pool_indices
    req_to_token = backend.req_to_token_pool.req_to_token
    device = kv_score_input.device

    page = backend.page_size
    slot_page = hp.slot_page_size  # == ring_size
    # off0=0 tile packing: the host state tile is packed at tile start; device
    # state rows are addressed independently via translate_from_swa_loc_to_state_loc,
    # so the host layout need not mirror the spec-padded SWA ring
    # (swa_ring = sliding_window + spec_extra may not divide page under EAGLE).
    win = ratio
    slot_bytes = hp.item_bytes // slot_page
    staging = hp._capture_staging
    host_layer_buf = hp.data_refs[li]

    # Stride/tail gate -- MUST match ``capture_swa_windows`` (the SWA carrier
    # capture) boundary-for-boundary: it stages a window only at every
    # ``stride``-th page boundary plus the TRUE sequence tail page ``tail_B``
    # (``deepseek_v4_backend_hip_radix.capture_swa_windows``). Staging state at
    # EVERY page boundary (the old behaviour) produced orphan state tiles at
    # non-stride boundaries that no SWA carrier binds; they pile up until
    # ``cleanup_after_caching_req`` and blow the staging capacity under peak
    # concurrency, excluding real boundaries from reuse (#cached-token=0).
    stride = max(1, int(getattr(token_to_kv_pool, "_swa_offload_page_stride", 1)))
    _orig = getattr(forward_batch, "orig_seq_lens", None)
    orig_l = _orig.tolist() if _orig is not None else None

    pt = 0
    for i in range(forward_batch.batch_size):
        ext = int(extend_lens[i])
        cs = int(prefix_lens[i])  # prefix_len == chunk start position
        new_tok = kv_score_input[pt : pt + ext]
        pt += ext
        if ext <= 0:
            continue
        total = cs + ext
        boundary = (total // page) * page
        # TRUE sequence tail page (page-aligned full seq len); captured even when
        # not stride-aligned, matching capture_swa_windows' forced tail.
        tail_B = (int(orig_l[i]) // page) * page if orig_l is not None else -1
        B = ((cs // page) + 1) * page
        if B > boundary:
            continue  # no page boundary crossed in this chunk -> nothing to stage

        # Per-request buffer == [pre_kv_state overlap prefix | new tokens], the
        # same object compress reads; pre_kv_state comes from the state ring.
        pre_state_indices = torch.arange(
            cs - (cs % ratio + ratio), cs, device=device
        ).clamp(min=-1)
        raw_loc = torch.where(
            pre_state_indices < 0,
            torch.full_like(pre_state_indices, -1),
            req_to_token[req_pool_indices[i], pre_state_indices],
        )
        swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(raw_loc)
        state_loc = state_pool.translate_from_swa_loc_to_state_loc(swa_loc)
        pre_kv_state = state_pool.get_state_by_state_loc(state_loc).kv_score
        state_buf = torch.cat([pre_kv_state, new_tok], dim=0)
        valid_kv_len = state_buf.size(0)
        pre_len = valid_kv_len - ext

        rid = int(req_pool_indices[i])
        while B <= boundary:
            # stride gate: keep every ``stride``-th page boundary plus the true
            # sequence tail page (identical to capture_swa_windows), so exactly
            # the SWA-carried boundaries get a state tile -- no orphan tiles.
            if (B // page) % stride != 0 and B != tail_B:
                B += page
                continue
            off0 = 0  # pack window at tile start (matches capture/restore)
            if off0 + win > slot_page:
                raise AssertionError(
                    f"unified state window out of range: B={B} win={win} cs={cs} "
                    f"off0={off0} ring_size={slot_page}"
                )
            # Under heavy reuse a tiny chunk crossing a page boundary can make the
            # window [B-win, B) reach back into the overlap prefix (pre_kv_state).
            # That is valid iff buf_lo >= 0; if the needed prefix is unavailable,
            # skip this boundary (excluded from reuse, correctness preserved)
            # rather than crash.
            if pre_len + (B - win - cs) < 0:
                B += page
                continue
            key = (rid, int(B))
            hidx = staging.get(key)
            if hidx is None:
                hidx = hp.alloc(slot_page)
                if hidx is None:
                    n = getattr(hp, "_state_alloc_fail", 0) + 1
                    hp._state_alloc_fail = n
                    if n & (n - 1) == 0:
                        logger.warning(
                            "[SWA-HiCache] c4 state staging exhausted (unified) "
                            "count=%d at (rid=%s,B=%d); boundary excluded from "
                            "reuse (recomputed) -- correctness-safe, reuse-only loss.",
                            n,
                            rid,
                            int(B),
                        )
                    B += page
                    continue
                staging[key] = hidx
            buf_lo = pre_len + (B - win - cs)
            buf_hi = pre_len + (B - cs)
            win_slice = state_buf[buf_lo:buf_hi]
            flat = win_slice.contiguous().view(torch.uint8).reshape(-1)
            if flat.numel() != win * slot_bytes:
                raise AssertionError(
                    f"unified state window bytes {flat.numel()} != {win * slot_bytes}"
                )
            page_row = int(hidx[0].item()) // slot_page
            dst = host_layer_buf[page_row]
            dst[off0 * slot_bytes : off0 * slot_bytes + flat.numel()].copy_(
                flat, non_blocking=True
            )
            if os.environ.get("SGLANG_SWA_DBG_CHECKSUM") == "1":
                _crc = getattr(hp, "_capture_state_crc", None)
                if _crc is not None:
                    _idx = (
                        torch.arange(
                            flat.numel(), device=flat.device, dtype=torch.int64
                        )
                        + 1
                    )
                    _crc[(rid, int(B), li)] = int(
                        (flat.to(torch.int64) * _idx).sum().item()
                    )
            B += page

    _rec = getattr(hp, "record_capture_done", None)
    if _rec is not None:
        _rec()


class DeepseekRefRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return rms_normalize_triton(x, self.eps, self.weight)


class CompressorHip(_CompressorBase):
    """HIP (ROCm) specific Compressor implementation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = DeepseekRefRMSNorm(self.head_dim, eps=self.norm.variance_epsilon)
        self._freqs_cis_real: torch.Tensor | None = None

    @cached_property
    def use_fused_compress(self) -> bool:
        return envs.SGLANG_OPT_USE_FUSED_COMPRESS.get()

    @cached_property
    def use_hip_fused_compress(self) -> bool:
        return envs.SGLANG_OPT_USE_FUSED_COMPRESS.get()

    @cached_property
    def use_fused_compress_triton(self) -> bool:
        # The fused Triton kernel only benefits non-overlap (HCA, ratio=128)
        # but HCA's K=128 loop is too sequential to outperform batched ops.
        # CSA (overlap=True) has a reshape/overlap-transform semantic mismatch.
        # Disabled until a tiled kernel for CSA overlap is implemented.
        return False

    def _get_states(
        self,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> KVAndScore:
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            return token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            return token_to_kv_pool.get_attention_compress_states(self.layer_id)

    def _get_state_pool(self, attn_backend: AttentionBackend) -> CompressStatePool:
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            ret = token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            ret = token_to_kv_pool.get_attention_compress_states(self.layer_id)
        assert isinstance(ret, CompressStatePool)
        return ret

    def overlap_transform(self, tensor: torch.Tensor, fill_value: Any) -> torch.Tensor:
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (self.ratio, 2 * self.head_dim)

        s, r, d = tensor.size(0), self.ratio, self.head_dim
        new_tensor = tensor.new_full((s, 2 * r, d), fill_value)
        new_tensor[:, r:] = tensor[:, :, d:]
        new_tensor[1:, :r] = tensor[:-1, :, :d]
        return new_tensor

    def overlap_transform_decode(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (2 * self.ratio, 2 * self.head_dim)
        r, d = self.ratio, self.head_dim
        ret = torch.cat((tensor[:, :r, :d], tensor[:, r:, d:]), dim=1)
        return ret

    @staticmethod
    def compute_state_len(seq_len: int, ratio: int):
        return seq_len % ratio + (ratio == 4) * ratio

    @staticmethod
    def compute_state_len_indices(seq_len: int, ratio: int):
        state_len = seq_len % ratio + (ratio == 4) * ratio
        return torch.arange(seq_len - state_len, seq_len).clamp(min=-1)

    def print_tensor(self, y: torch.Tensor, name: str):
        enable = int(os.environ.get("SGLANG_ENABLE_PRINT_TENSOR", 0))
        if enable:
            print(f"[sgl] {name}: shape={y.shape}, dtype={y.dtype}, device={y.device}")
            print(f"{y.flatten()[:10]}...{y.flatten()[-10:]}")

    def _capture_compress_state_windows(
        self,
        kv_and_score_buffer,
        valid_kv_len: int,
        prefix_len: int,
        extend_len: int,
        rid: int,
        backend,
        stride: int = 1,
        tail_B: int = -1,
    ) -> None:
        """Capture the c4 / c4-indexer overlap state [B-ratio, B) at each page boundary
        B into the host state pool.

        The device state ring is a small rolling buffer, so interior boundary states
        get overwritten during chunked prefill; snapshot them from kv_and_score_buffer
        instead. A window token at position s maps to ring offset
        (s % swa_ring_size) % ring_size, which is a pure function of s (the device
        ring row is pos % ring), so a reusing request recomputes the same slots
        regardless of page/ring divisibility. Captured before the
        in-place overlap transform, so the window is byte-identical to what
        set_state_by_state_loc persists. No-op unless the host state pool is wired.

        Boundaries match capture_swa_windows exactly: every stride-th page boundary
        plus the true tail page tail_B. This keeps every SWA carrier's (rid, B)
        present for the atomic co-lifetime bind and avoids staging orphan tiles at
        non-stride boundaries that would pile up and exhaust capacity under load.
        stride == 1 / tail_B == -1 keep every boundary.
        """
        if self.ratio != 4:
            return
        token_to_kv_pool = backend.token_to_kv_pool
        attr = (
            "_c4_indexer_state_host_pool"
            if self.is_in_indexer
            else "_c4_state_host_pool"
        )
        hp = getattr(token_to_kv_pool, attr, None)
        if hp is None:
            return
        layer_index = getattr(token_to_kv_pool, "_c4_state_layer_index", None)
        if layer_index is None:
            return
        li = layer_index.get(self.layer_id)
        if li is None:
            return
        if extend_len <= 0:
            return

        page = backend.page_size
        slot_page = hp.slot_page_size  # == ring_size
        # off0=0 tile packing: host tile is packed at tile start; device state
        # rows are addressed via translate_from_swa_loc_to_state_loc, so the host
        # layout need not mirror the spec-padded SWA ring (see
        # capture_c4_state_windows_unified).
        win = self.ratio  # compute_state_len(B, 4) == 4 at a page boundary
        slot_bytes = hp.item_bytes // slot_page
        staging = hp._capture_staging
        state_buf = kv_and_score_buffer.kv_score
        host_layer_buf = hp.data_refs[li]
        pre_len = valid_kv_len - extend_len

        stride = max(1, int(stride))
        cs = prefix_len
        total = prefix_len + extend_len
        boundary = (total // page) * page
        B = ((cs // page) + 1) * page
        while B <= boundary:
            # stride gate: keep every ``stride``-th page boundary plus the true
            # sequence tail page (identical to capture_swa_windows), so exactly
            # the SWA-carried boundaries get a state tile -- no orphan tiles.
            if (B // page) % stride != 0 and B != tail_B:
                B += page
                continue
            off0 = 0  # pack window at tile start (matches capture/restore)
            if off0 + win > slot_page:
                raise AssertionError(
                    f"state window out of range: B={B} win={win} cs={cs} "
                    f"off0={off0} ring_size={slot_page}"
                )
            # See capture_c4_state_windows_unified: allow windows reaching into the
            # overlap prefix (buf_lo >= 0); skip (do not crash) when unavailable.
            if pre_len + (B - win - prefix_len) < 0:
                B += page
                continue
            key = (rid, int(B))
            hidx = staging.get(key)
            if hidx is None:
                hidx = hp.alloc(slot_page)
                if hidx is None:
                    # Staging exhausted: this boundary's state is dropped, so its
                    # SWA window (if strided here) is excluded from strict reuse
                    # by the match validator. Count + throttle-warn so E2E can
                    # surface residual undersizing (see staging_slack sizing).
                    n = getattr(hp, "_state_alloc_fail", 0) + 1
                    hp._state_alloc_fail = n
                    if n & (n - 1) == 0:  # powers of two only
                        logger.warning(
                            "[SWA-HiCache] c4 state staging exhausted "
                            "(alloc None) count=%d at (rid=%s,B=%d); boundary "
                            "excluded from reuse (recomputed) -- "
                            "correctness-safe, reuse-only loss.",
                            n,
                            rid,
                            int(B),
                        )
                    B += page
                    continue
                staging[key] = hidx
            buf_lo = pre_len + (B - win - prefix_len)
            buf_hi = pre_len + (B - prefix_len)
            win_slice = state_buf[buf_lo:buf_hi]
            flat = win_slice.contiguous().view(torch.uint8).reshape(-1)
            if flat.numel() != win * slot_bytes:
                raise AssertionError(
                    f"state window bytes {flat.numel()} != {win * slot_bytes}"
                )
            page_row = int(hidx[0].item()) // slot_page
            dst = host_layer_buf[page_row]
            dst[off0 * slot_bytes : off0 * slot_bytes + flat.numel()].copy_(
                flat, non_blocking=True
            )
            if os.environ.get("SGLANG_SWA_DBG_CHECKSUM") == "1":
                _crc = getattr(hp, "_capture_state_crc", None)
                if _crc is not None:
                    _idx = (
                        torch.arange(
                            flat.numel(), device=flat.device, dtype=torch.int64
                        )
                        + 1
                    )
                    _crc[(rid, int(B), li)] = int(
                        (flat.to(torch.int64) * _idx).sum().item()
                    )
            B += page

    def compress_extend_paged(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ):
        backend = attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4HipRadixBackend)
        token_to_kv_pool = backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        state_pool = self._get_state_pool(backend)
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = backend.req_to_token_pool.req_to_token
        assert not self.forward_mode.is_target_verify()

        assert extend_lens is not None and prefix_lens is not None
        device = kv_and_scores.kv.device

        assert kv_and_scores.kv.shape[-1] == self.head_dim * self.coff
        compressed_kv_output = torch.full(
            (kv_and_scores.kv.size(0), self.head_dim),
            fill_value=10000.0,
            dtype=kv_and_scores.kv.dtype,
            device=device,
        )

        bs = forward_batch.batch_size
        # Strict-mode c4 overlap-state capture (Phase C): snapshot [B-ratio, B)
        # at each page boundary so a reusing request can restore the state ring
        # and avoid the boundary dirty read. Gate the (single) req_pool_indices
        # sync on the host pool actually being wired -- a no-op otherwise, so the
        # non-strict / best-effort path pays nothing.
        _cap_hp = (
            getattr(
                token_to_kv_pool,
                (
                    "_c4_indexer_state_host_pool"
                    if self.is_in_indexer
                    else "_c4_state_host_pool"
                ),
                None,
            )
            if self.ratio == 4
            else None
        )
        _cap_rids = req_pool_indices.tolist() if _cap_hp is not None else None
        # Stride/tail gate parameters (mirror capture_swa_windows): only stage the
        # SWA-carried boundaries so no orphan state tile is produced. Synced once,
        # gated on the host pool being wired so the non-strict path pays nothing.
        if _cap_rids is not None:
            _cap_stride = max(
                1, int(getattr(token_to_kv_pool, "_swa_offload_page_stride", 1))
            )
            _cap_orig = getattr(forward_batch, "orig_seq_lens", None)
            _cap_orig_l = _cap_orig.tolist() if _cap_orig is not None else None
        pt = 0
        for i in range(bs):
            kv_and_score = kv_and_scores[pt : pt + extend_lens[i]]
            pre_state_indices = self.compute_state_len_indices(
                seq_len=prefix_lens[i], ratio=self.ratio
            ).to(device)
            if self.ratio == 128:
                state_loc = state_pool.translate_from_req_position_to_state_loc(
                    req_pool_indices[i], pre_state_indices
                )
            else:
                raw_loc = torch.where(
                    pre_state_indices < 0,
                    -1,
                    req_to_token[req_pool_indices[i], pre_state_indices],
                )
                swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(raw_loc)
                state_loc = state_pool.translate_from_swa_loc_to_state_loc(swa_loc)
            pre_kv_state = state_pool.get_state_by_state_loc(state_loc)
            kv_and_score_buffer = KVAndScore.cat([pre_kv_state, kv_and_score], dim=0)
            valid_kv_len = kv_and_score_buffer.kv.size(0)

            post_state_indices = self.compute_state_len_indices(
                seq_len=prefix_lens[i] + extend_lens[i], ratio=self.ratio
            ).to(device)
            post_state_len = post_state_indices.size(0)

            assert post_state_len <= valid_kv_len
            if self.ratio == 128:
                post_state_loc = state_pool.translate_from_req_position_to_state_loc(
                    req_pool_indices[i], post_state_indices
                )
            else:
                post_raw_loc = torch.where(
                    post_state_indices < 0,
                    -1,
                    req_to_token[req_pool_indices[i], post_state_indices],
                )
                post_swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(
                    post_raw_loc
                )
                post_state_loc = state_pool.translate_from_swa_loc_to_state_loc(
                    post_swa_loc
                )
            post_state_to_set = kv_and_score_buffer[valid_kv_len - post_state_len :]
            state_pool.set_state_by_state_loc(post_state_loc, post_state_to_set)

            # Capture the boundary overlap state from the pre-transform buffer
            # (same bytes set_state_by_state_loc just persisted), BEFORE the
            # in-place compress transform below mutates kv_and_score_buffer.
            if _cap_rids is not None:
                _page = backend.page_size
                _tail_B = (
                    (int(_cap_orig_l[i]) // _page) * _page
                    if _cap_orig_l is not None
                    else -1
                )
                self._capture_compress_state_windows(
                    kv_and_score_buffer,
                    valid_kv_len,
                    int(prefix_lens[i]),
                    int(extend_lens[i]),
                    _cap_rids[i],
                    backend,
                    stride=_cap_stride,
                    tail_B=_tail_B,
                )

            compress_len = valid_kv_len // self.ratio * self.ratio
            if compress_len == 0:
                pt += extend_lens[i]
                continue

            kv_and_score_to_compress = kv_and_score_buffer[:compress_len].view(
                compress_len // self.ratio, self.ratio, -1
            )
            kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

            if self.overlap:
                new_kv = self.overlap_transform(
                    kv_and_score_to_compress.kv, fill_value=0
                )
                new_score = self.overlap_transform(
                    kv_and_score_to_compress.score, fill_value=float("-inf")
                )
                kv_and_score_to_compress = KVAndScore.from_kv_score(
                    kv=new_kv, score=new_score
                )
                del new_kv, new_score
                kv_and_score_to_compress = kv_and_score_to_compress[1:]

                if kv_and_score_to_compress.kv.size(0) == 0:
                    pt += extend_lens[i]
                    continue

            beg_idx = prefix_lens[i] // self.ratio * self.ratio
            end_idx = (prefix_lens[i] + extend_lens[i]) // self.ratio * self.ratio

            if self.use_hip_fused_compress:
                kv_compressed = fused_softmax_pool_triton(
                    kv_and_score_to_compress.kv_score,
                    kv_and_score_to_compress._item_size,
                )
            else:
                kv_compressed = (
                    kv_and_score_to_compress.kv
                    * kv_and_score_to_compress.score.softmax(dim=1)
                ).sum(dim=1)

            assert kv_compressed.dtype == torch.float32

            freqs_cis = self.freqs_cis[beg_idx : end_idx : self.ratio]
            assert freqs_cis.size(0) == kv_compressed.size(
                0
            ), f"{freqs_cis.shape=} {kv_compressed.shape=}"
            if self.use_hip_fused_compress:
                fused_norm_rope_inplace_triton(
                    kv_compressed, self.norm.weight, self.norm.eps, freqs_cis
                )
            else:
                kv_compressed = self.norm(kv_compressed)
                apply_rotary_emb_triton(
                    kv_compressed[..., -self.rope_head_dim :], freqs_cis
                )
            del beg_idx, end_idx

            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)

            start = prefix_lens[i]
            start = start + self.ratio - 1 - start % self.ratio
            indices_in_seq = torch.arange(
                start,
                prefix_lens[i] + extend_lens[i],
                self.ratio,
                device=kv_and_scores.kv.device,
            )
            assert indices_in_seq.size(0) == kv_compressed.size(0)
            compressed_kv_output[indices_in_seq - prefix_lens[i] + pt] = kv_compressed

            pt += extend_lens[i]

        # H2 (overlap safety): the prefill c4 state capture above enqueued its
        # D2H copies non_blocking on the current (forward) stream but -- unlike
        # capture_compress_state_windows_decode -- never recorded the pool's
        # capture-done event. A cross-stream reuse restore then wait_capture_done
        # on a stale (or absent) event, leaving its H2D free to read a
        # half-written host tile under overlap scheduling. Record here, symmetric
        # to the decode-source path, so restore orders strictly after these
        # copies. No-op unless the strict state pool is wired.
        if _cap_hp is not None:
            _rec = getattr(_cap_hp, "record_capture_done", None)
            if _rec is not None:
                _rec()

        return compressed_kv_output

    def compress_decode_paged(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ):
        """Paged and cudagraph compatible version of compress_decode"""
        assert self.ape_converted
        state_pool = self._get_state_pool(attn_backend)
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = attn_backend.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens

        if forward_batch.forward_mode.is_target_verify():
            draft_tokens = attn_backend.speculative_num_draft_tokens
            offsets = torch.arange(1, draft_tokens + 1, device=seq_lens.device)
            seq_lens_2d = seq_lens[:, None] + offsets[None, :]
            seq_lens = seq_lens_2d.view(-1)
            req_pool_indices = req_pool_indices.repeat_interleave(draft_tokens)

        if self.ratio == 128:
            state_locs = state_pool.translate_from_req_position_to_state_loc(
                req_pool_indices, seq_lens - 1
            )
        else:
            raw_locs = req_to_token[req_pool_indices, seq_lens - 1]
            swa_locs = token_to_kv_pool.translate_loc_from_full_to_swa(raw_locs)
            state_locs = state_pool.translate_from_swa_loc_to_state_loc(swa_locs)
        state_pool.set_state_by_state_loc(state_locs, kv_and_scores)

        compress_bulk_len = self.ratio * self.coff
        compress_indices = seq_lens[:, None] + torch.arange(
            -compress_bulk_len, 0, device=seq_lens.device
        )
        compress_indices.clamp_(min=-1)
        if self.ratio == 128:
            compress_indices_state = (
                state_pool.translate_from_req_position_to_state_loc(
                    req_pool_indices[:, None], compress_indices
                )
            )
        else:
            compress_indices_raw = torch.where(
                compress_indices < 0,
                -1,
                req_to_token[req_pool_indices[:, None], compress_indices],
            )
            compress_indices_swa = token_to_kv_pool.translate_loc_from_full_to_swa(
                compress_indices_raw
            )
            compress_indices_state = state_pool.translate_from_swa_loc_to_state_loc(
                compress_indices_swa
            )
        kv_and_score_to_compress = state_pool.get_state_by_state_loc(
            compress_indices_state.view(-1)
        ).view(-1, self.ratio, self.coff * self.head_dim)
        bs = seq_lens.size(0)

        if self.use_fused_compress_triton and not self.overlap:
            # Fused path for non-overlap (HCA, ratio=128, coff=1):
            # APE + softmax-pool + norm + RoPE in one kernel.
            # Overlap (CSA) is excluded because the overlap_transform_decode
            # rearranges A/B halves across the coff dimension in a way
            # that simple reshape cannot replicate correctly.
            raw = kv_and_score_to_compress.kv_score
            gathered = raw.reshape(bs, self.ratio, raw.shape[-1]).contiguous()

            comp_positions = (seq_lens - 1) // self.ratio * self.ratio
            freqs_real_table = self._get_freqs_cis_real()
            freqs_batch = freqs_real_table[comp_positions]

            kv_compressed = fused_ape_pool_norm_rope(
                kv_score_gathered=gathered,
                ape=self.ape,
                rms_weight=self.norm.weight,
                rms_eps=self.norm.eps,
                freqs_cis_real=freqs_batch,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                ratio=self.ratio,
                overlap=self.overlap,
            )
            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)
            return kv_compressed

        # Unfused reference path
        kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

        if self.overlap:
            kv_and_score_to_compress = kv_and_score_to_compress.view(
                bs, self.coff * self.ratio, self.coff * self.head_dim
            )
            kv_and_score_to_compress = KVAndScore.from_kv_score(
                kv=self.overlap_transform_decode(kv_and_score_to_compress.kv),
                score=self.overlap_transform_decode(kv_and_score_to_compress.score),
            )

        kv_and_score_to_compress = kv_and_score_to_compress.view(
            bs, self.ratio * self.coff, self.head_dim
        )

        if self.use_hip_fused_compress:
            kv_compressed = fused_softmax_pool_triton(
                kv_and_score_to_compress.kv_score,
                kv_and_score_to_compress._item_size,
            )
        else:
            kv_compressed = (
                kv_and_score_to_compress.kv
                * kv_and_score_to_compress.score.softmax(dim=1)
            ).sum(dim=1)
        if self.use_hip_fused_compress:
            freqs_cis = self._init_freqs_cis_per_decode_step(forward_batch, seq_lens)
            fused_norm_rope_inplace_triton(
                kv_compressed, self.norm.weight, self.norm.eps, freqs_cis
            )
        else:
            kv_compressed = self.norm(kv_compressed)
            freqs_cis = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
            apply_rotary_emb_triton(
                kv_compressed[..., -self.rope_head_dim :], freqs_cis
            )
        if self.rotate:
            kv_compressed = rotate_activation(kv_compressed)

        return kv_compressed

    def compress_fused(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        backend = attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4HipRadixBackend)
        kv_score_buffer = self._get_state_pool(backend)
        kv_score_buffer = kv_score_buffer.kv_score_buffer.kv_score

        return backend.forward_compress(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score,
            ape=self.ape.view(-1, self.head_dim),
            head_dim=self.head_dim,
            norm=self.norm,
            freqs_cis_cache=self.freqs_cis,
            rotate=self.rotate,
            compress_ratio=self.ratio,
            forward_batch=forward_batch,
            is_paged=True,
        )

    def _get_freqs_cis_real(self) -> torch.Tensor:
        """Cache the float32 view of freqs_cis (complex64 -> real interleaved)."""
        if self._freqs_cis_real is None:
            if self.freqs_cis.is_complex():
                self._freqs_cis_real = (
                    torch.view_as_real(self.freqs_cis).flatten(-2).contiguous()
                )
            else:
                self._freqs_cis_real = self.freqs_cis.contiguous()
        return self._freqs_cis_real

    def compress_dispatch(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        _take_fused = self.use_fused_compress and (
            envs.SGLANG_OPT_DPSK_V4_RADIX.get()
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_extend_without_speculative()
            )
        )
        if _take_fused:
            return self.compress_fused(
                kv_score, forward_batch, attn_backend=attn_backend
            )

        self.compress_decode = self.compress_decode_paged
        self.compress_extend = self.compress_extend_paged
        kv_and_scores = KVAndScore(kv_score)

        if TYPE_CHECKING:
            assert isinstance(kv_and_scores, KVAndScore)

        if (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        ):
            result = self.compress_decode(
                kv_and_scores=kv_and_scores,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
            )
        elif forward_batch.forward_mode.is_extend():
            result = self.compress_extend(
                kv_and_scores=kv_and_scores,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
            )
        else:
            msg = f"Forward mode {forward_batch.forward_mode} not supported in Compressor."
            raise NotImplementedError(msg)

        return result

    def _init_freqs_cis_per_decode_step(
        self,
        forward_batch: ForwardBatch,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        attr = f"freqs_cis_c{self.ratio}"
        cached = getattr(forward_batch, attr, None)
        if cached is not None:
            return cached
        decoded = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
        setattr(forward_batch, attr, decoded)
        return decoded

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)
        kv_score = self.compute_kv_score(x, forward_batch)
        self.forward_mode = forward_batch.forward_mode
        return self.compress_dispatch(
            kv_score, forward_batch, attn_backend=attn_backend
        )
