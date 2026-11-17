"""AITER attention paths that read an fp8 KV cache with calibrated KV scales, and
that apply the layer's softmax scale.

Each case stores ``K / k_scale`` and ``V / v_scale`` in an fp8 cache, runs one
``AiterAttnBackend`` path with its real kernel, and compares the output with
fp32 attention over the dequantized cache. With unit scales, mixing up q's scale
with ``k_scale`` or skipping the cache dequantization does not change the result, so
the scales are kept far from 1. For the same reason the softmax scale also runs at
``1 / head_dim`` (Granite's ``attention_multiplier``), far from ``head_dim**-0.5``.
The SHUFFLE 5D prefill cases check only the softmax scale, with a bf16 cache. The
gfx950 head-dim-256 fresh prefill runs through ``forward_extend``'s own cache write,
NHD and SHUFFLE 5D, so it reads what that writer stored. The fp8 prefills that
gather a cached prefix, SHUFFLE 5D and the gfx950 head-dim-256 context chunk, read
what the pool's own write stored. The unified-attention decode is covered by
``test_aiter_fp8_q_unified_attention.py``.
"""

import itertools
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")
# The head-dim-256 cases need gfx950.
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip()
if _RUNNABLE:
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
    from sglang.kernels.ops.kvcache.shuffle_5d_cache import (
        launch_reshape_and_cache_shuffle_5d,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        fp8_dtype,
        scaled_fp8_quant,
    )
    from sglang.srt.layers.attention.aiter_backend import (
        _AITER_PARTITION_SIZE_ROCM,
        AiterAttnBackend,
    )
    from sglang.srt.model_executor.forward_mode import ForwardMode

_DEVICE = "cuda"
_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 2
_HEAD_DIM = 128
# The default softmax scale, and Granite's attention_multiplier.
_SCALINGS = (_HEAD_DIM**-0.5, 1.0 / _HEAD_DIM)
_BLOCK_SIZE_5D = 16


class _FakeKVPool:
    def __init__(self, k_cache, v_cache):
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.start_layer = 0
        self.layer_num = 1
        self.layer_transfer_counter = None
        self.layer_router = PlainPoolRouter(self, "full")

    def get_kv_buffer(self, _layer_id):
        return self.k_cache, self.v_cache

    def get_key_buffer(self, _layer_id):
        return self.k_cache

    def get_value_buffer(self, _layer_id):
        return self.v_cache


def _offsets(lens):
    return [0, *itertools.accumulate(lens)]


def _reference_attention(q, k, v, causal, scaling):
    """fp32 attention for one request; q is [Lq, Hq, D], k / v are [Lk, Hkv, D].

    With ``causal`` the last query sees every key, so k / v may start with a
    cached prefix.
    """
    group = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(group, dim=1)
    v = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * scaling
    if causal:
        q_len, k_len = q.shape[0], k.shape[0]
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device).tril(
            k_len - q_len
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v)


@unittest.skipUnless(_RUNNABLE, "requires HIP with AITER")
class TestAiterFp8KvScalesPaths(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _make_inputs(self, num_q, num_kv):
        """Returns q and, for K and V, ``(fp8 cache, fp32 dequantized cache, scale)``."""
        # q is several times larger than k, as in real checkpoints, so q / k_scale
        # lies far outside the fp8 range while q itself does not.
        q = 4 * torch.randn(
            num_q, _NUM_Q_HEADS, _HEAD_DIM, device=_DEVICE, dtype=torch.bfloat16
        )
        k = 0.5 * torch.randn(
            num_kv, _NUM_KV_HEADS, _HEAD_DIM, device=_DEVICE, dtype=torch.bfloat16
        )
        kv = []
        for x in (k, torch.randn_like(k)):
            scale = (x.abs().float().amax() / torch.finfo(fp8_dtype).max).view(1)
            self.assertLess(scale.item(), 0.05)
            x_fp8, _ = scaled_fp8_quant(x.view(num_kv, -1), scale)
            x_fp8 = x_fp8.view(x.shape)
            kv.append((x_fp8, x_fp8.float() * scale, scale))
        return q, kv

    def _make_backend(self, k_cache, v_cache):
        backend = object.__new__(AiterAttnBackend)
        backend.use_mla = False
        backend.kv_cache_is_vectorized_5d = False
        backend.use_triton_unified_attention = False
        backend._use_unified_verify = False
        backend.dcp_world_size = 1
        backend.kv_cache_dtype = fp8_dtype
        backend.input_dtype = torch.bfloat16
        backend.page_size = 1
        backend.logits_soft_cap = 0.0
        backend.q_scale = torch.ones(1, dtype=torch.float32, device=_DEVICE)
        backend.k_scale = torch.ones(1, dtype=torch.float32, device=_DEVICE)
        backend.v_scale = torch.ones(1, dtype=torch.float32, device=_DEVICE)
        backend.token_to_kv_pool = _FakeKVPool(k_cache, v_cache)
        backend.extend_attention_fwd = extend_attention_fwd
        return backend

    def _make_layer(
        self, k_scale, v_scale, scaling=_HEAD_DIM**-0.5, head_dim=_HEAD_DIM
    ):
        return SimpleNamespace(
            layer_id=0,
            tp_q_head_num=_NUM_Q_HEADS,
            tp_k_head_num=_NUM_KV_HEADS,
            tp_v_head_num=_NUM_KV_HEADS,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            head_dim=head_dim,
            k_scale=k_scale,
            v_scale=v_scale,
            k_scale_float=k_scale.item() if k_scale is not None else None,
            v_scale_float=v_scale.item() if v_scale is not None else None,
            sliding_window_size=-1,
            scaling=scaling,
            logit_cap=0.0,
            is_cross_attention=False,
        )

    def _assert_close(self, actual, expected):
        actual = actual.float().flatten()
        expected = expected.flatten()
        self.assertTrue(bool(torch.isfinite(actual).all()))
        rel_err = ((actual - expected).norm() / expected.norm()).item()
        # The fp8 rounding of q and the kernels' reduced-precision softmax leave a
        # few percent of error; a misused scale is off by more than 50%.
        self.assertLess(rel_err, 0.15, f"relative L2 error {rel_err:.4f}")

    def test_paged_attention_ragged_decode(self):
        seq_lens = [384, 1000, 17]
        batch = len(seq_lens)
        offsets = _offsets(seq_lens)
        q, ((k_fp8, k_deq, k_scale), (v_fp8, v_deq, v_scale)) = self._make_inputs(
            batch, offsets[-1]
        )

        backend = self._make_backend(k_fp8, v_fp8)
        backend.max_num_partitions = math.ceil(
            max(seq_lens) / _AITER_PARTITION_SIZE_ROCM
        )
        rows = batch * _NUM_Q_HEADS * backend.max_num_partitions
        backend.workspace_buffer = torch.empty(
            rows * (_HEAD_DIM + 2) * 4, dtype=torch.uint8, device=_DEVICE
        )
        backend.kv_last_page_len = torch.ones(batch, dtype=torch.int32, device=_DEVICE)
        backend.forward_metadata = SimpleNamespace(
            kv_indptr=torch.tensor(offsets, dtype=torch.int32, device=_DEVICE),
            kv_indices=torch.arange(offsets[-1], dtype=torch.int32, device=_DEVICE),
            decode_num_partitions=None,
        )

        for scaling in _SCALINGS:
            with self.subTest(scaling=scaling):
                out = backend.forward_decode(
                    q,
                    None,
                    None,
                    self._make_layer(k_scale, v_scale, scaling),
                    SimpleNamespace(batch_size=batch),
                    save_kv_cache=False,
                )
                expected = torch.cat(
                    [
                        _reference_attention(
                            q[b : b + 1], k_deq[s:e], v_deq[s:e], False, scaling
                        )
                        for b, (s, e) in enumerate(zip(offsets, offsets[1:]))
                    ]
                )
                self._assert_close(out, expected)

    def test_mha_batch_prefill(self):
        seq_lens = [300, 77, 512]
        batch = len(seq_lens)
        offsets = _offsets(seq_lens)
        num_tokens = offsets[-1]
        q, ((k_fp8, k_deq, k_scale), (v_fp8, v_deq, v_scale)) = self._make_inputs(
            num_tokens, num_tokens
        )
        # In the model q is a strided view into the fused qkv projection output;
        # the k and v columns hold other values, so reading past q's rows shows up.
        qkv = torch.randn(
            num_tokens,
            (_NUM_Q_HEADS + 2 * _NUM_KV_HEADS) * _HEAD_DIM,
            device=_DEVICE,
            dtype=torch.bfloat16,
        )
        q_strided = qkv[:, : _NUM_Q_HEADS * _HEAD_DIM]
        q_strided.copy_(q.view(num_tokens, -1))
        self.assertFalse(q_strided.is_contiguous())

        backend = self._make_backend(k_fp8, v_fp8)
        indptr = torch.tensor(offsets, dtype=torch.int32, device=_DEVICE)
        backend.forward_metadata = SimpleNamespace(
            qo_indptr=indptr,
            kv_indptr=indptr,
            kv_indices=torch.arange(num_tokens, dtype=torch.int32, device=_DEVICE),
            swa_page_table=None,
            paged_kv_view=None,
            max_q_len=max(seq_lens),
            max_kv_len=max(seq_lens),
        )
        forward_batch = SimpleNamespace(
            batch_size=batch,
            forward_mode=ForwardMode.EXTEND,
            attn_attend_prefix_cache=False,
            extend_prefix_lens_cpu=[0] * batch,
            out_cache_loc=None,
        )

        for scaling in _SCALINGS:
            with self.subTest(scaling=scaling):
                # Keep gfx950 off the HD128 ASM prefill so the NHD path is
                # exercised on every architecture.
                with envs.SGLANG_AITER_ASM_PREFILL_HD128.override(False):
                    out = backend.forward_extend(
                        q_strided,
                        None,
                        None,
                        self._make_layer(k_scale, v_scale, scaling),
                        forward_batch,
                        save_kv_cache=False,
                    )
                expected = torch.cat(
                    [
                        _reference_attention(
                            q[s:e], k_deq[s:e], v_deq[s:e], True, scaling
                        )
                        for s, e in zip(offsets, offsets[1:])
                    ]
                )
                self._assert_close(out, expected)

    def test_mha_batch_prefill_vectorized_5d(self):
        # With no cached prefix in the batch the new K / V are read directly;
        # otherwise every request's K / V are gathered from the SHUFFLE 5D pool.
        extend_lens = [300, 77, 512]
        for prefix_lens in ([0, 0, 0], [128, 0, 1000]):
            with self.subTest(prefix_lens=prefix_lens):
                self._check_vectorized_5d_prefill(prefix_lens, extend_lens)

    def _check_vectorized_5d_prefill(self, prefix_lens, extend_lens):
        batch = len(extend_lens)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        qo_offsets, kv_offsets = _offsets(extend_lens), _offsets(seq_lens)
        num_q, num_kv = qo_offsets[-1], kv_offsets[-1]
        q = 4 * torch.randn(
            num_q, _NUM_Q_HEADS, _HEAD_DIM, device=_DEVICE, dtype=torch.bfloat16
        )
        k = torch.randn(
            num_kv, _NUM_KV_HEADS, _HEAD_DIM, device=_DEVICE, dtype=torch.bfloat16
        )
        v = torch.randn_like(k)

        x = 16 // k.element_size()
        num_blocks = math.ceil(num_kv / _BLOCK_SIZE_5D)
        k_buffer = torch.zeros(
            num_blocks,
            _NUM_KV_HEADS,
            _HEAD_DIM // x,
            _BLOCK_SIZE_5D,
            x,
            device=_DEVICE,
            dtype=k.dtype,
        )
        v_buffer = torch.zeros(
            num_blocks,
            _NUM_KV_HEADS,
            _BLOCK_SIZE_5D // x,
            _HEAD_DIM,
            x,
            device=_DEVICE,
            dtype=v.dtype,
        )
        slots = torch.randperm(num_blocks * _BLOCK_SIZE_5D, device=_DEVICE)[:num_kv]
        launch_reshape_and_cache_shuffle_5d(k, v, k_buffer, v_buffer, slots)

        backend = self._make_backend(k_buffer, v_buffer)
        backend.kv_cache_is_vectorized_5d = True
        backend.kv_cache_dtype = torch.bfloat16
        pool = backend.token_to_kv_pool
        pool.k_buffer, pool.v_buffer = [k_buffer], [v_buffer]
        pool.store_dtype = pool.dtype = torch.bfloat16
        backend.forward_metadata = SimpleNamespace(
            qo_indptr=torch.tensor(qo_offsets, dtype=torch.int32, device=_DEVICE),
            kv_indptr=torch.tensor(kv_offsets, dtype=torch.int32, device=_DEVICE),
            kv_indices=slots.to(torch.int32),
            swa_page_table=None,
            paged_kv_view=None,
            max_q_len=max(extend_lens),
            max_kv_len=max(seq_lens),
        )
        forward_batch = SimpleNamespace(
            batch_size=batch,
            forward_mode=ForwardMode.EXTEND,
            attn_attend_prefix_cache=False,
            extend_prefix_lens_cpu=prefix_lens,
            seq_lens_sum=num_kv,
            out_cache_loc=None,
        )
        new_tokens = torch.cat(
            [
                torch.arange(s + p, e, device=_DEVICE)
                for s, e, p in zip(kv_offsets, kv_offsets[1:], prefix_lens)
            ]
        )
        layer = self._make_layer(None, None, scaling=1.0 / _HEAD_DIM)

        out = backend.forward_extend(
            q.view(num_q, -1),
            k[new_tokens],
            v[new_tokens],
            layer,
            forward_batch,
            save_kv_cache=False,
        )

        expected = torch.cat(
            [
                _reference_attention(q[qs:qe], k[ks:ke], v[ks:ke], True, layer.scaling)
                for qs, qe, ks, ke in zip(
                    qo_offsets, qo_offsets[1:], kv_offsets, kv_offsets[1:]
                )
            ]
        )
        self._assert_close(out, expected)

    @unittest.skipUnless(
        _RUNNABLE and is_gfx95_supported(), "the head-dim-256 FP8 prefill is gfx950's"
    )
    def test_fresh_fp8_prefill_hd256_reads_what_the_cache_write_stored(self):
        for layout in ("nhd", "vectorized_5d"):
            with self.subTest(layout=layout):
                self._check_fresh_fp8_prefill_hd256(layout)

    def _check_fresh_fp8_prefill_hd256(self, layout):
        head_dim, page_size = 256, _BLOCK_SIZE_5D
        extend_lens = [300, 77, 512]
        batch, offsets = len(extend_lens), _offsets(extend_lens)
        num_tokens = offsets[-1]
        q = 4 * torch.randn(
            num_tokens, _NUM_Q_HEADS, head_dim, device=_DEVICE, dtype=torch.bfloat16
        )
        k = 0.5 * torch.randn(
            num_tokens, _NUM_KV_HEADS, head_dim, device=_DEVICE, dtype=torch.bfloat16
        )
        v = torch.randn_like(k)
        k_scale, v_scale = (
            (x.abs().float().amax() / torch.finfo(fp8_dtype).max).view(1)
            for x in (k, v)
        )
        num_slots = math.ceil(num_tokens / page_size) * page_size
        # The AITER pool takes its layout from the environment; only that path
        # sizes the SHUFFLE 5D buffers.
        with (
            patch("sglang.srt.mem_cache.memory_pool._use_aiter", True),
            envs.SGLANG_AITER_KV_CACHE_LAYOUT.override(layout),
        ):
            pool = MHATokenToKVPool(
                size=num_slots,
                page_size=page_size,
                dtype=fp8_dtype,
                head_num=_NUM_KV_HEADS,
                head_dim=head_dim,
                layer_num=1,
                device=_DEVICE,
                enable_memory_saver=False,
            )
        backend = self._make_backend(None, None)
        backend.token_to_kv_pool = pool
        backend.kv_cache_is_vectorized_5d = layout == "vectorized_5d"
        backend.use_sliding_window_kv_pool = False
        backend.page_size = page_size
        indptr = torch.tensor(offsets, dtype=torch.int32, device=_DEVICE)
        slots = page_size + torch.randperm(num_slots, device=_DEVICE)[:num_tokens]
        backend.forward_metadata = SimpleNamespace(
            qo_indptr=indptr,
            kv_indptr=indptr,
            kv_indices=slots.to(torch.int32),
            swa_page_table=None,
            swa_out_cache_loc=None,
            paged_kv_view=None,
            max_q_len=max(extend_lens),
            max_kv_len=max(extend_lens),
        )
        forward_batch = SimpleNamespace(
            batch_size=batch,
            forward_mode=ForwardMode.EXTEND,
            attn_attend_prefix_cache=False,
            extend_prefix_lens_cpu=[0] * batch,
            seq_lens_sum=num_tokens,
            out_cache_loc=slots,
            out_cache_loc_is_physical=True,
        )
        layer = self._make_layer(k_scale, v_scale, head_dim**-0.5, head_dim)
        # Before the forward: a cache write may divide k and v in place.
        expected = torch.cat(
            [
                _reference_attention(q[s:e], k[s:e], v[s:e], True, layer.scaling)
                for s, e in zip(offsets, offsets[1:])
            ]
        )

        out = backend.forward_extend(
            q.view(num_tokens, -1), k, v, layer, forward_batch, save_kv_cache=True
        )
        self._assert_close(out, expected)

    def test_fp8_prefill_with_a_prefix_vectorized_5d(self):
        # With a cached prefix the SHUFFLE 5D prefill gathers every request's
        # K and V from the pool and runs on fp8 q, K and V.
        self._check_fp8_prefill_from_the_pool("vectorized_5d", _HEAD_DIM)

    @unittest.skipUnless(
        _RUNNABLE and is_gfx95_supported(), "the head-dim-256 FP8 prefill is gfx950's"
    )
    def test_fp8_context_chunk_prefill_hd256(self):
        # A later chunk of a head-dim-256 prompt gathers the NHD cache for the
        # ASM varlen kernel.
        self._check_fp8_prefill_from_the_pool("nhd", 256)

    def _check_fp8_prefill_from_the_pool(self, layout, head_dim):
        prefix_lens, extend_lens = [128, 0, 1000], [300, 77, 512]
        batch = len(extend_lens)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        qo_offsets, kv_offsets = _offsets(extend_lens), _offsets(seq_lens)
        num_q, num_kv = qo_offsets[-1], kv_offsets[-1]
        q = 4 * torch.randn(
            num_q, _NUM_Q_HEADS, head_dim, device=_DEVICE, dtype=torch.bfloat16
        )
        k = 0.5 * torch.randn(
            num_kv, _NUM_KV_HEADS, head_dim, device=_DEVICE, dtype=torch.bfloat16
        )
        v = torch.randn_like(k)
        k_scale, v_scale = (
            (x.abs().float().amax() / torch.finfo(fp8_dtype).max).view(1)
            for x in (k, v)
        )
        layer = self._make_layer(k_scale, v_scale, scaling=head_dim**-0.5)
        layer.qk_head_dim = layer.v_head_dim = layer.head_dim = head_dim
        # Before the pool's write, which may divide k and v in place.
        expected = torch.cat(
            [
                _reference_attention(q[qs:qe], k[ks:ke], v[ks:ke], True, layer.scaling)
                for qs, qe, ks, ke in zip(
                    qo_offsets, qo_offsets[1:], kv_offsets, kv_offsets[1:]
                )
            ]
        )
        new_tokens = torch.cat(
            [
                torch.arange(s + p, e, device=_DEVICE)
                for s, e, p in zip(kv_offsets, kv_offsets[1:], prefix_lens)
            ]
        )
        k_new, v_new = k[new_tokens], v[new_tokens]

        page_size = _BLOCK_SIZE_5D
        num_slots = math.ceil(num_kv / page_size) * page_size
        # The AITER pool takes its layout from the environment; only that path
        # sizes the SHUFFLE 5D buffers.
        with (
            patch("sglang.srt.mem_cache.memory_pool._use_aiter", True),
            envs.SGLANG_AITER_KV_CACHE_LAYOUT.override(layout),
        ):
            pool = MHATokenToKVPool(
                size=num_slots,
                page_size=page_size,
                dtype=fp8_dtype,
                head_num=_NUM_KV_HEADS,
                head_dim=head_dim,
                layer_num=1,
                device=_DEVICE,
                enable_memory_saver=False,
            )
        slots = page_size + torch.randperm(num_slots, device=_DEVICE)[:num_kv]
        pool.set_kv_buffer(layer, slots, k, v, k_scale, v_scale)

        backend = self._make_backend(None, None)
        backend.token_to_kv_pool = pool
        backend.kv_cache_is_vectorized_5d = layout == "vectorized_5d"
        backend.use_sliding_window_kv_pool = False
        backend.page_size = page_size
        backend.forward_metadata = SimpleNamespace(
            qo_indptr=torch.tensor(qo_offsets, dtype=torch.int32, device=_DEVICE),
            kv_indptr=torch.tensor(kv_offsets, dtype=torch.int32, device=_DEVICE),
            kv_indices=slots.to(torch.int32),
            swa_page_table=None,
            swa_out_cache_loc=None,
            paged_kv_view=None,
            asm_ctx_ready=False,
            max_q_len=max(extend_lens),
            max_kv_len=max(seq_lens),
        )
        forward_batch = SimpleNamespace(
            batch_size=batch,
            forward_mode=ForwardMode.EXTEND,
            attn_attend_prefix_cache=False,
            extend_prefix_lens_cpu=prefix_lens,
            seq_lens_sum=num_kv,
            seq_lens_cpu=torch.tensor(seq_lens),
            seq_lens=torch.tensor(seq_lens, device=_DEVICE),
            spec_info=None,
            out_cache_loc=None,
        )

        out = backend.forward_extend(
            q.view(num_q, -1), k_new, v_new, layer, forward_batch, save_kv_cache=False
        )
        self._assert_close(out, expected)

    def test_extend_attention_target_verify(self):
        prefix_lens = [300, 77, 512]
        num_draft = 4
        batch = len(prefix_lens)
        offsets = _offsets(p + num_draft for p in prefix_lens)
        q, ((k_fp8, k_deq, k_scale), (v_fp8, v_deq, v_scale)) = self._make_inputs(
            batch * num_draft, offsets[-1]
        )
        # Target verify reads the committed prefix from the cache and the draft
        # tokens from k / v.
        prefix_loc = torch.cat(
            [
                torch.arange(s, s + p, dtype=torch.int32, device=_DEVICE)
                for s, p in zip(offsets, prefix_lens)
            ]
        )
        draft_loc = torch.cat(
            [torch.arange(e - num_draft, e, device=_DEVICE) for e in offsets[1:]]
        )

        backend = self._make_backend(k_fp8, v_fp8)
        backend.forward_metadata = SimpleNamespace(
            qo_indptr=torch.arange(
                0,
                (batch + 1) * num_draft,
                num_draft,
                dtype=torch.int32,
                device=_DEVICE,
            ),
            kv_indptr=torch.tensor(
                _offsets(prefix_lens), dtype=torch.int32, device=_DEVICE
            ),
            kv_indices=prefix_loc,
            custom_mask=None,
            mask_indptr=None,
            max_extend_len=num_draft,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            attn_attend_prefix_cache=False,
            out_cache_loc=None,
        )

        out = backend.forward_extend(
            q,
            k_deq[draft_loc].bfloat16(),
            v_deq[draft_loc].bfloat16(),
            self._make_layer(k_scale, v_scale),
            forward_batch,
            save_kv_cache=False,
        )

        expected = torch.cat(
            [
                _reference_attention(
                    q[b * num_draft : (b + 1) * num_draft],
                    k_deq[s:e],
                    v_deq[s:e],
                    True,
                    _HEAD_DIM**-0.5,
                )
                for b, (s, e) in enumerate(zip(offsets, offsets[1:]))
            ]
        )
        self._assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()
