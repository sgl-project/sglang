"""AITER attention paths that read an fp8 KV cache with calibrated KV scales.

Each case stores ``K / k_scale`` and ``V / v_scale`` in an fp8 cache, runs one
``AiterAttnBackend`` path with the real aiter kernel, and compares the output with
fp32 attention over the dequantized cache. With unit scales, mixing up q's scale
with ``k_scale`` or skipping the cache dequantization does not change the result, so
the scales are kept far from 1. The unified-attention decode is covered by
``test_aiter_fp8_q_unified_attention.py``.
"""

import itertools
import math
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

_RUNNABLE = is_hip()
if _RUNNABLE:
    from sglang.kernels.ops.quantization.fp8_kernel import (
        fp8_dtype,
        scaled_fp8_quant,
    )
    from sglang.srt.layers.attention.aiter_backend import (
        _AITER_PARTITION_SIZE_ROCM,
        AiterAttnBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

_DEVICE = "cuda"
_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 2
_HEAD_DIM = 128


class _FakeKVPool:
    def __init__(self, k_cache, v_cache):
        self.k_cache = k_cache
        self.v_cache = v_cache

    def get_kv_buffer(self, _layer_id):
        return self.k_cache, self.v_cache


def _offsets(lens):
    return [0, *itertools.accumulate(lens)]


def _reference_attention(q, k, v, causal):
    """fp32 attention for one request; q is [Lq, Hq, D], k / v are [Lk, Hkv, D]."""
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.ones(scores.shape[1:], dtype=torch.bool, device=q.device).tril()
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
        backend.scale = _HEAD_DIM**-0.5
        backend.logits_soft_cap = 0.0
        backend.q_scale = torch.ones(1, dtype=torch.float32, device=_DEVICE)
        backend.k_scale = torch.ones(1, dtype=torch.float32, device=_DEVICE)
        backend.v_scale = torch.ones(1, dtype=torch.float32, device=_DEVICE)
        backend.token_to_kv_pool = _FakeKVPool(k_cache, v_cache)
        return backend

    def _make_layer(self, k_scale, v_scale):
        return SimpleNamespace(
            layer_id=0,
            tp_q_head_num=_NUM_Q_HEADS,
            tp_k_head_num=_NUM_KV_HEADS,
            tp_v_head_num=_NUM_KV_HEADS,
            qk_head_dim=_HEAD_DIM,
            v_head_dim=_HEAD_DIM,
            head_dim=_HEAD_DIM,
            k_scale=k_scale,
            v_scale=v_scale,
            sliding_window_size=-1,
            scaling=_HEAD_DIM**-0.5,
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
        )

        out = backend.forward_decode(
            q,
            None,
            None,
            self._make_layer(k_scale, v_scale),
            SimpleNamespace(batch_size=batch),
            save_kv_cache=False,
        )

        expected = torch.cat(
            [
                _reference_attention(q[b : b + 1], k_deq[s:e], v_deq[s:e], False)
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

        # Keep gfx950 off the HD128 ASM prefill so the NHD path is exercised on
        # every architecture.
        with envs.SGLANG_AITER_ASM_PREFILL_HD128.override(False):
            out = backend.forward_extend(
                q_strided,
                None,
                None,
                self._make_layer(k_scale, v_scale),
                forward_batch,
                save_kv_cache=False,
            )

        expected = torch.cat(
            [
                _reference_attention(q[s:e], k_deq[s:e], v_deq[s:e], True)
                for s, e in zip(offsets, offsets[1:])
            ]
        )
        self._assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()
