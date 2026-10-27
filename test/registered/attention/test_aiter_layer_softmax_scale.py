"""AITER MHA paths must apply the layer's softmax scale, not head_dim**-0.5.

``RadixAttention.scaling`` differs from ``head_dim**-0.5`` for some models, e.g.
Granite passes ``attention_multiplier`` (1 / head_dim). Each case runs one
``AiterAttnBackend`` path with a bf16 KV cache and the real aiter kernel, and
compares the output with fp32 attention at ``layer.scaling``.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

_RUNNABLE = is_hip()
if _RUNNABLE:
    from sglang.srt.layers.attention.aiter_backend import (
        _AITER_PARTITION_SIZE_ROCM,
        AiterAttnBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

_DEVICE = "cuda"
_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 2
# Granite 3.x 2B and 8B.
_HEAD_DIMS = (64, 128)


class _FakeKVPool:
    def __init__(self, k_cache, v_cache):
        self.k_cache = k_cache
        self.v_cache = v_cache

    def get_kv_buffer(self, _layer_id):
        return self.k_cache, self.v_cache


def _reference_attention(q, k, v, scaling, causal):
    """fp32 attention for one request; q is [Lq, Hq, D], k / v are [Lk, Hkv, D]."""
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
class TestAiterLayerSoftmaxScale(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _make_qkv(self, num_q, num_kv, head_dim):
        q = 4 * torch.randn(
            num_q, _NUM_Q_HEADS, head_dim, device=_DEVICE, dtype=torch.bfloat16
        )
        k = torch.randn(
            num_kv, _NUM_KV_HEADS, head_dim, device=_DEVICE, dtype=torch.bfloat16
        )
        return q, k, torch.randn_like(k)

    def _make_backend(self, k_cache, v_cache, head_dim):
        backend = object.__new__(AiterAttnBackend)
        backend.use_mla = False
        backend.kv_cache_is_vectorized_5d = False
        backend.use_triton_unified_attention = False
        backend._use_unified_verify = False
        backend.dcp_world_size = 1
        backend.kv_cache_dtype = torch.bfloat16
        backend.input_dtype = torch.bfloat16
        backend.page_size = 1
        # What AiterAttnBackend.__init__ derives from the model-level head_dim.
        backend.scale = head_dim**-0.5
        backend.logits_soft_cap = 0.0
        backend.k_scale = torch.ones(1, dtype=torch.float32, device=_DEVICE)
        backend.v_scale = torch.ones(1, dtype=torch.float32, device=_DEVICE)
        backend.token_to_kv_pool = _FakeKVPool(k_cache, v_cache)
        return backend

    def _make_layer(self, head_dim):
        return SimpleNamespace(
            layer_id=0,
            tp_q_head_num=_NUM_Q_HEADS,
            tp_k_head_num=_NUM_KV_HEADS,
            tp_v_head_num=_NUM_KV_HEADS,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            head_dim=head_dim,
            k_scale=None,
            v_scale=None,
            sliding_window_size=-1,
            scaling=1.0 / head_dim,
            logit_cap=0.0,
            is_cross_attention=False,
        )

    def _assert_close_to_reference(self, actual, expected):
        actual = actual.float().flatten()
        expected = expected.float().flatten()
        self.assertTrue(bool(torch.isfinite(actual).all()))
        rel_err = ((actual - expected).norm() / expected.norm()).item()
        # bf16 rounding leaves under 1%; head_dim**-0.5 in place of 1 / head_dim
        # is off by more than 50%.
        self.assertLess(rel_err, 0.05, f"relative L2 error {rel_err:.4f}")

    def test_paged_attention_ragged_decode(self):
        seq_lens = [384, 1000, 17]
        batch = len(seq_lens)
        kv_indptr = torch.tensor(
            [0] + seq_lens, dtype=torch.int32, device=_DEVICE
        ).cumsum(0, dtype=torch.int32)
        for head_dim in _HEAD_DIMS:
            q, k, v = self._make_qkv(batch, sum(seq_lens), head_dim)
            backend = self._make_backend(k, v, head_dim)
            backend.max_num_partitions = (
                max(seq_lens) + _AITER_PARTITION_SIZE_ROCM - 1
            ) // _AITER_PARTITION_SIZE_ROCM
            backend.workspace_buffer = torch.empty(
                batch * _NUM_Q_HEADS * backend.max_num_partitions * head_dim * 4
                + 2 * batch * _NUM_Q_HEADS * backend.max_num_partitions * 4,
                dtype=torch.uint8,
                device=_DEVICE,
            )
            backend.kv_last_page_len = torch.ones(
                batch, dtype=torch.int32, device=_DEVICE
            )
            backend.forward_metadata = SimpleNamespace(
                kv_indptr=kv_indptr,
                kv_indices=torch.arange(
                    sum(seq_lens), dtype=torch.int32, device=_DEVICE
                ),
            )
            layer = self._make_layer(head_dim)

            out = backend.forward_decode(
                q,
                None,
                None,
                layer,
                SimpleNamespace(batch_size=batch),
                save_kv_cache=False,
            ).view(batch, _NUM_Q_HEADS, head_dim)

            for b in range(batch):
                start, end = kv_indptr[b].item(), kv_indptr[b + 1].item()
                expected = _reference_attention(
                    q[b : b + 1],
                    k[start:end],
                    v[start:end],
                    layer.scaling,
                    causal=False,
                )
                with self.subTest(head_dim=head_dim, request=b):
                    self._assert_close_to_reference(out[b : b + 1], expected)

    def _check_prefill(self, vectorized_5d):
        seq_lens = [300, 77, 512]
        batch = len(seq_lens)
        num_tokens = sum(seq_lens)
        indptr = torch.tensor([0] + seq_lens, dtype=torch.int32, device=_DEVICE).cumsum(
            0, dtype=torch.int32
        )
        for head_dim in _HEAD_DIMS:
            q, k, v = self._make_qkv(num_tokens, num_tokens, head_dim)
            backend = self._make_backend(k, v, head_dim)
            backend.kv_cache_is_vectorized_5d = vectorized_5d
            backend.forward_metadata = SimpleNamespace(
                qo_indptr=indptr,
                kv_indptr=indptr,
                kv_indices=torch.arange(num_tokens, dtype=torch.int32, device=_DEVICE),
                swa_page_table=None,
                paged_kv_view=None,
                max_q_len=max(seq_lens),
                max_kv_len=max(seq_lens),
            )
            layer = self._make_layer(head_dim)
            forward_batch = SimpleNamespace(
                batch_size=batch,
                forward_mode=ForwardMode.EXTEND,
                attn_attend_prefix_cache=False,
                extend_prefix_lens_cpu=[0] * batch,
                out_cache_loc=None,
            )

            out = backend.forward_extend(
                q.view(num_tokens, -1),
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=False,
            ).view(num_tokens, _NUM_Q_HEADS, head_dim)

            for b in range(batch):
                start, end = indptr[b].item(), indptr[b + 1].item()
                expected = _reference_attention(
                    q[start:end],
                    k[start:end],
                    v[start:end],
                    layer.scaling,
                    causal=True,
                )
                with self.subTest(head_dim=head_dim, request=b):
                    self._assert_close_to_reference(out[start:end], expected)

    def test_mha_batch_prefill(self):
        self._check_prefill(vectorized_5d=False)

    def test_mha_batch_prefill_vectorized_5d(self):
        self._check_prefill(vectorized_5d=True)


if __name__ == "__main__":
    unittest.main()
