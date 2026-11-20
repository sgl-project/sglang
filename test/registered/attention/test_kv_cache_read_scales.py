"""Attention backends apply a layer's KV-cache scales only to a cache that holds
K/V divided by them.

A checkpoint's FP8 KV-cache scales stay on the attention layer whatever
``--kv-cache-dtype`` is. The KV pool divides K/V by them only when it casts them
to another dtype (e.g. FP8), so a bf16 cache holds them unscaled. Each case runs
one backend's decode with its real kernel and compares the output with fp32
attention over the K/V the cache represents. The scales are kept far from 1, as
in calibrated checkpoints, so applying them to a bf16 cache cannot pass.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_DEVICE = "cuda"
_NUM_Q_HEADS, _NUM_KV_HEADS, _HEAD_DIM = 8, 2, 128
_K_SCALE, _V_SCALE = 0.05, 0.02
_SEQ_LENS = [37, 300, 129]
_CACHE_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)


class _FakeKVPool:
    def __init__(self, k_cache, v_cache):
        self.k_cache, self.v_cache = k_cache, v_cache
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


def _layer():
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=_NUM_Q_HEADS,
        tp_k_head_num=_NUM_KV_HEADS,
        tp_v_head_num=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        qk_head_dim=_HEAD_DIM,
        v_head_dim=_HEAD_DIM,
        scaling=_HEAD_DIM**-0.5,
        logit_cap=0.0,
        logit_capping_method="tanh",
        sliding_window_size=-1,
        is_cross_attention=False,
        xai_temperature_len=-1,
        k_scale=torch.tensor(_K_SCALE, device=_DEVICE),
        v_scale=torch.tensor(_V_SCALE, device=_DEVICE),
        k_scale_float=_K_SCALE,
        v_scale_float=_V_SCALE,
    )


def _inputs(cache_dtype):
    """q, kv indptr, the cache as the pool stores it, and the K/V it represents."""
    num_tokens = sum(_SEQ_LENS)
    q = torch.randn(
        len(_SEQ_LENS), _NUM_Q_HEADS, _HEAD_DIM, device=_DEVICE, dtype=torch.bfloat16
    )
    k = torch.randn(
        num_tokens, _NUM_KV_HEADS, _HEAD_DIM, device=_DEVICE, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    indptr = torch.tensor(
        [0, *torch.tensor(_SEQ_LENS).cumsum(0).tolist()],
        dtype=torch.int32,
        device=_DEVICE,
    )
    if cache_dtype == torch.bfloat16:
        return q, indptr, (k, v), (k.float(), v.float())
    k_fp8 = (k.float() / _K_SCALE).to(cache_dtype)
    v_fp8 = (v.float() / _V_SCALE).to(cache_dtype)
    return (
        q,
        indptr,
        (k_fp8, v_fp8),
        (k_fp8.float() * _K_SCALE, v_fp8.float() * _V_SCALE),
    )


def _reference(q, indptr, k, v):
    group = _NUM_Q_HEADS // _NUM_KV_HEADS
    out = []
    for b, (s, e) in enumerate(zip(indptr.tolist(), indptr.tolist()[1:])):
        kb = k[s:e].repeat_interleave(group, dim=1)
        vb = v[s:e].repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,khd->hk", q[b].float(), kb) * _HEAD_DIM**-0.5
        out.append(torch.einsum("hk,khd->hd", torch.softmax(scores, dim=-1), vb))
    return torch.stack(out)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestKvCacheReadScales(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def assert_matches(self, out, expected):
        out = out.float().view_as(expected)
        rel_err = ((out - expected).norm() / expected.norm()).item()
        # bf16/fp8 rounding stays within a few percent; applying the scales to an
        # unscaled cache, or skipping them on an FP8 one, is off by far more.
        self.assertLess(rel_err, 0.1, f"relative L2 error {rel_err:.3f}")

    def test_flashinfer_decode(self):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )

        for cache_dtype in _CACHE_DTYPES:
            with self.subTest(cache_dtype=cache_dtype):
                q, indptr, (k_cache, v_cache), (k, v) = _inputs(cache_dtype)
                wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=_DEVICE),
                    "NHD",
                )
                wrapper.plan(
                    indptr,
                    torch.arange(k_cache.shape[0], dtype=torch.int32, device=_DEVICE),
                    torch.ones(len(_SEQ_LENS), dtype=torch.int32, device=_DEVICE),
                    _NUM_Q_HEADS,
                    _NUM_KV_HEADS,
                    _HEAD_DIM,
                    1,
                    q_data_type=torch.bfloat16,
                    kv_data_type=cache_dtype,
                )
                backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
                backend.forward_metadata = SimpleNamespace(decode_wrappers=[wrapper])
                backend.num_wrappers = 1
                backend.decode_uses_dequant_workspace = False
                backend.flashinfer_kv_cache_dtype = cache_dtype
                backend.token_to_kv_pool = _FakeKVPool(k_cache, v_cache)
                out = backend.forward_decode(
                    q, None, None, _layer(), SimpleNamespace(), save_kv_cache=False
                )
                self.assert_matches(out, _reference(q, indptr, k, v))

    def test_triton_decode(self):
        from sglang.kernels.ops.attention.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        batch, max_kv_splits = len(_SEQ_LENS), 8
        for cache_dtype in _CACHE_DTYPES:
            with self.subTest(cache_dtype=cache_dtype):
                q, indptr, (k_cache, v_cache), (k, v) = _inputs(cache_dtype)
                backend = TritonAttnBackend.__new__(TritonAttnBackend)
                backend.kv_cache_dtype = cache_dtype
                backend.use_mla = False
                backend.dcp_size = 1
                backend.page_size = 1
                backend.use_pdl = False
                backend.enable_deterministic = True
                backend.max_kv_splits = max_kv_splits
                backend.swa_v_head_dim = None
                backend.decode_attention_fwd = decode_attention_fwd
                backend.token_to_kv_pool = _FakeKVPool(k_cache, v_cache)
                backend.forward_metadata = SimpleNamespace(
                    kv_indptr=indptr,
                    kv_indices=torch.arange(
                        k_cache.shape[0], dtype=torch.int64, device=_DEVICE
                    ),
                    attn_logits=torch.empty(
                        batch,
                        _NUM_Q_HEADS,
                        max_kv_splits,
                        _HEAD_DIM,
                        dtype=torch.float32,
                        device=_DEVICE,
                    ),
                    swa_attn_logits=None,
                    attn_lse=torch.empty(
                        batch,
                        _NUM_Q_HEADS,
                        max_kv_splits,
                        dtype=torch.float32,
                        device=_DEVICE,
                    ),
                    num_kv_splits=torch.full(
                        (batch,), max_kv_splits, dtype=torch.int32, device=_DEVICE
                    ),
                    lean_Mp=None,
                    lean_Lp=None,
                    lean_Op=None,
                    lean_locks=None,
                    uneven_batch=None,
                )
                out = backend.forward_decode(
                    q.view(batch, -1),
                    None,
                    None,
                    _layer(),
                    SimpleNamespace(batch_size=batch),
                    save_kv_cache=False,
                )
                self.assert_matches(out, _reference(q, indptr, k, v))


if __name__ == "__main__":
    unittest.main()
