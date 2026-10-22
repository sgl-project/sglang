"""When the aiter backend sends MLA decode to aiter's gfx1250 Gluon kernel."""

import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import aiter_gluon_mla_decode as gmd
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MLA_MODULE = "aiter.ops.triton.attention.mla"
_KV_LORA_RANK = 512
_ROPE_DIM = 64
_PAGE_SIZE = 64
_HEADS = 8
_SEGMENT_FIX_SRC = "if NUM_SEGMENTS_PER_SEQ > 1:"


def _fake_aiter(mla_decode_fwd, kernel_src=_SEGMENT_FIX_SRC):
    modules = {
        name: types.ModuleType(name)
        for name in (
            "aiter",
            "aiter.ops",
            "aiter.ops.triton",
            "aiter.ops.triton.attention",
        )
    }
    modules[_MLA_MODULE] = types.ModuleType(_MLA_MODULE)
    modules[_MLA_MODULE].mla_decode_fwd = mla_decode_fwd
    modules[_MLA_MODULE].gluon_mla_decode_fwd_kernel_non_pipelined = types.SimpleNamespace(
        src=kernel_src
    )
    return modules


def _backend(page_size=_PAGE_SIZE, dcp_size=1, kv_shape=(1024, 1, 576)):
    kv_buffer = torch.zeros(kv_shape, dtype=torch.bfloat16)
    pool = types.SimpleNamespace(
        start_layer=0,
        get_key_buffer=lambda layer_id: kv_buffer,
    )
    # Request r starts at page r, so every request has its own page table.
    req_to_token = (
        torch.arange(4096, dtype=torch.int32)[None, :]
        + page_size * torch.arange(4, dtype=torch.int32)[:, None]
    ) % kv_shape[0]
    return types.SimpleNamespace(
        use_mla=True,
        page_size=page_size,
        dcp_size=dcp_size,
        enable_deterministic=False,
        sliding_window_size=None,
        use_sliding_window_kv_pool=False,
        token_to_kv_pool=pool,
        req_to_token=req_to_token,
        device="cpu",
    )


def _layer(
    layer_id=0,
    qk_head_dim=_KV_LORA_RANK + _ROPE_DIM,
    sliding_window_size=-1,
    k_scale=None,
):
    return types.SimpleNamespace(
        layer_id=layer_id,
        xai_temperature_len=None,
        sliding_window_size=sliding_window_size,
        qk_head_dim=qk_head_dim,
        v_head_dim=_KV_LORA_RANK,
        tp_q_head_num=_HEADS,
        scaling=1.0,
        k_scale=None if k_scale is None else torch.tensor(k_scale),
        k_scale_float=k_scale,
    )


def _batch(req_pool_indices):
    bs = len(req_pool_indices)
    return types.SimpleNamespace(
        batch_size=bs,
        req_pool_indices=torch.tensor(req_pool_indices),
        seq_lens=torch.full((bs,), 100),
    )


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)


class TestGluonMlaDecodeGate(CustomTestCase):
    def setUp(self):
        self.kernel = _RecordingKernel()
        gmd._mla_decode_fwd.cache_clear()
        self.addCleanup(gmd._mla_decode_fwd.cache_clear)
        for patcher in (
            mock.patch.object(gmd, "is_hip", lambda: True),
            mock.patch.object(gmd, "is_gfx1250_supported", lambda: True),
            mock.patch.dict(sys.modules, _fake_aiter(self.kernel)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _create(self, backend, use_aiter=True):
        gmd._mla_decode_fwd.cache_clear()
        with envs.SGLANG_USE_AITER.override(use_aiter):
            return gmd.GluonMlaDecode.create(backend, _KV_LORA_RANK, _ROPE_DIM)

    def _forward(self, decode, layer, forward_batch):
        bs = forward_batch.batch_size
        q = torch.zeros((bs, _HEADS * layer.qk_head_dim), dtype=torch.bfloat16)
        o = torch.zeros((bs, _HEADS * layer.v_head_dim), dtype=torch.bfloat16)
        decode.forward(q=q, o=o, layer=layer, forward_batch=forward_batch)
        return self.kernel.calls[-1]

    def test_paged_pool_uses_the_kernel(self):
        decode = self._create(_backend())
        self.assertIsNotNone(decode)
        self.assertIs(decode.mla_decode_fwd, self.kernel)

    def test_off_unless_aiter_is_requested(self):
        self.assertIsNone(self._create(_backend(), use_aiter=False))

    def test_aiter_without_the_kernel_declines(self):
        with mock.patch.dict(sys.modules, {_MLA_MODULE: None}):
            self.assertIsNone(self._create(_backend()))

    def test_aiter_without_the_single_segment_fix_declines(self):
        with mock.patch.dict(sys.modules, _fake_aiter(self.kernel, kernel_src="")):
            self.assertIsNone(self._create(_backend()))

    def test_pool_layouts_the_kernel_cannot_read(self):
        for kwargs in (
            dict(page_size=1),
            dict(page_size=48),
            dict(dcp_size=2),
            dict(kv_shape=(1000, 1, 576)),
            dict(kv_shape=(1024, 2, 576)),
        ):
            with self.subTest(**kwargs):
                self.assertIsNone(self._create(_backend(**kwargs)))

    def test_layer_gate(self):
        decode = self._create(_backend())
        self.assertTrue(decode.supports(_layer(), logits_soft_cap=0))
        self.assertFalse(decode.supports(_layer(1), logits_soft_cap=30.0))
        self.assertFalse(decode.supports(_layer(2, sliding_window_size=128), 0))
        self.assertFalse(decode.supports(_layer(3, qk_head_dim=192), 0))

    def test_layer_gate_is_computed_once_per_layer(self):
        decode = self._create(_backend())
        layer = _layer()
        self.assertTrue(decode.supports(layer, 0))
        layer.qk_head_dim = 192
        self.assertTrue(decode.supports(layer, 0))

    def test_bf16_cache_keeps_a_non_unit_k_scale(self):
        """K is divided by k_scale before the write for any cache dtype."""
        decode = self._create(_backend())
        call = self._forward(decode, _layer(k_scale=2.0), _batch([1]))
        self.assertEqual(call["kv_descale"].item(), 2.0)

    def test_bf16_cache_with_unit_k_scale_needs_no_descale(self):
        decode = self._create(_backend())
        for k_scale in (None, 1.0):
            with self.subTest(k_scale=k_scale):
                call = self._forward(decode, _layer(k_scale=k_scale), _batch([1]))
                self.assertIsNone(call["kv_descale"])

    def test_page_table_is_rebuilt_at_the_first_kernel_layer(self):
        """The pool's first layer need not run this kernel; the page table
        must still follow each new batch."""
        backend = _backend()
        decode = self._create(backend)
        first = self._forward(decode, _layer(3), _batch([1]))
        self._forward(decode, _layer(4), _batch([1]))
        second = self._forward(decode, _layer(3), _batch([2]))

        expected = backend.req_to_token[2, ::_PAGE_SIZE] // _PAGE_SIZE
        self.assertFalse(torch.equal(first["block_tables"], second["block_tables"]))
        torch.testing.assert_close(second["block_tables"][0], expected)


if __name__ == "__main__":
    unittest.main()
