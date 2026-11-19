"""TRT-LLM MHA offers a draft worker's FP8 attention calls to a provider attention.

``set_draft_attention`` is the seam: without a provider the backend runs as before.
With one, a draft decode or an even draft extend on the fused FP8 KV path writes its
K / V rows and FP8 query first, hands the provider a ``DraftAttentionView``, and
attends over the written pool itself when the provider returns None.

CPU-only.

    python -m pytest test/registered/unit/layers/attention/test_trtllm_mha_draft_attention.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.plugin_kernels.nvidia.spec import DraftAttentionView
from sglang.srt.layers.attention import trtllm_mha_backend
from sglang.srt.layers.attention.trtllm_mha_backend import (
    TRTLLMHAAttnBackend,
    TRTLLMMHAMetadata,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEADS, HEAD_DIM = 2, 4


def _backend(*, draft=True, kv_dtype=torch.float8_e4m3fn):
    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    backend.is_draft_runner = draft
    backend.data_type = kv_dtype
    backend.q_data_type = torch.bfloat16
    backend.is_nvfp4_kvcache = False
    backend.is_xqa_impl = False
    backend.decode_uses_native_fp4 = False
    backend.page_size = 64
    backend._draft_attention = None
    backend.k_pool = torch.zeros(128, 1, HEAD_DIM)
    backend.v_pool = torch.ones(128, 1, HEAD_DIM)
    pools = SimpleNamespace(kv=lambda: (backend.k_pool, backend.v_pool))
    backend.token_to_kv_pool = SimpleNamespace(
        layer_router=SimpleNamespace(read=lambda layer_id: pools)
    )
    backend.forward_metadata = TRTLLMMHAMetadata(
        cache_seqlens_int32=torch.tensor([9, 17], dtype=torch.int32),
        max_seq_len_q=1,
    )
    return backend


def _layer(**fields):
    base = dict(
        layer_id=3,
        tp_q_head_num=HEADS,
        head_dim=HEAD_DIM,
        attn_type=AttentionType.DECODER,
        k_scale=None,
        v_scale=None,
        k_scale_float=0.5,
        v_scale_float=2.0,
        sliding_window_size=-1,
        scaling=1.0,
    )
    base.update(fields)
    return SimpleNamespace(**base)


def _batch(mode=ForwardMode.DECODE):
    return SimpleNamespace(forward_mode=mode)


def _qkv(rows):
    q = torch.randn(rows, HEADS * HEAD_DIM)
    k = torch.randn(rows, HEAD_DIM)
    v = torch.randn(rows, HEAD_DIM)
    return q, k, v


class _Stock:
    """The stock path's two launches on this backend, recorded."""

    def __init__(self, backend):
        self.writes, self.decodes = [], []
        self.page_table = torch.arange(4, dtype=torch.int32).view(2, 2)

        def write(q, k, v, layer, forward_batch):
            self.writes.append((q, k, v))
            return torch.full(q.shape, 7.0).to(torch.float8_e4m3fn)

        def decode(q, kv_cache, page_table, cache_seqlens, **kwargs):
            self.decodes.append(q)
            return torch.full((q.shape[0], HEADS, HEAD_DIM), 3.0)

        self.patches = [
            patch.object(backend, "_fused_fp8_qkv_kv_cache", side_effect=write),
            patch.object(backend, "_run_fixed_q_len_decode", side_effect=decode),
            patch.object(
                backend, "_get_layer_page_table", return_value=self.page_table
            ),
            patch.object(backend, "_reshape_paged_kv_cache", return_value=None),
            patch.object(backend, "_get_bmm_scales", return_value=(1.0, 1.0)),
            patch.object(
                trtllm_mha_backend, "is_cp_active", side_effect=lambda fb: False
            ),
        ]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()


class TestDraftAttentionSeam(CustomTestCase):
    def test_only_a_draft_instance_on_an_fp8_kv_cache_offers_its_calls(self):
        attention = lambda view: None
        cases = {
            "draft fp8": (_backend(), True),
            "target": (_backend(draft=False), False),
            "bf16 kv": (_backend(kv_dtype=torch.bfloat16), False),
        }
        nvfp4 = _backend()
        nvfp4.is_nvfp4_kvcache = True
        xqa = _backend()
        xqa.is_xqa_impl = True
        cases.update({"nvfp4 kv": (nvfp4, False), "xqa": (xqa, False)})
        for name, (backend, offers) in cases.items():
            with self.subTest(name):
                self.assertIs(backend.set_draft_attention(attention), offers)
                self.assertIs(backend._draft_attention, attention if offers else None)
        backend = _backend()
        backend.set_draft_attention(attention)
        self.assertFalse(backend.set_draft_attention(None))
        self.assertIsNone(backend._draft_attention)

    def test_taken_decode_writes_first_and_returns_the_provider_rows(self):
        backend, layer = _backend(), _layer()
        views = []
        out = torch.full((2, HEADS * HEAD_DIM), 5.0, dtype=torch.bfloat16)

        def attention(view):
            views.append(view)
            return out

        backend.set_draft_attention(attention)
        q, k, v = _qkv(2)
        batch = _batch()
        with _Stock(backend) as stock:
            o = backend.forward_decode(q, k, v, layer, batch)
        self.assertIs(o, out)
        self.assertEqual(len(stock.writes), 1)
        self.assertIs(stock.writes[0][0], q)
        self.assertEqual(stock.decodes, [])
        (view,) = views
        self.assertIsInstance(view, DraftAttentionView)
        self.assertEqual(view.q.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(view.q.shape), (2, HEADS * HEAD_DIM))
        self.assertIs(view.layer, layer)
        self.assertIs(view.k_pool, backend.k_pool)
        self.assertIs(view.v_pool, backend.v_pool)
        self.assertEqual((view.k_scale, view.v_scale), (0.5, 2.0))
        self.assertEqual(view.page_size, 64)
        self.assertIs(view.page_table, stock.page_table)
        self.assertIs(view.cache_seqlens, backend.forward_metadata.cache_seqlens_int32)
        self.assertEqual(view.q_len, 1)
        self.assertEqual(view.out_dtype, torch.bfloat16)
        self.assertIs(view.step, batch)

    def test_declined_call_attends_over_the_written_pool(self):
        """The decline costs no second K / V write; the stock kernel reads the FP8
        query the write produced, as the provider-less fused path does."""
        backend, layer = _backend(), _layer()
        backend.set_draft_attention(lambda view: None)
        q, k, v = _qkv(2)
        with _Stock(backend) as stock:
            o = backend.forward_decode(q, k, v, layer, _batch())
        self.assertEqual(len(stock.writes), 1)
        (q8,) = stock.decodes
        self.assertEqual(q8.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(o.shape), (2, HEADS * HEAD_DIM))

        plain = _backend()
        with _Stock(plain) as stock_plain:
            o_plain = plain.forward_decode(q, k, v, layer, _batch())
        self.assertEqual(len(stock_plain.writes), 1)
        self.assertTrue(torch.equal(stock_plain.decodes[0].float(), q8.float()))
        self.assertTrue(torch.equal(o_plain, o))

    def test_calls_the_provider_cannot_take_keep_the_stock_path(self):
        q, k, v = _qkv(2)
        cases = {
            "no K / V write": dict(save_kv_cache=False),
            "no K rows": dict(k=None),
            "no V rows": dict(v=None),
            "attention sinks": dict(sinks=torch.zeros(HEADS)),
            "encoder-only layer": dict(
                layer=_layer(attn_type=AttentionType.ENCODER_ONLY)
            ),
            "context parallelism": dict(cp=True),
            "no forward batch": dict(batch=None),
            "autotune dummy run": dict(autotune=True),
        }
        for name, case in cases.items():
            with (
                self.subTest(name),
                patch.object(
                    trtllm_mha_backend,
                    "is_cp_active",
                    side_effect=lambda fb: case.get("cp", False),
                ),
                patch(
                    "sglang.srt.layers.logits_processor.get_in_autotune_dummy_run",
                    return_value=case.get("autotune", False),
                ),
            ):
                got = _backend()._draft_attention_q_len(
                    case.get("k", k),
                    case.get("v", v),
                    case.get("layer", _layer()),
                    case.get("batch", _batch()),
                    case.get("save_kv_cache", True),
                    case.get("sinks"),
                    decode=True,
                )
                self.assertEqual(got, 0)

        backend = _backend()
        calls = []
        backend.set_draft_attention(calls.append)
        with _Stock(backend) as stock:
            backend.forward_decode(q, k, v, _layer(), _batch(), save_kv_cache=False)
        self.assertEqual(calls, [])
        self.assertEqual(stock.writes, [])
        self.assertEqual(len(stock.decodes), 1)

    def test_draft_extend_offers_its_rows_a_request_and_nothing_else(self):
        backend = _backend()
        backend.forward_metadata.max_seq_len_q = 4
        q, k, v = _qkv(8)
        cases = {
            "draft extend": (ForwardMode.DRAFT_EXTEND_V2, False, 4),
            "ragged draft extend": (ForwardMode.DRAFT_EXTEND_V2, True, 0),
            "target verify": (ForwardMode.TARGET_VERIFY, False, 0),
            "extend": (ForwardMode.EXTEND, False, 0),
        }
        for name, (mode, ragged, q_len) in cases.items():
            with self.subTest(name):
                backend.forward_metadata.is_ragged_verify = ragged
                with patch.object(
                    trtllm_mha_backend, "is_cp_active", side_effect=lambda fb: False
                ):
                    got = backend._draft_attention_q_len(
                        k, v, _layer(), _batch(mode), True, None, decode=False
                    )
                self.assertEqual(got, q_len)


if __name__ == "__main__":
    unittest.main()
