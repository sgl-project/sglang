"""Scheduling tests for DeepSeek-V4 CP multi-stream prepare."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.attention.dsv4.compressor as dsv4_compressor
import sglang.srt.layers.attention.dsv4.compressor_v2 as dsv4_compressor_v2
import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeStream:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait_stream(self, other):
        self.log.append(f"wait:{self.name}:{other.name}")

    def wait_event(self, event):
        self.log.append(f"wait_event:{self.name}:{event}")

    def record_event(self):
        event = f"event:{self.name}:{len(self.log)}"
        self.log.append(f"record:{event}")
        return event


class _FakeCompressor:
    compute_kv_score = dsv4_compressor.Compressor.compute_kv_score

    def __init__(self, name, value, log):
        self.name = name
        self.value = value
        self.log = log

    def _compute_wkv_gate(self, x):
        self.log.append(f"project:{self.name}")
        return torch.full((x.shape[0], 1), self.value)


class _FakeIndexer:
    def __init__(self, compressor, log):
        self.compressor = compressor
        self.log = log
        self.precomputed_kv_score = None

    def __call__(self, **kwargs):
        self.log.append("consume:indexer")
        self.precomputed_kv_score = kwargs["precomputed_kv_score"]


class _FakeBackend:
    def __init__(self, log):
        self.log = log
        self.swa_k = None
        self.compressor_kv_score = None

    def store_cache(self, *, layer_id, swa_k, forward_batch):
        self.log.append("consume:swa")
        self.swa_k = swa_k

    def forward_core_compressor(
        self,
        x,
        forward_batch,
        layer_id,
        compressor,
        precomputed_kv_score=None,
    ):
        self.log.append("consume:core")
        self.compressor_kv_score = precomputed_kv_score


class _FakeLayer:
    def __init__(self, log):
        self.log = log
        self.alt_streams = [
            _FakeStream("kv", log),
            _FakeStream("core", log),
            _FakeStream("indexer", log),
        ]
        self.fuse_wqa_wkv = False
        self.dsa_enable_prefill_cp = True
        self.layer_id = 7
        self.indexer = _FakeIndexer(_FakeCompressor("indexer", 2, log), log)
        self.compressor = _FakeCompressor("core", 3, log)
        self.used_fused_kv_store = False

    def _compute_q_a(self, x, qkv_a=None):
        self.log.append("compute:q_a")
        return x

    def _compute_q_b(self, q_lora, positions, q_out=None):
        self.log.append("compute:q_b")
        return q_lora

    def _compute_kv_bf16(self, x, positions, qkv_a=None):
        self.log.append("compute:swa")
        return torch.full((x.shape[0], 1), 1.0)

    def _compute_kv_to_cache(
        self, x, positions, forward_batch, attn_backend, qkv_a=None
    ):
        self.log.append("consume:fused_swa")
        self.used_fused_kv_store = True


class TestDeepseekV4CPMultiStream(CustomTestCase):
    def _run_prepare(self, use_cp):
        log = []
        main_stream = _FakeStream("main", log)
        active_stream = main_stream

        @contextmanager
        def use_stream(stream):
            nonlocal active_stream
            previous = active_stream
            active_stream = stream
            try:
                yield
            finally:
                active_stream = previous

        def current_stream():
            return active_stream

        def gather(value, forward_batch, stream=None):
            name = {1: "swa", 2: "indexer", 3: "core"}[int(value[0, 0])]
            log.append(f"gather:{name}:{current_stream().name}")
            return value + 10

        layer = _FakeLayer(log)
        backend = _FakeBackend(log)
        forward_batch = SimpleNamespace()
        x = torch.ones(2, 1)
        positions = torch.arange(2)
        with (
            patch.object(torch.cuda, "current_stream", side_effect=current_stream),
            patch.object(torch.cuda, "stream", side_effect=use_stream),
            patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=use_cp),
            patch.object(dsv4_compressor, "dsa_use_prefill_cp", return_value=use_cp),
            patch.object(
                deepseek_v4, "cp_materialize_global_token_order", side_effect=gather
            ),
            patch.object(
                dsv4_compressor,
                "cp_materialize_global_token_order",
                side_effect=gather,
            ),
        ):
            output = deepseek_v4.MQALayer._forward_prepare_multi_stream(
                layer, x, positions, forward_batch, backend
            )
        return log, layer, backend, output

    def test_cp_collectives_finish_before_worker_stream_consumers(self):
        log, layer, backend, output = self._run_prepare(use_cp=True)

        gather_events = [event for event in log if event.startswith("gather:")]
        self.assertEqual(
            gather_events,
            ["gather:swa:main", "gather:indexer:main", "gather:core:main"],
        )
        last_gather = max(log.index(event) for event in gather_events)
        first_consumer = min(
            log.index(event)
            for event in ("consume:indexer", "consume:swa", "consume:core")
        )
        self.assertLess(last_gather, first_consumer)
        self.assertFalse(layer.used_fused_kv_store)
        torch.testing.assert_close(backend.swa_k, torch.full((2, 1), 11.0))
        torch.testing.assert_close(
            layer.indexer.precomputed_kv_score, torch.full((2, 1), 12.0)
        )
        torch.testing.assert_close(
            backend.compressor_kv_score, torch.full((2, 1), 13.0)
        )
        torch.testing.assert_close(output, torch.ones(2, 1))

    def test_non_cp_keeps_fused_swa_store_and_compressor_owned_scores(self):
        log, layer, backend, _ = self._run_prepare(use_cp=False)

        self.assertFalse(any(event.startswith("gather:") for event in log))
        self.assertTrue(layer.used_fused_kv_store)
        self.assertIsNone(backend.swa_k)
        self.assertIsNone(layer.indexer.precomputed_kv_score)
        self.assertIsNone(backend.compressor_kv_score)

    def test_unified_compressor_consumes_precomputed_score(self):
        precomputed = torch.ones(2, 4)

        class TokenPool:
            uniform_fp8 = False

            @staticmethod
            def get_index_k_page_size(ratio):
                return 64

            @staticmethod
            def get_index_k_with_scale_buffer(layer_id):
                return torch.empty(1, dtype=torch.uint8)

        class Backend:
            token_to_kv_pool = TokenPool()
            enable_deepseek_v4_fp4_indexer = False

            def __init__(self):
                self.received_kv_score = None

            @staticmethod
            def _get_out_loc(ratio):
                return torch.zeros(1, dtype=torch.int32)

            def _forward_compress_all_in_one(self, **kwargs):
                self.received_kv_score = kwargs["kv_score_input"]

        compressor = SimpleNamespace(
            is_in_indexer=True,
            ratio=4,
            head_dim=128,
            rotate=True,
            ape=torch.empty(4, 256),
            norm=SimpleNamespace(weight=torch.ones(128), variance_epsilon=1e-6),
            freqs_cis=torch.empty(1),
            get_state_pool=lambda backend: SimpleNamespace(
                kv_score_buffer=SimpleNamespace(kv_score=torch.empty(1))
            ),
            compute_kv_score=lambda *args: self.fail(
                "precomputed score must bypass compute_kv_score"
            ),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False)
        )
        backend = Backend()

        dsv4_compressor_v2.CompressorBackendMixin.forward_unified(
            backend,
            torch.empty(2, 1),
            forward_batch,
            layer_id=0,
            compressor=compressor,
            precomputed_kv_score=precomputed,
        )
        self.assertIs(backend.received_kv_score, precomputed)


if __name__ == "__main__":
    unittest.main()
