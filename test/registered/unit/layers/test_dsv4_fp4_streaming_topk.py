import sys
import types
import unittest
from unittest import mock

from sglang.kernels.ops.attention.dsv4 import fp4_indexer_hip
from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.layers.attention.dsv4 import indexer
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fake_aiter_modules(entrypoint=None, allocator=None, result_type=None):
    aiter = types.ModuleType("aiter")
    ops = types.ModuleType("aiter.ops")
    flydsl = types.ModuleType("aiter.ops.flydsl")
    if entrypoint is not None:
        flydsl.flydsl_pa_mqa_topk_fp4_prefill = entrypoint
    if allocator is not None:
        flydsl.allocate_fp4_prefill_topk_workspace = allocator
    if result_type is not None:
        flydsl.FP4PrefillTopKResult = result_type
    aiter.ops = ops
    ops.flydsl = flydsl
    return {
        "aiter": aiter,
        "aiter.ops": ops,
        "aiter.ops.flydsl": flydsl,
    }


class TestAiterFP4StreamingTopKCapability(unittest.TestCase):
    def tearDown(self):
        fp4_indexer_hip.get_aiter_fp4_streaming_topk.cache_clear()

    def test_detects_exact_entrypoint(self):
        entrypoint = lambda: None
        with mock.patch.dict(
            sys.modules,
            _fake_aiter_modules(entrypoint, lambda: None, lambda: None),
        ):
            self.assertIs(
                fp4_indexer_hip.get_aiter_fp4_streaming_topk(),
                entrypoint,
            )
            self.assertTrue(fp4_indexer_hip.aiter_fp4_streaming_topk_available())

    def test_missing_entrypoint_uses_legacy_path(self):
        with mock.patch.dict(sys.modules, _fake_aiter_modules()):
            self.assertIsNone(fp4_indexer_hip.get_aiter_fp4_streaming_topk())
            self.assertFalse(fp4_indexer_hip.aiter_fp4_streaming_topk_available())

    def test_capability_is_frozen_after_first_probe(self):
        entrypoint = lambda: None
        with mock.patch.dict(sys.modules, _fake_aiter_modules()):
            self.assertIsNone(fp4_indexer_hip.get_aiter_fp4_streaming_topk())
        with mock.patch.dict(
            sys.modules,
            _fake_aiter_modules(entrypoint, lambda: None, lambda: None),
        ):
            self.assertIsNone(fp4_indexer_hip.get_aiter_fp4_streaming_topk())

    def test_partial_api_uses_legacy_path(self):
        entrypoint = lambda: None
        cases = (
            _fake_aiter_modules(entrypoint),
            _fake_aiter_modules(entrypoint, lambda: None),
            _fake_aiter_modules(entrypoint, result_type=lambda: None),
        )
        for modules in cases:
            with self.subTest(exports=tuple(vars(modules["aiter.ops.flydsl"]))):
                fp4_indexer_hip.get_aiter_fp4_streaming_topk.cache_clear()
                with mock.patch.dict(sys.modules, modules):
                    self.assertIsNone(fp4_indexer_hip.get_aiter_fp4_streaming_topk())


class TestAiterFP4StreamingTopKSelection(unittest.TestCase):
    def _is_selected(
        self,
        *,
        mode=ForwardMode.EXTEND,
        prefill_graph=False,
        capture_mode=False,
        piecewise_graph=False,
        breakable_graph=False,
        capturing=False,
    ):
        with (
            mock.patch.object(
                indexer, "get_is_capture_mode", return_value=capture_mode
            ),
            mock.patch.object(
                indexer,
                "is_in_tc_piecewise_cuda_graph",
                return_value=piecewise_graph,
            ),
            mock.patch.object(
                indexer,
                "is_in_breakable_cuda_graph",
                return_value=breakable_graph,
            ),
            mock.patch(
                "torch.cuda.is_current_stream_capturing",
                return_value=capturing,
            ),
        ):
            return indexer._should_use_aiter_fp4_streaming_topk(
                mode,
                True,
                prefill_graph,
            )

    def test_eager_prefill_uses_streaming_topk(self):
        self.assertTrue(self._is_selected())

    def test_capture_warmup_uses_legacy_topk(self):
        self.assertFalse(self._is_selected(capture_mode=True))

    def test_prefill_graph_uses_legacy_topk(self):
        self.assertFalse(self._is_selected(prefill_graph=True))

    def test_piecewise_graph_uses_legacy_topk(self):
        self.assertFalse(self._is_selected(piecewise_graph=True))

    def test_breakable_graph_uses_legacy_topk(self):
        self.assertFalse(self._is_selected(breakable_graph=True))

    def test_active_capture_uses_legacy_topk(self):
        self.assertFalse(self._is_selected(capturing=True))

    def test_decode_uses_legacy_topk(self):
        self.assertFalse(self._is_selected(mode=ForwardMode.DECODE))

    def test_hip_indexer_metadata_preserves_prefill_graph_mode(self):
        backend = types.SimpleNamespace(
            page_size=256,
            dsa_topk_backend=types.SimpleNamespace(should_use_topk_v2=lambda: False),
        )
        core_metadata = types.SimpleNamespace(
            page_table=mock.sentinel.page_table,
            c4_topk_lengths_raw=mock.sentinel.c4_seq_lens,
        )
        with mock.patch.object(
            sys.modules["sglang.srt.layers.attention.deepseek_v4_backend_hip_radix"],
            "PagedIndexerMetadata",
        ) as metadata_type:
            DeepseekV4HipRadixBackend.init_forward_metadata_indexer(
                backend,
                core_metadata,
                use_prefill_cuda_graph=True,
            )

        metadata_type.assert_called_once_with(
            page_size=256,
            page_table=mock.sentinel.page_table,
            c4_seq_lens=mock.sentinel.c4_seq_lens,
            use_topk_v2=False,
            use_prefill_cuda_graph=True,
        )


if __name__ == "__main__":
    unittest.main()
