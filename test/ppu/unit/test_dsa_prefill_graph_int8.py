"""CPU-only regression tests for DSA prefill graph quantization dispatch.

Load the real split-op module with mocked runtime/kernel boundaries so these
tests can run without PyTorch, Triton, or PPU hardware. Kernel numerics and
actual CUDA graph replay require a separate device-side regression.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


def _identity_decorator(*args, **kwargs):
    return lambda fn: fn


class TestDsaPrefillGraphInt8(unittest.TestCase):
    def setUp(self):
        self.query = MagicMock(name="query")
        self.key = MagicMock(name="key")
        self.quantized = MagicMock(name="quantized")
        self.scale = Mock(name="scale")
        self.weights = Mock(name="weights")
        self.topk = MagicMock(name="topk")
        self.topk.numel.return_value = 4096 * 2048
        self.batch = SimpleNamespace(
            extend_num_tokens=3000, out_cache_loc=MagicMock(name="cache_loc")
        )
        self.indexer = SimpleNamespace(
            use_int8=True,
            use_dsa_indexer_fusion=False,
            dsa_enable_prefill_cp=False,
            block_size=128,
            scale_fmt=None,
            _should_skip_logits_computation=Mock(return_value=False),
            _forward_cuda_k_only=Mock(),
            _get_q_k_bf16=Mock(return_value=(self.query, self.key, None)),
            _get_logits_head_gate=Mock(return_value=self.weights),
            _store_index_k_cache=Mock(),
            _get_topk_ragged=Mock(),
        )
        self.fp8_quant = Mock(return_value=(self.quantized, self.scale))
        self.int8_quant = Mock(return_value=(self.quantized, self.scale))
        context = SimpleNamespace(
            forward_batch=self.batch, dsa_indexers={0: self.indexer}
        )
        self.metadata = Mock(name="metadata")
        backend = SimpleNamespace(get_indexer_metadata=Mock(return_value=self.metadata))
        runtime = "sglang.srt.model_executor.runner_backend_utils"
        dependencies = {
            "torch": {},
            "sglang.srt.compilation.compilation_config": {
                "register_split_op": _identity_decorator,
            },
            "sglang.srt.model_executor.forward_context": {
                "get_attn_backend": lambda: backend,
            },
            f"{runtime}.breakable_cuda_graph": {
                "eager_on_graph": _identity_decorator,
            },
            f"{runtime}.breakable_cuda_graph.context": {
                "is_in_breakable_cuda_graph": lambda: True,
            },
            f"{runtime}.tc_piecewise_cuda_graph": {
                "get_tc_piecewise_forward_context": lambda: context,
                "is_in_tc_piecewise_cuda_graph": lambda: False,
            },
            "sglang.srt.utils": {"is_cuda": lambda: True},
            "sglang.srt.utils.custom_op": {
                "register_custom_op": _identity_decorator,
            },
            "sglang.kernels.ops.attention.dsa.triton_kernel": {
                "act_quant": self.fp8_quant,
            },
            "sglang.kernels.ops.quantization.int8_kernel": {
                "per_token_quant_int8": self.int8_quant,
            },
        }
        modules = {}
        for name, attributes in dependencies.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            modules[name] = module
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        source = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/layers/attention/dsa/dsa_prefill_cuda_graph.py"
        )
        spec = importlib.util.spec_from_file_location("dsa_graph_under_test", source)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def _run_prefill(self):
        x = Mock(name="x")
        result = self.module.pcg_dsa_indexer_prefill_split(
            0, x, Mock(name="q_lora"), Mock(name="positions"), self.topk
        )
        self.assertIsNone(result)
        return x

    def test_long_int8_prefill_never_calls_fp8_quant(self):
        self.fp8_quant.side_effect = AssertionError("FP8 is unavailable on SM80")
        x = self._run_prefill()
        self.int8_quant.assert_called_once_with(self.query.contiguous.return_value)
        self.fp8_quant.assert_not_called()
        self.indexer._get_logits_head_gate.assert_called_once_with(x, self.scale)
        self.key.__getitem__.assert_called_once_with(slice(None, 3000))
        store_args = self.indexer._store_index_k_cache.call_args.kwargs
        self.assertIs(store_args["key"], self.key.__getitem__.return_value)
        self.assertIs(
            store_args["out_cache_loc"],
            self.batch.out_cache_loc.__getitem__.return_value,
        )
        self.quantized.__getitem__.assert_called_once_with(slice(None, 3000))
        self.indexer._get_topk_ragged.assert_called_once_with(
            False,
            self.batch,
            0,
            self.quantized.__getitem__.return_value,
            self.weights,
            self.metadata,
            self.topk,
        )

    def test_fp8_prefill_preserves_quantization(self):
        self.indexer.use_int8 = False
        self._run_prefill()
        self.fp8_quant.assert_called_once_with(self.query, 128, None)
        self.int8_quant.assert_not_called()

    def test_short_prefill_skips_query_quantization(self):
        self.indexer._should_skip_logits_computation.return_value = True
        self._run_prefill()
        self.indexer._forward_cuda_k_only.assert_called_once()
        self.indexer._get_q_k_bf16.assert_not_called()
        self.fp8_quant.assert_not_called()
        self.int8_quant.assert_not_called()


if __name__ == "__main__":
    unittest.main()
