import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.hardware_backend.npu.utils import set_default_server_args
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")


class TestJetNPUServerArgs(unittest.TestCase):
    def make_args(self, architecture):
        return SimpleNamespace(
            _model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture])
            ),
            cuda_graph_config=CudaGraphConfig(),
            page_size=None,
            chunked_prefill_size=None,
            tp_size=1,
            disable_radix_cache=False,
            disable_overlap_schedule=False,
            enable_hierarchical_cache=False,
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_mode="null",
            disaggregation_decode_retraction_backup=None,
        )

    @patch(
        "sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity",
        return_value=32 * 1024,
    )
    def test_jet_keeps_prefix_reuse_and_overlap_enabled(self, _memory):
        for architecture in (
            "JetNemotronForCausalLM",
            "JetVLMForConditionalGeneration",
        ):
            with self.subTest(architecture=architecture):
                args = self.make_args(architecture)
                original_graphs = args.cuda_graph_config.to_dict()
                set_default_server_args(args)
                cfg = resolving_view(args)
                self.assertGreaterEqual(cfg.page_size, 16)
                self.assertEqual(cfg.page_size % 16, 0)
                self.assertFalse(cfg.disable_radix_cache)
                self.assertFalse(cfg.disable_overlap_schedule)
                self.assertEqual(cfg.cuda_graph_config.decode.backend, Backend.DISABLED)
                self.assertEqual(
                    cfg.cuda_graph_config.prefill.backend, Backend.DISABLED
                )
                self.assertEqual(cfg.cuda_graph_config.decode.max_bs, 16)
                self.assertEqual(cfg.attention_backend, "ascend")
                self.assertEqual(args.cuda_graph_config.to_dict(), original_graphs)

    @patch(
        "sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity",
        return_value=32 * 1024,
    )
    def test_other_models_keep_cache_and_graph_settings(self, _memory):
        args = self.make_args("Qwen2ForCausalLM")
        args.page_size = 16
        args.cuda_graph_config.decode.max_bs = 16
        original_graphs = args.cuda_graph_config.to_dict()
        set_default_server_args(args)
        cfg = resolving_view(args)
        self.assertEqual(cfg.page_size, 16)
        self.assertFalse(cfg.disable_radix_cache)
        self.assertFalse(cfg.disable_overlap_schedule)
        self.assertEqual(cfg.cuda_graph_config.to_dict(), original_graphs)


if __name__ == "__main__":
    unittest.main()
