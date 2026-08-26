import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sglang.srt.server_args as server_args_module
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHopperEagleAttentionBackend(unittest.TestCase):
    def test_unresolved_eagle_topk_skips_fa3(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            has_asymmetric_kv=False,
            has_attention_sinks=False,
        )

        with patch.multiple(
            server_args_module,
            current_platform=SimpleNamespace(is_out_of_tree=lambda: False),
            is_hopper_with_cuda_12_3=MagicMock(return_value=True),
            is_sm100_supported=MagicMock(return_value=False),
            is_hip=MagicMock(return_value=False),
            is_mps=MagicMock(return_value=False),
            is_flashinfer_available=MagicMock(return_value=True),
        ):
            cases = (
                ("no_spec", None, None, "fa3"),
                ("eagle_topk_one", "EAGLE", 1, "fa3"),
                ("eagle_auto_pending", "EAGLE", None, "flashinfer"),
                ("eagle_topk_four", "EAGLE", 4, "flashinfer"),
            )
            for name, algorithm, topk, expected in cases:
                with self.subTest(name=name):
                    args = ServerArgs(
                        model_path="dummy",
                        speculative_algorithm=algorithm,
                        speculative_eagle_topk=topk,
                        page_size=1,
                    )
                    self.assertEqual(
                        args._get_default_attn_backend(False, model_config), expected
                    )


if __name__ == "__main__":
    unittest.main()
