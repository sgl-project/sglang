import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
from sglang.srt.arg_groups.deepseek_v4_hook import apply_deepseek_v4_defaults
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def mega_args(model_type="deepseek_v41", **fields):
    args = ServerArgs(
        **{
            "model_path": "local-deepseek-v41",
            "device": "cuda",
            "dsv4_attn_backend": "auto",
            "cuda_graph_backend_prefill": Backend.DISABLED,
            "max_running_requests": 32,
            **fields,
        }
    )
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"], model_type=model_type
        ),
        hf_text_config=SimpleNamespace(model_type=model_type),
        is_multimodal=False,
    )
    parse_cuda_graph_config(args)
    return args


def resolved_backend(args):
    apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")
    return resolution_result(args, "dsv4_attn_backend")


class TestDSV41MegaAttentionPolicy(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(override_platform(is_cuda=True, is_hip=False, is_npu=False))
        self.enterContext(
            patch("sglang.srt.utils.common.is_sm100_supported", return_value=True)
        )
        self.enterContext(envs.SGLANG_DSV4_KV_LAYOUT.override("auto"))

    def test_auto_keeps_flashmla_where_mega_would_silently_change_the_pool(self):
        self.assertEqual(resolved_backend(mega_args()), "flashmla_mega")
        for fields in (
            dict(model_type="deepseek_v4"),
            dict(disaggregation_mode="prefill"),
        ):
            with self.subTest(fields=fields):
                self.assertEqual(resolved_backend(mega_args(**fields)), "flashmla")


if __name__ == "__main__":
    unittest.main()
