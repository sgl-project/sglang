import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
from sglang.srt.arg_groups.deepseek_v4_hook import apply_deepseek_v4_defaults
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
            "dsv4_attn_backend": "flashmla_mega",
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


class TestDSV41MegaAttentionPolicy(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(override_platform(is_cuda=True, is_hip=False, is_npu=False))

    def test_rejects_layout_and_execution_incompatibilities(self):
        cases = (
            ("DeepSeek-V4.1", dict(model_type="deepseek_v4")),
            ("BF16 activations", dict(dtype="float16")),
            ("packed KV storage", dict(kv_cache_dtype="bfloat16")),
            ("context parallelism", dict(attn_cp_size=2)),
            ("context parallelism", dict(enable_prefill_cp=True)),
            ("PD disaggregation", dict(disaggregation_mode="decode")),
            ("prefill CUDA graphs", dict(cuda_graph_backend_prefill=Backend.BREAKABLE)),
        )
        with patch("sglang.srt.utils.common.is_sm100_supported", return_value=True):
            for message, fields in cases:
                with self.subTest(fields=fields):
                    with self.assertRaisesRegex(ValueError, message):
                        apply_deepseek_v4_defaults(
                            mega_args(**fields), "DeepseekV4ForCausalLM"
                        )

    def test_accepts_dp_attention_and_decode_cuda_graphs(self):
        with patch("sglang.srt.utils.common.is_sm100_supported", return_value=True):
            args = mega_args(tp_size=4, attn_dp_size=4)
            apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")

    def test_rejects_non_sm100_hardware(self):
        with patch("sglang.srt.utils.common.is_sm100_supported", return_value=False):
            with self.assertRaisesRegex(ValueError, "SM100/SM103"):
                apply_deepseek_v4_defaults(mega_args(), "DeepseekV4ForCausalLM")


if __name__ == "__main__":
    unittest.main()
