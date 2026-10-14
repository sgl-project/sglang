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
        hf_text_config=SimpleNamespace(
            model_type=model_type, num_attention_heads=64, o_groups=8
        ),
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
        self.enterContext(
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook._mega_attention_available",
                return_value=True,
            )
        )
        self.enterContext(envs.SGLANG_DSV4_KV_LAYOUT.override("auto"))

    def test_auto_selects_mega_only_where_the_fused_path_runs(self):
        args = mega_args(
            dsv4_attn_backend="auto",
            tp_size=2,
            attn_dp_size=2,
            cuda_graph_backend_prefill=Backend.BREAKABLE,
            enable_decoder_swa_bounded_replay=True,
        )
        self.assertEqual(resolved_backend(args), "flashmla_mega")
        self.assertEqual(args.dsv4_attn_backend, "auto")
        self.assertEqual(
            resolved_backend(mega_args(dsv4_attn_backend="flashmla")), "flashmla"
        )
        for fields in (
            dict(model_type="deepseek_v4"),
            dict(cuda_graph_backend_prefill=Backend.FULL),
            dict(disaggregation_mode="prefill"),
            dict(tp_size=16),
        ):
            with self.subTest(fields=fields):
                args = mega_args(dsv4_attn_backend="auto", **fields)
                self.assertEqual(resolved_backend(args), "flashmla")
        with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(True):
            args = mega_args(dsv4_attn_backend="auto")
            args._model_config.hf_config.quantization_config = {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            self.assertEqual(resolved_backend(args), "flashmla")

    def test_explicit_mega_rejects_only_what_the_fused_path_cannot_run(self):
        rejected = (
            ("DeepSeek-V4.1", dict(model_type="deepseek_v4")),
            ("prefill CUDA graphs", dict(cuda_graph_backend_prefill=Backend.FULL)),
            (
                "encoder SWA bounded replay",
                dict(enable_encoder_swa_bounded_replay=True),
            ),
        )
        for message, fields in rejected:
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, message):
                    apply_deepseek_v4_defaults(
                        mega_args(**fields), "DeepseekV4ForCausalLM"
                    )
        with patch("sglang.srt.utils.common.is_sm100_supported", return_value=False):
            with self.assertRaisesRegex(ValueError, "SM100/SM103"):
                apply_deepseek_v4_defaults(mega_args(), "DeepseekV4ForCausalLM")


if __name__ == "__main__":
    unittest.main()
