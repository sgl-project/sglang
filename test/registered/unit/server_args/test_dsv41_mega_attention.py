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
            model_type=model_type,
            q_head_norm=False,
            head_dim=512,
            qk_rope_head_dim=64,
            num_attention_heads=64,
            o_groups=8,
        ),
        is_multimodal=False,
    )
    parse_cuda_graph_config(args)
    return args


class TestDSV41MegaAttentionPolicy(CustomTestCase):
    def test_auto_backend_resolves_without_changing_input(self):
        with (
            override_platform(is_cuda=True, is_hip=False, is_npu=False),
            patch("sglang.srt.utils.common.is_sm100_supported", return_value=True),
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook._mega_attention_available",
                return_value=True,
            ),
            envs.SGLANG_DSV4_KV_LAYOUT.override("auto"),
        ):
            args = mega_args(
                dsv4_attn_backend="auto",
                tp_size=2,
                attn_dp_size=2,
                ep_size=2,
                cuda_graph_backend_prefill=Backend.BREAKABLE,
                enable_decoder_swa_bounded_replay=True,
            )
            apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")
            self.assertEqual(args.dsv4_attn_backend, "auto")
            self.assertEqual(
                resolution_result(args, "dsv4_attn_backend"), "flashmla_mega"
            )
            for backend in ("flashmla", "flashmla_mega", "trtllm"):
                with self.subTest(backend=backend):
                    args = mega_args(
                        dsv4_attn_backend=backend, chunked_prefill_size=32768
                    )
                    apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")
                    self.assertEqual(
                        resolution_result(args, "dsv4_attn_backend"), backend
                    )

    def test_auto_backend_preserves_unsupported_configurations(self):
        cases = (
            dict(model_type="deepseek_v4"),
            dict(device="cpu"),
            dict(kv_cache_dtype="bfloat16"),
            dict(cuda_graph_backend_prefill=Backend.FULL),
            dict(enable_encoder_swa_bounded_replay=True),
            dict(enable_hisparse=True),
            dict(disaggregation_mode="prefill"),
            dict(attn_cp_size=2),
            dict(tp_size=3),
            dict(tp_size=2, attn_dp_size=4),
        )
        with (
            override_platform(is_cuda=True, is_hip=False, is_npu=False),
            patch("sglang.srt.utils.common.is_sm100_supported", return_value=True),
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook._mega_attention_available",
                return_value=True,
            ),
            envs.SGLANG_DSV4_KV_LAYOUT.override("auto"),
        ):
            for fields in cases:
                with self.subTest(fields=fields):
                    args = mega_args(dsv4_attn_backend="auto", **fields)
                    apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")
                    self.assertEqual(
                        resolution_result(args, "dsv4_attn_backend"), "flashmla"
                    )
            for text_fields in (
                dict(q_head_norm=True),
                dict(head_dim=256),
                dict(num_attention_heads=256, o_groups=32),
                dict(o_groups=16),
            ):
                with self.subTest(text_fields=text_fields):
                    args = mega_args(dsv4_attn_backend="auto")
                    args._model_config.hf_text_config.__dict__.update(text_fields)
                    apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")
                    self.assertEqual(
                        resolution_result(args, "dsv4_attn_backend"), "flashmla"
                    )
            with envs.SGLANG_DSV4_KV_LAYOUT.override("v4"):
                args = mega_args(dsv4_attn_backend="auto")
                apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")
                self.assertEqual(
                    resolution_result(args, "dsv4_attn_backend"), "flashmla"
                )
            with patch(
                "sglang.srt.arg_groups.deepseek_v4_hook._mega_attention_available",
                return_value=False,
            ):
                args = mega_args(dsv4_attn_backend="auto")
                apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")
                self.assertEqual(
                    resolution_result(args, "dsv4_attn_backend"), "flashmla"
                )
            with patch(
                "sglang.srt.utils.common.is_sm100_supported", return_value=False
            ):
                args = mega_args(dsv4_attn_backend="auto")
                apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")
                self.assertEqual(
                    resolution_result(args, "dsv4_attn_backend"), "flashmla"
                )
            with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(True):
                args = mega_args(dsv4_attn_backend="auto")
                args._model_config.hf_config.quantization_config = {
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")
                self.assertEqual(
                    resolution_result(args, "dsv4_attn_backend"), "flashmla"
                )

    def test_flashmla_mega_launch_policy(self):
        rejected = (
            ("DeepSeek-V4.1", dict(model_type="deepseek_v4")),
            ("prefill CUDA graphs", dict(cuda_graph_backend_prefill=Backend.FULL)),
            (
                "encoder SWA bounded replay",
                dict(enable_encoder_swa_bounded_replay=True),
            ),
        )
        with (
            override_platform(is_cuda=True, is_hip=False, is_npu=False),
            patch("sglang.srt.utils.common.is_sm100_supported", return_value=True),
        ):
            for message, fields in rejected:
                with self.subTest(fields=fields):
                    with self.assertRaisesRegex(ValueError, message):
                        apply_deepseek_v4_defaults(
                            mega_args(**fields), "DeepseekV4ForCausalLM"
                        )
            apply_deepseek_v4_defaults(
                mega_args(tp_size=4, attn_dp_size=4), "DeepseekV4ForCausalLM"
            )
            apply_deepseek_v4_defaults(
                mega_args(
                    tp_size=2,
                    attn_dp_size=2,
                    ep_size=2,
                    moe_a2a_backend="megamoe",
                    cuda_graph_backend_prefill=Backend.BREAKABLE,
                    enable_decoder_swa_bounded_replay=True,
                ),
                "DeepseekV4ForCausalLM",
            )
        with (
            override_platform(is_cuda=True, is_hip=False, is_npu=False),
            patch("sglang.srt.utils.common.is_sm100_supported", return_value=False),
        ):
            with self.assertRaisesRegex(ValueError, "SM100/SM103"):
                apply_deepseek_v4_defaults(mega_args(), "DeepseekV4ForCausalLM")


if __name__ == "__main__":
    unittest.main()
