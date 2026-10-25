"""Phase overrides inherit the global runner and reject incompatible layouts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import moe_hook
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMoePhaseArgs(unittest.TestCase):
    def resolve(self, **kwargs):
        cfg = SimpleNamespace(
            **{
                "moe_runner_backend": "auto",
                "moe_prefill_runner_backend": None,
                "moe_decode_runner_backend": None,
                "quantization": "modelopt_fp4",
                "moe_a2a_backend": "none",
                "nnodes": 1,
                "speculative_algorithm": None,
                "tp_size": 4,
                "attn_dp_size": 4,
                **kwargs,
            }
        )
        writes = {}
        with (
            patch.object(moe_hook, "resolving_view", return_value=cfg),
            patch.object(moe_hook, "resolved_view", return_value=cfg),
            patch.object(moe_hook, "attn_dp_enabled_of", return_value=True),
            patch.object(moe_hook, "_set_flashinfer_megamoe_transport_defaults"),
            patch.object(
                moe_hook,
                "model_config_of",
                return_value=SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]),
                    dtype=torch.bfloat16,
                    is_fp4_experts=False,
                    nvfp4_moe_meta=None,
                ),
            ),
            patch.object(
                moe_hook,
                "get_platform",
                return_value=SimpleNamespace(is_sm100=True),
            ),
            patch(
                "sglang.srt.arg_groups.overrides.get_platform",
                return_value=SimpleNamespace(
                    is_sm100=True, is_sm120=False, is_sm110=False
                ),
            ),
            patch.object(
                moe_hook,
                "declare_resolution",
                side_effect=lambda _, __, **values: writes.update(values),
            ),
        ):
            moe_hook.handle_moe_phase_backends(cfg)
        return writes

    def test_no_overrides_preserves_existing_resolution(self):
        self.assertEqual(self.resolve(), {})

    def test_one_override_inherits_global_default(self):
        self.assertEqual(
            self.resolve(moe_decode_runner_backend="flashinfer_cutedsl"),
            {
                "moe_runner_backend": "flashinfer_trtllm",
                "moe_prefill_runner_backend": "flashinfer_trtllm",
                "moe_decode_runner_backend": "flashinfer_cutedsl",
            },
        )
        self.assertEqual(
            self.resolve(
                moe_runner_backend="flashinfer_cutedsl",
                moe_prefill_runner_backend="flashinfer_trtllm",
            )["moe_decode_runner_backend"],
            "flashinfer_cutedsl",
        )

    def test_matching_overrides_use_existing_single_runner(self):
        result = self.resolve(
            quantization=None,
            moe_prefill_runner_backend="triton",
            moe_decode_runner_backend="triton",
        )
        self.assertEqual(result["moe_runner_backend"], "triton")

    def test_flashinfer_a2a_normalizes_trtllm_in_both_phases(self):
        for prefill, decode, expected_prefill, expected_decode in (
            (
                None,
                "flashinfer_cutedsl",
                "flashinfer_trtllm_routed",
                "flashinfer_cutedsl",
            ),
            (
                "flashinfer_cutedsl",
                "flashinfer_trtllm",
                "flashinfer_cutedsl",
                "flashinfer_trtllm_routed",
            ),
            (
                "flashinfer_trtllm",
                "flashinfer_trtllm_routed",
                "flashinfer_trtllm_routed",
                "flashinfer_trtllm_routed",
            ),
        ):
            with self.subTest(prefill=prefill, decode=decode):
                result = self.resolve(
                    moe_a2a_backend="flashinfer",
                    moe_prefill_runner_backend=prefill,
                    moe_decode_runner_backend=decode,
                )
                self.assertEqual(result["moe_runner_backend"], expected_prefill)
                self.assertEqual(result["moe_prefill_runner_backend"], expected_prefill)
                self.assertEqual(result["moe_decode_runner_backend"], expected_decode)

    def test_cutedsl_online_prefill_and_megamoe_w4a16_decode(self):
        with envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override("decode"):
            result = self.resolve(
                quantization="nvfp4_online",
                moe_a2a_backend="flashinfer",
                moe_runner_backend="flashinfer_cutedsl",
                moe_decode_runner_backend="flashinfer_megamoe",
            )
        self.assertEqual(result["moe_prefill_runner_backend"], "flashinfer_cutedsl")
        self.assertEqual(result["moe_decode_runner_backend"], "flashinfer_megamoe")
        with (
            envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override("prefill"),
            self.assertRaises(ValueError),
        ):
            self.resolve(
                quantization="nvfp4_online",
                moe_a2a_backend="flashinfer",
                moe_runner_backend="flashinfer_cutedsl",
                moe_decode_runner_backend="flashinfer_megamoe",
            )

    def test_kernel_hook_resolves_phase_overrides_before_global_constraints(self):
        from sglang.srt.arg_groups.overrides import resolution_result
        from sglang.srt.server_args import ServerArgs

        for default, decode, expected_decode in (
            ("flashinfer_megamoe", None, "flashinfer_megamoe"),
            ("triton", "flashinfer_trtllm", "flashinfer_trtllm_routed"),
        ):
            with self.subTest(default=default, decode=decode):
                args = ServerArgs(
                    model_path="dummy",
                    quantization="nvfp4_online",
                    moe_a2a_backend="flashinfer",
                    moe_runner_backend=default,
                    moe_prefill_runner_backend="flashinfer_cutedsl",
                    moe_decode_runner_backend=decode,
                    tp_size=4,
                    ep_size=4,
                    attn_dp_size=4,
                )
                args._model_config = SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]),
                    dtype=torch.bfloat16,
                    is_fp4_experts=False,
                    nvfp4_moe_meta=None,
                )
                with (
                    envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override("decode"),
                    patch.object(
                        moe_hook, "_set_flashinfer_megamoe_transport_defaults"
                    ),
                    patch.object(
                        moe_hook,
                        "get_platform",
                        return_value=SimpleNamespace(is_sm100=True),
                    ),
                    patch(
                        "sglang.srt.arg_groups.overrides.get_platform",
                        return_value=SimpleNamespace(is_sm100=True),
                    ),
                ):
                    moe_hook.handle_moe_kernel_config(args)
                self.assertEqual(
                    resolution_result(args, "moe_runner_backend"), "flashinfer_cutedsl"
                )
                self.assertEqual(
                    resolution_result(args, "moe_decode_runner_backend"),
                    expected_decode,
                )

    def test_unsupported_combinations_fail_at_configuration(self):
        cases = [
            {"moe_prefill_runner_backend": "triton"},
            {"moe_a2a_backend": "deepep"},
            {"quantization": "fp8", "moe_runner_backend": "flashinfer_trtllm"},
            {"enable_lora": True},
            {"enable_eplb": True},
            {"enable_paged_experts": True},
            {"enable_two_batch_overlap": True},
            {"enable_single_batch_overlap": True},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(
                    moe_decode_runner_backend="flashinfer_cutedsl", **overrides
                )

    def test_online_megamoe_still_requires_all_phase_w4a16(self):
        from sglang.srt.arg_groups.overrides import (
            _moe_runner_backend_quant_constraints,
        )

        view = SimpleNamespace(
            quantization="nvfp4_online",
            moe_a2a_backend="flashinfer_megamoe",
            moe_runner_backend="flashinfer_megamoe",
        )
        with patch(
            "sglang.srt.arg_groups.overrides.get_platform",
            return_value=SimpleNamespace(is_sm100=True),
        ):
            for value in ("prefill", "decode", False):
                with (
                    self.subTest(value=value),
                    envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(value),
                    self.assertRaises(ValueError),
                ):
                    _moe_runner_backend_quant_constraints(view)
            with envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(True):
                self.assertEqual(_moe_runner_backend_quant_constraints(view), {})


if __name__ == "__main__":
    unittest.main()
