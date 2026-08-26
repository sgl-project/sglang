import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.arg_groups.deepseek_v4_hook import (
    validate_deepseek_v4_fp4_indexer,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekV4Fp4IndexerValidation(unittest.TestCase):
    def _server_args(self, **kwargs):
        return ServerArgs(
            model_path="dummy", enable_deepseek_v4_fp4_indexer=True, **kwargs
        )

    @patch(
        "sglang.srt.arg_groups.deepseek_v4_hook._deepseek_v4_fp4_indexer_aiter_error",
        side_effect=AssertionError("CUDA validation imported AITER"),
    )
    @patch("sglang.srt.arg_groups.deepseek_v4_hook.is_hip", return_value=False)
    def test_cuda_sm100_and_sm120_are_accepted(self, _mock_is_hip, mock_aiter_probe):
        for sm100, sm120 in ((True, False), (False, True)):
            with (
                self.subTest(sm100=sm100, sm120=sm120),
                patch(
                    "sglang.srt.arg_groups.deepseek_v4_hook.is_sm100_supported",
                    return_value=sm100,
                ),
                patch(
                    "sglang.srt.arg_groups.deepseek_v4_hook.is_sm120_supported",
                    return_value=sm120,
                ),
            ):
                validate_deepseek_v4_fp4_indexer(self._server_args())

        mock_aiter_probe.assert_not_called()

    @patch(
        "sglang.srt.arg_groups.deepseek_v4_hook.is_sm120_supported",
        return_value=False,
    )
    @patch(
        "sglang.srt.arg_groups.deepseek_v4_hook.is_sm100_supported",
        return_value=False,
    )
    @patch("sglang.srt.arg_groups.deepseek_v4_hook.is_hip", return_value=False)
    def test_cuda_other_capabilities_are_rejected(
        self, _mock_is_hip, _mock_sm100, _mock_sm120
    ):
        with self.assertRaisesRegex(ValueError, "requires SM100 or SM120"):
            validate_deepseek_v4_fp4_indexer(self._server_args())

    @patch(
        "sglang.srt.arg_groups.deepseek_v4_hook.is_gfx950_supported",
        return_value=True,
    )
    @patch("sglang.srt.arg_groups.deepseek_v4_hook.is_hip", return_value=True)
    def test_hip_ordinary_single_node_with_required_aiter_is_accepted(
        self, _mock_is_hip, _mock_gfx950
    ):
        with patch(
            "sglang.srt.arg_groups.deepseek_v4_hook.importlib.import_module",
            side_effect=self._import_complete_aiter_module,
        ):
            validate_deepseek_v4_fp4_indexer(self._server_args())

    @patch("sglang.srt.arg_groups.deepseek_v4_hook.is_hip", return_value=True)
    def test_hip_unsupported_config_or_arch_is_rejected_before_aiter_probe(
        self, _mock_is_hip
    ):
        cases = (
            (
                {"enable_hierarchical_cache": True},
                True,
                "HiCache",
            ),
            (
                {"disaggregation_mode": "prefill"},
                True,
                "PD disaggregation",
            ),
            (
                {"disaggregation_mode": "decode"},
                True,
                "PD disaggregation",
            ),
            ({}, False, "requires an AMD gfx950 GPU"),
        )
        for server_args_kwargs, gfx950_supported, error_pattern in cases:
            with (
                self.subTest(
                    server_args_kwargs=server_args_kwargs,
                    gfx950_supported=gfx950_supported,
                ),
                patch(
                    "sglang.srt.arg_groups.deepseek_v4_hook.is_gfx950_supported",
                    return_value=gfx950_supported,
                ),
                patch(
                    "sglang.srt.arg_groups.deepseek_v4_hook.importlib.import_module",
                    side_effect=AssertionError(
                        "unsupported HIP configuration imported AITER"
                    ),
                ) as mock_import,
                self.assertRaisesRegex(ValueError, error_pattern),
            ):
                validate_deepseek_v4_fp4_indexer(
                    self._server_args(**server_args_kwargs)
                )

            mock_import.assert_not_called()

    @patch(
        "sglang.srt.arg_groups.deepseek_v4_hook.is_gfx950_supported",
        return_value=True,
    )
    @patch("sglang.srt.arg_groups.deepseek_v4_hook.is_hip", return_value=True)
    def test_hip_invalid_aiter_dependency_is_rejected(self, _mock_is_hip, _mock_gfx950):
        cases = tuple(
            (
                {"missing": missing_api},
                f"missing AITER callables: {missing_api}",
            )
            for missing_api in (
                "rope_rotate_activation",
                "rmsnorm_rope_rotate_activation_fp4quant_kvcache",
                "flydsl_pa_mqa_logits_fp4",
                "flydsl_pa_mqa_logits_fp4_prefill",
            )
        ) + (
            ({"marker_version": None}, "known multi-row K-cache scatter bug"),
            ({"marker_version": 0}, "known multi-row K-cache scatter bug"),
            ({"include_fp4x2": False}, "aiter.dtypes.fp4x2"),
        )
        for module_kwargs, error_pattern in cases:
            with (
                self.subTest(module_kwargs=module_kwargs),
                patch(
                    "sglang.srt.arg_groups.deepseek_v4_hook.importlib.import_module",
                    side_effect=lambda module_name, module_kwargs=module_kwargs: self._import_complete_aiter_module(
                        module_name, **module_kwargs
                    ),
                ),
                self.assertRaisesRegex(ValueError, error_pattern),
            ):
                validate_deepseek_v4_fp4_indexer(self._server_args())

    @patch(
        "sglang.srt.arg_groups.deepseek_v4_hook.is_gfx950_supported",
        return_value=True,
    )
    @patch("sglang.srt.arg_groups.deepseek_v4_hook.is_hip", return_value=True)
    def test_hip_aiter_import_failure_is_rejected(self, _mock_is_hip, _mock_gfx950):
        with (
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.importlib.import_module",
                side_effect=RuntimeError("AITER initialization failed"),
            ),
            self.assertRaisesRegex(
                ValueError, "could not import aiter.*Install a matching AITER build"
            ),
        ):
            validate_deepseek_v4_fp4_indexer(self._server_args())

    @staticmethod
    def _import_complete_aiter_module(
        module_name, missing=None, marker_version=1, include_fp4x2=True
    ):
        api_names = {
            "aiter": (
                "rope_rotate_activation",
                "rmsnorm_rope_rotate_activation_fp4quant_kvcache",
            ),
            "aiter.ops.flydsl": (
                "flydsl_pa_mqa_logits_fp4",
                "flydsl_pa_mqa_logits_fp4_prefill",
            ),
        }[module_name]
        attributes = {
            api_name: MagicMock() for api_name in api_names if api_name != missing
        }
        if module_name == "aiter":
            if marker_version is not None:
                attributes["DSV4_FP4_KVCACHE_SCATTER_VERSION"] = marker_version
            attributes["dtypes"] = SimpleNamespace(
                **({"fp4x2": object()} if include_fp4x2 else {})
            )
        return SimpleNamespace(**attributes)


if __name__ == "__main__":
    unittest.main()
