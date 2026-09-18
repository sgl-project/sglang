import unittest
from unittest.mock import Mock, patch, sentinel

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

HOOK_MODULE = "sglang.srt.arg_groups.speculative_hook"


def _platform(
    default_backend: str = "custom_backend",
    supported_backends: set[str] | None = None,
    cache_result=None,
) -> Mock:
    supported_backends = supported_backends or {"custom_backend"}
    platform = Mock(spec=SRTPlatform)
    platform.is_out_of_tree.return_value = True
    platform.supports_speculative_algorithm.return_value = True
    platform.supports_speculative_draft_attention_backend.side_effect = (
        lambda algorithm, backend: (
            algorithm == "DFLASH" and backend in supported_backends
        )
    )
    platform.get_default_speculative_draft_attention_backend.return_value = (
        default_backend
    )
    platform.get_speculative_cache_locs_fn.return_value = (
        None if cache_result is None else lambda **_: cache_result
    )
    return platform


def _make_dflash_args(draft_backend: str | None) -> ServerArgs:
    return ServerArgs(
        model_path="dummy",
        device="custom",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="draft",
        speculative_num_draft_tokens=4,
        speculative_draft_attention_backend=draft_backend,
    )


class TestOOTDFlashHooks(CustomTestCase):
    def _resolve_backend(self, draft_backend: str | None, platform: Mock) -> str | None:
        args = _make_dflash_args(draft_backend)
        with (
            patch(f"{HOOK_MODULE}.current_platform", platform),
            patch(f"{HOOK_MODULE}.attention_backends_of", return_value=(None, None)),
        ):
            handle_speculative_decoding(args)
        return args.speculative_draft_attention_backend

    def test_explicit_backends_follow_platform_capabilities(self):
        cases = (
            ("custom_backend", {"custom_backend"}, "custom_backend"),
            ("flashinfer", {"custom_backend"}, "custom_backend"),
            ("flashinfer", {"flashinfer"}, "flashinfer"),
            ("trtllm_mha", {"custom_backend"}, "custom_backend"),
        )
        for draft_backend, supported_backends, expected in cases:
            with self.subTest(
                draft_backend=draft_backend, supported_backends=supported_backends
            ):
                self.assertEqual(
                    self._resolve_backend(
                        draft_backend,
                        _platform(supported_backends=supported_backends),
                    ),
                    expected,
                )

    def test_unknown_backend_warns_and_falls_back(self):
        with self.assertLogs(
            "sglang.srt.arg_groups.speculative_hook", "WARNING"
        ) as logs:
            resolved = self._resolve_backend("typo", _platform())

        self.assertEqual(resolved, "custom_backend")
        self.assertIn("attention_backend 'typo'", "\n".join(logs.output))

    def test_invalid_platform_defaults_raise_actionable_errors(self):
        missing_default = _platform()
        missing_default.get_default_speculative_draft_attention_backend.side_effect = (
            NotImplementedError
        )
        for platform, error in (
            (_platform(default_backend="flashinfer"), "returned unsupported"),
            (
                missing_default,
                "get_default_speculative_draft_attention_backend",
            ),
        ):
            with (
                self.subTest(error=error),
                self.assertRaisesRegex(ValueError, error),
            ):
                self._resolve_backend(None, platform)

    def test_cache_location_dispatch_tracks_platform_changes(self):
        for expected in (sentinel.first, sentinel.second):
            with patch(
                "sglang.srt.platforms.current_platform",
                _platform(cache_result=expected),
            ):
                result = assign_extend_cache_locs_func(
                    None, None, None, None, 0, 0, None
                )
            self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()
