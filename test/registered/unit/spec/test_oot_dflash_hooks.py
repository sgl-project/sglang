import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _OOTDFlashPlatform(SRTPlatform):
    _enum = PlatformEnum.OOT
    device_name = "custom"
    device_type = "custom"

    def __init__(self, default_backend: str = "custom_backend") -> None:
        self.default_backend = default_backend

    def get_default_attention_backend(self) -> str:
        return self.default_backend

    def supports_speculative_algorithm(self, algorithm: str) -> bool:
        return algorithm == "DFLASH"

    def supports_speculative_draft_attention_backend(self, backend: str) -> bool:
        return backend == "custom_backend"


def _make_dflash_args(draft_backend: str | None) -> ServerArgs:
    args = ServerArgs(model_path="dummy")
    args.device = "custom"
    args.speculative_algorithm = "DFLASH"
    args.speculative_draft_model_path = "draft"
    args.speculative_draft_model_revision = "main"
    args.speculative_num_steps = 1
    args.speculative_eagle_topk = 1
    args.speculative_num_draft_tokens = 4
    args.speculative_draft_attention_backend = draft_backend
    return args


class TestOOTDFlashHooks(CustomTestCase):
    def test_custom_draft_backend_is_accepted(self):
        args = _make_dflash_args("custom_backend")
        platform = _OOTDFlashPlatform()

        with patch("sglang.srt.arg_groups.speculative_hook.current_platform", platform):
            handle_speculative_decoding(args)

        self.assertEqual(args.speculative_draft_attention_backend, "custom_backend")

    def test_unknown_draft_backend_falls_back_to_platform_default(self):
        args = _make_dflash_args("typo")
        platform = _OOTDFlashPlatform()

        with (
            patch("sglang.srt.arg_groups.speculative_hook.current_platform", platform),
            self.assertLogs(
                "sglang.srt.arg_groups.speculative_hook", "WARNING"
            ) as logs,
        ):
            handle_speculative_decoding(args)

        self.assertEqual(args.speculative_draft_attention_backend, "custom_backend")
        self.assertTrue(any("got 'typo'" in message for message in logs.output))

    def test_missing_platform_default_has_actionable_error(self):
        args = _make_dflash_args(None)

        class PlatformWithoutDefault(SRTPlatform):
            _enum = PlatformEnum.OOT
            device_name = "custom"
            device_type = "custom"

            def supports_speculative_algorithm(self, algorithm: str) -> bool:
                return algorithm == "DFLASH"

        with (
            patch(
                "sglang.srt.arg_groups.speculative_hook.current_platform",
                PlatformWithoutDefault(),
            ),
            patch(
                "sglang.srt.arg_groups.overrides.attention_backends_of",
                return_value=(None, None),
            ),
            self.assertRaisesRegex(ValueError, "get_default_attention_backend"),
        ):
            handle_speculative_decoding(args)

    def test_cache_location_dispatch_tracks_platform_changes(self):
        class CacheLocPlatform(_OOTDFlashPlatform):
            def __init__(self, value: int) -> None:
                super().__init__()
                self.value = value

            def get_speculative_cache_locs_fn(self):
                def assign(**kwargs):
                    del kwargs
                    return torch.tensor([self.value])

                return assign

        first = CacheLocPlatform(1)
        second = CacheLocPlatform(2)
        kwargs = dict(
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            req_to_token=torch.empty((0, 0), dtype=torch.int64),
            start_offset=torch.empty(0, dtype=torch.int64),
            end_offset=torch.empty(0, dtype=torch.int64),
            batch_size=0,
            draft_token_num=0,
            device=torch.device("cpu"),
        )

        with patch("sglang.srt.platforms.current_platform", first):
            first_result = assign_extend_cache_locs_func(**kwargs)
        with patch("sglang.srt.platforms.current_platform", second):
            second_result = assign_extend_cache_locs_func(**kwargs)

        self.assertEqual(first_result.item(), 1)
        self.assertEqual(second_result.item(), 2)


if __name__ == "__main__":
    unittest.main()
