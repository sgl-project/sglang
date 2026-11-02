"""Model defaults must not count as an operator-supplied SWA ratio."""

import pickle
import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
from sglang.srt.arg_groups.model_overrides.deepseek_v4 import _deepseek_v4_overrides
from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
from sglang.srt.runtime_context import get_schedule, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSwaRatioExplicitInput(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)

    def _record(self, ratio=None, device="cuda", **kwargs):
        args = ServerArgs(
            model_path="dummy",
            device=device,
            moe_runner_backend="triton",
            swa_full_tokens_ratio=ratio,
            **kwargs,
        )
        args.resolve_once()
        return args

    def _dsv4_defaults(self, args):
        hf_config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v4"
        )
        declare_resolution(
            args,
            "_handle_model_specific_adjustments",
            **_deepseek_v4_overrides(args, hf_config),
        )

    def _check_published(self, args, explicit, ratio):
        self.assertIs(
            resolution_result(args, "_swa_full_tokens_ratio_explicitly_set"),
            explicit,
        )
        publish(args, role="test")
        self.assertIs(get_schedule()._swa_full_tokens_ratio_explicitly_set, explicit)
        self.assertEqual(get_schedule().swa_full_tokens_ratio, ratio)

    def test_dsv4_automatic_ratio_is_not_explicit(self):
        for device in ("cuda", "npu"):
            with self.subTest(device=device):
                args = self._record(device=device)
                self._dsv4_defaults(args)
                handle_cache_compatibility(args)
                self.assertIsNone(args._raw_input["swa_full_tokens_ratio"])
                self._check_published(args, explicit=False, ratio=0.1)

    def test_dsv4_explicit_ratio_is_preserved_including_default_value(self):
        for device in ("cuda", "npu"):
            for ratio in (0.1, 0.25, 1.0):
                with self.subTest(device=device, ratio=ratio):
                    args = self._record(ratio=ratio, device=device)
                    self._dsv4_defaults(args)
                    handle_cache_compatibility(args)
                    self._check_published(args, explicit=True, ratio=ratio)

    def test_generic_fallback_is_not_explicit(self):
        args = self._record()
        handle_cache_compatibility(args)
        self._check_published(args, explicit=False, ratio=0.8)

    def test_model_reset_keeps_the_original_user_intent(self):
        args = self._record(ratio=0.25)
        declare_resolution(args, "model_reset", swa_full_tokens_ratio=1.0)
        handle_cache_compatibility(args)
        self.assertEqual(args._raw_input["swa_full_tokens_ratio"], 0.25)
        self._check_published(args, explicit=True, ratio=1.0)

    def test_effective_ratio_validation_is_preserved(self):
        for ratio in (0.0, -0.1, 1.01):
            with self.subTest(ratio=ratio):
                args = self._record()
                declare_resolution(
                    args, "invalid_model_default", swa_full_tokens_ratio=ratio
                )
                with self.assertRaisesRegex(ValueError, "swa-full-tokens-ratio"):
                    handle_cache_compatibility(args)

    def test_explicit_invalid_ratio_is_rejected(self):
        for ratio in (0.0, -0.1, 1.01):
            with self.subTest(ratio=ratio):
                args = self._record(ratio=ratio)
                with self.assertRaisesRegex(ValueError, "swa-full-tokens-ratio"):
                    handle_cache_compatibility(args)

    def test_negative_prefix_tails_is_rejected(self):
        args = self._record(swa_prefix_tails=-1)
        with self.assertRaisesRegex(ValueError, "swa-prefix-tails"):
            handle_cache_compatibility(args)

    def test_origin_survives_serialization_and_publish(self):
        for ratio in (None, 0.1):
            with self.subTest(ratio=ratio):
                args = self._record(ratio=ratio)
                self._dsv4_defaults(args)
                handle_cache_compatibility(args)
                restored = pickle.loads(pickle.dumps(args))
                self.assertEqual(restored._raw_input["swa_full_tokens_ratio"], ratio)
                self._check_published(restored, explicit=ratio is not None, ratio=0.1)


if __name__ == "__main__":
    unittest.main()
