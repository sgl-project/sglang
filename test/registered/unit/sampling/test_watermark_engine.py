"""Algorithm dispatch, config admission and request-policy compatibility."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.fields.exec_ import ExecFeatures, ExecKernel
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.arg_groups.serving_hook import handle_watermark_config
from sglang.srt.arg_groups.validation_hook import check_watermark_server_args
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.runtime_context import get_context, get_exec
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.watermarking.config import WatermarkConfigError
from sglang.srt.sampling.watermarking.core import (
    build_watermark_batch_config,
    redact_watermark_command_line,
    redact_watermark_secrets,
)
from sglang.srt.sampling.watermarking.server import validate_watermark_request
from sglang.srt.server_args import ServerArgs
from sglang.srt.watermarking.synthid import get_synthid_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestWatermarkEngine(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.key_file = Path(self.directory.name) / "keys.json"
        self.key_file.write_text(json.dumps({"keys": [1, 2]}))
        self.key_file.chmod(0o600)

    def resolve(self, **kwargs):
        args = ServerArgs(
            **{
                "model_path": "dummy",
                "device": "cuda",
                "sampling_backend": "pytorch",
                **kwargs,
            }
        )
        with patch("sglang.srt.arg_groups.validation_hook.is_cuda", return_value=True):
            handle_watermark_config(args)
            check_watermark_server_args(args)
        return resolving_view(args)

    def test_legacy_keys_and_generic_synthid_preserve_same_config(self):
        """Selecting the new API must not replace the existing key/tournament law."""
        legacy = self.resolve(
            synthid_watermark_key_file=str(self.key_file), synthid_watermark_top_k=512
        )
        config = Path(self.directory.name) / "synthid.json"
        config.write_text(json.dumps({"key_file": str(self.key_file), "top_k": 512}))
        config.chmod(0o600)
        generic = self.resolve(
            enable_watermark=True,
            watermark_algorithm="synthid",
            watermark_config=str(config),
        )
        self.assertEqual(
            generic.synthid_watermark_key_file, legacy.synthid_watermark_key_file
        )
        for args in (legacy, generic):
            self.assertTrue(args.enable_watermark)
            self.assertEqual(args.watermark_algorithm, "synthid")
            self.assertFalse(args.watermark_default_enabled)
            self.assertFalse(args.watermark_enforce_all)
            with get_context().override_server_args(
                **{
                    name: getattr(args, name)
                    for name in (
                        *ExecFeatures.__struct_fields__,
                        *ExecKernel.__struct_fields__,
                    )
                    if name.startswith(("synthid_", "watermark_"))
                    or name == "enable_watermark"
                }
            ):
                resolved = get_synthid_config()
                self.assertEqual(
                    (resolved.keys, resolved.top_k, resolved.ngram_len),
                    ((1, 2), 512, 5),
                )
                self.assertEqual(get_exec().features.watermark_algorithm, "synthid")

    def test_generic_file_can_reuse_the_original_key_fixture(self):
        args = self.resolve(
            enable_watermark=True,
            watermark_algorithm="synthid",
            watermark_config=str(self.key_file),
        )
        self.assertEqual(args.synthid_watermark_key_file, str(self.key_file))
        self.assertEqual(
            (args.synthid_watermark_top_k, args.synthid_watermark_ngram_len),
            (1024, 5),
        )

    def test_algorithm_specific_rejection_sampling_admission(self):
        """Gumbel's target-only restriction must never disable SynthID MTP."""
        self.resolve(
            enable_watermark=True,
            watermark_algorithm="synthid",
            watermark_config=str(self.key_file),
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_use_rejection_sampling=True,
        )
        with self.assertRaisesRegex(ValueError, "incompatible.*rejection-sampling"):
            self.resolve(enable_watermark=True, speculative_use_rejection_sampling=True)
        with self.assertRaisesRegex(ValueError, "requires.*rejection-sampling"):
            self.resolve(
                enable_watermark=True,
                watermark_algorithm="synthid",
                watermark_config=str(self.key_file),
                speculative_algorithm="EAGLE",
                speculative_eagle_topk=1,
            )

    def test_invalid_selectors_and_conflicting_config_fail_closed(self):
        for kwargs in (
            {"enable_watermark": True, "watermark_algorithm": "unknown"},
            {"enable_watermark": True, "device": "cpu"},
            {
                "enable_watermark": True,
                "watermark_algorithm": "synthid",
                "watermark_config": str(self.key_file),
                "is_embedding": True,
            },
            {"watermark_algorithm": "synthid"},
            {"watermark_config": '{"key":"1234"}'},
            {"enable_watermark": True, "watermark_algorithm": "synthid"},
            {
                "enable_watermark": True,
                "watermark_algorithm": "gumbel",
                "synthid_watermark_key_file": str(self.key_file),
            },
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.resolve(**kwargs)
        with self.assertRaises(WatermarkConfigError):
            self.resolve(
                enable_watermark=True,
                watermark_algorithm="synthid",
                watermark_config='{"keys":[1,2]}',
            )
        for overrides in (
            {"speculative_algorithm": "NGRAM"},
            {"speculative_algorithm": "EAGLE", "speculative_eagle_topk": 2},
            {"speculative_algorithm": "EAGLE", "speculative_eagle_topk": 1},
            {
                "speculative_algorithm": "EAGLE",
                "speculative_eagle_topk": 1,
                "speculative_token_map": "map.txt",
            },
            {
                "speculative_algorithm": "EAGLE",
                "speculative_eagle_topk": 1,
                "speculative_use_rejection_sampling": True,
                "disaggregation_mode": "decode",
            },
            {"synthid_watermark_top_k": 65537},
            {"synthid_watermark_ngram_len": 1},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(synthid_watermark_key_file=str(self.key_file), **overrides)

    def test_gumbel_request_policy_and_disabled_batch_gate(self):
        args = self.resolve(
            enable_watermark=True,
            watermark_config='{"key":"1234","default_enabled":true}',
        )
        self.assertEqual(args.watermark_algorithm, "gumbel")
        requests = [
            Mock(sampling_params=SamplingParams(top_k=20)),
            Mock(
                sampling_params=SamplingParams(top_k=20, watermark={"enabled": False})
            ),
            Mock(sampling_params=SamplingParams(temperature=0)),
        ]
        for request in requests:
            validate_watermark_request(request.sampling_params, args)
        batch = build_watermark_batch_config(
            requests,
            default_key=args.watermark_key,
            default_context_window=4,
            default_enabled=True,
            enforce_all=False,
            device="cpu",
        )
        self.assertEqual(batch.candidates_host, [True, False, False])
        self.assertTrue(batch.has_candidates)
        self.assertEqual(batch.enabled.tolist(), [True, False, True])
        with self.assertRaisesRegex(ValueError, "requires.*enable-watermark"):
            validate_watermark_request(
                SamplingParams(watermark={"enabled": True}),
                ServerArgs(model_path="dummy"),
            )

    def test_synthid_legacy_flags_and_generic_api_are_opt_in(self):
        args = self.resolve(synthid_watermark_key_file=str(self.key_file))
        validate_watermark_request(SamplingParams(), args)
        validate_watermark_request(SamplingParams(watermark={"enabled": True}), args)
        validate_watermark_request(SamplingParams(watermark={"enabled": False}), args)
        for watermark in (
            {"key": "1234"},
            {"enabled": True, "context_window": 2},
        ):
            with self.subTest(watermark=watermark), self.assertRaises(ValueError):
                validate_watermark_request(SamplingParams(watermark=watermark), args)
        generic = self.resolve(
            enable_watermark=True,
            watermark_algorithm="synthid",
            watermark_config=str(self.key_file),
        )
        self.assertFalse(generic.watermark_default_enabled)
        self.assertFalse(generic.watermark_enforce_all)
        for control in (None, {"enabled": False}, {"enabled": True}):
            validate_watermark_request(SamplingParams(watermark=control), generic)
        policy = Path(self.directory.name) / "force.json"
        policy.write_text(
            json.dumps({"key_file": str(self.key_file), "enforce_all": True})
        )
        policy.chmod(0o600)
        forced = self.resolve(
            enable_watermark=True,
            watermark_algorithm="synthid",
            watermark_config=str(policy),
        )
        self.assertTrue(forced.watermark_enforce_all)
        with self.assertRaisesRegex(ValueError, "enforce_all"):
            validate_watermark_request(
                SamplingParams(watermark={"enabled": False}), forced
            )

    def test_watermark_secrets_are_redacted_without_mutating_live_config(self):
        source = '{"key":"123456","key_b":"654321","default_enabled":true}'
        args = self.resolve(enable_watermark=True, watermark_config=source)
        self.assertNotIn("123456", repr(args))
        self.assertNotIn("654321", repr(args))
        rendered = redact_watermark_secrets(
            {name: getattr(args, name) for name in ExecFeatures.__struct_fields__}
        )
        self.assertEqual(rendered["watermark_key"], "<redacted>")
        self.assertEqual(rendered["watermark_config"], "<redacted>")
        self.assertEqual(args.watermark_key, "123456")
        self.assertNotIn(
            "123456", redact_watermark_command_line(["--watermark-config", source])
        )
        request = redact_watermark_secrets(
            {"sampling_params": {"watermark": {"key": "123456"}}}
        )
        self.assertEqual(request["sampling_params"]["watermark"]["key"], "<redacted>")

    def test_filter_and_merge_preserve_watermark_row_policy(self):
        """A filtered disabled batch must not keep the enabled-batch host gate."""

        def info(keys, enabled):
            count = len(keys)
            return SamplingBatchInfo(
                temperatures=torch.ones(count, 1),
                top_ps=torch.ones(count),
                top_ks=torch.full((count,), 20, dtype=torch.int32),
                min_ps=torch.zeros(count),
                is_all_greedy=False,
                is_any_greedy=False,
                need_top_k_sampling=True,
                need_top_p_sampling=False,
                need_min_p_sampling=False,
                vocab_size=32,
                device="cpu",
                penalizer_orchestrator=Mock(is_required=False),
                watermark_keys=torch.tensor(keys),
                watermark_enabled=torch.tensor(enabled),
                watermark_context_windows=torch.full((count,), 4),
                watermark_candidates_host=enabled,
                has_watermark_candidates=any(enabled),
                max_top_k=20,
            )

        batch = info([1, 2], [True, False])
        batch.filter_batch([1], torch.tensor([1]))
        self.assertEqual(batch.watermark_keys.tolist(), [2])
        self.assertFalse(batch.has_watermark_candidates)
        batch.merge_batch(info([3], [True]))
        self.assertEqual(batch.watermark_keys.tolist(), [2, 3])
        self.assertEqual(batch.watermark_enabled.tolist(), [False, True])
        self.assertEqual(batch.watermark_candidates_host, [False, True])
        self.assertTrue(batch.has_watermark_candidates)

    def test_native_batch_and_parallel_samples_keep_request_policy(self):
        """Request fan-out must replicate watermark controls alongside prompts."""
        off = {"enabled": False}
        on = {"key": "1234"}
        request = GenerateReqInput(
            text=["one", "two"],
            sampling_params={"n": 2},
            watermark=[off, on],
        )
        request.normalize_batch_and_arguments()
        self.assertEqual(request.watermark, [off, on, off, on])
        self.assertEqual(request[0].watermark, off)
        self.assertEqual(request[1].watermark, on)
        invalid = GenerateReqInput(text=["one", "two"], watermark=[off])
        with self.assertRaisesRegex(ValueError, "length of watermark"):
            invalid.normalize_batch_and_arguments()


if __name__ == "__main__":
    unittest.main()
