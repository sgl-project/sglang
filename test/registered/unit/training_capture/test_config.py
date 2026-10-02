"""Capture capability gates against the actual ServerArgs defaults."""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.training_capture.config import (
    CaptureConfig,
    validate_capture_server_args,
)
from sglang.srt.training_capture.protocol import ContractError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCaptureConfiguration(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "config.json"
        self.config = dict(
            dataset_id="test",
            model_id="test",
            producer_revision="test",
            selected_layer_ids=[0, 2],
            catalog_endpoint="http://localhost:1234",
            journal_directory=self.directory.name + "/journal",
            store=dict(
                local_hostname="localhost", master_server_addr="localhost:50051"
            ),
        )
        self.path.write_text(json.dumps(self.config))
        self.args = SimpleNamespace(
            **{
                field.name: field.default
                for field in dataclasses.fields(ServerArgs)
                if field.default is not dataclasses.MISSING
            }
        )
        for field in dataclasses.fields(ServerArgs):
            if field.default_factory is not dataclasses.MISSING:
                setattr(self.args, field.name, field.default_factory())
        self.args.training_capture_config = str(self.path)
        self.args.disable_overlap_schedule = True

    def tearDown(self):
        self.directory.cleanup()

    def test_plain_ar_and_disabled_mode(self):
        validate_capture_server_args(self.args)
        self.args.disable_overlap_schedule = False
        validate_capture_server_args(self.args)
        self.args.training_capture_config = None
        self.args.tp_size = 8
        self.args.disable_overlap_schedule = False
        validate_capture_server_args(self.args)

    def test_unsupported_modes_fail_before_loading_weights(self):
        cases = {
            "dp_size": 2,
            "dcp_size": 2,
            "speculative_algorithm": "DSpark",
            "enable_unified_memory": True,
            "enable_single_batch_overlap": True,
            "enable_hisparse": True,
            "load_format": "dummy",
            "custom_weight_loader": ["unbound-loader"],
            "enable_lora": True,
            "quantization": "fp8",
        }
        for name, value in cases.items():
            with self.subTest(option=name):
                previous = getattr(self.args, name)
                setattr(self.args, name, value)
                try:
                    with self.assertRaisesRegex(
                        ValueError, "training capture does not yet support"
                    ):
                        validate_capture_server_args(self.args)
                finally:
                    setattr(self.args, name, previous)

    def test_pd_requires_mooncake_and_waits_for_capture_context(self):
        for role in ("prefill", "decode"):
            self.args.disaggregation_mode = role
            self.args.disaggregation_transfer_backend = "mooncake"
            self.args.optimistic_prefill_attempts = 0
            for tp, pp in ((1, 1), (2, 1), (4, 1), (1, 2), (2, 2)):
                self.args.tp_size, self.args.pp_size = tp, pp
                self.args.speculative_algorithm = None
                validate_capture_server_args(self.args)
                self.args.speculative_algorithm = "DSPARK"
                validate_capture_server_args(self.args)
            self.args.pp_size = 1
            for name, value in (
                ("disaggregation_transfer_backend", "nixl"),
                ("optimistic_prefill_attempts", 1),
            ):
                with self.subTest(role=role, option=name):
                    previous = getattr(self.args, name)
                    setattr(self.args, name, value)
                    try:
                        with self.assertRaisesRegex(ValueError, "PD capture topology"):
                            validate_capture_server_args(self.args)
                    finally:
                        setattr(self.args, name, previous)

    def test_colocated_dspark_accepts_pipeline_capture(self):
        for tp, pp in ((2, 1), (1, 2), (2, 2)):
            with self.subTest(tp=tp, pp=pp):
                self.args.tp_size, self.args.pp_size = tp, pp
                self.args.speculative_algorithm = None
                validate_capture_server_args(self.args)
                self.args.speculative_algorithm = "DSPARK"
                with (
                    patch.object(
                        envs.SGLANG_RAGGED_VERIFY_MODE, "get", return_value="static"
                    ),
                ):
                    validate_capture_server_args(self.args)

    def test_dspark_verify_modes_still_reject_simulated_acceptance(self):
        self.args.speculative_algorithm = "DSPARK"
        with (
            patch.object(envs.SGLANG_RAGGED_VERIFY_MODE, "get", return_value="static"),
            patch.object(envs.SGLANG_SIMULATE_ACC_LEN, "get", return_value=0),
        ):
            validate_capture_server_args(self.args)
            self.args.disable_overlap_schedule = False
            validate_capture_server_args(self.args)
            self.args.disable_overlap_schedule = True
            for mode in ("compact", "cap-accept"):
                with (
                    self.subTest(mode=mode),
                    patch.object(
                        envs.SGLANG_RAGGED_VERIFY_MODE, "get", return_value=mode
                    ),
                ):
                    validate_capture_server_args(self.args)
            with (
                patch.object(envs.SGLANG_SIMULATE_ACC_LEN, "get", return_value=2),
                self.assertRaisesRegex(ValueError, "simulated speculative acceptance"),
            ):
                validate_capture_server_args(self.args)
            self.args.speculative_algorithm = "EAGLE"
            with self.assertRaisesRegex(ValueError, "speculative algorithm"):
                validate_capture_server_args(self.args)

    def test_invalid_contract_and_budgets_are_rejected(self):
        for override in (
            {"selected_layer_ids": []},
            {"selected_layer_ids": [0, 0]},
            {"max_host_bytes": 0},
            {"kv_d2h_batch_tokens": 0},
            {"kv_d2h_batch_tokens": 16},
            {"teacher_d2h_batch_tokens": 0},
            {"teacher_d2h_batch_tokens": 16},
            {"teacher_topk_backend": "unknown"},
            {"max_device_bytes": -1},
            {"sample_ratio": 2},
            {"catalog_endpoint": "file:///tmp/catalog"},
            {"journal_directory": "relative/journal"},
            {"unknown_option": True},
            {"adaptive": {"low_watermark": 0.8, "high_watermark": 0.5}},
            {"adaptive": {"interval_seconds": 0}},
            {"adaptive": {"writer_stall_seconds": -1}},
            {"adaptive": {"cooldown_seconds": 0}},
            {"adaptive": {"high_watermark": 2}},
            {"adaptive": {"unknown_option": True}},
        ):
            with self.subTest(override=override):
                self.path.write_text(json.dumps(self.config | override))
                with self.assertRaises((ContractError, msgspec.ValidationError)):
                    CaptureConfig.load(str(self.path))

    def test_adaptive_sampling_is_explicit_and_changes_config_identity(self):
        baseline = CaptureConfig.load(str(self.path))
        self.assertIsNone(baseline.adaptive)
        self.path.write_text(json.dumps(self.config | {"adaptive": {}}))
        adaptive = CaptureConfig.load(str(self.path))
        self.assertEqual(adaptive.adaptive.writer_stall_seconds, 10.0)
        self.assertNotEqual(baseline.fingerprint, adaptive.fingerprint)

    def test_staging_budget_is_explicit_and_changes_capture_identity(self):
        baseline = CaptureConfig.load(str(self.path))
        self.assertEqual(baseline.kv_d2h_batch_tokens, 1)
        self.assertEqual(baseline.teacher_d2h_batch_tokens, 1)
        self.assertEqual(baseline.max_device_bytes, 0)
        self.path.write_text(
            json.dumps(
                self.config | {"kv_d2h_batch_tokens": 16, "max_device_bytes": 1 << 20}
            )
        )
        staged = CaptureConfig.load(str(self.path))
        self.assertNotEqual(baseline.fingerprint, staged.fingerprint)
        self.path.write_text(
            json.dumps(
                self.config
                | {"teacher_d2h_batch_tokens": 16, "max_device_bytes": 1 << 20}
            )
        )
        teacher_staged = CaptureConfig.load(str(self.path))
        self.assertNotEqual(baseline.fingerprint, teacher_staged.fingerprint)
        self.assertNotEqual(staged.startup_policy, teacher_staged.startup_policy)


if __name__ == "__main__":
    unittest.main()
