"""Unit tests for SchedulerProfilerManager profiler_kwargs forwarding."""

import argparse
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang import profiler as cli_profiler
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ProfileReq, msgpack_decode, msgpack_encode
from sglang.srt.managers.scheduler_components import profiler_manager as pm_mod
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.observability import profile_utils as pu_mod
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeOOTPlatform(SRTPlatform):
    _enum = PlatformEnum.OOT
    device_name = "oot"
    device_type = "cpu"

    def get_torch_profiler_activity_str(self) -> str:
        return "OOT"

    def get_torch_profiler_activity(self) -> torch.profiler.ProfilerActivity:
        return torch.profiler.ProfilerActivity.PrivateUse1

    def get_torch_profiler_experimental_config(
        self,
        *,
        profiler_kwargs=None,
        with_cpu: bool = True,
        output_dir: str | None = None,
        profile_id: str | None = None,
        profile_prefix: str | None = None,
        stage: str | None = None,
    ):
        if profiler_kwargs and profiler_kwargs.get("bad_option"):
            raise ValueError("invalid experimental option: bad_option")
        if (
            stage is not None
            and profiler_kwargs
            and profiler_kwargs.get("fail_on_stage") == stage
        ):
            raise ValueError(f"unsupported stage: {stage}")
        return {
            "kwargs": profiler_kwargs,
            "with_cpu": with_cpu,
            "output_dir": output_dir,
            "profile_id": profile_id,
            "profile_prefix": profile_prefix,
            "stage": stage,
        }


class _DefaultOOTPlatform(SRTPlatform):
    _enum = PlatformEnum.OOT
    device_name = "oot"
    device_type = "cpu"

    def get_torch_profiler_activity_str(self) -> str:
        return "OOT"

    def get_torch_profiler_activity(self) -> torch.profiler.ProfilerActivity:
        return torch.profiler.ProfilerActivity.PrivateUse1


class TestProfilerKwargs(CustomTestCase):
    def _make_manager(
        self, *, v2: bool = False, get_forward_ct=lambda: 0
    ) -> SchedulerProfilerManager:
        dev = types.SimpleNamespace(gpu_id=0, base_gpu_id=0)
        with (
            envs.SGLANG_PROFILE_V2.override(v2),
            patch.object(
                pm_mod,
                "get_dp_tp_group",
                return_value=types.SimpleNamespace(cpu_group=None),
            ),
            patch.object(pu_mod, "get_device", return_value=dev),
        ):
            return SchedulerProfilerManager(get_forward_ct=get_forward_ct)

    def test_profile_req_msgpack_roundtrip(self):
        for kwargs in (
            None,
            {"host_tracer_level": 2, "mode": "COMPUTE", "flag": True},
        ):
            req = ProfileReq(profile_prefix="r1", profiler_kwargs=kwargs)
            self.assertEqual(msgpack_decode(msgpack_encode(req)), req)

    def test_oot_experimental_config_forwarding_and_stage_context(self):
        oot = _FakeOOTPlatform()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            envs.SGLANG_PROFILE_V2.override(False),
            patch.object(pm_mod, "current_platform", oot),
            patch.object(
                torch.profiler.ProfilerActivity,
                "OOT",
                torch.profiler.ProfilerActivity.PrivateUse1,
                create=True,
            ),
            patch("torch.profiler.profile") as mock_profile,
        ):
            mgr = self._make_manager()
            ok = mgr._profile(
                ProfileReq(
                    output_dir=tmpdir,
                    profile_id="pid-1",
                    profile_prefix="runA",
                    activities=["CPU", "OOT"],
                    profiler_kwargs={"level": 3},
                )
            )
            self.assertTrue(ok.success, ok.message)
            self.assertEqual(
                mock_profile.call_args.kwargs["experimental_config"],
                {
                    "kwargs": {"level": 3},
                    "with_cpu": True,
                    "output_dir": tmpdir,
                    "profile_id": "pid-1",
                    "profile_prefix": "runA",
                    "stage": None,
                },
            )

            mock_profile.reset_mock()
            mgr_staged = self._make_manager()
            staged = mgr_staged._profile(
                ProfileReq(
                    output_dir=tmpdir,
                    num_steps=1,
                    profile_by_stage=True,
                    profile_id="pid-2",
                    profile_prefix="staged",
                    activities=["OOT"],
                    profiler_kwargs={"level": 1},
                )
            )
            self.assertTrue(staged.success, staged.message)
            mock_profile.assert_not_called()

            mgr_staged._profile_batch_predicate(
                types.SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND, batch_size=lambda: 1
                )
            )
            self.assertEqual(
                mock_profile.call_args.kwargs["experimental_config"],
                {
                    "kwargs": {"level": 1},
                    "with_cpu": False,
                    "output_dir": tmpdir,
                    "profile_id": "pid-2",
                    "profile_prefix": "staged",
                    "stage": "EXTEND",
                },
            )

            # An OOT platform that inherits DeviceMixin's default returns None.
            mock_profile.reset_mock()
            with patch.object(pm_mod, "current_platform", _DefaultOOTPlatform()):
                mgr_default = self._make_manager()
                res_default = mgr_default._profile(
                    ProfileReq(
                        output_dir=tmpdir,
                        activities=["OOT"],
                        profiler_kwargs={"level": 1},
                    )
                )
            self.assertTrue(res_default.success, res_default.message)
            self.assertIsNone(mock_profile.call_args.kwargs["experimental_config"])

    def test_invalid_kwargs_and_runtime_start_errors_disarm_cleanly(self):
        oot = _FakeOOTPlatform()
        forward_ct = [0]
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            envs.SGLANG_PROFILE_V2.override(False),
            patch.object(pm_mod, "current_platform", oot),
            patch.object(
                torch.profiler.ProfilerActivity,
                "OOT",
                torch.profiler.ProfilerActivity.PrivateUse1,
                create=True,
            ),
            patch("torch.profiler.profile") as mock_profile,
        ):
            mgr = self._make_manager(get_forward_ct=lambda: forward_ct[0])

            # Non-dict profiler_kwargs passed programmatically is rejected cleanly.
            bad_type = mgr._profile(
                ProfileReq(
                    output_dir=tmpdir,
                    num_steps=1,
                    activities=["OOT"],
                    profiler_kwargs=["not", "a", "dict"],  # type: ignore[arg-type]
                )
            )
            self.assertFalse(bad_type.success)
            self.assertIn("must be a dict", bad_type.message)
            self.assertFalse(mgr.profile_in_progress)

            # Immediate, deferred start_step, and profile_by_stage all fail upfront.
            for extra in (
                {"num_steps": 2},
                {"start_step": 5, "num_steps": 2},
                {"profile_by_stage": True, "num_steps": 2},
            ):
                with self.subTest(mode=extra):
                    bad = mgr._profile(
                        ProfileReq(
                            output_dir=tmpdir,
                            activities=["OOT"],
                            profiler_kwargs={"bad_option": 1},
                            **extra,
                        )
                    )
                    self.assertFalse(bad.success)
                    self.assertFalse(mgr.profile_in_progress)
                    self.assertFalse(mgr.profile_by_stage)
                    self.assertIsNone(mgr.profiler_start_forward_ct)
                    self.assertIsNone(mgr.profiler_target_forward_ct)
            mock_profile.assert_not_called()

            # Low-level RuntimeError from profiler.start() also disarms target counters.
            mock_profile.return_value.start.side_effect = RuntimeError(
                "unrecognized native flag"
            )
            rt_fail = mgr._profile(
                ProfileReq(
                    output_dir=tmpdir,
                    num_steps=1,
                    activities=["OOT"],
                    profiler_kwargs={"level": 2},
                )
            )
            self.assertFalse(rt_fail.success)
            self.assertIn("unrecognized native flag", rt_fail.message)
            self.assertIsNone(mgr.torch_profiler)
            self.assertFalse(mgr.profile_in_progress)
            self.assertIsNone(mgr.profiler_target_forward_ct)

            forward_ct[0] = 10
            with patch.object(mgr, "_stop_profile") as mock_stop:
                mgr._profile_batch_predicate(
                    types.SimpleNamespace(forward_mode=ForwardMode.DECODE)
                )
                mock_stop.assert_not_called()

            # Stage-time ValueError inside _start_profile disarms staged profiling
            # without crashing _profile_batch_predicate on the same batch.
            mock_profile.reset_mock()
            mock_profile.return_value.start.side_effect = None
            mgr_stage_err = self._make_manager()
            armed = mgr_stage_err._profile(
                ProfileReq(
                    output_dir=tmpdir,
                    num_steps=1,
                    profile_by_stage=True,
                    activities=["OOT"],
                    profiler_kwargs={"fail_on_stage": "DECODE"},
                )
            )
            self.assertTrue(armed.success, armed.message)
            mgr_stage_err._profile_batch_predicate(
                types.SimpleNamespace(
                    forward_mode=ForwardMode.DECODE, batch_size=lambda: 1
                )
            )
            self.assertFalse(mgr_stage_err.profile_in_progress)
            self.assertFalse(mgr_stage_err.profile_by_stage)
            mock_profile.assert_not_called()

    def test_v2_rejects_profiler_kwargs_explicitly(self):
        mgr_v2 = self._make_manager(v2=True)
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            envs.SGLANG_PROFILE_V2.override(True),
        ):
            rejected = mgr_v2._profile(
                ProfileReq(
                    output_dir=tmpdir,
                    num_steps=1,
                    profile_by_stage=True,
                    activities=["CPU"],
                    profiler_kwargs={"level": 1},
                )
            )
            self.assertFalse(rejected.success)
            self.assertIn("SGLANG_PROFILE_V2", rejected.message)

            accepted = mgr_v2._profile(
                ProfileReq(
                    output_dir=tmpdir,
                    num_steps=1,
                    profile_by_stage=True,
                    activities=["CPU"],
                )
            )
            self.assertTrue(accepted.success, accepted.message)

    def test_cli_profiler_kwargs_validation_and_forwarding(self):
        self.assertEqual(
            cli_profiler._parse_profiler_kwargs('{"level": 2}'), {"level": 2}
        )
        for bad_raw in ("not-json", "[1, 2]", '"str"', "123", "null"):
            with self.subTest(bad_raw=bad_raw):
                with self.assertRaises(argparse.ArgumentTypeError):
                    cli_profiler._parse_profiler_kwargs(bad_raw)

        posted = []
        fake_get = MagicMock()
        fake_get.return_value.json.return_value = {}
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(cli_profiler.requests, "get", fake_get),
            patch.object(
                cli_profiler.requests,
                "post",
                side_effect=lambda url, json, **_: (posted.append(json), MagicMock())[
                    1
                ],
            ),
        ):
            with self.assertRaises(TypeError):
                cli_profiler.run_profile(
                    "http://localhost:30000",
                    3,
                    ["CPU"],
                    output_dir=tmpdir,
                    profiler_kwargs=[1, 2],  # type: ignore[arg-type]
                )
            cli_profiler.run_profile(
                "http://localhost:30000", 3, ["CPU"], output_dir=tmpdir
            )
            cli_profiler.run_profile(
                "http://localhost:30000",
                3,
                ["CPU"],
                output_dir=tmpdir,
                profiler_kwargs={"level": 2},
            )
        self.assertNotIn("profiler_kwargs", posted[0])
        self.assertEqual(posted[1]["profiler_kwargs"], {"level": 2})


if __name__ == "__main__":
    unittest.main()
