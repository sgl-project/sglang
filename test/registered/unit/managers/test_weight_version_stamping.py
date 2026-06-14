"""Unit tests for weight-version admission stamping and the scheduler-side
version bump: Req.__init__ stamps weight_version_start and (behind
--enable-weight-version-kv-isolation) composes a fixed-position version
namespace into extra_key that stays unambiguous under the delimiter-free
lora_id suffix; SchedulerWeightUpdaterManager drains the forward pipeline
before mutating weights and adopts recv_req.weight_version only on success.
Pure CPU with mocked workers."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(extra_key=None, lora_id=None):
    return Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=SamplingParams(max_new_tokens=8),
        extra_key=extra_key,
        lora_id=lora_id,
    )


def _fake_server_args(version="7", isolation=True):
    return SimpleNamespace(
        weight_version=version,
        enable_weight_version_kv_isolation=isolation,
    )


class TestReqAdmissionStamping(CustomTestCase):
    def test_stamps_version_and_namespaces_extra_key(self):
        with patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=_fake_server_args(),
        ):
            req = _make_req()
            self.assertEqual(req.weight_version_start, "7")
            self.assertEqual(req.extra_key, "wv1:7;")

            req = _make_req(extra_key="salt")
            self.assertEqual(req.extra_key, "wv1:7;salt")

    def test_version_prefix_stays_fixed_under_lora_suffix(self):
        with patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=_fake_server_args(),
        ):
            req = _make_req(extra_key="salt", lora_id="loraX")
            # lora_id concatenates after the user key, so the version segment
            # keeps its fixed, delimiter-terminated position at the front.
            self.assertEqual(req.extra_key, "wv1:7;saltloraX")
            self.assertEqual(req.lora_id, "loraX")

    def test_delimiter_in_version_cannot_collide_namespaces(self):
        # Without the length prefix, version "a;b" + extra_key "c" and
        # version "a" + extra_key "b;c" would compose the same namespace
        # and share KV entries across different weights.
        with patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=_fake_server_args(version="a;b"),
        ):
            key_one = _make_req(extra_key="c").extra_key
        with patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=_fake_server_args(version="a"),
        ):
            key_two = _make_req(extra_key="b;c").extra_key
        self.assertNotEqual(key_one, key_two)

    def test_isolation_disabled_keeps_extra_key_but_still_stamps(self):
        with patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=_fake_server_args(isolation=False),
        ):
            req = _make_req(extra_key="salt")
            self.assertEqual(req.weight_version_start, "7")
            self.assertEqual(req.extra_key, "salt")

    def test_bare_unit_context_without_global_args(self):
        with patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            side_effect=ValueError("Global server args is not set yet!"),
        ):
            req = _make_req(extra_key="salt")
            self.assertIsNone(req.weight_version_start)
            self.assertEqual(req.extra_key, "salt")


def _make_updater(tp_worker, **overrides):
    kwargs = dict(
        tp_worker=tp_worker,
        draft_worker=None,
        tp_cpu_group=None,
        memory_saver_adapter=None,
        flush_cache=MagicMock(return_value=True),
        is_fully_idle=MagicMock(return_value=True),
        drain_forward_pipeline=MagicMock(),
        on_weight_version_bump=MagicMock(),
        metrics_collector=None,
    )
    kwargs.update(overrides)
    return SchedulerWeightUpdaterManager(**kwargs)


class TestWeightUpdaterVersionBump(CustomTestCase):
    def _recv_req(self, weight_version="9"):
        return UpdateWeightFromDiskReqInput(
            model_path="dummy",
            flush_cache=False,
            weight_version=weight_version,
        )

    def test_successful_update_bumps_version_and_runs_hooks(self):
        parent = MagicMock()
        parent.tp_worker.update_weights_from_disk.return_value = (True, "ok")
        updater = _make_updater(parent.tp_worker, drain_forward_pipeline=parent.drain)
        fake_args = _fake_server_args(version="old")
        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater.get_global_server_args",
            return_value=fake_args,
        ), patch(
            "sglang.srt.managers.mm_utils.embedding_cache", MagicMock()
        ) as mm_cache:
            out = updater.update_weights_from_disk(self._recv_req())

        self.assertTrue(out.success)
        self.assertEqual(fake_args.weight_version, "9")
        updater.on_weight_version_bump.assert_called_once()
        mm_cache.clear.assert_called_once()
        # The forward pipeline must be quiesced before weights mutate.
        call_names = [c[0] for c in parent.mock_calls]
        self.assertLess(
            call_names.index("drain"),
            call_names.index("tp_worker.update_weights_from_disk"),
        )

    def test_failed_update_does_not_bump(self):
        tp_worker = MagicMock()
        tp_worker.update_weights_from_disk.return_value = (False, "load failed")
        updater = _make_updater(tp_worker)
        fake_args = _fake_server_args(version="old")
        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater.get_global_server_args",
            return_value=fake_args,
        ):
            out = updater.update_weights_from_disk(self._recv_req())

        self.assertFalse(out.success)
        self.assertEqual(fake_args.weight_version, "old")
        updater.on_weight_version_bump.assert_not_called()

    def test_update_without_version_clears_mm_cache_but_keeps_version(self):
        tp_worker = MagicMock()
        tp_worker.update_weights_from_disk.return_value = (True, "ok")
        updater = _make_updater(tp_worker)
        fake_args = _fake_server_args(version="old")
        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater.get_global_server_args",
            return_value=fake_args,
        ), patch(
            "sglang.srt.managers.mm_utils.embedding_cache", MagicMock()
        ) as mm_cache:
            out = updater.update_weights_from_disk(self._recv_req(weight_version=None))

        self.assertTrue(out.success)
        self.assertEqual(fake_args.weight_version, "old")
        updater.on_weight_version_bump.assert_not_called()
        mm_cache.clear.assert_called_once()

    def test_draft_failure_blocks_bump(self):
        tp_worker = MagicMock()
        tp_worker.update_weights_from_disk.return_value = (True, "ok")
        draft_worker = MagicMock()
        draft_worker.update_weights_from_disk.return_value = (False, "draft failed")
        updater = _make_updater(tp_worker, draft_worker=draft_worker)
        fake_args = _fake_server_args(version="old")
        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater.get_global_server_args",
            return_value=fake_args,
        ):
            out = updater.update_weights_from_disk(self._recv_req())

        self.assertFalse(out.success)
        # Bump only on full success: target *and* draft.
        self.assertEqual(fake_args.weight_version, "old")
        updater.on_weight_version_bump.assert_not_called()


if __name__ == "__main__":
    unittest.main()
