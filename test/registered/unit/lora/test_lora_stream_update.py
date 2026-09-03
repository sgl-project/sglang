"""Streamed LoRA updates: prefix routing, the session stash, and the
whole-adapter apply at end_weight_update."""

import unittest
from unittest.mock import MagicMock, Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.io_struct import LoRAUpdateOutput
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
    _sha256_tensor,
    _split_lora_named_tensors,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    LocalSerializedTensor,
)
from sglang.srt.utils import MultiprocessingSerializer

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


BASE_NAME = "model.layers.0.self_attn.q_proj.weight"
LORA_A = "model.layers.0.self_attn.q_proj.lora_A.weight"
LORA_B = "model.layers.0.self_attn.q_proj.lora_B.weight"


def _make_manager(**overrides) -> SchedulerWeightUpdaterManager:
    kwargs = dict(
        tp_worker=MagicMock(),
        draft_worker=None,
        tp_cpu_group=MagicMock(),
        memory_saver_adapter=MagicMock(),
        flush_cache=Mock(return_value=True),
        is_fully_idle=Mock(return_value=True),
    )
    kwargs.update(overrides)
    return SchedulerWeightUpdaterManager(**kwargs)


class TestSplitLoraNamedTensors(CustomTestCase):
    def test_partitions_on_prefix(self):
        t = torch.zeros(1)
        base, lora = _split_lora_named_tensors(
            [
                (BASE_NAME, t),
                (f"miles_lora:{LORA_A}", t),
                (f"__miles_slot_3:{LORA_B}", t),
            ]
        )
        self.assertEqual([n for n, _ in base], [BASE_NAME])
        self.assertEqual(
            [n for n, _ in lora],
            [f"miles_lora:{LORA_A}", f"__miles_slot_3:{LORA_B}"],
        )

    def test_unprefixed_lora_tensor_is_a_sender_bug(self):
        with self.assertRaisesRegex(AssertionError, "without a"):
            _split_lora_named_tensors([(LORA_A, torch.zeros(1))])

    def test_pure_base_payload_passes_through(self):
        base, lora = _split_lora_named_tensors([(BASE_NAME, torch.zeros(1))])
        self.assertEqual(len(base), 1)
        self.assertEqual(lora, [])


class TestLoraStash(CustomTestCase):
    def test_stash_groups_by_adapter(self):
        mgr = _make_manager()
        a, b = torch.zeros(2), torch.ones(2)
        mgr._stash_lora_tensors(
            [(f"miles_lora:{LORA_A}", a), (f"__miles_slot_1:{LORA_A}", b)]
        )
        self.assertEqual(set(mgr._lora_stash), {"miles_lora", "__miles_slot_1"})
        self.assertIs(mgr._lora_stash["miles_lora"][LORA_A], a)
        self.assertIs(mgr._lora_stash["__miles_slot_1"][LORA_A], b)

    def test_split_at_first_colon_only(self):
        mgr = _make_manager()
        mgr._stash_lora_tensors([("name:with:colon.lora_A.weight", torch.zeros(1))])
        self.assertIn("with:colon.lora_A.weight", mgr._lora_stash["name"])

    def test_per_rank_payload_is_unwrapped(self):
        """A LoRA tensor never reaches the base loader, so the stash is the last
        place a per-rank payload can be reduced to this rank's tensor."""
        tp_worker = MagicMock()
        tp_worker.ps.tp_rank = 1
        mgr = _make_manager(tp_worker=tp_worker)
        rank_tensors = [torch.zeros(2), torch.ones(2)]
        payload = LocalSerializedTensor(
            values=[MultiprocessingSerializer.serialize(t) for t in rank_tensors]
        )
        mgr._stash_lora_tensors([(f"miles_lora:{LORA_A}", payload)])
        self.assertTrue(
            torch.equal(mgr._lora_stash["miles_lora"][LORA_A], rank_tensors[1])
        )


class TestApplyLoraStash(CustomTestCase):
    def _manager_with_lora(self, apply_result=None):
        mgr = _make_manager()
        lora_manager = MagicMock()
        lora_manager.apply_streamed_adapter.return_value = (
            apply_result if apply_result is not None else LoRAUpdateOutput(success=True)
        )
        mgr.tp_worker.model_runner.lora_manager = lora_manager
        return mgr, lora_manager

    def test_empty_stash_is_noop(self):
        mgr = _make_manager()
        mgr.tp_worker.model_runner.lora_manager = None
        success, _ = mgr._apply_lora_stash(None)
        self.assertTrue(success)

    def test_stash_without_lora_manager_fails(self):
        mgr = _make_manager()
        mgr.tp_worker.model_runner.lora_manager = None
        mgr._stash_lora_tensors([(f"miles_lora:{LORA_A}", torch.zeros(1))])
        success, message = mgr._apply_lora_stash(None)
        self.assertFalse(success)
        self.assertIn("--enable-lora", message)

    def test_applies_each_adapter_and_clears_stash(self):
        mgr, lora_manager = self._manager_with_lora()
        mgr._stash_lora_tensors(
            [(f"a:{LORA_A}", torch.zeros(1)), (f"b:{LORA_A}", torch.zeros(1))]
        )
        success, _ = mgr._apply_lora_stash(None)
        self.assertTrue(success)
        self.assertEqual(lora_manager.apply_streamed_adapter.call_count, 2)
        applied = [
            c.args[0] for c in lora_manager.apply_streamed_adapter.call_args_list
        ]
        self.assertEqual(applied, ["a", "b"])  # sorted, deterministic
        self.assertEqual(mgr._lora_stash, {})

    def test_manager_failure_propagates(self):
        mgr, _ = self._manager_with_lora(
            apply_result=LoRAUpdateOutput(success=False, error_message="unregistered")
        )
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(1))])
        success, message = mgr._apply_lora_stash(None)
        self.assertFalse(success)
        self.assertIn("unregistered", message)

    def test_checksum_match_passes(self):
        mgr, _ = self._manager_with_lora()
        t = torch.arange(4, dtype=torch.float32)
        mgr._stash_lora_tensors([(f"a:{LORA_A}", t)])
        success, _ = mgr._apply_lora_stash({"a": {LORA_A: _sha256_tensor(t)}})
        self.assertTrue(success)

    def test_checksum_mismatch_fails(self):
        mgr, lora_manager = self._manager_with_lora()
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(4))])
        success, message = mgr._apply_lora_stash(
            {"a": {LORA_A: _sha256_tensor(torch.ones(4))}}
        )
        self.assertFalse(success)
        self.assertIn("checksum mismatch", message)
        lora_manager.apply_streamed_adapter.assert_not_called()

    def test_checksum_name_set_mismatch_fails(self):
        mgr, _ = self._manager_with_lora()
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(1))])
        success, message = mgr._apply_lora_stash({"a": {LORA_A: "x", LORA_B: "y"}})
        self.assertFalse(success)
        self.assertIn("expected manifest", message)

    def test_manifest_adapter_set_mismatch_fails(self):
        """An adapter the sender promised but streamed nothing for is a sender bug
        reported to the caller, not a scheduler kill."""
        mgr, lora_manager = self._manager_with_lora()
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(1))])
        success, message = mgr._apply_lora_stash(
            {"a": {LORA_A: "x"}, "b": {LORA_A: "y"}}
        )
        self.assertFalse(success)
        self.assertIn("expected manifest", message)
        self.assertIn("'b'", message)
        lora_manager.apply_streamed_adapter.assert_not_called()

    def test_forget_admits_a_different_tensor_set(self):
        """Without the forget on re-register/unload, a name that legitimately
        changes its tensor set is rejected as a partial stream forever."""
        mgr, _ = self._manager_with_lora()
        mgr._stash_lora_tensors(
            [(f"a:{LORA_A}", torch.zeros(1)), (f"a:{LORA_B}", torch.zeros(1))]
        )
        self.assertTrue(mgr._apply_lora_stash(None)[0])
        mgr.forget_lora_adapter("a")
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(1))])
        self.assertTrue(mgr._apply_lora_stash(None)[0])

    def test_shrunken_tensor_set_rejected(self):
        # The adapter's tensor set is static for its lifetime; a change means a
        # partial stream, which the whole-adapter replace would silently zero.
        mgr, _ = self._manager_with_lora()
        mgr._stash_lora_tensors(
            [(f"a:{LORA_A}", torch.zeros(1)), (f"a:{LORA_B}", torch.zeros(1))]
        )
        self.assertTrue(mgr._apply_lora_stash(None)[0])
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(1))])
        success, message = mgr._apply_lora_stash(None)
        self.assertFalse(success)
        self.assertIn("different tensor", message)


class TestLoRAManagerStreamEntryPoints(CustomTestCase):
    def _bare_manager(self):
        manager = object.__new__(LoRAManager)
        manager.lora_refs = {}
        return manager

    def test_require_registered(self):
        manager = self._bare_manager()
        ref = LoRARef(lora_name="miles_lora", lora_path="__stream__")
        manager.lora_refs[ref.lora_id] = ref
        self.assertIs(manager.require_registered("miles_lora"), ref)
        with self.assertRaisesRegex(ValueError, "unregistered"):
            manager.require_registered("missing")

    def test_register_delegates_with_empty_tensors_and_upsert(self):
        manager = self._bare_manager()
        manager.load_lora_adapter_from_tensors = MagicMock(
            return_value=LoRAUpdateOutput(success=True)
        )
        ref = LoRARef(lora_name="miles_lora", lora_path="__stream__")
        result = manager.register_lora_adapter(ref, {"r": 8})
        self.assertTrue(result.success)
        kwargs = manager.load_lora_adapter_from_tensors.call_args.kwargs
        self.assertEqual(kwargs["tensors"], {})
        self.assertEqual(kwargs["config_dict"], {"r": 8})
        self.assertTrue(kwargs["upsert"])

    def test_apply_streamed_adapter_reuses_registered_config(self):
        manager = self._bare_manager()
        ref = LoRARef(lora_name="miles_lora", lora_path="__stream__")
        manager.lora_refs[ref.lora_id] = ref
        manager.load_lora_adapter_from_tensors = MagicMock(
            return_value=LoRAUpdateOutput(success=True)
        )
        tensors = {LORA_A: torch.zeros(1)}
        result = manager.apply_streamed_adapter("miles_lora", tensors)
        self.assertTrue(result.success)
        args, kwargs = manager.load_lora_adapter_from_tensors.call_args
        self.assertIs(args[0], ref)
        self.assertIs(args[1], tensors)
        self.assertIsNone(kwargs["config_dict"])
        self.assertTrue(kwargs["upsert"])

    def test_apply_streamed_adapter_unregistered_fails(self):
        manager = self._bare_manager()
        result = manager.apply_streamed_adapter("missing", {LORA_A: torch.zeros(1)})
        self.assertFalse(result.success)
        self.assertIn("unregistered", result.error_message)


if __name__ == "__main__":
    unittest.main()


class TestWeightVersionDeferredToCommit(CustomTestCase):
    """Inside a session the version reported by a bucket must not reach the
    scheduler until end_weight_update commits; a failed commit drops it."""

    def _manager(self):
        mgr = _make_manager(scheduler=Mock())
        mgr._weight_update_in_progress = True
        return mgr

    def test_bucket_version_is_held_while_the_session_is_open(self):
        mgr = self._manager()
        mgr.record_weight_version_after_update("7")
        mgr.scheduler.record_weight_version_change.assert_not_called()
        self.assertEqual(mgr._weight_update_pending_version, "7")

    def test_outside_a_session_the_version_is_recorded_at_once(self):
        mgr = _make_manager(scheduler=Mock())
        mgr.record_weight_version_after_update("7")
        mgr.scheduler.record_weight_version_change.assert_called_once_with(
            new_version="7"
        )

    def test_commit_records_the_held_version(self):
        mgr = self._manager()
        mgr.record_weight_version_after_update("7")
        mgr._weight_update_in_progress = False
        mgr.record_weight_version_after_update(mgr._weight_update_pending_version)
        mgr.scheduler.record_weight_version_change.assert_called_once_with(
            new_version="7"
        )
