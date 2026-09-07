import ctypes
import hashlib
import sys
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    EndWeightUpdateReqInput,
    LoRAUpdateOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
    _sha256_tensor,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


PARAM_A = "model.layers.0.self_attn.q_proj.weight"
PARAM_B = "model.layers.0.mlp.down_proj.weight"
LORA_A = "model.layers.0.self_attn.q_proj.lora_A.weight"


class _FakeModel:
    def __init__(self, params: Dict[str, torch.Tensor]):
        self._params = params

    def named_parameters(self):
        return list(self._params.items())


class _FakeRunner:
    def __init__(self, params: Dict[str, torch.Tensor]):
        self._params = params
        self.end_calls: List[bool] = []

    def end_weight_update(self, run_post_load: bool) -> None:
        self.end_calls.append(run_post_load)
        for tensor in self._params.values():
            tensor.mul_(-1.0)


@contextmanager
def _fake_tp_collectives(
    other_statuses: Tuple[Tuple[bool, str], ...] = (),
) -> Iterator[List[Tuple[bool, str]]]:
    """Stand in for the engine TP CPU group: rank 0 is local, the rest are given."""
    gathered: List[Tuple[bool, str]] = []

    def fake_all_gather_object(object_list, obj, group=None):
        gathered.append(obj)
        object_list[0] = obj
        for i, status in enumerate(other_statuses):
            object_list[i + 1] = status

    with patch.object(
        torch.distributed, "get_world_size", return_value=1 + len(other_statuses)
    ), patch.object(
        torch.distributed, "all_gather_object", side_effect=fake_all_gather_object
    ), patch.object(
        torch.distributed, "barrier"
    ):
        yield gathered


def _make_manager(
    params: Optional[Dict[str, torch.Tensor]] = None,
    tp_rank: int = 0,
    model: Optional[object] = None,
) -> Tuple[SchedulerWeightUpdaterManager, _FakeRunner]:
    params = {} if params is None else params
    runner = _FakeRunner(params)
    tp_worker = MagicMock()
    tp_worker.ps.tp_rank = tp_rank
    tp_worker.model_runner.model = _FakeModel(params) if model is None else model
    tp_worker.iter_runners.return_value = [("", runner)]
    manager = SchedulerWeightUpdaterManager(
        tp_worker=tp_worker,
        draft_worker=None,
        tp_cpu_group=MagicMock(),
        memory_saver_adapter=MagicMock(),
        flush_cache=Mock(return_value=True),
        is_fully_idle=Mock(return_value=True),
        scheduler=MagicMock(),
    )
    manager._weight_update_in_progress = True
    manager._weight_update_sync_base = True
    manager._weight_update_pending_version = "v7"
    return manager, runner


def _manifest(params: Dict[str, torch.Tensor], tp_rank: int = 0) -> Dict:
    return {str(tp_rank): {n: _sha256_tensor(t) for n, t in params.items()}}


class TestBaseWeightChecksumVerification(CustomTestCase):
    def test_matching_manifest_finalizes_and_publishes_the_version(self):
        """A manifest matching every registered parameter lets the session finalize."""
        params = {PARAM_A: torch.arange(4, dtype=torch.float32)}
        manager, runner = _make_manager(params)
        with _fake_tp_collectives():
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(
                    expected_base_weight_checksums=_manifest(params)
                )
            )
        self.assertTrue(output.success)
        self.assertEqual(runner.end_calls, [True])
        manager.scheduler.record_weight_version_change.assert_called_once_with(
            new_version="v7"
        )

    def test_corrupted_tensor_fails_without_finalizing_or_publishing(self):
        """A single flipped byte in a received tensor blocks finalization and the version."""
        params = {PARAM_A: torch.arange(4, dtype=torch.float32)}
        expected = _manifest(params)
        params[PARAM_A][2] = 99.0
        manager, runner = _make_manager(params)
        with _fake_tp_collectives():
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(expected_base_weight_checksums=expected)
            )
        self.assertFalse(output.success)
        self.assertIn("checksum mismatch", output.message)
        self.assertEqual(runner.end_calls, [])
        manager.scheduler.record_weight_version_change.assert_not_called()

    def test_missing_tensor_name_fails(self):
        """A tensor the sender promised but never wrote is reported as missing."""
        params = {PARAM_A: torch.arange(4, dtype=torch.float32)}
        expected = _manifest(params)
        expected["0"][PARAM_B] = _sha256_tensor(torch.ones(2))
        manager, runner = _make_manager(params)
        with _fake_tp_collectives():
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(expected_base_weight_checksums=expected)
            )
        self.assertFalse(output.success)
        self.assertIn(PARAM_B, output.message)
        self.assertEqual(runner.end_calls, [])

    def test_unlisted_registered_parameter_fails(self):
        """A sparse manifest that omits a registered parameter is rejected, not ignored."""
        params = {
            PARAM_A: torch.arange(4, dtype=torch.float32),
            PARAM_B: torch.ones(3),
        }
        expected = {"0": {PARAM_A: _sha256_tensor(params[PARAM_A])}}
        manager, runner = _make_manager(params)
        with _fake_tp_collectives():
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(expected_base_weight_checksums=expected)
            )
        self.assertFalse(output.success)
        self.assertIn("unexpected", output.message)
        self.assertEqual(runner.end_calls, [])

    def test_manifest_without_this_rank_fails(self):
        """A manifest that skips this engine TP rank must not pass the rank unverified."""
        params = {PARAM_A: torch.arange(4, dtype=torch.float32)}
        manager, runner = _make_manager(params, tp_rank=3)
        with _fake_tp_collectives():
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(
                    expected_base_weight_checksums=_manifest(params, tp_rank=0)
                )
            )
        self.assertFalse(output.success)
        self.assertIn("no manifest entry", output.message)
        self.assertEqual(runner.end_calls, [])

    def test_empty_model_with_empty_rank_manifest_passes(self):
        """A rank that registers no parameters is verified by an empty manifest entry."""
        manager, runner = _make_manager({})
        with _fake_tp_collectives():
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(expected_base_weight_checksums={"0": {}})
            )
        self.assertTrue(output.success)
        self.assertEqual(runner.end_calls, [True])

    def test_verification_reads_the_bytes_before_post_load_transformation(self):
        """The manifest is compared against the received bytes, not the transformed ones."""
        params = {PARAM_A: torch.arange(4, dtype=torch.float32)}
        expected = _manifest(params)
        manager, runner = _make_manager(params)
        with _fake_tp_collectives():
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(expected_base_weight_checksums=expected)
            )
        self.assertTrue(output.success)
        self.assertEqual(runner.end_calls, [True])
        self.assertTrue(
            torch.equal(params[PARAM_A], -torch.arange(4, dtype=torch.float32))
        )

    def test_manifest_in_a_sync_base_false_session_fails(self):
        """A base-weight manifest is meaningless for an adapter-only session."""
        params = {PARAM_A: torch.arange(4, dtype=torch.float32)}
        manager, runner = _make_manager(params)
        manager._weight_update_sync_base = False
        with _fake_tp_collectives():
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(
                    expected_base_weight_checksums=_manifest(params)
                )
            )
        self.assertFalse(output.success)
        self.assertIn("sync_base=True", output.message)
        manager.scheduler.record_weight_version_change.assert_not_called()


class TestBaseWeightChecksumAcrossRanks(CustomTestCase):
    def test_another_rank_failure_fails_this_rank_too(self):
        """One rank's mismatch must stop finalization on every rank of the group."""
        params = {PARAM_A: torch.arange(4, dtype=torch.float32)}
        manager, runner = _make_manager(params)
        with _fake_tp_collectives(
            other_statuses=((False, "tp_rank 1 checksum mismatch for 'x'"),)
        ):
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(
                    expected_base_weight_checksums=_manifest(params)
                )
            )
        self.assertFalse(output.success)
        self.assertIn("group rank 1", output.message)
        self.assertEqual(runner.end_calls, [])
        manager.scheduler.record_weight_version_change.assert_not_called()

    def test_local_failure_still_reaches_the_collective(self):
        """A locally failing rank must still gather, or the healthy ranks hang."""
        params = {PARAM_A: torch.arange(4, dtype=torch.float32)}
        expected = _manifest(params)
        expected["0"][PARAM_A] = _sha256_tensor(torch.zeros(4))
        manager, _ = _make_manager(params)
        with _fake_tp_collectives(other_statuses=((True, "Success"),)) as gathered:
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(expected_base_weight_checksums=expected)
            )
        self.assertFalse(output.success)
        self.assertEqual(len(gathered), 1)
        self.assertFalse(gathered[0][0])

    def test_raising_local_verification_is_reported_not_propagated(self):
        """A local exception becomes a gathered failure instead of skipping the collective."""
        model = MagicMock()
        model.named_parameters.side_effect = RuntimeError("cuda is gone")
        manager, runner = _make_manager({}, model=model)
        with _fake_tp_collectives(other_statuses=((True, "Success"),)) as gathered:
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(expected_base_weight_checksums={"0": {}})
            )
        self.assertFalse(output.success)
        self.assertIn("cuda is gone", output.message)
        self.assertEqual(len(gathered), 1)
        self.assertEqual(runner.end_calls, [])


class TestBaseWeightChecksumCompatibility(CustomTestCase):
    def test_absent_manifest_keeps_the_previous_behavior(self):
        """Without a manifest the session finalizes and never touches the parameters."""
        model = MagicMock()
        model.named_parameters.side_effect = AssertionError(
            "parameters must not be hashed when no manifest is supplied"
        )
        manager, runner = _make_manager({}, model=model)
        with _fake_tp_collectives() as gathered:
            output = manager.end_weight_update(EndWeightUpdateReqInput())
        self.assertTrue(output.success)
        self.assertEqual(runner.end_calls, [True])
        self.assertEqual(gathered, [])
        manager.scheduler.record_weight_version_change.assert_called_once_with(
            new_version="v7"
        )

    def test_base_and_lora_manifests_are_both_verified(self):
        """A base manifest coexists with the LoRA manifest in one end_weight_update."""
        params = {PARAM_A: torch.arange(4, dtype=torch.float32)}
        manager, runner = _make_manager(params)
        lora_tensor = torch.ones(2)
        manager._stash_lora_tensors([(f"adapter:{LORA_A}", lora_tensor)])
        lora_manager = MagicMock()
        lora_manager.apply_streamed_adapter.return_value = LoRAUpdateOutput(
            success=True
        )
        manager.tp_worker.model_runner.lora_manager = lora_manager
        with _fake_tp_collectives():
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(
                    expected_base_weight_checksums=_manifest(params),
                    expected_lora_checksums={
                        "adapter": {LORA_A: _sha256_tensor(lora_tensor)}
                    },
                )
            )
        self.assertTrue(output.success)
        self.assertEqual(runner.end_calls, [True])
        lora_manager.apply_streamed_adapter.assert_called_once()

    def test_base_failure_skips_the_lora_apply(self):
        """A corrupted base weight must not let the streamed adapter land either."""
        params = {PARAM_A: torch.arange(4, dtype=torch.float32)}
        expected = _manifest(params)
        params[PARAM_A][0] = 42.0
        manager, runner = _make_manager(params)
        manager._stash_lora_tensors([(f"adapter:{LORA_A}", torch.ones(2))])
        lora_manager = MagicMock()
        manager.tp_worker.model_runner.lora_manager = lora_manager
        with _fake_tp_collectives():
            output = manager.end_weight_update(
                EndWeightUpdateReqInput(expected_base_weight_checksums=expected)
            )
        self.assertFalse(output.success)
        self.assertEqual(runner.end_calls, [])
        lora_manager.apply_streamed_adapter.assert_not_called()


class TestEndWeightUpdateReqInputWire(CustomTestCase):
    def test_base_checksum_field_survives_serialization(self):
        """The Miles-facing field is part of the request's serialized wire form."""
        obj = EndWeightUpdateReqInput(
            expected_base_weight_checksums={"0": {PARAM_A: "aa"}, "1": {PARAM_A: "bb"}}
        )
        decoded = msgpack_decode(msgpack_encode(obj))
        self.assertEqual(
            decoded.expected_base_weight_checksums,
            {"0": {PARAM_A: "aa"}, "1": {PARAM_A: "bb"}},
        )
        self.assertIsNone(decoded.expected_lora_checksums)


class TestChecksumStorage:
    def test_contiguous_cpu_weights_are_hashed_without_copying_their_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hashing a CPU weight must not allocate a second full tensor of bytes."""
        tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        expected = hashlib.sha256(
            tensor.flatten().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        sha256 = hashlib.sha256

        def hash_buffer(data: object) -> object:
            assert (
                ctypes.addressof(ctypes.c_char.from_buffer(data)) == tensor.data_ptr()
            )
            return sha256(data)

        monkeypatch.setattr(hashlib, "sha256", hash_buffer)

        assert _sha256_tensor(tensor) == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
