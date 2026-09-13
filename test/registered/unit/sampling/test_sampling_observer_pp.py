"""Unit tests for sampling-observer pipeline-parallel transport validation."""

import unittest

import torch

from sglang.srt.sampling import sampling_observer_pp as observer_pp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _PPAuxiliaryOutput:
    def __init__(self, tensors):
        self.tensors = tensors

    def to_pp_tensors(self):
        return self.tensors


class _RecordingObserver:
    def __init__(self, result):
        self.result = result
        self.received = None

    def from_pp_tensors(self, tensors):
        self.received = dict(tensors)
        return self.result


def _aux_key(name):
    return f"{observer_pp._OUTPUT_PREFIX}{name}"


class TestAddAuxiliaryOutputToPPTensors(CustomTestCase):
    def test_none_output_leaves_transport_unchanged(self):
        hidden_states = torch.tensor([1.0])
        tensors = {"hidden_states": hidden_states}

        observer_pp.add_auxiliary_output_to_pp_tensors(tensors, None)

        self.assertEqual(set(tensors), {"hidden_states"})
        torch.testing.assert_close(tensors["hidden_states"], hidden_states)

    def test_add_namespaces_multiple_auxiliary_tensors(self):
        hidden_states = torch.tensor([1.0])
        scores = torch.tensor([0.2, 0.8])
        token_ids = torch.tensor([3, 7])
        tensors = {"hidden_states": hidden_states}

        observer_pp.add_auxiliary_output_to_pp_tensors(
            tensors,
            _PPAuxiliaryOutput({"scores": scores, "token_ids": token_ids}),
        )

        self.assertEqual(
            set(tensors),
            {"hidden_states", _aux_key("scores"), _aux_key("token_ids")},
        )
        torch.testing.assert_close(tensors["hidden_states"], hidden_states)
        torch.testing.assert_close(tensors[_aux_key("scores")], scores)
        torch.testing.assert_close(tensors[_aux_key("token_ids")], token_ids)

    def test_add_rejects_malformed_payloads(self):
        cases = (
            ("empty mapping", {}, "must contain at least one tensor"),
            ("empty name", {"": torch.tensor([1.0])}, "non-empty strings"),
            ("non-string name", {7: torch.tensor([1.0])}, "non-empty strings"),
            ("non-tensor value", {"scores": [0.2, 0.8]}, "is not a tensor"),
        )

        for label, payload, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, message):
                    observer_pp.add_auxiliary_output_to_pp_tensors(
                        {}, _PPAuxiliaryOutput(payload)
                    )

    def test_add_rejects_reserved_key_collision_without_overwriting(self):
        key = _aux_key("scores")
        existing = torch.tensor([9.0])
        tensors = {key: existing}

        with self.assertRaisesRegex(RuntimeError, "duplicate auxiliary PP tensor"):
            observer_pp.add_auxiliary_output_to_pp_tensors(
                tensors,
                _PPAuxiliaryOutput({"scores": torch.tensor([1.0])}),
            )

        self.assertEqual(set(tensors), {key})
        torch.testing.assert_close(tensors[key], existing)


class TestPopAuxiliaryOutputFromPPTensors(CustomTestCase):
    def test_pop_without_auxiliary_data_skips_observer(self):
        hidden_states = torch.tensor([1.0])
        tensors = {"hidden_states": hidden_states}
        observer = _RecordingObserver(result=object())

        result = observer_pp.pop_auxiliary_output_from_pp_tensors(tensors, observer)

        self.assertIsNone(result)
        self.assertIsNone(observer.received)
        self.assertEqual(set(tensors), {"hidden_states"})
        torch.testing.assert_close(tensors["hidden_states"], hidden_states)

    def test_pop_rejects_non_tensor_payload_before_observer(self):
        tensors = {_aux_key("scores"): [0.4, 0.6]}
        observer = _RecordingObserver(result=object())

        with self.assertRaisesRegex(RuntimeError, "received a non-tensor"):
            observer_pp.pop_auxiliary_output_from_pp_tensors(tensors, observer)

        self.assertIsNone(observer.received)

    def test_pop_rejects_failed_reconstruction(self):
        scores = torch.tensor([0.4, 0.6])
        observer = _RecordingObserver(result=None)

        with self.assertRaisesRegex(RuntimeError, "did not reconstruct its PP output"):
            observer_pp.pop_auxiliary_output_from_pp_tensors(
                {_aux_key("scores"): scores}, observer
            )

        self.assertEqual(set(observer.received), {"scores"})
        torch.testing.assert_close(observer.received["scores"], scores)


if __name__ == "__main__":
    unittest.main()
