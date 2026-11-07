"""Prepared metadata is copied from CPU mirrors and scoped to one result."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from sglang_simulator.simulation.sglang.prepared_batch import (
    SNAPSHOT_ATTRIBUTE,
    capture_prepared_batch,
    predictor_batch_from_result,
)
from sglang_simulator.time_predictor.base import PredictorError


def native_batch(mode="MIXED", **overrides):
    fields = {
        "forward_mode": SimpleNamespace(name=mode),
        "batch_size": 3,
        "seq_lens_cpu": [101, 8, 66],
        "extend_seq_lens_cpu": [1, 8, 2],
        "extend_prefix_lens_cpu": [100, 0, 64],
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_snapshot_keeps_raw_lengths_order_and_independent_predictor_batches():
    batch = native_batch()
    snapshot = capture_prepared_batch(batch)
    batch.seq_lens_cpu[:] = [1, 1, 1]
    batch.extend_seq_lens_cpu[:] = [1, 1, 1]
    batch.extend_prefix_lens_cpu[:] = [0, 0, 0]

    assert snapshot.seq_lens_cpu == (101, 8, 66)
    assert snapshot.extend_seq_lens_cpu == (1, 8, 2)
    assert snapshot.extend_prefix_lens_cpu == (100, 0, 64)
    first = snapshot.predictor_batch()
    assert first.request_info() == [[1, 100], [8, 0], [2, 64]]
    first.reqs[0].past_kv_length = 999
    assert snapshot.predictor_batch().request_info() == [[1, 100], [8, 0], [2, 64]]
    with pytest.raises(FrozenInstanceError):
        snapshot.seq_lens_cpu = (1, 1, 1)


def test_decode_raw_visible_length_is_converted_to_past_exactly_once():
    snapshot = capture_prepared_batch(
        native_batch("DECODE", seq_lens_cpu=[1, 100, 201])
    )
    assert snapshot.seq_lens_cpu == (1, 100, 201)
    assert snapshot.predictor_batch().request_info() == [[1, 0], [1, 99], [1, 200]]


@pytest.mark.parametrize(
    "field",
    ["seq_lens_cpu", "extend_seq_lens_cpu", "extend_prefix_lens_cpu"],
)
@pytest.mark.parametrize("value", [None, [], [1], [[1], [2], [3]]])
def test_missing_or_malformed_cpu_mirrors_fail(field, value):
    with pytest.raises(PredictorError) as error:
        capture_prepared_batch(native_batch(**{field: value}))
    assert error.value.code == "invalid_batch"


@pytest.mark.parametrize(
    "fields",
    [
        {"batch_size": True},
        {"batch_size": 0},
        {"batch_size": 3.0},
        {"seq_lens_cpu": [0, 8, 66]},
        {"seq_lens_cpu": [102, 8, 66]},
        {"extend_seq_lens_cpu": [0, 8, 2]},
        {"extend_prefix_lens_cpu": [-1, 0, 64]},
        {"seq_lens_cpu": [True, 8, 66]},
        {"seq_lens_cpu": [101.0, 8, 66]},
    ],
)
def test_inconsistent_or_invalid_prepared_shapes_fail(fields):
    with pytest.raises(PredictorError) as error:
        capture_prepared_batch(native_batch(**fields))
    assert error.value.code == "invalid_batch"


class DeviceValue:
    def __init__(self, device, ndim):
        self.device = SimpleNamespace(type=device)
        self.ndim = ndim

    def item(self):
        raise AssertionError("device item() must never be called")

    def tolist(self):
        raise AssertionError("device tolist() must never be called")

    def __len__(self):
        raise AssertionError("device mirror must be rejected before inspecting it")


@pytest.mark.parametrize("device", ["cuda", "meta"])
@pytest.mark.parametrize(
    "field",
    ["seq_lens_cpu", "extend_seq_lens_cpu", "extend_prefix_lens_cpu"],
)
def test_device_mirrors_are_rejected_without_synchronizing(device, field):
    with pytest.raises(PredictorError, match="must be on CPU"):
        capture_prepared_batch(native_batch(**{field: DeviceValue(device, 1)}))


@pytest.mark.parametrize("device", ["cuda", "meta"])
def test_device_scalar_inside_host_list_is_rejected_before_item(device):
    with pytest.raises(PredictorError, match="must be on CPU"):
        capture_prepared_batch(
            native_batch(seq_lens_cpu=[DeviceValue(device, 0), 8, 66])
        )


def test_result_snapshots_cannot_be_replaced_by_the_latest_forward():
    first = capture_prepared_batch(native_batch())
    second = capture_prepared_batch(native_batch("DECODE", seq_lens_cpu=[2, 10, 70]))
    outputs = []
    for snapshot in (first, second):
        logits = SimpleNamespace()
        setattr(logits, SNAPSHOT_ATTRIBUTE, snapshot)
        outputs.append(SimpleNamespace(logits_output=logits))
    assert predictor_batch_from_result(
        outputs[1], expected_batch_size=3
    ).request_info() == [[1, 1], [1, 9], [1, 69]]
    assert predictor_batch_from_result(
        outputs[0], expected_batch_size=3
    ).request_info() == [[1, 100], [8, 0], [2, 64]]


@pytest.mark.parametrize(
    "result", [None, SimpleNamespace(), SimpleNamespace(logits_output=None)]
)
def test_missing_result_snapshot_is_explicit(result):
    with pytest.raises(PredictorError, match="prepared snapshot"):
        predictor_batch_from_result(result, expected_batch_size=3)


def test_result_batch_count_must_match_the_forward_snapshot():
    logits = SimpleNamespace()
    setattr(logits, SNAPSHOT_ATTRIBUTE, capture_prepared_batch(native_batch()))
    with pytest.raises(PredictorError, match="result batch size"):
        predictor_batch_from_result(
            SimpleNamespace(logits_output=logits), expected_batch_size=2
        )


def test_native_cpu_tensor_mirror_is_copied_before_mutation():
    torch = pytest.importorskip("torch")
    lengths = torch.tensor([101, 8, 66], device="cpu")
    snapshot = capture_prepared_batch(native_batch(seq_lens_cpu=lengths))
    lengths.add_(100)
    assert snapshot.seq_lens_cpu == (101, 8, 66)
    assert snapshot.predictor_batch().request_info() == [[1, 100], [8, 0], [2, 64]]


def test_native_meta_tensor_mirror_fails_without_materialization():
    torch = pytest.importorskip("torch")
    with pytest.raises(PredictorError, match="must be on CPU"):
        capture_prepared_batch(native_batch(seq_lens_cpu=torch.empty(3, device="meta")))
