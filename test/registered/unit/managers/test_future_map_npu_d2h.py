from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.managers import overlap_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeEvent:
    def __init__(self):
        self.record_count = 0
        self.wait_count = 0
        self.synchronize_count = 0

    def record(self):
        self.record_count += 1

    def wait(self):
        self.wait_count += 1

    def synchronize(self):
        self.synchronize_count += 1


class _FakeStream:
    def __init__(self):
        self.waited_events = []

    def wait_event(self, event):
        self.waited_events.append(event)


class _FakeDeviceModule:
    def __init__(self):
        self.events = []
        self.entered_streams = []

    def Event(self):
        event = _FakeEvent()
        self.events.append(event)
        return event

    def Stream(self):
        return _FakeStream()

    @contextmanager
    def stream(self, stream):
        self.entered_streams.append(stream)
        yield


def _make_future_map():
    future_map = object.__new__(overlap_utils.FutureMap)
    future_map.device = torch.device("cpu")
    future_map.spec_algo = SimpleNamespace(
        is_some=lambda: True,
        is_eagle=lambda: True,
        is_frozen_kv_mtp=lambda: False,
    )
    future_map.needs_cpu_seq_lens = True
    future_map.needs_confidence_relay = False
    future_map.confidence_relay = Mock()
    future_map.new_seq_lens_buf = torch.full((4,), -1, dtype=torch.int64)
    future_map.new_seq_lens_cpu_pinned = torch.full((4,), -1, dtype=torch.int64)
    future_map.fwd_prepare_d2h_stream = None
    future_map.npu_seq_lens_d2h_stream = _FakeStream()
    future_map.npu_seq_lens_d2h_done = _FakeEvent()
    future_map._npu_seq_lens_publish_generation = 0
    future_map._npu_seq_lens_consumed_generation = 0
    future_map.publish_ready = None
    future_map._publish_fresh = False
    return future_map


def _make_batch(indices):
    indices = torch.tensor(indices, dtype=torch.int64)
    return SimpleNamespace(
        spec_info=SimpleNamespace(future_indices=indices),
        req_pool_indices_cpu=indices.clone(),
        seq_lens=None,
        seq_lens_cpu=None,
        seq_lens_sum=None,
    )


@contextmanager
def _npu_mode(device_module):
    with (
        patch.object(overlap_utils, "_is_npu", True),
        patch.object(overlap_utils, "_DEBUG_ASSERT", False),
        patch.object(
            overlap_utils.torch,
            "get_device_module",
            return_value=device_module,
        ),
    ):
        yield


@pytest.mark.parametrize(
    ("is_eagle", "is_frozen_kv_mtp", "needs_cpu_seq_lens", "expected"),
    (
        (True, False, True, True),  # EAGLE / EAGLE3
        (True, True, True, False),  # FROZEN_KV_MTP
        (False, False, True, False),  # DSpark / DFlash / NGRAM
        (True, False, False, False),
    ),
)
def test_npu_d2h_scope_is_limited_to_non_frozen_eagle(
    is_eagle, is_frozen_kv_mtp, needs_cpu_seq_lens, expected
):
    device_module = _FakeDeviceModule()
    spec_algo = SimpleNamespace(
        is_eagle=lambda: is_eagle,
        is_frozen_kv_mtp=lambda: is_frozen_kv_mtp,
    )
    req_pool = SimpleNamespace(req_to_token=torch.empty((4, 1)))
    torch_empty = torch.empty

    def empty_without_pinning(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch_empty(*args, **kwargs)

    with (
        patch.object(overlap_utils, "_is_cuda", False),
        patch.object(overlap_utils, "_is_npu", True),
        patch.object(overlap_utils.torch, "empty", side_effect=empty_without_pinning),
        patch.object(
            overlap_utils.torch,
            "get_device_module",
            return_value=device_module,
        ),
    ):
        future_map = overlap_utils.FutureMap(
            device=torch.device("cpu"),
            spec_algo=spec_algo,
            req_to_token_pool=req_pool,
            needs_cpu_seq_lens=needs_cpu_seq_lens,
        )

    assert (future_map.npu_seq_lens_d2h_stream is not None) is expected
    assert (future_map.npu_seq_lens_d2h_done is not None) is expected


def test_publish_starts_npu_d2h_and_resolve_returns_owned_rows():
    """The NPU transfer must start at publish, not block later at resolve."""
    device_module = _FakeDeviceModule()
    future_map = _make_future_map()
    future_map.needs_confidence_relay = True

    with _npu_mode(device_module):
        future_map.publish(
            torch.tensor([1, 3]),
            torch.tensor([11, 13]),
            confidence=torch.tensor([0.25, 0.75]),
        )

        assert future_map.new_seq_lens_cpu_pinned[[1, 3]].tolist() == [11, 13]
        assert future_map.npu_seq_lens_d2h_done.record_count == 1
        assert future_map._npu_seq_lens_publish_generation == 1
        assert future_map.npu_seq_lens_d2h_stream.waited_events == [
            future_map.publish_ready
        ]
        assert device_module.entered_streams == [future_map.npu_seq_lens_d2h_stream]
        assert future_map.confidence_relay.scatter.call_count == 1
        assert (
            future_map.confidence_relay.issue_ring_copy.call_args.kwargs["stream"]
            is None
        )

        batch = _make_batch([3, 1])
        future_map.resolve_seq_lens_cpu(batch)

    assert batch.seq_lens.tolist() == [13, 11]
    assert batch.seq_lens_cpu.tolist() == [13, 11]
    assert batch.seq_lens_sum == 24
    assert future_map.publish_ready.wait_count == 1
    assert future_map.npu_seq_lens_d2h_done.synchronize_count == 1
    assert future_map._npu_seq_lens_consumed_generation == 1

    future_map.new_seq_lens_cpu_pinned.fill_(99)
    assert batch.seq_lens_cpu.tolist() == [13, 11]


def test_idle_resolve_drains_generation_without_waiting_stale_publish():
    """An empty DP iteration drains an issued copy without waiting an old publish."""
    device_module = _FakeDeviceModule()
    future_map = _make_future_map()

    with _npu_mode(device_module):
        future_map.publish(torch.tensor([1]), torch.tensor([10]))
        publish_ready = future_map.publish_ready

        idle_batch = _make_batch([])
        future_map.resolve_seq_lens_cpu(idle_batch)
        future_map.resolve_seq_lens_cpu(_make_batch([]))

        assert idle_batch.seq_lens.numel() == 0
        assert idle_batch.seq_lens_cpu.numel() == 0
        assert idle_batch.seq_lens_sum == 0
        assert publish_ready.wait_count == 0
        assert future_map.npu_seq_lens_d2h_done.synchronize_count == 1
        assert future_map._npu_seq_lens_consumed_generation == 1

        future_map.publish(torch.tensor([2]), torch.tensor([20]))
        active_batch = _make_batch([2])
        future_map.resolve_seq_lens_cpu(active_batch)

    assert active_batch.seq_lens_cpu.tolist() == [20]
    assert publish_ready.wait_count == 1
    assert future_map.npu_seq_lens_d2h_done.synchronize_count == 2
    assert future_map._npu_seq_lens_consumed_generation == 2


def test_consecutive_publishes_resolve_latest_generation():
    """A later publish must supersede an unconsumed copy without stale rows."""
    device_module = _FakeDeviceModule()
    future_map = _make_future_map()

    with _npu_mode(device_module):
        future_map.publish(torch.tensor([1]), torch.tensor([10]))
        future_map.publish(torch.tensor([1, 2]), torch.tensor([11, 20]))

        batch = _make_batch([2, 1])
        future_map.resolve_seq_lens_cpu(batch)

    assert batch.seq_lens.tolist() == [20, 11]
    assert batch.seq_lens_cpu.tolist() == [20, 11]
    assert batch.seq_lens_sum == 31
    assert future_map.publish_ready.record_count == 2
    assert future_map.npu_seq_lens_d2h_done.record_count == 2
    assert future_map.npu_seq_lens_d2h_done.wait_count == 1
    assert future_map.npu_seq_lens_d2h_done.synchronize_count == 1
    assert future_map._npu_seq_lens_publish_generation == 2
    assert future_map._npu_seq_lens_consumed_generation == 2


def test_npu_bootstrap_without_publish_uses_device_gather():
    """Bootstrap has no completion event and must retain the blocking fallback."""
    device_module = _FakeDeviceModule()
    future_map = _make_future_map()
    future_map.new_seq_lens_buf[2] = 17
    batch = _make_batch([2])

    with _npu_mode(device_module):
        future_map.resolve_seq_lens_cpu(batch)

    assert batch.seq_lens_cpu.tolist() == [17]
    assert batch.seq_lens_sum == 17
    assert future_map.publish_ready is None
    assert future_map.npu_seq_lens_d2h_done.synchronize_count == 0
