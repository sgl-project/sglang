import threading
import time
from unittest.mock import Mock, patch

import pytest

from sglang.multimodal_gen.runtime.managers.memory_managers.prefetch_traffic import (
    PrefetchTrafficCoordinator,
    collective_prefetch_guard,
)


def test_prefetch_coordinator_registration_lifecycle():
    coordinator = PrefetchTrafficCoordinator()
    assert not coordinator.active
    coordinator.register()
    assert coordinator.active
    coordinator.unregister()
    assert not coordinator.active

    with pytest.raises(RuntimeError, match="unregister imbalance"):
        coordinator.unregister()


def test_prefetch_coordinator_blocks_and_transfers_event_dependency():
    coordinator = PrefetchTrafficCoordinator()
    stream = Mock()
    completion = Mock()
    generation = coordinator.begin_collective()
    result = []

    thread = threading.Thread(
        target=lambda: result.append(coordinator.before_submit_block(stream, 0))
    )
    thread.start()
    time.sleep(0.02)
    assert thread.is_alive()

    coordinator.end_collective(generation, completion)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert result[0][0] == generation
    stream.wait_event.assert_called_once_with(completion)
    coordinator.cancel_block(result[0][1])


def test_prefetch_coordinator_skips_observed_completion():
    coordinator = PrefetchTrafficCoordinator()
    stream = Mock()
    completion = Mock()
    generation = coordinator.begin_collective()
    coordinator.end_collective(generation, completion)

    observed, token = coordinator.before_submit_block(stream, generation)
    assert observed == generation
    stream.wait_event.assert_not_called()
    coordinator.cancel_block(token)


def test_prefetch_coordinator_hard_drains_admitted_block():
    coordinator = PrefetchTrafficCoordinator()
    stream = Mock()
    block_completion = Mock()
    _, token = coordinator.before_submit_block(stream, 0)
    coordinator.publish_block(token, block_completion)

    result = []
    thread = threading.Thread(
        target=lambda: result.append(coordinator.begin_collective())
    )
    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert result == [1]
    block_completion.synchronize.assert_called_once_with()


def test_prefetch_coordinator_waits_for_reserved_block_to_publish():
    coordinator = PrefetchTrafficCoordinator()
    stream = Mock()
    block_completion = Mock()
    _, token = coordinator.before_submit_block(stream, 0)

    result = []
    thread = threading.Thread(
        target=lambda: result.append(coordinator.begin_collective())
    )
    thread.start()
    time.sleep(0.02)
    assert thread.is_alive()

    coordinator.publish_block(token, block_completion)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert result == [1]
    block_completion.synchronize.assert_called_once_with()


def test_prefetch_coordinator_keeps_newest_completion():
    coordinator = PrefetchTrafficCoordinator()
    stream = Mock()
    first = Mock()
    second = Mock()
    generation_1 = coordinator.begin_collective()
    generation_2 = coordinator.begin_collective()
    coordinator.end_collective(generation_2, second)
    coordinator.end_collective(generation_1, first)

    observed, token = coordinator.before_submit_block(stream, 0)
    assert observed == generation_2
    stream.wait_event.assert_called_once_with(second)
    coordinator.cancel_block(token)


def test_collective_guard_is_noop_without_registered_producer():
    coordinator = PrefetchTrafficCoordinator()
    device = Mock()
    with (
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers.prefetch_traffic._get_active_prefetch_traffic_coordinator",
            return_value=None,
        ),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        patch.object(coordinator, "begin_collective") as begin_collective,
    ):
        with collective_prefetch_guard(device):
            pass
    begin_collective.assert_not_called()


def test_collective_guard_reopens_admission_after_exception():
    coordinator = PrefetchTrafficCoordinator()
    coordinator.register()
    device = Mock()
    with (
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers.prefetch_traffic._get_active_prefetch_traffic_coordinator",
            return_value=coordinator,
        ),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        pytest.raises(RuntimeError, match="collective failed"),
    ):
        with collective_prefetch_guard(device):
            raise RuntimeError("collective failed")

    stream = Mock()
    _, token = coordinator.before_submit_block(stream, 0)
    coordinator.cancel_block(token)


def test_collective_guard_records_current_stream_completion():
    coordinator = PrefetchTrafficCoordinator()
    coordinator.register()
    device = Mock()
    stream = Mock()
    completion = Mock()
    with (
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers.prefetch_traffic._get_active_prefetch_traffic_coordinator",
            return_value=coordinator,
        ),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        patch("torch.cuda.current_stream", return_value=stream),
        patch("torch.cuda.Event", return_value=completion),
        collective_prefetch_guard(device),
    ):
        pass

    completion.record.assert_called_once_with(stream)
    copy_stream = Mock()
    generation, token = coordinator.before_submit_block(copy_stream, 0)
    assert generation == 1
    copy_stream.wait_event.assert_called_once_with(completion)
    coordinator.cancel_block(token)
