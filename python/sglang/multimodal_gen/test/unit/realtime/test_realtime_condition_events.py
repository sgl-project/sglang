# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.realtime.condition_events import (
    ConditionEvent,
    ConditionEventQueue,
    ConditionSamplingParams,
    ControlSignal,
    ControlStateSamplingQueue,
    ControlStateTransition,
)


def test_condition_event_queue_samples_chunk_and_repeats_last_item():
    queue = ConditionEventQueue()
    queue.push(ConditionEvent(kind="camera_actions", payload=[["w"], ["d"]]))

    chunk = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=4, default_item=[]),
    )

    assert chunk == [["w"], ["d"], ["d"], ["d"]]


def test_condition_event_contains_multiple_same_kind_control_signals():
    event = ConditionEvent(
        kind="camera_actions",
        payload=[
            ControlSignal(kind="camera_actions", payload=["w"]),
            ControlSignal(kind="camera_actions", payload=["d"]),
        ],
    )
    signals = list(event.iter_signals())

    assert [signal.kind for signal in signals] == [
        "camera_actions",
        "camera_actions",
    ]
    assert [signal.payload for signal in signals] == [["w"], ["d"]]


def test_condition_event_queue_samples_control_signal_payloads():
    queue = ConditionEventQueue()
    queue.push(
        ConditionEvent(
            kind="camera_actions",
            payload=[
                ControlSignal(kind="camera_actions", payload=["w"]),
                ControlSignal(kind="camera_actions", payload=["a"]),
                ControlSignal(kind="camera_actions", payload=["s"]),
            ],
        )
    )

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )

    assert first == [["w"], ["a"]]
    assert second == [["s"], ["s"]]


def test_condition_event_queue_preserves_event_remainder_across_chunks():
    queue = ConditionEventQueue()
    queue.push(ConditionEvent(kind="camera_actions", payload=[["w"], ["a"], ["s"]]))

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )

    assert first == [["w"], ["a"]]
    assert second == [["s"], ["s"]]


def test_condition_event_queue_does_not_persist_last_signal_across_empty_chunks():
    queue = ConditionEventQueue()
    queue.push(ConditionEvent(kind="camera_actions", payload=[["w"]]))

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )

    assert first == [["w"], ["w"]]
    assert second == [[], []]


def test_condition_event_queue_can_repeat_last_signal_across_empty_chunks():
    queue = ConditionEventQueue()
    queue.push(ConditionEvent(kind="camera_actions", payload=[["w"]]))

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(
            chunk_size=2,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(
            chunk_size=2,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )
    queue.push(ConditionEvent(kind="camera_actions", payload=[[]]))
    third = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(
            chunk_size=2,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )

    assert first == [["w"], ["w"]]
    assert second == [["w"], ["w"]]
    assert third == [[], []]


def test_condition_event_queue_tracks_sampled_signal_seq_id():
    queue = ConditionEventQueue()
    queue.push(
        ConditionEvent(
            kind="camera_actions",
            payload=[
                ControlSignal(kind="camera_actions", payload=["w"], seq_id=7),
                ControlSignal(kind="camera_actions", payload=[], seq_id=8),
            ],
        )
    )

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(
            chunk_size=1,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )
    first_seq_id = queue.last_sampled_seq_id("camera_actions")
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(
            chunk_size=1,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )
    second_seq_id = queue.last_sampled_seq_id("camera_actions")

    assert first == [["w"]]
    assert first_seq_id == 7
    assert second == [[]]
    assert second_seq_id == 8


def test_condition_event_queue_replace_clears_pending_signals():
    queue = ConditionEventQueue()
    queue.push(
        ConditionEvent(kind="camera_actions", payload=[["w"], ["w"], ["w"], ["w"]])
    )

    first = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=2, default_item=[]),
    )
    queue.replace(ConditionEvent(kind="camera_actions", payload=[["d"]]))
    second = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=3, default_item=[]),
    )
    third = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=3, default_item=[]),
    )

    assert first == [["w"], ["w"]]
    assert second == [["d"], ["d"], ["d"]]
    assert third == [[], [], []]


def test_condition_event_queue_returns_none_without_default_item():
    queue = ConditionEventQueue()

    chunk = queue.sample_chunk("audio", ConditionSamplingParams(chunk_size=2))

    assert chunk is None


def test_condition_event_queue_empty_event_switches_to_default_item():
    queue = ConditionEventQueue()
    queue.push(ConditionEvent(kind="camera_actions", payload=[]))

    chunk = queue.sample_chunk(
        "camera_actions",
        ConditionSamplingParams(chunk_size=3, default_item=[]),
    )

    assert chunk == [[], [], []]


def test_control_state_sampling_queue_preserves_short_pulse():
    queue = ControlStateSamplingQueue(default_item=[], min_pulse_items=1)
    queue.push(ControlStateTransition(payload=["w"], seq_id=7))
    queue.push(ControlStateTransition(payload=[], seq_id=8))

    chunk = queue.sample_chunk(3)

    assert chunk == [["w"], [], []]
    assert queue.latest_sampled_seq_id() == 8
    assert queue.sample_chunk(3) == [[], [], []]


def test_control_state_sampling_queue_holds_current_state_without_backlog():
    queue = ControlStateSamplingQueue(default_item=[], min_pulse_items=1)
    queue.push(ControlStateTransition(payload=["w"], seq_id=7))

    assert queue.sample_chunk(3) == [["w"], ["w"], ["w"]]
    assert queue.latest_sampled_seq_id() == 7
    assert queue.sample_chunk(3) == [["w"], ["w"], ["w"]]
    assert queue.latest_sampled_seq_id() == 7


def test_control_state_sampling_queue_compacts_many_transitions():
    queue = ControlStateSamplingQueue(default_item=[], min_pulse_items=1)
    queue.push(ControlStateTransition(payload=["w"], seq_id=7))
    queue.push(ControlStateTransition(payload=["w", "d"], seq_id=8))
    queue.push(ControlStateTransition(payload=["d"], seq_id=9))

    chunk = queue.sample_chunk(3)

    assert chunk == [["d"], ["d"], ["d"]]
    assert queue.latest_sampled_seq_id() == 9
