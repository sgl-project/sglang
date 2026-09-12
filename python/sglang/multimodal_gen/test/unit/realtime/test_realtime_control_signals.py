# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.realtime.control_signals import (
    ControlScriptQueue,
    ControlSignal,
    ControlSignalQueue,
    ControlSignalSamplingParams,
    ControlStateQueue,
    ControlStateTransition,
    parse_control_event_payload,
)


def test_control_signal_queue_samples_chunk_and_repeats_last_item():
    queue = ControlSignalQueue()
    queue.push("camera_actions", [["w"], ["d"]])

    chunk = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=4, default_item=[]),
    )

    assert chunk == [["w"], ["d"], ["d"], ["d"]]


def test_control_signal_queue_accepts_multiple_same_kind_control_signals():
    queue = ControlSignalQueue()
    queue.push(
        "camera_actions",
        [
            ControlSignal(kind="camera_actions", payload=["w"]),
            ControlSignal(kind="camera_actions", payload=["d"]),
        ],
    )

    chunk = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=2, default_item=[]),
    )

    assert chunk == [["w"], ["d"]]


def test_control_signal_queue_samples_control_signal_payloads():
    queue = ControlSignalQueue()
    queue.push(
        "camera_actions",
        [
            ControlSignal(kind="camera_actions", payload=["w"]),
            ControlSignal(kind="camera_actions", payload=["a"]),
            ControlSignal(kind="camera_actions", payload=["s"]),
        ],
    )

    first = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=2, default_item=[]),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=2, default_item=[]),
    )

    assert first == [["w"], ["a"]]
    assert second == [["s"], ["s"]]


def test_control_signal_queue_preserves_event_remainder_across_chunks():
    queue = ControlSignalQueue()
    queue.push("camera_actions", [["w"], ["a"], ["s"]])

    first = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=2, default_item=[]),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=2, default_item=[]),
    )

    assert first == [["w"], ["a"]]
    assert second == [["s"], ["s"]]


def test_control_signal_queue_does_not_persist_last_signal_across_empty_chunks():
    queue = ControlSignalQueue()
    queue.push("camera_actions", [["w"]])

    first = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=2, default_item=[]),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=2, default_item=[]),
    )

    assert first == [["w"], ["w"]]
    assert second == [[], []]


def test_control_signal_queue_can_repeat_last_signal_across_empty_chunks():
    queue = ControlSignalQueue()
    queue.push("camera_actions", [["w"]])

    first = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(
            chunk_size=2,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )
    second = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(
            chunk_size=2,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )
    queue.push("camera_actions", [[]])
    third = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(
            chunk_size=2,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )

    assert first == [["w"], ["w"]]
    assert second == [["w"], ["w"]]
    assert third == [[], []]


def test_control_signal_queue_tracks_sampled_signal_seq_id():
    queue = ControlSignalQueue()
    queue.push(
        "camera_actions",
        [
            ControlSignal(kind="camera_actions", payload=["w"], seq_id=7),
            ControlSignal(kind="camera_actions", payload=[], seq_id=8),
        ],
    )

    first = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(
            chunk_size=1,
            default_item=[],
            repeat_last_across_empty_chunks=True,
        ),
    )
    first_seq_id = queue.last_sampled_seq_id("camera_actions")
    second = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(
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


def test_control_signal_queue_replace_clears_pending_signals():
    queue = ControlSignalQueue()
    queue.push("camera_actions", [["w"], ["w"], ["w"], ["w"]])

    first = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=2, default_item=[]),
    )
    queue.replace("camera_actions", [["d"]])
    second = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=3, default_item=[]),
    )
    third = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=3, default_item=[]),
    )

    assert first == [["w"], ["w"]]
    assert second == [["d"], ["d"], ["d"]]
    assert third == [[], [], []]


def test_control_signal_queue_returns_none_without_default_item():
    queue = ControlSignalQueue()

    chunk = queue.sample_chunk("audio", ControlSignalSamplingParams(chunk_size=2))

    assert chunk is None


def test_control_signal_queue_empty_event_switches_to_default_item():
    queue = ControlSignalQueue()
    queue.push("camera_actions", [])

    chunk = queue.sample_chunk(
        "camera_actions",
        ControlSignalSamplingParams(chunk_size=3, default_item=[]),
    )

    assert chunk == [[], [], []]


def test_control_script_queue_samples_script_and_pads_default_item():
    queue = ControlScriptQueue("camera_actions", default_item=[])
    queue.push_script([["w"], ["d"]], event_id=7)

    chunk = queue.sample_script(3)

    assert chunk == [["w"], ["d"], []]
    assert queue.last_sampled_seq_id() == 7
    assert not queue.has_script()


def test_parse_control_event_payload_normalizes_state_transitions():
    parsed = parse_control_event_payload(
        {
            "mode": "state",
            "transitions": [
                {"actions": ["W"], "client_ts_ms": 100},
                {"actions": [], "client_ts_ms": 120},
            ],
        },
        event_id=11,
        kind="camera_actions",
        normalize_state_payload=lambda actions: [
            str(action).lower() for action in actions
        ],
        validate_script_payload=lambda payload: payload,
    )

    assert parsed.mode == "state"
    assert parsed.payload == [
        ControlStateTransition(payload=["w"], timestamp_ms=100, seq_id=11),
        ControlStateTransition(payload=[], timestamp_ms=120, seq_id=11),
    ]


def test_parse_control_event_payload_validates_script_payload():
    parsed = parse_control_event_payload(
        [["w"], ["d"]],
        event_id=7,
        kind="camera_actions",
        normalize_state_payload=lambda actions: actions,
        validate_script_payload=lambda payload: [list(actions) for actions in payload],
    )

    assert parsed.mode == "script"
    assert parsed.payload == [["w"], ["d"]]


def test_control_state_queue_preserves_short_pulse():
    queue = ControlStateQueue(default_item=[], min_pulse_items=1)
    queue.push(ControlStateTransition(payload=["w"], seq_id=7))
    queue.push(ControlStateTransition(payload=[], seq_id=8))

    chunk = queue.sample_chunk(3)

    assert chunk == [["w"], [], []]
    assert queue.latest_sampled_seq_id() == 8
    assert queue.sample_chunk(3) == [[], [], []]


def test_control_state_queue_holds_current_state_without_backlog():
    queue = ControlStateQueue(default_item=[], min_pulse_items=1)
    queue.push(ControlStateTransition(payload=["w"], seq_id=7))

    assert queue.sample_chunk(3) == [["w"], ["w"], ["w"]]
    assert queue.latest_sampled_seq_id() == 7
    assert queue.sample_chunk(3) == [["w"], ["w"], ["w"]]
    assert queue.latest_sampled_seq_id() == 7


def test_control_state_queue_compacts_many_transitions():
    queue = ControlStateQueue(default_item=[], min_pulse_items=1)
    queue.push(ControlStateTransition(payload=["w"], seq_id=7))
    queue.push(ControlStateTransition(payload=["w", "d"], seq_id=8))
    queue.push(ControlStateTransition(payload=["d"], seq_id=9))

    chunk = queue.sample_chunk(3)

    assert chunk == [["d"], ["d"], ["d"]]
    assert queue.latest_sampled_seq_id() == 9
