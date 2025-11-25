"""
Streaming events tests for Response API.
Tests for streaming event validation including:
- Zero-based output_index for reasoning content
- OutputItemDone event emission and output array construction
"""

import pytest


@pytest.mark.parametrize("setup_backend", ["grpc", "grpc_harmony"], indirect=True)
class TestStreamingEventsTest:
    """Tests for streaming event validation."""

    def test_output_index_zero_based(self, setup_backend):
        """
        Test that output_index is zero-based in streaming responses.
        Verifies that the first output item has output_index: 0.
        """
        _, model, client = setup_backend

        resp = client.responses.create(
            model=model,
            input="Count from 1 to 3",
            stream=True,
            max_output_tokens=50,
        )

        events = [event for event in resp]
        assert len(events) > 0

        # Find output_item.added events
        output_item_added_events = [
            event for event in events if event.type == "response.output_item.added"
        ]
        assert len(output_item_added_events) > 0, "Should have output_item.added events"

        # Verify first output item has output_index: 0
        first_item_event = output_item_added_events[0]
        assert first_item_event.item is not None
        assert first_item_event.output_index is not None
        assert (
            first_item_event.output_index == 0
        ), "First output item must have output_index: 0 (zero-based indexing)"

        # Verify subsequent items increment correctly
        for i, event in enumerate(output_item_added_events):
            assert (
                event.output_index == i
            ), f"Output item {i} should have output_index: {i}"

    def test_output_item_done_event_emitted(self, setup_backend):
        """
        Test that response.output_item.done event is emitted in streaming.
        Verifies that output_item.done events are emitted for each output item.
        """
        _, model, client = setup_backend

        resp = client.responses.create(
            model=model,
            input="Say hello",
            stream=True,
            max_output_tokens=50,
        )

        events = [event for event in resp]
        assert len(events) > 0

        event_types = [e.type for e in events]

        # Verify output_item.done event exists
        assert (
            "response.output_item.done" in event_types
        ), "Should emit response.output_item.done event"

        # Verify output_item.done event structure
        output_item_done_events = [
            event for event in events if event.type == "response.output_item.done"
        ]
        assert len(output_item_done_events) > 0

        for event in output_item_done_events:
            assert event.item is not None
            assert event.output_index is not None
            assert event.item.type is not None

    def test_output_array_in_completed_event(self, setup_backend):
        """
        Test that output array is properly constructed in response.completed event.
        Verifies that the completed event contains all output items in the output array.
        """
        _, model, client = setup_backend

        resp = client.responses.create(
            model=model,
            input="What is 2+2?",
            stream=True,
            max_output_tokens=50,
        )

        events = [event for event in resp]
        assert len(events) > 0

        # Find response.completed event
        completed_events = [
            event for event in events if event.type == "response.completed"
        ]
        assert len(completed_events) == 1, "Should have exactly one completed event"

        # Verify output array exists and contains items
        completed_event = completed_events[0]

        assert completed_event.response.output is not None
        output_array = completed_event.response.output
        assert isinstance(output_array, list)
        assert len(output_array) > 0, "Output array should contain at least one item"

        # Verify each item in output array has proper structure
        for i, item in enumerate(output_array):
            assert item.type is not None

        # Verify output_item.added events match items in final output array
        output_item_added_events = [
            event for event in events if event.type == "response.output_item.added"
        ]

        assert len(output_item_added_events) == len(
            output_array
        ), "Number of output_item.added events should match output array length"


@pytest.mark.parametrize("setup_backend", ["grpc_harmony"], indirect=True)
class TestHarmonyStreamingEvents:
    """Tests for Harmony-specific streaming events (reasoning content)."""

    def test_reasoning_content_output_index(self, setup_backend):
        """
        Test that reasoning content has correct zero-based output_index.
        Specifically tests that reasoning item has output_index: 0
        and message item has output_index: 1.
        """
        _, model, client = setup_backend

        resp = client.responses.create(
            model=model,
            input="What is the capital of France? Think step by step.",
            stream=True,
            max_output_tokens=200,
        )

        events = [event for event in resp]
        assert len(events) > 0

        # Find output_item.added events
        output_item_added_events = [
            event for event in events if event.type == "response.output_item.added"
        ]
        assert len(output_item_added_events) > 0

        reasoning_items = [
            item for item in output_item_added_events if item.item.type == "reasoning"
        ]
        message_items = [
            item for item in output_item_added_events if item.item.type == "message"
        ]

        # If reasoning is present, verify it has output_index: 0
        if reasoning_items:
            reasoning_item = reasoning_items[0]
            assert (
                reasoning_item.output_index == 0
            ), "Reasoning item should have output_index: 0"

        # If message is present after reasoning, verify it has output_index: 1
        if reasoning_items and message_items:
            message_item = message_items[0]
            assert (
                message_item.output_index == 1
            ), "Message item after reasoning should have output_index: 1"

    def test_reasoning_content_in_output_array(self, setup_backend):
        """
        Test that reasoning content is properly included in final output array.
        Verifies that reasoning items are stored and included in the
        response.completed event's output array.
        """
        _, model, client = setup_backend

        resp = client.responses.create(
            model=model,
            input="Explain why 2+2=4. Show your reasoning.",
            stream=True,
            max_output_tokens=200,
        )

        events = [event for event in resp]
        assert len(events) > 0

        # Find response.completed event
        completed_events = [
            event for event in events if event.type == "response.completed"
        ]
        assert len(completed_events) == 1

        # Get output array from completed event
        output_array = completed_events[0].response.output
        assert len(output_array) > 0

        # Check if reasoning items are in output array
        reasoning_items_in_output = [
            item for item in output_array if item.type == "reasoning"
        ]
        assert len(reasoning_items_in_output) > 0
