"""
Streaming events tests for Response API.

Tests for streaming event validation including:
- Zero-based output_index for reasoning content
- OutputItemDone event emission and output array construction
"""

from basic_crud import ResponseAPIBaseTest


class StreamingEventsTests(ResponseAPIBaseTest):
    """Tests for streaming event validation."""

    def test_output_index_zero_based(self):
        """
        Test that output_index is zero-based in streaming responses.

        Verifies that the first output item has output_index: 0.
        """
        resp = self.create_response(
            "Count from 1 to 3",
            stream=True,
            max_output_tokens=50,
        )

        self.assertEqual(resp.status_code, 200)

        events = self.parse_sse_events(resp)
        self.assertGreater(len(events), 0)

        # Find output_item.added events
        output_item_added_events = [
            e for e in events if e.get("event") == "response.output_item.added"
        ]
        self.assertGreater(
            len(output_item_added_events), 0, "Should have output_item.added events"
        )

        # Verify first output item has output_index: 0
        first_item_event = output_item_added_events[0]
        first_item_data = first_item_event.get("data", {})
        self.assertIn("item", first_item_data)

        first_item = first_item_data["item"]
        self.assertIn("output_index", first_item)
        self.assertEqual(
            first_item["output_index"],
            0,
            "First output item must have output_index: 0 (zero-based indexing)",
        )

        # Verify subsequent items increment correctly
        for i, event in enumerate(output_item_added_events):
            item_data = event.get("data", {}).get("item", {})
            self.assertEqual(
                item_data.get("output_index"),
                i,
                f"Output item {i} should have output_index: {i}",
            )

    def test_output_item_done_event_emitted(self):
        """
        Test that response.output_item.done event is emitted in streaming.

        Verifies that output_item.done events are emitted for each output item.
        """
        resp = self.create_response(
            "Say hello",
            stream=True,
            max_output_tokens=50,
        )

        self.assertEqual(resp.status_code, 200)

        events = self.parse_sse_events(resp)
        self.assertGreater(len(events), 0)

        event_types = [e.get("event") for e in events]

        # Verify output_item.done event exists
        self.assertIn(
            "response.output_item.done",
            event_types,
            "Should emit response.output_item.done event",
        )

        # Verify output_item.done event structure
        output_item_done_events = [
            e for e in events if e.get("event") == "response.output_item.done"
        ]
        self.assertGreater(len(output_item_done_events), 0)

        for event in output_item_done_events:
            data = event.get("data", {})
            self.assertIn("item", data, "output_item.done should contain item")

            item = data["item"]
            self.assertIn("output_index", item)
            self.assertIn("type", item)

    def test_output_array_in_completed_event(self):
        """
        Test that output array is properly constructed in response.completed event.

        Verifies that the completed event contains all output items in the output array.
        """
        resp = self.create_response(
            "What is 2+2?",
            stream=True,
            max_output_tokens=50,
        )

        self.assertEqual(resp.status_code, 200)

        events = self.parse_sse_events(resp)
        self.assertGreater(len(events), 0)

        # Find response.completed event
        completed_events = [e for e in events if e.get("event") == "response.completed"]
        self.assertEqual(
            len(completed_events), 1, "Should have exactly one completed event"
        )

        # Verify output array exists and contains items
        completed_event = completed_events[0]
        response_data = completed_event.get("data", {}).get("response", {})

        self.assertIn("output", response_data)
        output_array = response_data["output"]
        self.assertIsInstance(output_array, list)
        self.assertGreater(
            len(output_array), 0, "Output array should contain at least one item"
        )

        # Verify each item in output array has proper structure
        for i, item in enumerate(output_array):
            self.assertIn("type", item, f"Output item {i} should have type")
            self.assertIn(
                "output_index", item, f"Output item {i} should have output_index"
            )

        # Verify output_item.added events match items in final output array
        output_item_added_events = [
            e for e in events if e.get("event") == "response.output_item.added"
        ]

        self.assertEqual(
            len(output_item_added_events),
            len(output_array),
            "Number of output_item.added events should match output array length",
        )


class HarmonyStreamingEventsTests(ResponseAPIBaseTest):
    """Tests for Harmony-specific streaming events (reasoning content)."""

    def test_reasoning_content_output_index(self):
        """
        Test that reasoning content has correct zero-based output_index.

        Specifically tests that reasoning item has output_index: 0
        and message item has output_index: 1.
        """
        resp = self.create_response(
            "What is the capital of France? Think step by step.",
            stream=True,
            max_output_tokens=200,
        )

        self.assertEqual(resp.status_code, 200)

        events = self.parse_sse_events(resp)
        self.assertGreater(len(events), 0)

        # Find output_item.added events
        output_item_added_events = [
            e for e in events if e.get("event") == "response.output_item.added"
        ]
        self.assertGreater(len(output_item_added_events), 0)

        output_items = [
            e.get("data", {}).get("item", {}) for e in output_item_added_events
        ]
        reasoning_items = [
            item for item in output_items if item.get("type") == "reasoning"
        ]
        message_items = [item for item in output_items if item.get("type") == "message"]

        # If reasoning is present, verify it has output_index: 0
        if reasoning_items:
            reasoning_item = reasoning_items[0]
            self.assertEqual(
                reasoning_item.get("output_index"),
                0,
                "Reasoning item should have output_index: 0",
            )

        # If message is present after reasoning, verify it has output_index: 1
        if reasoning_items and message_items:
            message_item = message_items[0]
            self.assertEqual(
                message_item.get("output_index"),
                1,
                "Message item after reasoning should have output_index: 1",
            )

    def test_reasoning_content_in_output_array(self):
        """
        Test that reasoning content is properly included in final output array.

        Verifies that reasoning items are stored and included in the
        response.completed event's output array.
        """
        resp = self.create_response(
            "Explain why 2+2=4. Show your reasoning.",
            stream=True,
            max_output_tokens=200,
        )

        self.assertEqual(resp.status_code, 200)

        events = self.parse_sse_events(resp)
        self.assertGreater(len(events), 0)

        # Find response.completed event
        completed_events = [e for e in events if e.get("event") == "response.completed"]
        self.assertEqual(len(completed_events), 1)

        # Get output array from completed event
        response_data = completed_events[0].get("data", {}).get("response", {})
        output_array = response_data.get("output", [])

        self.assertGreater(len(output_array), 0)

        # Check if reasoning items are in output array
        reasoning_items_in_output = [
            item for item in output_array if item.get("type") == "reasoning"
        ]

        # If model produced reasoning, verify it's in the output array
        if reasoning_items_in_output:
            reasoning_item = reasoning_items_in_output[0]
            self.assertIn("content", reasoning_item)
            self.assertIn("output_index", reasoning_item)

            # Verify reasoning has correct index
            self.assertEqual(
                reasoning_item["output_index"],
                0,
                "First reasoning item should have output_index: 0",
            )
