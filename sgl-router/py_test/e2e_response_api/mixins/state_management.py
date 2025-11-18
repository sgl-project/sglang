"""
State management tests for Response API.

Tests both previous_response_id and conversation-based state management.
These tests should work across all backends (OpenAI, XAI, gRPC).
"""

import unittest

import openai
from basic_crud import ResponseAPIBaseTest


class StateManagementTests(ResponseAPIBaseTest):
    """Tests for state management using previous_response_id and conversation."""

    def test_previous_response_id_chaining(self):
        """Test chaining responses using previous_response_id."""
        # First response
        resp1 = self.create_response(
            "My name is Alice and my friend is Bob. Remember it."
        )
        self.assertIsNone(resp1.error)
        self.assertEqual(resp1.status, "completed")
        response1_id = resp1.id

        # Second response referencing first
        resp2 = self.create_response(
            "What is my name", previous_response_id=response1_id
        )
        self.assertIsNone(resp2.error)
        self.assertEqual(resp2.status, "completed")

        # The model should remember the name from previous response
        self.assertIn("Alice", resp2.output_text)

        # Third response referencing second
        resp3 = self.create_response(
            "What is my friend name?",
            previous_response_id=resp2.id,
        )
        self.assertIsNone(resp3.error)
        self.assertEqual(resp3.status, "completed")
        self.assertIn("Bob", resp3.output_text)

    @unittest.skip("TODO: Add the invalid previous_response_id check")
    def test_previous_response_id_invalid(self):
        """Test using invalid previous_response_id."""
        with self.assertRaises(openai.BadRequestError):
            self.create_response(
                "Test", previous_response_id="resp_invalid123", max_output_tokens=50
            )

    def test_conversation_with_multiple_turns(self):
        """Test state management using conversation ID."""
        # Create conversation
        conv_resp = self.create_conversation(metadata={"topic": "math"})
        self.assertIsNotNone(conv_resp.id)
        self.assertIsNotNone(conv_resp.created_at)

        conversation_id = conv_resp.id

        # First response in conversation
        resp1 = self.create_response("I have 5 apples.", conversation=conversation_id)
        self.assertIsNone(resp1.error)
        self.assertEqual(resp1.status, "completed")

        # Second response in same conversation
        resp2 = self.create_response(
            "How many apples do I have?",
            conversation=conversation_id,
        )
        self.assertIsNone(resp2.error)
        self.assertEqual(resp2.status, "completed")
        output_text = resp2.output_text

        # Should remember "5 apples"
        self.assertTrue("5" in output_text or "five" in output_text.lower())

        # Third response in same conversation
        resp3 = self.create_response(
            "If I get 3 more, how many total?",
            conversation=conversation_id,
        )
        self.assertIsNone(resp3.error)
        self.assertEqual(resp3.status, "completed")
        output_text = resp3.output_text

        # Should calculate 5 + 3 = 8
        self.assertTrue("8" in output_text or "eight" in output_text.lower())
        list_resp = self.list_conversation_items(conversation_id)
        self.assertIsNotNone(list_resp.data)
        items = list_resp.data
        # Should have at least 6 items (3 inputs + 3 outputs)
        self.assertGreaterEqual(len(items), 6)

    def test_mutually_exclusive_parameters(self):
        """Test that previous_response_id and conversation are mutually exclusive."""
        # TODO: Remove this once the conversation API is implemented for GRPC backend
        conversation_id = "conv_123"

        resp1 = self.create_response("Test")
        response1_id = resp1.id

        # Try to use both parameters
        with self.assertRaises(openai.BadRequestError):
            self.create_response(
                "This should fail",
                previous_response_id=response1_id,
                conversation=conversation_id,
            )

    # Helper methods

    def _extract_output_text(self, response_data: dict) -> str:
        """Extract text content from response output."""
        output = response_data.get("output", [])
        if not output:
            return ""

        text_parts = []
        for item in output:
            content = item.get("content", [])
            for part in content:
                if part.get("type") == "output_text":
                    text_parts.append(part.get("text", ""))

        return " ".join(text_parts)
