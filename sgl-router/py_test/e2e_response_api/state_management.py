"""
State management tests for Response API.

Tests both previous_response_id and conversation-based state management.
These tests should work across all backends (OpenAI, XAI, gRPC).
"""

import unittest

from base import ResponseAPIBaseTest


class StateManagementTests(ResponseAPIBaseTest):
    """Tests for state management using previous_response_id and conversation."""

    def test_previous_response_id_chaining(self):
        """Test chaining responses using previous_response_id."""
        # First response
        resp1 = self.create_response(
            "My name is Alice and my friend is Bob. Remember it."
        )
        self.assertEqual(resp1.status_code, 200)
        response1_id = resp1.json()["id"]

        # Second response referencing first
        resp2 = self.create_response(
            "What is my name", previous_response_id=response1_id
        )
        self.assertEqual(resp2.status_code, 200)
        response2_data = resp2.json()

        # The model should remember the name from previous response
        output_text = self._extract_output_text(response2_data)
        self.assertIn("Alice", output_text)

        # Third response referencing second
        resp3 = self.create_response(
            "What is my friend name?",
            previous_response_id=response2_data["id"],
        )
        response3_data = resp3.json()
        output_text = self._extract_output_text(response3_data)
        self.assertEqual(resp3.status_code, 200)
        self.assertIn("Bob", output_text)

    @unittest.skip("TODO: Add the invalid previous_response_id check")
    def test_previous_response_id_invalid(self):
        """Test using invalid previous_response_id."""
        resp = self.create_response(
            "Test", previous_response_id="resp_invalid123", max_output_tokens=50
        )
        self.assertIn(resp.status_code, [400, 404])

    def test_conversation_with_multiple_turns(self):
        """Test state management using conversation ID."""
        # Create conversation
        conv_resp = self.create_conversation(metadata={"topic": "math"})
        self.assertEqual(conv_resp.status_code, 200)

        conversation_id = conv_resp.json()["id"]

        # First response in conversation
        resp1 = self.create_response("I have 5 apples.", conversation=conversation_id)
        self.assertEqual(resp1.status_code, 200)

        # Second response in same conversation
        resp2 = self.create_response(
            "How many apples do I have?",
            conversation=conversation_id,
        )
        self.assertEqual(resp2.status_code, 200)
        output_text = self._extract_output_text(resp2.json())

        # Should remember "5 apples"
        self.assertTrue("5" in output_text or "five" in output_text.lower())

        # Third response in same conversation
        resp3 = self.create_response(
            "If I get 3 more, how many total?",
            conversation=conversation_id,
        )
        self.assertEqual(resp3.status_code, 200)
        output_text = self._extract_output_text(resp3.json())

        # Should calculate 5 + 3 = 8
        self.assertTrue("8" in output_text or "eight" in output_text.lower())
        list_resp = self.list_conversation_items(conversation_id)
        self.assertEqual(list_resp.status_code, 200)
        items = list_resp.json()["data"]
        # Should have at least 6 items (3 inputs + 3 outputs)
        self.assertGreaterEqual(len(items), 6)

    def test_mutually_exclusive_parameters(self):
        """Test that previous_response_id and conversation are mutually exclusive."""
        # Create conversation and response
        conv_resp = self.create_conversation()
        conversation_id = conv_resp.json()["id"]

        resp1 = self.create_response("Test")
        response1_id = resp1.json()["id"]

        # Try to use both parameters
        resp = self.create_response(
            "This should fail",
            previous_response_id=response1_id,
            conversation=conversation_id,
        )

        # Should return 400 Bad Request
        self.assertEqual(resp.status_code, 400)
        error_data = resp.json()
        self.assertIn("error", error_data)
        self.assertIn("mutually exclusive", error_data["error"]["message"].lower())

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
