
import json
import unittest
import uuid

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestGrammar(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--grammar-backend",
                "llguidance",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def setUp(self):
        """Set up each test with clean state and track created resources."""
        # requests.post(self.base_url + "/flush_cache")
        self.created_grammars = []

    def tearDown(self):
        """Clean up any resources created during the test."""
        # Clean up grammars
        for grammar_id in self.created_grammars:
            try:
                requests.post(
                    self.base_url + "/delete_grammar",
                    json={"grammar_id": grammar_id}
                )
            except:
                pass  # Grammar might already be deleted

    def _create_grammar(self, json_schema, grammar_id=None):
        """Helper method to create a grammar and track it for cleanup."""
        if grammar_id is None:
            grammar_id = str(uuid.uuid4())

        response = requests.post(
            self.base_url + "/create_grammar",
            json={
                "json_schema": json_schema,
                "grammar_id": grammar_id
            }
        )

        if response.status_code == 200 and response.json() is not None:
            self.created_grammars.append(grammar_id)

        return response



    def test_create_grammar_with_json_schema(self):
        """Test creating a grammar with a JSON schema."""
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        })

        grammar_id = str(uuid.uuid4())
        response = self._create_grammar(json_schema, grammar_id)

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result, grammar_id)

    def test_use_grammar_in_generation(self):
        """Test using a created grammar in text generation."""
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        })

        grammar_id = str(uuid.uuid4())
        create_response = self._create_grammar(json_schema, grammar_id)
        self.assertEqual(create_response.status_code, 200)

        # Test using the grammar in generation
        gen_response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Generate a person object:",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 50,
                    "grammar_id": grammar_id,
                },
                "stream": False,
            },
        )
        self.assertEqual(gen_response.status_code, 200)
        gen_result = gen_response.json()
        self.assertIn("text", gen_result)

        gen_text = gen_result["text"]
        self.assertIn("name", gen_text)
        self.assertIn("age", gen_text)

    def test_create_grammar_auto_id(self):
        """Test creating a grammar with auto-generated ID."""

        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]}
            }
        })

        # Create grammar without specifying ID
        response = requests.post(
            self.base_url + "/create_grammar",
            json={
                "json_schema": json_schema
            }
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)  # Should be a valid UUID string

        # Track for cleanup
        if result:
            self.created_grammars.append(result)

    def test_create_grammar_duplicate_id_error(self):
        """Test creating a grammar with a duplicate ID should fail."""
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            }
        })

        grammar_id = "duplicate_test_grammar"

        # Create first grammar
        response1 = self._create_grammar(json_schema, grammar_id)
        self.assertEqual(response1.status_code, 200)
        result1 = response1.json()
        self.assertEqual(result1, grammar_id)

        # Try to create second grammar with same ID
        response2 = requests.post(
            self.base_url + "/create_grammar",
            json={
                "json_schema": json_schema,
                "grammar_id": grammar_id
            }
        )
        # Should fail
        self.assertEqual(response2.status_code, 400)

    def test_create_grammar_missing_content_error(self):
        """Test creating a grammar without any grammar content should fail."""

        # Try to create grammar with only ID, no content
        response = requests.post(
            self.base_url + "/create_grammar",
            json={
                "grammar_id": "empty_grammar"
            }
        )

        # Should fail - expecting None return
        self.assertEqual(response.status_code, 400)

    def test_delete_grammar(self):
        """Test deleting a grammar."""
        # Create a grammar first
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "temp": {"type": "string"}
            }
        })

        grammar_id = "deletable_grammar"
        create_response = self._create_grammar(json_schema, grammar_id)
        self.assertEqual(create_response.status_code, 200)

        # Delete the grammar
        delete_response = requests.post(
            self.base_url + "/delete_grammar",
            json={
                "grammar_id": grammar_id
            }
        )
        self.assertEqual(delete_response.status_code, 200)

        # Remove from tracking since we manually deleted it
        if grammar_id in self.created_grammars:
            self.created_grammars.remove(grammar_id)

    def test_use_deleted_grammar_fails(self):
        """Test that using a deleted grammar fails."""
        # Create a grammar first
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "temp": {"type": "string"}
            }
        })

        grammar_id = "grammar_to_delete"
        create_response = self._create_grammar(json_schema, grammar_id)
        self.assertEqual(create_response.status_code, 200)

        # Delete the grammar
        delete_response = requests.post(
            self.base_url + "/delete_grammar",
            json={"grammar_id": grammar_id}
        )
        self.assertEqual(delete_response.status_code, 200)

        # Remove from tracking since we manually deleted it
        if grammar_id in self.created_grammars:
            self.created_grammars.remove(grammar_id)

        # Try to use the deleted grammar - should fail
        gen_response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Test with deleted grammar:",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 10,
                    "grammar_id": grammar_id,
                },
                "stream": False,
            },
        )
        # Should fail because grammar no longer exists
        self.assertNotEqual(gen_response.status_code, 200)

    def test_delete_nonexistent_grammar(self):
        """Test deleting a non-existent grammar."""

        # Try to delete a grammar that doesn't exist
        response = requests.post(
            self.base_url + "/delete_grammar",
            json={
                "grammar_id": "nonexistent_grammar"
            }
        )
        # Should succeed (no-op) or return appropriate status
        self.assertEqual(response.status_code, 200)

    def test_use_nonexistent_grammar_id(self):
        """Test using a non-existent grammar ID in generation should fail."""

        # Try to use a grammar that doesn't exist
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Test with nonexistent grammar:",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 10,
                    "grammar_id": "nonexistent_grammar_id",
                },
                "stream": False,
            },
        )
        # Should fail because grammar doesn't exist
        self.assertNotEqual(response.status_code, 200)

    def test_grammar_with_stop_sequences(self):
        """Test using grammar with stop sequences in generation."""
        # Create a simple JSON schema
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        })

        grammar_id = str(uuid.uuid4())
        create_response = self._create_grammar(json_schema, grammar_id)
        self.assertEqual(create_response.status_code, 200)

        # Generate with stop sequences
        gen_response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Here is a person object: ",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 50,
                    "grammar_id": grammar_id,
                    "stop": [",", "\n"],  # Stop after the first field
                },
                "stream": False,
            },
        )
        self.assertEqual(gen_response.status_code, 200)
        result = gen_response.json()
        self.assertIn("text", result)

        # Should start JSON generation
        generated_text = result["text"]
        self.assertIn("{", generated_text)

if __name__ == "__main__":
    unittest.main()
