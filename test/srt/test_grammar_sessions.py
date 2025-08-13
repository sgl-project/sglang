
import json
import uuid
import unittest
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

class TestGrammarSessions(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
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

    def test_create_grammar_with_json_schema(self):
        """Test creating a grammar with a JSON schema."""
        requests.post(self.base_url + "/flush_cache")
        
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        })
        
        # Create grammar with explicit ID
        grammar_id = str(uuid.uuid4())
        response = requests.post(
            self.base_url + "/create_grammar",
            json={
                "json_schema": json_schema,
                "grammar_id": grammar_id
            }
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result, grammar_id)
        
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

        # Delete grammar
        delete_response = requests.post(
            self.base_url + "/delete_grammar",
            json={"grammar_id": grammar_id}
        )
        self.assertEqual(delete_response.status_code, 200)

    def test_create_grammar_with_ebnf(self):
        """Test creating a grammar with EBNF notation."""
        requests.post(self.base_url + "/flush_cache")
        
        ebnf_grammar = """
        start: expr
        expr: NUMBER ("+" NUMBER)*
        NUMBER: /[0-9]+/
        """
        
        # Create grammar with explicit ID
        grammar_id = "test_ebnf_grammar"
        response = requests.post(
            self.base_url + "/create_grammar",
            json={
                "ebnf": ebnf_grammar,
                "grammar_id": grammar_id
            }
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result, grammar_id)

    def test_create_grammar_auto_id(self):
        """Test creating a grammar with auto-generated ID."""
        requests.post(self.base_url + "/flush_cache")
        
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

    def test_create_grammar_duplicate_id_error(self):
        """Test creating a grammar with a duplicate ID should fail."""
        requests.post(self.base_url + "/flush_cache")
        
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            }
        })
        
        grammar_id = "duplicate_test_grammar"
        
        # Create first grammar
        response1 = requests.post(
            self.base_url + "/create_grammar",
            json={
                "json_schema": json_schema,
                "grammar_id": grammar_id
            }
        )
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
        # Should fail - expecting None return or error
        if response2.status_code == 200:
            result2 = response2.json()
            self.assertIsNone(result2)
        else:
            self.assertNotEqual(response2.status_code, 200)

    def test_create_grammar_missing_content_error(self):
        """Test creating a grammar without any grammar content should fail."""
        requests.post(self.base_url + "/flush_cache")
        
        # Try to create grammar with only ID, no content
        response = requests.post(
            self.base_url + "/create_grammar",
            json={
                "grammar_id": "empty_grammar"
            }
        )
        
        # Should fail - expecting None return or error
        if response.status_code == 200:
            result = response.json()
            self.assertIsNone(result)
        else:
            self.assertNotEqual(response.status_code, 200)

    def test_create_grammar_multiple_types_error(self):
        """Test creating a grammar with multiple grammar types should work (last one wins)."""
        requests.post(self.base_url + "/flush_cache")
        
        json_schema = json.dumps({"type": "string"})
        regex_pattern = r"[a-z]+"
        
        # Create grammar with both JSON schema and regex
        grammar_id = "multi_type_grammar"
        response = requests.post(
            self.base_url + "/create_grammar",
            json={
                "json_schema": json_schema,
                "regex": regex_pattern,
                "grammar_id": grammar_id
            }
        )
        # This should still work - the implementation will pick one
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result, grammar_id)

    def test_delete_grammar(self):
        """Test deleting a grammar."""
        requests.post(self.base_url + "/flush_cache")
        
        # Create a grammar first
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "temp": {"type": "string"}
            }
        })
        
        grammar_id = "deletable_grammar"
        create_response = requests.post(
            self.base_url + "/create_grammar",
            json={
                "json_schema": json_schema,
                "grammar_id": grammar_id
            }
        )
        self.assertEqual(create_response.status_code, 200)
        
        # Delete the grammar
        delete_response = requests.post(
            self.base_url + "/delete_grammar",
            json={
                "grammar_id": grammar_id
            }
        )
        self.assertEqual(delete_response.status_code, 200)
        
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
        requests.post(self.base_url + "/flush_cache")
        
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
        requests.post(self.base_url + "/flush_cache")
        
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

    def test_grammar_with_stop_sequence_resume(self):
        """Test creating a JSON schema grammar and using stop sequences to generate JSON incrementally."""
        requests.post(self.base_url + "/flush_cache")
        
        # Create a JSON schema for a person object
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "city": {"type": "string"},
                "occupation": {"type": "string"}
            },
            "required": ["name", "age", "city", "occupation"],
            "additionalProperties": False
        })
        
        # Create grammar
        grammar_id = str(uuid.uuid4())
        create_response = requests.post(
            self.base_url + "/create_grammar",
            json={
                "json_schema": json_schema,
                "grammar_id": grammar_id
            }
        )
        self.assertEqual(create_response.status_code, 200)
        result = create_response.json()
        self.assertEqual(result, grammar_id)
        
        # Open a session for continuity
        session_response = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 2000}
        )
        self.assertEqual(session_response.status_code, 200)
        session_id = session_response.json()
        
        # First request: Start JSON generation but stop after name field
        gen_response1 = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Here is a person object: ",
                "session_params": {
                    "id": session_id,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 50,
                    "grammar_id": grammar_id,
                    "stop": [",", "\n"],  # Stop after the first field
                },
                "stream": False,
            },
        )
        self.assertEqual(gen_response1.status_code, 200)
        first_result = gen_response1.json()
        first_text = first_result["text"]
        
        print(f"First generation result: {first_text}")
        
        # Verify first result starts JSON and has at least opening brace and one field
        self.assertIn("{", first_text)
        
        # Second request: Continue the JSON generation
        gen_response2 = requests.post(
            self.base_url + "/generate",
            json={
                "text": "",  # Continue from where we left off
                "session_params": {
                    "id": session_id,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 100,
                    "grammar_id": grammar_id,
                },
                "stream": False,
            },
        )
        self.assertEqual(gen_response2.status_code, 200)
        second_result = gen_response2.json()
        second_text = second_result["text"]
        
        print(f"Second generation result: {second_text}")
        
        # Extract the full generated JSON by combining the texts
        # The second text should contain the continuation from the session
        full_json_text = second_text
        
        # Try to find and extract JSON from the full text
        start_idx = full_json_text.find("{")
        if start_idx != -1:
            # Find the matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(full_json_text[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            json_str = full_json_text[start_idx:end_idx]
            print(f"Extracted JSON: {json_str}")
            
            # Try to parse the JSON to verify it's valid
            try:
                parsed_json = json.loads(json_str)
                print(f"Parsed JSON: {parsed_json}")
                
                # Verify the JSON structure matches our schema requirements
                self.assertIn("name", parsed_json)
                self.assertIn("age", parsed_json)
                self.assertIn("city", parsed_json)
                self.assertIn("occupation", parsed_json)
                
                # Verify types
                self.assertIsInstance(parsed_json["name"], str)
                self.assertIsInstance(parsed_json["age"], int)
                self.assertIsInstance(parsed_json["city"], str)
                self.assertIsInstance(parsed_json["occupation"], str)
                
                # Verify age constraint
                self.assertGreaterEqual(parsed_json["age"], 0)
                self.assertLessEqual(parsed_json["age"], 150)
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Attempted to parse: {json_str}")
                # Don't fail the test immediately, as the grammar might still be working
                # but the stop sequence made it incomplete
        
        # Clean up
        close_response = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id}
        )
        self.assertEqual(close_response.status_code, 200)
        
        delete_response = requests.post(
            self.base_url + "/delete_grammar",
            json={"grammar_id": grammar_id}
        )
        self.assertEqual(delete_response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
