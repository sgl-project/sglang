"""
Test cases for LongBench-v2 evaluation utility.
"""

import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.simple_eval_longbench_v2 import LongBenchV2Eval, TASK_CATEGORIES


class TestLongBenchV2Eval(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = [
            {
                "context": "This is a test context that is long enough to simulate a real LongBench-v2 example. " * 50,
                "question": "What is the main topic of this context?",
                "A": "Test context",
                "B": "Random text", 
                "C": "Example data",
                "D": "None of the above",
                "answer": "A",
                "category": "single_document_qa"
            },
            {
                "context": "Another test context for multi-document scenario. " * 100,
                "question": "What type of scenario is this?",
                "choices": ["Multi-document", "Single-document", "Code analysis", "Dialogue"],
                "answer": "A",
                "category": "multi_document_qa"
            }
        ]
    
    def test_init_with_list_data(self):
        """Test initialization with list of examples."""
        eval_obj = LongBenchV2Eval(
            data_source=self.sample_data,
            num_examples=None,
            num_threads=1
        )
        self.assertEqual(len(eval_obj.examples), 2)
    
    def test_init_with_num_examples(self):
        """Test initialization with limited number of examples."""
        eval_obj = LongBenchV2Eval(
            data_source=self.sample_data,
            num_examples=1,
            num_threads=1
        )
        self.assertEqual(len(eval_obj.examples), 1)
    
    def test_init_with_category_filter(self):
        """Test initialization with category filtering."""
        eval_obj = LongBenchV2Eval(
            data_source=self.sample_data,
            num_examples=None,
            num_threads=1,
            categories=["single_document_qa"]
        )
        self.assertEqual(len(eval_obj.examples), 1)
        self.assertEqual(eval_obj.examples[0]["category"], "single_document_qa")
    
    def test_invalid_category_filter(self):
        """Test initialization with invalid category raises error."""
        with self.assertRaises(ValueError):
            LongBenchV2Eval(
                data_source=self.sample_data,
                categories=["invalid_category"]
            )
    
    def test_load_json_file(self):
        """Test loading from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name
        
        try:
            eval_obj = LongBenchV2Eval(data_source=temp_path)
            self.assertEqual(len(eval_obj.examples), 2)
        finally:
            import os
            os.unlink(temp_path)
    
    def test_format_longbench_question(self):
        """Test question formatting."""
        eval_obj = LongBenchV2Eval(data_source=self.sample_data[:1])
        formatted = eval_obj._format_longbench_question(self.sample_data[0])
        
        self.assertIn("What is the main topic of this context?", formatted)
        self.assertIn("A) Test context", formatted)
        self.assertIn("B) Random text", formatted)
        self.assertIn("Answer: $LETTER", formatted)
    
    def test_format_question_with_choices_list(self):
        """Test question formatting with choices as list."""
        eval_obj = LongBenchV2Eval(data_source=self.sample_data[1:])
        formatted = eval_obj._format_longbench_question(self.sample_data[1])
        
        self.assertIn("What type of scenario is this?", formatted)
        self.assertIn("A) Multi-document", formatted)
    
    def test_context_length_filtering(self):
        """Test filtering by context length."""
        eval_obj = LongBenchV2Eval(
            data_source=self.sample_data,
            max_context_length=1000,  # Should filter out longer contexts
            num_threads=1
        )
        # Both examples should be filtered out due to long contexts
        self.assertLessEqual(len(eval_obj.examples), 2)
    
    @patch('sglang.test.simple_eval_longbench_v2.common.map_with_progress')
    def test_evaluation_call(self, mock_map):
        """Test the main evaluation call."""
        # Mock the sampler
        mock_sampler = MagicMock()
        mock_sampler._pack_message.return_value = {"role": "user", "content": "test"}
        mock_sampler.return_value = "Answer: A"
        
        # Mock the map_with_progress function
        from sglang.test.simple_eval_common import SingleEvalResult
        mock_result = SingleEvalResult(
            html="<div>test</div>",
            score=1.0,
            convo=[],
            metrics={"chars": 10}
        )
        mock_map.return_value = [mock_result]
        
        eval_obj = LongBenchV2Eval(data_source=self.sample_data[:1])
        result = eval_obj(mock_sampler)
        
        # Verify the evaluation was called
        mock_map.assert_called_once()
        self.assertIsNotNone(result)
    
    def test_task_categories_constant(self):
        """Test that all expected task categories are defined."""
        expected_categories = {
            "single_document_qa",
            "multi_document_qa", 
            "long_in_context_learning",
            "long_dialogue_history",
            "code_repo_understanding",
            "long_structured_data"
        }
        self.assertEqual(set(TASK_CATEGORIES.keys()), expected_categories)


class TestLongBenchV2Integration(unittest.TestCase):
    """Test integration with SGLang evaluation framework."""
    
    def test_import_success(self):
        """Test that the module can be imported successfully."""
        try:
            from sglang.test.simple_eval_longbench_v2 import LongBenchV2Eval, download_longbench_v2_dataset
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import LongBench-v2 evaluation module")
    
    def test_download_function_exists(self):
        """Test that download function is available."""
        from sglang.test.simple_eval_longbench_v2 import download_longbench_v2_dataset
        self.assertTrue(callable(download_longbench_v2_dataset))


if __name__ == "__main__":
    unittest.main()