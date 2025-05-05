"""
Tests for the separate_reasoning functionality in sglang.

Usage:
python3 -m unittest test/lang/test_separate_reasoning.py
"""

import unittest

from sglang import assistant, gen, separate_reasoning, user
from sglang.lang.ir import SglExprList, SglSeparateReasoning
from sglang.test.test_utils import CustomTestCase


class TestSeparateReasoning(CustomTestCase):
    def test_separate_reasoning_creation(self):
        """Test that SglSeparateReasoning objects are created correctly."""
        # Test with valid model type and gen expression
        test_gen = gen("test")
        expr = separate_reasoning(test_gen, model_type="deepseek-r1")
        self.assertIsInstance(expr, SglExprList)
        self.assertEqual(len(expr.expr_list), 2)
        self.assertEqual(expr.expr_list[0], test_gen)
        reasoning_expr = expr.expr_list[1]
        self.assertIsInstance(reasoning_expr, SglSeparateReasoning)
        self.assertEqual(reasoning_expr.model_type, "deepseek-r1")
        self.assertEqual(reasoning_expr.name, "test_reasoning_content")

        # Test with another valid model type
        expr = separate_reasoning(test_gen, model_type="qwen3")
        self.assertIsInstance(expr, SglExprList)
        self.assertEqual(expr.expr_list[1].model_type, "qwen3")

    def test_separate_reasoning_name_processing(self):
        """Test that separate_reasoning correctly processes names."""
        test_gen = gen("test_var")
        expr = separate_reasoning(test_gen, model_type="deepseek-r1")
        reasoning_expr = expr.expr_list[1]
        self.assertEqual(reasoning_expr.name, "test_var_reasoning_content")

        # Test the process_name_for_reasoning method
        self.assertEqual(
            reasoning_expr.process_name_for_reasoning("another_var"),
            "another_var_reasoning_content",
        )

    def test_separate_reasoning_repr(self):
        """Test the string representation of SglSeparateReasoning."""
        test_gen = gen("test_var")
        expr = separate_reasoning(test_gen, model_type="deepseek-r1")
        reasoning_expr = expr.expr_list[1]
        self.assertEqual(
            repr(reasoning_expr),
            "SeparateReasoning(model_type=deepseek-r1, name=test_var_reasoning_content)",
        )

    def test_separate_reasoning_with_invalid_model_type(self):
        """Test that separate_reasoning accepts any model type during creation."""
        # Create with invalid model type
        test_gen = gen("test")
        expr = separate_reasoning(test_gen, model_type="invalid-model")
        self.assertIsInstance(expr, SglExprList)
        self.assertEqual(expr.expr_list[1].model_type, "invalid-model")
        # The actual validation happens in the ReasoningParser constructor


if __name__ == "__main__":
    unittest.main()
