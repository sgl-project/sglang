"""
Test cases for LongBench-v2 evaluation utility.
"""

import json
import os
import tempfile

from sglang.test.simple_eval_longbench_v2 import (
    LongBenchV2Eval,
    extract_longbench_v2_answer,
    format_longbench_v2_question,
)


def test_format_longbench_v2_question():
    """Test the official LongBench-v2 question formatting."""
    sample_row = {
        "context": "This is a sample context about environmental issues.",
        "question": "What is the main theme?",
        "A": "Technology",
        "B": "Environment",
        "C": "Economics",
        "D": "Politics",
        "answer": "B",
    }

    formatted = format_longbench_v2_question(sample_row)

    # Verify official template structure
    assert "This is a sample context about environmental issues." in formatted
    assert (
        "What is the correct answer to this question: What is the main theme?"
        in formatted
    )
    assert "(A) Technology" in formatted
    assert "(B) Environment" in formatted
    assert "(C) Economics" in formatted
    assert "(D) Politics" in formatted
    assert "The correct answer is" in formatted
    print("✓ Question formatting works correctly")


def test_extract_longbench_v2_answer():
    """Test the official LongBench-v2 answer extraction."""

    # Test official format: "The correct answer is (A)"
    response1 = "After analyzing the context, The correct answer is (B)."
    assert extract_longbench_v2_answer(response1) == "B"

    # Test alternative format: "The correct answer is A"
    response2 = "Based on the evidence, The correct answer is C."
    assert extract_longbench_v2_answer(response2) == "C"

    # Test with asterisks
    response3 = "*The correct answer is (D)*"
    assert extract_longbench_v2_answer(response3) == "D"

    # Test fallback to standard pattern
    response4 = "I think the answer is A."
    assert extract_longbench_v2_answer(response4) == "A"

    # Test no answer
    response5 = "I'm not sure about this."
    assert extract_longbench_v2_answer(response5) is None

    print("✓ Answer extraction works correctly")


def test_longbench_v2_eval_initialization():
    """Test LongBench-v2 evaluation class initialization."""

    # Create a temporary JSON file with sample data
    sample_data = [
        {
            "_id": "test_001",
            "domain": "single_document_qa",
            "question": "What is X?",
            "choice_A": "Option A1",
            "choice_B": "Option B1",
            "choice_C": "Option C1",
            "choice_D": "Option D1",
            "answer": "A",
            "context": "Context 1",
        },
        {
            "_id": "test_002",
            "domain": "multi_document_qa",
            "question": "What is Y?",
            "A": "Option A2",
            "B": "Option B2",
            "C": "Option C2",
            "D": "Option D2",
            "answer": "B",
            "context": "Context 2",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        temp_file = f.name

    try:
        # Test initialization with new data_source parameter
        eval_instance = LongBenchV2Eval(data_source=temp_file, num_examples=1)
        assert len(eval_instance.examples) == 1
        first_example = eval_instance.examples[0]
        assert first_example.get("category") in {
            "single_document_qa",
            "multi_document_qa",
        }
        assert first_example.get("A") in {"Option A1", "Option A2"}
        print("✓ Evaluation class initialization works correctly")

    finally:
        os.unlink(temp_file)


def test_category_filtering():
    """Ensure category filtering keeps only requested domains."""

    sample_data = [
        {
            "_id": "test_001",
            "domain": "single_document_qa",
            "question": "What is X?",
            "choice_A": "Option A1",
            "choice_B": "Option B1",
            "choice_C": "Option C1",
            "choice_D": "Option D1",
            "answer": "A",
            "context": "Context 1",
        },
        {
            "_id": "test_002",
            "domain": "multi_document_qa",
            "question": "What is Y?",
            "choice_A": "Option A2",
            "choice_B": "Option B2",
            "choice_C": "Option C2",
            "choice_D": "Option D2",
            "answer": "B",
            "context": "Context 2",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        temp_file = f.name

    try:
        eval_instance = LongBenchV2Eval(
            data_source=temp_file,
            categories=["multi_document_qa"],
        )
        assert len(eval_instance.examples) == 1
        assert eval_instance.examples[0]["category"] == "multi_document_qa"
        print("✓ Category filtering works correctly")
    finally:
        os.unlink(temp_file)


def main():
    """Run all tests."""
    print("Testing simplified LongBench-v2 evaluation utility...\n")

    test_format_longbench_v2_question()
    test_extract_longbench_v2_answer()
    test_longbench_v2_eval_initialization()
    test_category_filtering()

    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("The simplified implementation follows SGLang patterns")
    print("while maintaining LongBench-v2 compatibility.")
    print("=" * 50)


if __name__ == "__main__":
    main()
