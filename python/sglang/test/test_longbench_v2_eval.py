"""
Test cases for LongBench-v2 evaluation utility.
"""

import tempfile
import os
import pandas as pd
from sglang.test.simple_eval_longbench_v2 import (
    LongBenchV2Eval, 
    format_longbench_v2_question,
    extract_longbench_v2_answer
)


def test_format_longbench_v2_question():
    """Test the official LongBench-v2 question formatting."""
    sample_row = {
        "context": "This is a sample context about environmental issues.",
        "question": "What is the main theme?",
        "choice_A": "Technology",
        "choice_B": "Environment", 
        "choice_C": "Economics",
        "choice_D": "Politics",
        "answer": "B"
    }
    
    formatted = format_longbench_v2_question(sample_row)
    
    # Verify official template structure
    assert "This is a sample context about environmental issues." in formatted
    assert "What is the correct answer to this question: What is the main theme?" in formatted
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
    
    # Create a temporary CSV file with sample data
    sample_data = {
        "_id": ["test_001", "test_002"],
        "domain": ["single_document_qa", "multi_document_qa"],
        "question": ["What is X?", "What is Y?"],
        "choice_A": ["Option A1", "Option A2"],
        "choice_B": ["Option B1", "Option B2"],
        "choice_C": ["Option C1", "Option C2"],
        "choice_D": ["Option D1", "Option D2"],
        "answer": ["A", "B"],
        "context": ["Context 1", "Context 2"]
    }
    
    df = pd.DataFrame(sample_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Test initialization
        eval_instance = LongBenchV2Eval(temp_file, num_examples=1)
        assert len(eval_instance.examples) == 1
        assert eval_instance.examples[0]["domain"] == "single_document_qa"
        print("✓ Evaluation class initialization works correctly")
        
    finally:
        os.unlink(temp_file)


def main():
    """Run all tests."""
    print("Testing simplified LongBench-v2 evaluation utility...\n")
    
    test_format_longbench_v2_question()
    test_extract_longbench_v2_answer()
    test_longbench_v2_eval_initialization()
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("The simplified implementation follows SGLang patterns")
    print("while maintaining LongBench-v2 compatibility.")
    print("="*50)


if __name__ == "__main__":
    main()