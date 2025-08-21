#!/usr/bin/env python3
"""
Utility script to run the specific memory leak tests requested in issue #9365.

This script addresses the comment by @mickqian asking for:
1. A text model test
2. A VLM without initializing mm_processor test

Usage:
    # Test 1: Text model memory tracking
    python run_memory_tests.py --test text-model

    # Test 2: VLM without mm_processor (text-only requests to VLM)
    python run_memory_tests.py --test vlm-no-mm-processor

    # Test 3: VLM with image processing (reference)
    python run_memory_tests.py --test vlm-with-images

Based on: https://github.com/sgl-project/sglang/issues/9365
"""

import argparse
import subprocess
import sys
import os


def run_text_model_test():
    """Run memory tracking test for a text-only model."""
    print("=== Running Text Model Memory Test ===")
    print("This test addresses the request for testing a text model.")
    print("It will send text-only requests to a LLM and track memory usage.\n")
    
    # Use the memory leak monitor script
    cmd = [
        sys.executable, 
        os.path.join(os.path.dirname(__file__), "memory_leak_monitor.py"),
        "--model", "meta-llama/Llama-3.2-1B-Instruct",
        "--test-type", "text",
        "--num-requests", "100",
        "--request-interval", "0.1"
    ]
    
    return subprocess.run(cmd).returncode == 0


def run_vlm_no_mm_processor_test():
    """Run memory tracking test for VLM without mm_processor usage."""
    print("=== Running VLM Without MM Processor Test ===")
    print("This test addresses the request for testing a VLM without initializing mm_processor.")
    print("It will send text-only requests to a VLM server (no image processing).\n")
    
    # Use the memory leak monitor script with VLM but text-only requests
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "memory_leak_monitor.py"),
        "--model", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--test-type", "vlm-text",  # VLM server, but text-only requests
        "--num-requests", "100",
        "--request-interval", "0.1"
    ]
    
    return subprocess.run(cmd).returncode == 0


def run_vlm_with_images_test():
    """Run memory tracking test for VLM with image processing (reference)."""
    print("=== Running VLM With Images Test ===")
    print("This test serves as a reference to demonstrate the memory leak")
    print("issue described in the GitHub issue when processing images.\n")
    
    # Use the memory leak monitor script with VLM and image processing
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "memory_leak_monitor.py"),
        "--model", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--test-type", "vlm-image",
        "--num-requests", "50",  # Fewer requests as this may hit OOM faster
        "--request-interval", "0.2"
    ]
    
    return subprocess.run(cmd).returncode == 0


def run_unit_tests():
    """Run the unit tests for memory leak tracking."""
    print("=== Running Memory Leak Unit Tests ===")
    print("Running the test suite in test_memory_leak_tracking.py\n")
    
    # Find the test file
    test_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "test", "srt", "test_memory_leak_tracking.py"
    )
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    # Run the test file directly
    cmd = [sys.executable, test_file]
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run specific memory leak tests for SGLang",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Types:
  text-model          : Test memory usage with text-only model (addresses comment request #1)
  vlm-no-mm-processor : Test VLM with text-only requests (addresses comment request #2)
  vlm-with-images     : Test VLM with image processing (reference/baseline)
  unit-tests          : Run the full unit test suite
  all                 : Run all tests

Examples:
  python run_memory_tests.py --test text-model
  python run_memory_tests.py --test vlm-no-mm-processor
  python run_memory_tests.py --test all
        """
    )
    
    parser.add_argument(
        "--test", 
        type=str, 
        required=True,
        choices=["text-model", "vlm-no-mm-processor", "vlm-with-images", "unit-tests", "all"],
        help="Type of memory test to run"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    missing_deps = []
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import GPUtil
    except ImportError:
        missing_deps.append("GPUtil")
    
    if missing_deps:
        print(f"Warning: Missing optional dependencies: {', '.join(missing_deps)}")
        print("Some features may not be available. Install with: pip install " + " ".join(missing_deps))
        print("Continuing with basic functionality...\n")
    
    print(f"SGLang Memory Leak Test Suite")
    print(f"============================")
    print(f"Running test: {args.test}\n")
    
    success = True
    
    if args.test == "text-model":
        success = run_text_model_test()
    elif args.test == "vlm-no-mm-processor":
        success = run_vlm_no_mm_processor_test()
    elif args.test == "vlm-with-images":
        success = run_vlm_with_images_test()
    elif args.test == "unit-tests":
        success = run_unit_tests()
    elif args.test == "all":
        print("Running all memory leak tests...\n")
        tests = [
            ("Text Model", run_text_model_test),
            ("VLM No MM Processor", run_vlm_no_mm_processor_test),
            ("VLM With Images", run_vlm_with_images_test),
            ("Unit Tests", run_unit_tests)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"Starting: {test_name}")
            print(f"{'='*50}")
            
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"Test {test_name} failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        print(f"\n{'='*50}")
        print("TEST SUMMARY")
        print(f"{'='*50}")
        
        for test_name, result in results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name:<25}: {status}")
        
        success = all(results.values())
    
    if success:
        print(f"\n✅ Test '{args.test}' completed successfully!")
        print(f"📊 Check /tmp/ directory for memory usage data and plots")
    else:
        print(f"\n❌ Test '{args.test}' failed!")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())