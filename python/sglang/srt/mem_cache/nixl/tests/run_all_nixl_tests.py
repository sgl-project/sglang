#!/usr/bin/env python3

import sys
import os
import subprocess

def run_test(test_file):
    """Run a single test file and return success status."""
    print(f"\n{'='*50}")
    print(f"Running {test_file}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=60)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        success = result.returncode == 0
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"\n{status}: {test_file}")
        
        return success
        
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {test_file}")
        return False
    except Exception as e:
        print(f"✗ ERROR running {test_file}: {e}")
        return False

def main():
    """Run all NIXL tests."""
    print("NIXL Component Tests")
    print("=" * 50)
    
    # List of test files to run
    test_files = [
        "test_nixl_backend_selection.py",
        "test_nixl_file_management.py", 
        "test_nixl_registration.py",
        "test_hicache_nixl_integration.py"
    ]
    
    # Check which test files exist
    existing_tests = [f for f in test_files if os.path.exists(f)]
    
    if not existing_tests:
        print("No test files found!")
        return False
    
    print(f"Found {len(existing_tests)} test files to run:")
    for test in existing_tests:
        print(f"  - {test}")
    
    # Run all tests
    results = {}
    for test_file in existing_tests:
        results[test_file] = run_test(test_file)
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_file, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_file}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 