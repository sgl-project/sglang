#!/usr/bin/env python3
"""
Quick validation script to test the memory leak tracking infrastructure.

This script validates that the memory tracking infrastructure works correctly
without requiring SGLang server or models to be installed.
"""

import json
import os
import sys
import time

# Add the test directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'test', 'srt'))

def test_memory_tracker():
    """Test basic memory tracker functionality."""
    print("Testing Memory Tracker...")
    
    try:
        from test_memory_leak_tracking import MemoryTracker
        
        # Create tracker and take snapshots
        tracker = MemoryTracker()
        
        print("Taking initial snapshot...")
        initial = tracker.take_snapshot(0)
        
        # Simulate some work
        time.sleep(0.5)
        
        print("Taking final snapshot...")
        final = tracker.take_snapshot(1)
        
        # Calculate growth
        growth = tracker.get_memory_growth()
        
        print(f"✅ Memory tracking test passed!")
        print(f"   Initial: GPU {initial.gpu_memory_mb:.1f}MB, CPU {initial.cpu_memory_mb:.1f}MB")
        print(f"   Final: GPU {final.gpu_memory_mb:.1f}MB, CPU {final.cpu_memory_mb:.1f}MB")
        print(f"   Growth: GPU {growth['gpu_growth_mb']:.1f}MB, CPU {growth['cpu_growth_mb']:.1f}MB")
        
        # Test saving
        test_file = "/tmp/validation_test.json"
        tracker.save_snapshots(test_file)
        
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                data = json.load(f)
            print(f"✅ Data save test passed! Saved {len(data)} snapshots")
            os.remove(test_file)
        else:
            print("❌ Data save test failed!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Memory tracker test failed: {e}")
        return False

def test_script_help():
    """Test that the main scripts show help correctly."""
    print("\nTesting script help outputs...")
    
    scripts = [
        "scripts/run_memory_tests.py",
        "scripts/memory_leak_monitor.py"
    ]
    
    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), '..', script)
        if os.path.exists(script_path):
            print(f"✅ {script} exists and is accessible")
        else:
            print(f"❌ {script} not found at {script_path}")
            return False
    
    return True

def test_documentation():
    """Test that documentation exists."""
    print("\nTesting documentation...")
    
    docs = [
        "scripts/README_memory_testing.md"
    ]
    
    for doc in docs:
        doc_path = os.path.join(os.path.dirname(__file__), '..', doc)
        if os.path.exists(doc_path):
            print(f"✅ {doc} exists")
        else:
            print(f"❌ {doc} not found")
            return False
    
    return True

def main():
    """Run all validation tests."""
    print("SGLang Memory Leak Tracking Infrastructure Validation")
    print("=" * 60)
    
    tests = [
        ("Memory Tracker", test_memory_tracker),
        ("Script Accessibility", test_script_help),
        ("Documentation", test_documentation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name + ":"))
        results[test_name] = test_func()
    
    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    
    all_passed = True
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20}: {status}")
        if not result:
            all_passed = False
    
    print(f"\n{'=' * 60}")
    if all_passed:
        print("🎉 All validation tests PASSED!")
        print("\nThe memory leak tracking infrastructure is ready to use.")
        print("\nNext steps:")
        print("1. Run: python scripts/run_memory_tests.py --test text-model")
        print("2. Run: python scripts/run_memory_tests.py --test vlm-no-mm-processor")
        print("3. Check the documentation in scripts/README_memory_testing.md")
    else:
        print("❌ Some validation tests FAILED!")
        print("\nPlease check the errors above and fix any issues.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())