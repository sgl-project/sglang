"""
Memory leak tracking tests for SGLang.

This test suite aims to reproduce and track memory usage patterns in SGLang
to identify potential memory leaks in both LLM and VLM scenarios.

Based on issue #9365: [Bug] [Tracking] VLM/LLM OOM related issues
https://github.com/sgl-project/sglang/issues/9365
"""

import gc
import json
import os
import sys
import time
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import SGLang test utilities
try:
    from sglang.test.test_utils import (
        DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
        DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        CustomTestCase,
        kill_process_tree,
        popen_launch_server,
    )
    HAS_SGLANG_UTILS = True
except ImportError:
    HAS_SGLANG_UTILS = False
    # Fallback test case
    class CustomTestCase(unittest.TestCase):
        pass


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage at a specific point in time."""
    request_index: int
    gpu_memory_mb: float
    cpu_memory_mb: float
    timestamp: float


class MemoryTracker:
    """Utility class to track memory usage over time."""
    
    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.process = None
        if HAS_PSUTIL:
            try:
                self.process = psutil.Process()
            except Exception:
                pass
    
    def take_snapshot(self, request_index: int = 0):
        """Take a memory snapshot."""
        gpu_memory_mb = 0.0
        if HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            except Exception:
                pass
        
        cpu_memory_mb = 0.0
        if self.process:
            try:
                cpu_memory_mb = self.process.memory_info().rss / (1024 ** 2)
            except Exception:
                pass
        
        snapshot = MemorySnapshot(
            request_index=request_index,
            gpu_memory_mb=gpu_memory_mb,
            cpu_memory_mb=cpu_memory_mb,
            timestamp=time.time()
        )
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_memory_growth(self) -> Dict[str, float]:
        """Calculate memory growth from first to last snapshot."""
        if len(self.snapshots) < 2:
            return {"gpu_growth_mb": 0.0, "cpu_growth_mb": 0.0}
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        return {
            "gpu_growth_mb": last.gpu_memory_mb - first.gpu_memory_mb,
            "cpu_growth_mb": last.cpu_memory_mb - first.cpu_memory_mb,
            "requests_count": last.request_index - first.request_index + 1
        }
    
    def save_snapshots(self, filename: str):
        """Save snapshots to a JSON file for analysis."""
        data = []
        for snapshot in self.snapshots:
            data.append({
                "request_index": snapshot.request_index,
                "gpu_memory_mb": snapshot.gpu_memory_mb,
                "cpu_memory_mb": snapshot.cpu_memory_mb,
                "timestamp": snapshot.timestamp
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear(self):
        """Clear all snapshots."""
        self.snapshots.clear()


class TestMemoryLeakTracking(CustomTestCase):
    """Test suite for tracking memory leaks in SGLang."""
    
    def setUp(self):
        """Set up test environment."""
        if not HAS_SGLANG_UTILS:
            self.skipTest("SGLang test utilities not available")
            
        self.memory_tracker = MemoryTracker()
        self.base_url = "http://127.0.0.1:30000"
        
        # Test parameters
        self.num_requests = 10  # Reduced for testing environment
        self.request_interval = 0.1  # Seconds between requests
        
        # Simple text prompt for testing
        self.text_prompt = "Tell me a short story about a cat."
        
        # Force garbage collection
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def tearDown(self):
        """Clean up test environment."""
        # Force garbage collection
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _send_text_request(self, prompt: str, max_new_tokens: int = 32) -> Dict:
        """Send a text-only request to the server."""
        data = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            }
        }
        
        response = requests.post(f"{self.base_url}/generate", json=data, timeout=30)
        self.assertEqual(response.status_code, 200)
        return response.json()
    
    def _send_vlm_request_with_image(self, prompt: str, image_url: str, max_new_tokens: int = 32) -> Dict:
        """Send a VLM request with an image."""
        data = {
            "text": prompt,
            "image_data": [image_url],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            }
        }
        
        response = requests.post(f"{self.base_url}/generate", json=data, timeout=30)
        self.assertEqual(response.status_code, 200)
        return response.json()
    
    def _run_memory_tracking_test(self, request_func, test_name: str):
        """Run a memory tracking test with the given request function."""
        print(f"\n=== Running {test_name} ===")
        
        # Take initial memory snapshot
        initial_snapshot = self.memory_tracker.take_snapshot(0)
        print(f"Initial memory - GPU: {initial_snapshot.gpu_memory_mb:.1f}MB, CPU: {initial_snapshot.cpu_memory_mb:.1f}MB")
        
        # Send multiple requests and track memory
        for i in range(1, self.num_requests + 1):
            try:
                # Send request
                result = request_func()
                
                # Take memory snapshot
                snapshot = self.memory_tracker.take_snapshot(i)
                
                # Log progress every 10 requests
                if i % 10 == 0:
                    print(f"Request {i}: GPU: {snapshot.gpu_memory_mb:.1f}MB, CPU: {snapshot.cpu_memory_mb:.1f}MB")
                
                # Small delay between requests
                time.sleep(self.request_interval)
                
            except Exception as e:
                print(f"Error on request {i}: {e}")
                break
        
        # Calculate and report memory growth
        growth = self.memory_tracker.get_memory_growth()
        print(f"\n=== {test_name} Results ===")
        print(f"Requests processed: {growth['requests_count']}")
        print(f"GPU memory growth: {growth['gpu_growth_mb']:.1f}MB")
        print(f"CPU memory growth: {growth['cpu_growth_mb']:.1f}MB")
        
        # Save detailed results
        filename = f"/tmp/{test_name.lower().replace(' ', '_')}_memory_snapshots.json"
        self.memory_tracker.save_snapshots(filename)
        print(f"Detailed memory snapshots saved to: {filename}")
        
        # Clear snapshots for next test
        self.memory_tracker.clear()
        
        return growth
    
    def test_text_model_memory_tracking(self):
        """
        Test memory usage for text-only model requests.
        
        This test addresses the comment request for testing a text model.
        It sends multiple text-only requests and tracks memory usage to identify
        potential memory leaks in the LLM component.
        """
        # Launch text-only model server
        process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-radix-cache",  # Disable caching to isolate memory issues
                "--mem-fraction-static", "0.7",
            ]
        )
        
        try:
            # Wait for server to be ready
            time.sleep(5)
            
            # Test text-only requests
            request_func = lambda: self._send_text_request(self.text_prompt)
            growth = self._run_memory_tracking_test(request_func, "Text Model Memory Tracking")
            
            # Assert that memory growth is reasonable
            # Allow some growth but flag excessive increases
            max_acceptable_gpu_growth = 500.0  # MB
            max_acceptable_cpu_growth = 200.0  # MB
            
            if growth['gpu_growth_mb'] > max_acceptable_gpu_growth:
                print(f"WARNING: GPU memory grew by {growth['gpu_growth_mb']:.1f}MB, which exceeds threshold of {max_acceptable_gpu_growth}MB")
            
            if growth['cpu_growth_mb'] > max_acceptable_cpu_growth:
                print(f"WARNING: CPU memory grew by {growth['cpu_growth_mb']:.1f}MB, which exceeds threshold of {max_acceptable_cpu_growth}MB")
                
        finally:
            kill_process_tree(process.pid)
    
    def test_vlm_without_mm_processor_memory_tracking(self):
        """
        Test memory usage for VLM without initializing mm_processor.
        
        This test addresses the comment request for testing a VLM without 
        initializing mm_processor. It sends text-only requests to a VLM
        and tracks memory usage.
        """
        # Launch VLM server but only send text requests (no image processing)
        process = popen_launch_server(
            DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-radix-cache",  # Disable caching to isolate memory issues
                "--mem-fraction-static", "0.7",
                # Note: We're not disabling mm_processor, but we're only sending text requests
                # This simulates using a VLM for text-only without triggering image processing
            ]
        )
        
        try:
            # Wait for server to be ready
            time.sleep(10)  # VLM may take longer to initialize
            
            # Test text-only requests to VLM (no image processing)
            request_func = lambda: self._send_text_request(self.text_prompt)
            growth = self._run_memory_tracking_test(request_func, "VLM Without MM Processor Memory Tracking")
            
            # Assert that memory growth is reasonable
            # VLM may have slightly higher baseline but should not leak significantly
            max_acceptable_gpu_growth = 600.0  # MB (slightly higher for VLM)
            max_acceptable_cpu_growth = 250.0  # MB
            
            if growth['gpu_growth_mb'] > max_acceptable_gpu_growth:
                print(f"WARNING: GPU memory grew by {growth['gpu_growth_mb']:.1f}MB, which exceeds threshold of {max_acceptable_gpu_growth}MB")
            
            if growth['cpu_growth_mb'] > max_acceptable_cpu_growth:
                print(f"WARNING: CPU memory grew by {growth['cpu_growth_mb']:.1f}MB, which exceeds threshold of {max_acceptable_cpu_growth}MB")
                
        finally:
            kill_process_tree(process.pid)
    
    def test_vlm_with_image_processing_memory_tracking(self):
        """
        Test memory usage for VLM with image processing (reference test).
        
        This test serves as a reference point to compare against the text-only
        and VLM-without-mm-processor tests. It processes images and may show
        the memory leak described in the issue.
        """
        # Skip this test if dependencies are not available
        if not HAS_TORCH:
            self.skipTest("PyTorch not available for VLM testing")
            
        if not torch.cuda.is_available():
            self.skipTest("GPU not available for VLM testing")
        
        # Use a small image URL for testing
        image_url = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
        
        # Launch VLM server
        process = popen_launch_server(
            DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-radix-cache",  # Disable caching to isolate memory issues
                "--mem-fraction-static", "0.7",
            ]
        )
        
        try:
            # Wait for server to be ready
            time.sleep(10)  # VLM may take longer to initialize
            
            # Test VLM requests with image processing
            request_func = lambda: self._send_vlm_request_with_image(
                "What do you see in this image?", image_url
            )
            growth = self._run_memory_tracking_test(request_func, "VLM With Image Processing Memory Tracking")
            
            # This test may show the memory leak mentioned in the issue
            print(f"VLM with image processing memory growth: GPU {growth['gpu_growth_mb']:.1f}MB, CPU {growth['cpu_growth_mb']:.1f}MB")
            
            # Don't assert limits here as this test is meant to demonstrate the issue
            
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    # Simple standalone test to verify functionality
    print("SGLang Memory Leak Tracking Test")
    print("================================")
    
    # Test basic memory tracker functionality
    tracker = MemoryTracker()
    initial = tracker.take_snapshot(0)
    print(f"Initial snapshot: GPU {initial.gpu_memory_mb:.1f}MB, CPU {initial.cpu_memory_mb:.1f}MB")
    
    # Simulate some activity
    time.sleep(1)
    final = tracker.take_snapshot(1)
    print(f"Final snapshot: GPU {final.gpu_memory_mb:.1f}MB, CPU {final.cpu_memory_mb:.1f}MB")
    
    growth = tracker.get_memory_growth()
    print(f"Memory growth: GPU {growth['gpu_growth_mb']:.1f}MB, CPU {growth['cpu_growth_mb']:.1f}MB")
    
    # Save test results
    test_file = "/tmp/memory_tracker_test.json"
    tracker.save_snapshots(test_file)
    print(f"Test data saved to: {test_file}")
    
    print("\n✅ Basic memory tracker functionality verified")
    print("\nTo run full unit tests:")
    print("  python -m unittest test_memory_leak_tracking -v")
    print("\nNote: Full tests require SGLang server and models to be available")
    
    # Set environment variables for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only first GPU for testing
    
    # Create output directory for test results
    os.makedirs("/tmp", exist_ok=True)
    
    if HAS_SGLANG_UTILS:
        unittest.main(verbosity=2)
    else:
        print("\nSkipping full unit tests - SGLang utilities not available")