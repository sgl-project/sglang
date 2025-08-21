#!/usr/bin/env python3
"""
SGLang Memory Leak Monitoring Script

This script reproduces the experimental setup described in issue #9365 
for tracking memory leaks in SGLang servers.

Usage:
    python memory_leak_monitor.py --model meta-llama/Llama-3.2-1B-Instruct --test-type text
    python memory_leak_monitor.py --model Qwen/Qwen2.5-VL-3B-Instruct --test-type vlm-text
    python memory_leak_monitor.py --model Qwen/Qwen2.5-VL-3B-Instruct --test-type vlm-image

Based on: https://github.com/sgl-project/sglang/issues/9365
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

# Optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class MemoryReading:
    """A single memory reading at a specific point in time."""
    request_index: int
    timestamp: float
    gpu_memory_mb: float
    gpu_memory_used_mb: float
    cpu_memory_mb: float
    peak_memory_mb: float


class MemoryMonitor:
    """Monitors memory usage during SGLang server operation."""
    
    def __init__(self):
        self.readings: List[MemoryReading] = []
        self.peak_memory = 0.0
        
    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Get current GPU memory usage in MB."""
        if not HAS_GPUTIL:
            return 0.0, 0.0
            
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                total_mb = gpu.memoryTotal
                used_mb = gpu.memoryUsed
                return total_mb, used_mb
            else:
                return 0.0, 0.0
        except Exception:
            return 0.0, 0.0
    
    def get_cpu_memory_usage(self) -> float:
        """Get current CPU memory usage in MB."""
        if not HAS_PSUTIL:
            return 0.0
            
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 ** 2)
        except Exception:
            return 0.0
    
    def take_reading(self, request_index: int):
        """Take a memory reading."""
        timestamp = time.time()
        gpu_total, gpu_used = self.get_gpu_memory_usage()
        cpu_memory = self.get_cpu_memory_usage()
        
        # Update peak memory
        self.peak_memory = max(self.peak_memory, gpu_used)
        
        reading = MemoryReading(
            request_index=request_index,
            timestamp=timestamp,
            gpu_memory_mb=gpu_total,
            gpu_memory_used_mb=gpu_used,
            cpu_memory_mb=cpu_memory,
            peak_memory_mb=self.peak_memory
        )
        
        self.readings.append(reading)
        return reading
    
    def save_results(self, filename: str):
        """Save memory readings to a JSON file."""
        data = []
        for reading in self.readings:
            data.append({
                "request_index": reading.request_index,
                "timestamp": reading.timestamp,
                "gpu_memory_mb": reading.gpu_memory_mb,
                "gpu_memory_used_mb": reading.gpu_memory_used_mb,
                "cpu_memory_mb": reading.cpu_memory_mb,
                "peak_memory_mb": reading.peak_memory_mb
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Memory readings saved to: {filename}")
    
    def plot_memory_usage(self, filename: str, title: str):
        """Create a plot of memory usage over time."""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available - skipping plot generation")
            return
            
        if not self.readings:
            print("No memory readings to plot")
            return
        
        requests = [r.request_index for r in self.readings]
        gpu_memory = [r.gpu_memory_used_mb for r in self.readings]
        peak_memory = [r.peak_memory_mb for r in self.readings]
        
        plt.figure(figsize=(12, 8))
        plt.plot(requests, gpu_memory, label='Memory (MB)', linewidth=2)
        plt.plot(requests, peak_memory, label='Peak Memory (MB)', linestyle='--', linewidth=2)
        
        plt.xlabel('Request Index')
        plt.ylabel('Memory (MB)')
        plt.title(f'GPU Memory Usage Over Requests ({title})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate and display memory growth
        if len(gpu_memory) > 1:
            initial_memory = gpu_memory[0]
            final_memory = gpu_memory[-1]
            growth = final_memory - initial_memory
            growth_rate = growth / len(gpu_memory)
            
            plt.text(0.02, 0.98, 
                    f'Initial: {initial_memory:.1f}MB\n'
                    f'Final: {final_memory:.1f}MB\n'
                    f'Growth: {growth:.1f}MB\n'
                    f'Rate: {growth_rate:.2f}MB/req',
                    transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Memory usage plot saved to: {filename}")


class SGLangServerManager:
    """Manages SGLang server processes for testing."""
    
    def __init__(self):
        self.process = None
        
    def start_server(self, model: str, port: int = 30000, extra_args: List[str] = None) -> bool:
        """Start SGLang server with the specified model."""
        if extra_args is None:
            extra_args = []
            
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model,
            "--port", str(port),
            "--disable-radix-cache",  # Disable caching to isolate memory issues
            "--host", "0.0.0.0"
        ] + extra_args
        
        print(f"Starting server with command: {' '.join(cmd)}")
        
        try:
            # Start server process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            base_url = f"http://127.0.0.1:{port}"
            for attempt in range(60):  # Wait up to 60 seconds
                try:
                    response = requests.get(f"{base_url}/health", timeout=5)
                    if response.status_code == 200:
                        print(f"Server started successfully on port {port}")
                        return True
                except:
                    pass
                time.sleep(1)
                
            print("Server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the SGLang server."""
        if self.process:
            try:
                # Try graceful shutdown first
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self.process.kill()
                self.process.wait()
            except Exception as e:
                print(f"Error stopping server: {e}")
            finally:
                self.process = None


class RequestSender:
    """Sends requests to SGLang server for testing."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        
    def send_text_request(self, prompt: str, max_new_tokens: int = 32) -> Dict:
        """Send a text-only request."""
        data = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            }
        }
        
        response = self.session.post(f"{self.base_url}/generate", json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def send_vlm_request(self, prompt: str, image_url: str, max_new_tokens: int = 32) -> Dict:
        """Send a VLM request with an image."""
        data = {
            "text": prompt,
            "image_data": [image_url],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            }
        }
        
        response = self.session.post(f"{self.base_url}/generate", json=data, timeout=30)
        response.raise_for_status()
        return response.json()


def run_memory_test(args):
    """Run the memory leak test with the specified parameters."""
    print(f"=== Starting Memory Leak Test ===")
    print(f"Model: {args.model}")
    print(f"Test Type: {args.test_type}")
    print(f"Number of requests: {args.num_requests}")
    print(f"Request interval: {args.request_interval}s")
    
    # Initialize components
    server_manager = SGLangServerManager()
    memory_monitor = MemoryMonitor()
    
    # Start server
    extra_args = ["--mem-fraction-static", str(args.mem_fraction)]
    if args.test_type.startswith("vlm") and args.mm_attention_backend:
        extra_args.extend(["--mm-attention-backend", args.mm_attention_backend])
    
    if not server_manager.start_server(args.model, args.port, extra_args):
        print("Failed to start server")
        return False
    
    try:
        # Initialize request sender
        base_url = f"http://127.0.0.1:{args.port}"
        request_sender = RequestSender(base_url)
        
        # Test prompts and images
        text_prompt = "Tell me a short story about artificial intelligence."
        image_url = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
        
        # Take initial memory reading
        initial_reading = memory_monitor.take_reading(0)
        print(f"Initial GPU memory: {initial_reading.gpu_memory_used_mb:.1f}MB")
        
        # Send requests and monitor memory
        for i in range(1, args.num_requests + 1):
            try:
                # Choose request type based on test configuration
                if args.test_type == "text":
                    result = request_sender.send_text_request(text_prompt)
                elif args.test_type == "vlm-text":
                    # VLM server, but text-only requests (no mm_processor usage)
                    result = request_sender.send_text_request(text_prompt)
                elif args.test_type == "vlm-image":
                    # VLM server with image processing
                    result = request_sender.send_vlm_request(
                        "What do you see in this image?", image_url
                    )
                else:
                    raise ValueError(f"Unknown test type: {args.test_type}")
                
                # Take memory reading
                reading = memory_monitor.take_reading(i)
                
                # Log progress
                if i % 10 == 0:
                    memory_growth = reading.gpu_memory_used_mb - initial_reading.gpu_memory_used_mb
                    print(f"Request {i}: GPU memory: {reading.gpu_memory_used_mb:.1f}MB "
                          f"(+{memory_growth:.1f}MB from start)")
                
                # Wait between requests
                time.sleep(args.request_interval)
                
            except Exception as e:
                print(f"Error on request {i}: {e}")
                if args.stop_on_error:
                    break
        
        # Calculate final statistics
        final_reading = memory_monitor.readings[-1]
        total_growth = final_reading.gpu_memory_used_mb - initial_reading.gpu_memory_used_mb
        growth_rate = total_growth / len(memory_monitor.readings) if memory_monitor.readings else 0
        
        print(f"\n=== Test Results ===")
        print(f"Total requests: {len(memory_monitor.readings) - 1}")
        print(f"Initial GPU memory: {initial_reading.gpu_memory_used_mb:.1f}MB")
        print(f"Final GPU memory: {final_reading.gpu_memory_used_mb:.1f}MB")
        print(f"Total memory growth: {total_growth:.1f}MB")
        print(f"Memory growth rate: {growth_rate:.3f}MB/request")
        print(f"Peak GPU memory: {final_reading.peak_memory_mb:.1f}MB")
        
        # Save results
        timestamp = int(time.time())
        test_name = f"{args.test_type}_{args.model.replace('/', '_')}_{timestamp}"
        
        # Save data
        data_filename = f"/tmp/{test_name}_memory_data.json"
        memory_monitor.save_results(data_filename)
        
        # Create plot
        plot_filename = f"/tmp/{test_name}_memory_plot.png"
        plot_title = f"{args.test_type} - {args.model}"
        memory_monitor.plot_memory_usage(plot_filename, plot_title)
        
        return True
        
    finally:
        # Clean up
        server_manager.stop_server()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SGLang Memory Leak Monitoring")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Model path (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    
    parser.add_argument("--test-type", type=str, required=True,
                       choices=["text", "vlm-text", "vlm-image"],
                       help="Type of test to run")
    
    parser.add_argument("--num-requests", type=int, default=100,
                       help="Number of requests to send (default: 100)")
    
    parser.add_argument("--request-interval", type=float, default=0.1,
                       help="Interval between requests in seconds (default: 0.1)")
    
    parser.add_argument("--port", type=int, default=30000,
                       help="Server port (default: 30000)")
    
    parser.add_argument("--mem-fraction", type=float, default=0.7,
                       help="Memory fraction for server (default: 0.7)")
    
    parser.add_argument("--mm-attention-backend", type=str, default="sdpa",
                       choices=["sdpa", "fa3"],
                       help="MM attention backend for VLM (default: sdpa)")
    
    parser.add_argument("--stop-on-error", action="store_true",
                       help="Stop test on first error")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.test_type.startswith("vlm") and "VL" not in args.model:
        print("Warning: Using VLM test type with non-VLM model")
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\nReceived interrupt signal, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the test
    try:
        success = run_memory_test(args)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()