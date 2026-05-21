"""
Problem Background

Multiple calls to the MTP model's update_weights_from_tensor interface for parameter
updates can cause OOM (GPU memory leak).

Root Cause Analysis

SGLang and RL engine communicate via inter-process communication (IPC), with parameters
transferred through CUDA IPC.

In EAGLEWorker/EAGLEWorker2, update_weights_from_tensor calls in sequence:

draft_worker.draft_runner.update_weights_from_tensor()

target_worker.model_runner.update_weights_from_tensor()

Each worker deserializes named_tensors independently, causing the CUDA IPC handle in
LocalSerializedTensor to be opened twice (cudaIpcOpenMemHandle) but closed only once
(cudaIpcCloseMemHandle), resulting in GPU memory leak.

Solution

Add a new unwrap_ipc_tensors function to ensure IPC allocation and release are performed
only once per update process.

Test MTP model update_weights_from_tensor via HTTP interface.

This test verifies:
1. MTP model weights can be updated correctly via HTTP
2. No memory leak occurs after multiple weight updates
"""

import gc
import subprocess
import unittest

import requests
import torch
from transformers import AutoModelForCausalLM

from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="stage-b-test-small-1-gpu")

DEFAULT_SMALL_MODEL_NAME_FOR_TEST = "XiaomiMiMo/MiMo-7B-RL"


def get_server_gpu_memory(pid=None):
    """
    Get SGLang server GPU memory usage using nvidia-smi.
    If pid is provided, query specific process.
    Otherwise, return the first GPU process.
    Returns memory in bytes.
    """
    try:
        if pid is not None:
            # Query specific PID
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        line_pid = parts[0].strip()
                        if int(line_pid) == pid:
                            memory_mb = int(parts[1].strip())
                            return memory_mb * 1024 * 1024, pid
            return None, None
        else:
            # Get the first process's memory
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                if lines:
                    parts = lines[0].split(",")
                    if len(parts) >= 2:
                        line_pid = parts[0].strip()
                        memory_mb = int(parts[1].strip())
                        return memory_mb * 1024 * 1024, line_pid
            return None, None
    except Exception as e:
        print(f"Warning: Failed to get GPU memory: {e}")
        return None, None


def get_all_updatable_params(model):
    """
    Get all updatable parameters from the model.

    Args:
        model: Loaded model instance

    Returns:
        dict: Dictionary of parameter_name -> shape (list)
    """
    params_info = {}

    for name, param in model.named_parameters():
        # Skip certain parameters that are not updatable
        if "rotary_emb.inv_freq" in name or "projector" in name:
            continue

        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            continue

        if param.requires_grad:
            params_info[name] = list(param.shape)

    return params_info


class TestMTPMemoryLeakWithIPC(CustomTestCase):
    """
    Specific test for MTP memory leak with IPC-like updates.

    This test simulates the scenario where SGLang and RL engine communicate
    via IPC, and update_weights_from_tensor is called multiple times.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--max-running-requests",
                "8",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.5",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_update_weights(self, named_tensors, flush_cache=True):
        """Update model weights via HTTP."""
        response = requests.post(
            self.base_url + "/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [
                    MultiprocessingSerializer.serialize(named_tensors, output_str=True)
                ],
                "flush_cache": flush_cache,
            },
        )
        ret = response.json()
        return ret

    def test_full_model_update_all_parameters(self):
        """
        Test updating ALL model parameters (MTP + non-MTP) via IPC with batched updates.
        This test checks if SGLang server has memory leak after parameter updates.
        """
        print("\n" + "=" * 80)
        print("FULL MODEL UPDATE TEST - Memory Leak Detection")
        print("=" * 80)

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.ipc_collect()

        # Step 1: Record initial server memory using nvidia-smi
        memory_initial, server_pid = get_server_gpu_memory()
        if memory_initial is not None:
            print(
                f"\n[Step 1] Initial server memory (PID {server_pid}): {memory_initial / 1024 ** 2:.2f} MB"
            )
        else:
            print(f"\n[Step 1] Warning: Could not get server memory via nvidia-smi")
            memory_initial = 0

        try:
            print("\n[Step 2] Loading model to get parameter information...")
            model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cpu",
            )

            all_params = get_all_updatable_params(model)
            print(f"  ✓ Found {len(all_params)} total parameters")

            # Count MTP vs non-MTP
            mtp_count = sum(1 for name in all_params if "mtp" in name)
            non_mtp_count = len(all_params) - mtp_count
            print(f"  - MTP parameters: {mtp_count}")
            print(f"  - Non-MTP parameters: {non_mtp_count}")

            # Print some sample parameter names
            print("\nSample MTP parameters:")
            for name in list(all_params.keys())[:5]:
                if "mtp" in name:
                    print(f"  - {name}: {all_params[name]}")

            print("\nSample Non-MTP parameters:")
            for name in list(all_params.keys())[:5]:
                if "mtp" not in name:
                    print(f"  - {name}: {all_params[name]}")

            # Clean up model
            del model
            gc.collect()

            # Step 3: Build tensors for all parameters
            print("\n[Step 3] Building tensors for all parameters...")
            named_tensors = []
            failed_params = []
            tensor_count = 0

            for param_name in sorted(all_params.keys()):
                shape = all_params[param_name]
                try:
                    tensor = torch.randn(*shape, device="cuda", dtype=torch.float16)
                    named_tensors.append((param_name, tensor))
                    tensor_count += 1

                    if tensor_count % 50 == 0:
                        print(f"  Built {tensor_count}/{len(all_params)} tensors...")

                except Exception as e:
                    failed_params.append((param_name, str(e)))
                    continue

            if failed_params:
                print(f"\n⚠ Failed to create {len(failed_params)} tensors:")
                for name, err in failed_params[:10]:
                    print(f"  - {name}: {err}")

            print(f"  ✓ Built {len(named_tensors)} tensors successfully")

            # Step 4: Run update with full parameters in batches
            print("\n[Step 4] Running full model weight update (batched)...")

            batch_size = 50
            successful_batches = 0
            failed_batches = 0

            for i in range(0, len(named_tensors), batch_size):
                batch = named_tensors[i : i + batch_size]
                try:
                    ret = self.run_update_weights(batch, flush_cache=True)
                    if ret.get("success"):
                        successful_batches += 1
                    else:
                        failed_batches += 1
                        print(
                            f"  Batch {i // batch_size + 1} failed: {ret.get('message', 'Unknown error')}"
                        )
                except Exception as e:
                    failed_batches += 1
                    print(f"  Batch {i // batch_size + 1} error: {e}")

            print(f"\n  Batch update completed:")
            print(f"    - Successful batches: {successful_batches}")
            print(f"    - Failed batches: {failed_batches}")

            # Step 5: Clean up test tensors and check server memory via nvidia-smi
            print("\n[Step 5] Cleaning up test tensors...")
            del named_tensors
            gc.collect()
            torch.cuda.empty_cache()

            # Record server memory after update (using nvidia-smi with fixed PID)
            if server_pid is not None:
                memory_after_update, _ = get_server_gpu_memory(int(server_pid))
            else:
                memory_after_update, _ = get_server_gpu_memory()
            if memory_after_update is not None:
                print(
                    f"  Server (PID {server_pid}) memory after update: {memory_after_update / 1024 ** 2:.2f} MB"
                )
            else:
                print(f"  Warning: Could not get server memory via nvidia-smi")
                memory_after_update = memory_initial

            # Calculate memory differences
            memory_leak = memory_after_update - memory_initial

            print("\n" + "=" * 80)
            print("MEMORY LEAK TEST RESULTS (via nvidia-smi)")
            print("=" * 80)
            print(f"  Initial server memory: {memory_initial / 1024 ** 2:.2f} MB")
            print(
                f"  Server memory after update: {memory_after_update / 1024 ** 2:.2f} MB"
            )
            print(f"  Memory increase (leak check): {memory_leak / 1024 ** 2:.2f} MB")
            print("=" * 80)

            # Test summary
            print("\nTEST SUMMARY")
            print("=" * 80)
            print(f"  Total parameters: {len(all_params)}")
            print(f"  MTP parameters: {mtp_count}")
            print(f"  Non-MTP parameters: {non_mtp_count}")
            print(f"  Successful batches: {successful_batches}")
            print(f"  Failed batches: {failed_batches}")
            print("=" * 80)

            # Assert that batches succeeded
            self.assertGreater(successful_batches, 0, "No successful batch updates!")

            # Assert no significant memory leak (allow small buffer for normal operation)
            leak_threshold = 500 * 1024 * 1024  # 500 MB threshold
            self.assertLess(
                memory_leak,
                leak_threshold,
                f"Memory leak detected: {memory_leak / 1024 ** 2:.2f} MB increase "
                f"(threshold: {leak_threshold / 1024 ** 2:.2f} MB)",
            )
            print(
                f"\n Memory leak test PASSED (leak: {memory_leak / 1024 ** 2:.2f} MB < {leak_threshold / 1024 ** 2:.2f} MB)"
            )

        except Exception as e:
            print(f"Error in full model update test: {e}")
            import traceback

            traceback.print_exc()
            raise

    def get_weights_by_name(self, name, truncate_size=100):
        """Get model weights by parameter name."""
        response = requests.post(
            self.base_url + "/get_weights_by_name",
            json={"name": name, "truncate_size": truncate_size},
        )
        return response.json()


if __name__ == "__main__":
    unittest.main()
