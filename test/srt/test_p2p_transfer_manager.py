"""
Test P2PTransferManager with Qwen3-32B model weight transfer simulation.

This test simulates real-world RL training scenario with Qwen3-32B model:
- Rank 0: Training process (sender) - updates model weights and sends via RDMA
- Rank 1+: Rollout processes (receivers) - receive weight updates via P2P RDMA

The test measures end-to-end transfer time from when training finishes update
until all rollout workers complete receiving weights.

Qwen3-32B Model Architecture:
- 64 transformer layers
- Hidden size: 5120
- Intermediate size: 27648
- Vocab size: 152064
- Total parameters: ~32B
- Total memory (fp32): ~128GB
- Total memory (fp16): ~64GB
"""

import gc
import logging
import os
import threading
import time
import unittest
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.multiprocessing as mp
import zmq

from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.utils.common import format_tcp_address
from sglang.srt.weight_sync.p2p_transfer import P2PTransferManager

logger = logging.getLogger(__name__)

# Set multiprocessing start method
mp.set_start_method("spawn", force=True)


@dataclass
class Qwen3_32BConfig:
    """Qwen3-32B model configuration."""
    num_layers: int = 64
    hidden_size: int = 5120
    intermediate_size: int = 27648
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    vocab_size: int = 152064

    def get_layer_param_counts(self) -> Dict[str, int]:
        """Get parameter counts for each layer type."""
        return {
            "qkv_proj": self.hidden_size * (self.hidden_size + 2 * (self.hidden_size // self.num_attention_heads) * self.num_key_value_heads),
            "o_proj": self.hidden_size * self.hidden_size,
            "gate_up_proj": self.hidden_size * self.intermediate_size * 2,
            "down_proj": self.intermediate_size * self.hidden_size,
            "ln": self.hidden_size * 2,  # LayerNorm weights and bias
        }

    def get_total_params(self) -> int:
        """Calculate total parameter count."""
        layer_params = self.get_layer_param_counts()
        params_per_layer = sum(layer_params.values())

        # Embedding layer
        embedding_params = self.vocab_size * self.hidden_size

        # LM head
        lm_head_params = self.vocab_size * self.hidden_size

        # Total
        total = embedding_params + (params_per_layer * self.num_layers) + lm_head_params
        return total

    def get_total_size_bytes(self, dtype=torch.float16) -> int:
        """Get total model size in bytes."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        return self.get_total_params() * element_size


def create_qwen3_layer_weights(config: Qwen3_32BConfig, device: torch.device, dtype=torch.float16) -> Dict[str, torch.Tensor]:
    """Create mock weights for a single Qwen3-32B transformer layer."""
    weights = {}

    # QKV projection
    qkv_size = config.hidden_size + 2 * (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads
    weights["qkv_proj"] = torch.randn(qkv_size, config.hidden_size, device=device, dtype=dtype)

    # O projection
    weights["o_proj"] = torch.randn(config.hidden_size, config.hidden_size, device=device, dtype=dtype)

    # Gate and Up projection (merged)
    weights["gate_up_proj"] = torch.randn(config.intermediate_size * 2, config.hidden_size, device=device, dtype=dtype)

    # Down projection
    weights["down_proj"] = torch.randn(config.hidden_size, config.intermediate_size, device=device, dtype=dtype)

    # Layer norm
    weights["ln1"] = torch.randn(config.hidden_size, device=device, dtype=dtype)
    weights["ln2"] = torch.randn(config.hidden_size, device=device, dtype=dtype)

    return weights


def create_qwen3_full_model(config: Qwen3_32BConfig, device: torch.device, dtype=torch.float16) -> Dict[str, torch.Tensor]:
    """Create full Qwen3-32B model weights (for testing purposes, we'll use a subset)."""
    weights = {}

    # Embedding layer
    weights["embedding"] = torch.randn(config.vocab_size, config.hidden_size, device=device, dtype=dtype)

    # Transformer layers (for testing, we'll create a few representative layers)
    # In production, this would be all 64 layers
    num_test_layers = min(config.num_layers, 8)  # Use 8 layers for testing
    for layer_idx in range(num_test_layers):
        layer_weights = create_qwen3_layer_weights(config, device, dtype)
        for name, weight in layer_weights.items():
            weights[f"layer_{layer_idx}_{name}"] = weight

    # LM head
    weights["lm_head"] = torch.randn(config.vocab_size, config.hidden_size, device=device, dtype=dtype)

    return weights


class TrainingWeightSenderManager:
    """
    Training-side weight sender using multiple transfer engines.
    Similar to TrainingWeightSender but handles multiple concurrent RDMA operations.
    """

    def __init__(self, hostname: str, port: int, gpu_id: int):
        self.hostname = hostname
        self.port = port
        self.gpu_id = gpu_id
        self.running = False

        # Initialize Mooncake transfer engine for RDMA writes
        self.engine = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=None,
        )

        # ZMQ socket for receiving sync_status and sending confirmations
        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{hostname}:{port}")

        # Track registered weight buffers
        self.weight_buffers = {}  # name -> (ptr, length, tensor)

        # Track transfer timing
        self.transfer_times = []
        self.transfer_lock = threading.Lock()

        logger.info(f"[TrainingSender] Initialized on {hostname}:{port}, GPU {gpu_id}")

    def register_weights(self, weights_dict: Dict[str, torch.Tensor]):
        """Register all weight tensors for sending."""
        for name, tensor in weights_dict.items():
            ptr = tensor.data_ptr()
            length = tensor.numel() * tensor.element_size()

            if name in self.weight_buffers:
                existing_ptr = self.weight_buffers[name]['ptr']
                if existing_ptr == ptr:
                    self.weight_buffers[name]['tensor'] = tensor
                    continue

            # Register with Mooncake engine
            self.engine.register(ptr, length)

            self.weight_buffers[name] = {
                'ptr': ptr,
                'length': length,
                'tensor': tensor,
            }

        logger.info(f"[TrainingSender] Registered {len(weights_dict)} weight tensors")

    def start(self):
        """Start the message handling loop."""
        if self.running:
            return

        self.running = True
        self.handler_thread = threading.Thread(target=self._message_loop, daemon=True)
        self.handler_thread.start()
        logger.info("[TrainingSender] Started message handler")

    def stop(self):
        """Stop the sender."""
        self.running = False
        if hasattr(self, 'handler_thread'):
            self.handler_thread.join(timeout=2)
        self.router_socket.close()
        self.context.term()
        logger.info("[TrainingSender] Stopped")

    def _message_loop(self):
        """Handle incoming sync_status messages."""
        while self.running:
            try:
                if self.router_socket.poll(timeout=100):
                    frames = self.router_socket.recv_multipart()

                    if len(frames) < 2:
                        continue

                    identity = frames[0]
                    message = None
                    for i in range(1, len(frames)):
                        try:
                            if len(frames[i]) > 0:
                                message = zmq.utils.jsonapi.loads(frames[i])
                                break
                        except Exception:
                            continue

                    if message is None:
                        continue

                    msg_type = message.get("type", "")
                    if msg_type == "sync_status":
                        self._handle_sync_status(identity, message)

            except zmq.ZMQError as e:
                if self.running:
                    logger.error(f"[TrainingSender] ZMQ error: {e}")
                break
            except Exception as e:
                logger.error(f"[TrainingSender] Error: {e}", exc_info=True)

    def _handle_sync_status(self, identity: bytes, message: dict):
        """Handle sync_status and perform RDMA transfer."""
        rollout_transfer_session_id = message.get("transfer_session_id", "")
        rollout_ptr = message.get("ptr", 0)
        rollout_length = message.get("length", 0)
        task_id = message.get("task_id", "")
        weight_name = message.get("weight_name", "")  # Which weight to transfer

        try:
            # Get the specific weight to send
            if weight_name not in self.weight_buffers:
                # Fallback to first weight if not specified
                weight_name = list(self.weight_buffers.keys())[0]

            weight_info = self.weight_buffers[weight_name]
            src_ptr = weight_info['ptr']
            src_length = weight_info['length']

            if src_length != rollout_length:
                raise RuntimeError(f"Length mismatch: src={src_length}, dst={rollout_length}")

            # Record transfer start time
            transfer_start = time.time()

            # Perform RDMA write
            status = self.engine.transfer_sync(
                session_id=rollout_transfer_session_id,
                buffer=src_ptr,
                peer_buffer_address=rollout_ptr,
                length=src_length,
            )

            # Record transfer end time
            transfer_end = time.time()
            transfer_duration = transfer_end - transfer_start

            with self.transfer_lock:
                self.transfer_times.append({
                    'task_id': task_id,
                    'weight_name': weight_name,
                    'size_bytes': src_length,
                    'duration_seconds': transfer_duration,
                    'bandwidth_gbps': (src_length * 8) / (transfer_duration * 1e9) if transfer_duration > 0 else 0
                })

            if status != 0:
                raise RuntimeError(f"RDMA transfer failed with status {status}")

            logger.info(
                f"[TrainingSender] RDMA completed: task={task_id}, weight={weight_name}, "
                f"size={src_length/1e6:.2f}MB, time={transfer_duration*1000:.2f}ms, "
                f"bandwidth={((src_length * 8) / (transfer_duration * 1e9)):.2f}Gbps"
            )

            # Send success confirmation
            response_data = zmq.utils.jsonapi.dumps({
                "type": "transfer_complete",
                "status": "success",
                "task_id": task_id,
            })
            self.router_socket.send_multipart([identity, b"", response_data])

        except Exception as e:
            logger.error(f"[TrainingSender] Error handling sync_status: {e}", exc_info=True)
            response_data = zmq.utils.jsonapi.dumps({
                "type": "transfer_complete",
                "status": "failed",
                "error": str(e),
                "task_id": task_id,
            })
            self.router_socket.send_multipart([identity, b"", response_data])

    def get_transfer_stats(self) -> Dict:
        """Get transfer statistics."""
        with self.transfer_lock:
            if not self.transfer_times:
                return {}

            total_bytes = sum(t['size_bytes'] for t in self.transfer_times)
            total_duration = sum(t['duration_seconds'] for t in self.transfer_times)
            avg_bandwidth = np.mean([t['bandwidth_gbps'] for t in self.transfer_times])

            return {
                'num_transfers': len(self.transfer_times),
                'total_bytes': total_bytes,
                'total_duration': total_duration,
                'avg_bandwidth_gbps': avg_bandwidth,
                'transfers': self.transfer_times.copy()
            }


def training_process_manager(
    rank: int,
    world_size: int,
    config: Qwen3_32BConfig,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    hostname: str = "127.0.0.1",
    port: int = 50000,
):
    """Training process using manager."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[Training] Initializing on GPU {rank}")

        # Initialize sender
        sender = TrainingWeightSenderManager(hostname=hostname, port=port, gpu_id=rank)

        # Create model weights
        logger.info(f"[Training] Creating Qwen3-32B model weights...")
        weights = create_qwen3_full_model(config, device, dtype=torch.float16)

        total_params = sum(w.numel() for w in weights.values())
        total_size_mb = sum(w.numel() * w.element_size() for w in weights.values()) / 1e6
        logger.info(f"[Training] Model created: {total_params/1e9:.2f}B params, {total_size_mb:.2f}MB")

        # Register weights
        sender.register_weights(weights)
        sender.start()

        # Wait for rollout processes
        logger.info("[Training] Waiting for rollout processes...")
        barrier.wait()

        # Simulate training updates
        for update_idx in range(num_updates):
            update_start = time.time()

            # Simulate weight update
            for weight in weights.values():
                weight += torch.randn_like(weight) * 0.01
            torch.cuda.synchronize()

            update_end = time.time()
            logger.info(
                f"[Training] Update {update_idx + 1}/{num_updates} completed in "
                f"{(update_end - update_start)*1000:.2f}ms"
            )

            # Re-register updated weights
            sender.register_weights(weights)

            # Wait for all rollout workers to complete transfer
            barrier.wait()

        # Get transfer statistics
        stats = sender.get_transfer_stats()
        result_queue.put(("training_stats", stats))

        logger.info("[Training] All updates completed")
        sender.stop()

    except Exception as e:
        logger.error(f"[Training] Error: {e}", exc_info=True)
        result_queue.put(("training_error", str(e)))


def rollout_process_manager(
    rank: int,
    world_size: int,
    config: Qwen3_32BConfig,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    training_hostname: str = "127.0.0.1",
    training_port: int = 50000,
):
    """Rollout process using P2PTransferManager."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[Rollout-{rank}] Initializing on GPU {rank}")

        # Initialize P2PTransferManager
        manager = P2PTransferManager(
            hostname="127.0.0.1",
            gpu_id=rank,
            ib_device=None,
        )

        logger.info(f"[Rollout-{rank}] Initialized with {manager.engine_pool_size} transfer engines")

        # Allocate buffers for model weights
        weights = create_qwen3_full_model(config, device, dtype=torch.float16)

        total_params = sum(w.numel() for w in weights.values())
        total_size_mb = sum(w.numel() * w.element_size() for w in weights.values()) / 1e6
        logger.info(f"[Rollout-{rank}] Allocated buffers: {total_params/1e9:.2f}B params, {total_size_mb:.2f}MB")

        # Signal ready
        barrier.wait()

        # Track timing for all transfers
        all_transfer_times = []

        # Receive weight updates
        for update_idx in range(num_updates):
            update_start = time.time()

            session_id = f"{training_hostname}:{training_port}"
            handles = []

            # Submit all weight transfer tasks
            submission_start = time.time()
            for weight_name, weight_tensor in weights.items():
                ptr = weight_tensor.data_ptr()
                length = weight_tensor.numel() * weight_tensor.element_size()

                handle = manager.submit_transfer_task(
                    session_id=session_id,
                    ptr=ptr,
                    length=length,
                )
                handles.append((weight_name, handle))

            submission_end = time.time()

            logger.info(
                f"[Rollout-{rank}] Submitted {len(handles)} transfer tasks in "
                f"{(submission_end - submission_start)*1000:.2f}ms"
            )

            # Wait for all transfers to complete
            wait_start = time.time()
            for weight_name, handle in handles:
                handle.wait()

            torch.cuda.synchronize()
            wait_end = time.time()

            update_end = time.time()

            total_time = update_end - update_start
            wait_time = wait_end - wait_start

            all_transfer_times.append({
                'update_idx': update_idx,
                'total_time': total_time,
                'submission_time': submission_end - submission_start,
                'wait_time': wait_time,
                'num_weights': len(handles),
                'total_bytes': sum(w.numel() * w.element_size() for w in weights.values()),
            })

            logger.info(
                f"[Rollout-{rank}] Update {update_idx + 1}/{num_updates} completed: "
                f"total={total_time*1000:.2f}ms, submission={((submission_end - submission_start)*1000):.2f}ms, "
                f"wait={wait_time*1000:.2f}ms, bandwidth={(sum(w.numel() * w.element_size() for w in weights.values()) * 8) / (total_time * 1e9):.2f}Gbps"
            )

            # Synchronize with training
            barrier.wait()

        # Calculate and report statistics
        avg_total_time = np.mean([t['total_time'] for t in all_transfer_times])
        avg_wait_time = np.mean([t['wait_time'] for t in all_transfer_times])
        total_bytes = all_transfer_times[0]['total_bytes']
        avg_bandwidth = (total_bytes * 8) / (avg_total_time * 1e9)

        stats = {
            'rank': rank,
            'num_updates': num_updates,
            'avg_total_time': avg_total_time,
            'avg_wait_time': avg_wait_time,
            'avg_bandwidth_gbps': avg_bandwidth,
            'total_bytes': total_bytes,
            'engine_pool_size': manager.engine_pool_size,
            'all_transfers': all_transfer_times,
        }

        result_queue.put((f"rollout_{rank}_stats", stats))

        logger.info(
            f"[Rollout-{rank}] All updates completed. Avg time: {avg_total_time*1000:.2f}ms, "
            f"Avg bandwidth: {avg_bandwidth:.2f}Gbps"
        )

    except Exception as e:
        logger.error(f"[Rollout-{rank}] Error: {e}", exc_info=True)
        result_queue.put((f"rollout_{rank}_error", str(e)))


def worker_process_manager(
    rank: int,
    world_size: int,
    config: Qwen3_32BConfig,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    training_port: int = 50000,
):
    """Entry point for each worker process."""
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank == 0:
        training_process_manager(
            rank, world_size, config, num_updates,
            result_queue, barrier, "127.0.0.1", training_port
        )
    else:
        rollout_process_manager(
            rank, world_size, config, num_updates,
            result_queue, barrier, "127.0.0.1", training_port
        )


class TestP2PTransferManagerQwen3(unittest.TestCase):
    """Test P2PTransferManager with Qwen3-32B model simulation."""

    def test_qwen3_single_training_single_rollout(self):
        """Test 1 training + 1 rollout with Qwen3-32B model weights."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("Requires at least 2 CUDA devices")

        world_size = 2
        config = Qwen3_32BConfig()
        num_updates = 3
        training_port = 50100

        logger.info(
            f"\n{'='*80}\n"
            f"Starting Qwen3-32B P2PTransferManager Test\n"
            f"Config: {world_size} processes, {num_updates} updates\n"
            f"Model: {config.get_total_params()/1e9:.2f}B params, "
            f"{config.get_total_size_bytes()/1e6:.2f}MB (fp16)\n"
            f"{'='*80}\n"
        )

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        context = mp.spawn(
            worker_process_manager,
            args=(world_size, config, num_updates, result_queue, barrier, training_port),
            nprocs=world_size,
            join=False,
        )

        # Collect results
        results = {}
        timeout = 120
        start_time = time.time()

        expected_results = 1 + (world_size - 1)  # training stats + rollout stats
        while len(results) < expected_results:
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                key, value = result_queue.get(timeout=min(10, remaining_time))
                results[key] = value
            except Exception:
                if all(not p.is_alive() for p in context.processes):
                    break

        context.join()

        # Check for errors
        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        # Print statistics
        if "training_stats" in results:
            training_stats = results["training_stats"]
            logger.info(
                f"\n{'='*80}\n"
                f"Training Statistics:\n"
                f"  Total transfers: {training_stats.get('num_transfers', 0)}\n"
                f"  Total bytes: {training_stats.get('total_bytes', 0)/1e6:.2f}MB\n"
                f"  Total duration: {training_stats.get('total_duration', 0):.3f}s\n"
                f"  Avg bandwidth: {training_stats.get('avg_bandwidth_gbps', 0):.2f}Gbps\n"
                f"{'='*80}\n"
            )

        for rank in range(1, world_size):
            key = f"rollout_{rank}_stats"
            if key in results:
                stats = results[key]
                logger.info(
                    f"\n{'='*80}\n"
                    f"Rollout-{rank} Statistics:\n"
                    f"  Engine pool size: {stats['engine_pool_size']}\n"
                    f"  Avg total time: {stats['avg_total_time']*1000:.2f}ms\n"
                    f"  Avg wait time: {stats['avg_wait_time']*1000:.2f}ms\n"
                    f"  Avg bandwidth: {stats['avg_bandwidth_gbps']:.2f}Gbps\n"
                    f"  Total data: {stats['total_bytes']/1e6:.2f}MB\n"
                    f"{'='*80}\n"
                )

        # Cleanup
        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()

    def test_qwen3_single_training_multiple_rollouts(self):
        """Test 1 training + multiple rollouts with Qwen3-32B."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            self.skipTest("Requires at least 4 CUDA devices")

        world_size = 4  # 1 training + 3 rollouts
        config = Qwen3_32BConfig()
        num_updates = 2
        training_port = 50101

        logger.info(
            f"\n{'='*80}\n"
            f"Starting Multi-Rollout Qwen3-32B Test\n"
            f"Config: {world_size} processes (1 training + {world_size-1} rollouts), {num_updates} updates\n"
            f"{'='*80}\n"
        )

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        context = mp.spawn(
            worker_process_manager,
            args=(world_size, config, num_updates, result_queue, barrier, training_port),
            nprocs=world_size,
            join=False,
        )

        # Collect results
        results = {}
        timeout = 120
        start_time = time.time()

        expected_results = 1 + (world_size - 1)
        while len(results) < expected_results:
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                key, value = result_queue.get(timeout=min(10, remaining_time))
                results[key] = value
            except Exception:
                if all(not p.is_alive() for p in context.processes):
                    break

        context.join()

        # Check for errors
        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        # Verify all rollouts completed
        for rank in range(1, world_size):
            self.assertIn(f"rollout_{rank}_stats", results)

        # Print comparative statistics
        logger.info(f"\n{'='*80}\nComparative Statistics:\n")
        for rank in range(1, world_size):
            stats = results[f"rollout_{rank}_stats"]
            logger.info(
                f"Rollout-{rank}: "
                f"avg_time={stats['avg_total_time']*1000:.2f}ms, "
                f"bandwidth={stats['avg_bandwidth_gbps']:.2f}Gbps"
            )
        logger.info(f"{'='*80}\n")

        # Cleanup
        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    unittest.main()
