"""
Test suite for P2PTransferManager with Qwen3-32B model.

This test validates:
1. Manager connectivity test (simple correctness)
2. Manager performance test with Qwen3-32B model

Test scenarios:
- 1 training + 1 rollout process
- 1 training + 2 rollout processes

Model: Qwen3-32B
- 64 transformer layers (8 layers for testing)
- Hidden size: 5120
- Intermediate size: 27648
- Vocab size: 152064
- Total parameters: ~32B
- Total memory (fp16): ~64GB
"""

import gc
import logging
import os
import threading
import time
import unittest
from dataclasses import dataclass
from typing import Dict

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


def create_simple_weight(size: int, device: torch.device, dtype=torch.float16) -> torch.Tensor:
    """Create a simple weight tensor for correctness testing."""
    return torch.randn(size, size, device=device, dtype=dtype)


def simple_training_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    hostname: str = "127.0.0.1",
    port: int = 50000,
):
    """Simplified training process for correctness testing."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        sender = TrainingWeightSender(hostname=hostname, port=port, gpu_id=rank)

        # Single small weight (128x128 fp16 = 32KB)
        weight = create_simple_weight(128, device, dtype=torch.float16)
        weights = {"test_weight": weight}

        total_size_kb = weight.numel() * weight.element_size() / 1e3
        logger.info(f"[SimpleTrain] Weight size: {total_size_kb:.2f}KB")

        sender.register_weights(weights)
        sender.start()

        barrier.wait()

        # Single update
        weight += torch.randn_like(weight) * 0.01
        torch.cuda.synchronize()
        sender.register_weights(weights)

        barrier.wait()
        sender.stop()

    except Exception as e:
        logger.error(f"[SimpleTrain] Error: {e}", exc_info=True)
        result_queue.put(("training_error", str(e)))


def simple_rollout_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    training_hostname: str = "127.0.0.1",
    training_port: int = 50000,
):
    """Simplified rollout process for correctness testing."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[SimpleRollout-{rank}] Using Manager on GPU {rank}")

        transfer_impl = P2PTransferManager(
            hostname="127.0.0.1",
            gpu_id=rank,
            ib_device=None,
        )

        weight = create_simple_weight(128, device, dtype=torch.float16)
        original_data = weight.clone()

        barrier.wait()

        session_id = f"{training_hostname}:{training_port}"
        ptr = weight.data_ptr()
        length = weight.numel() * weight.element_size()

        handle = transfer_impl.submit_transfer_task(
            session_id=session_id,
            ptr=ptr,
            length=length,
        )
        handle.wait()
        torch.cuda.synchronize()

        # Verify data changed
        data_changed = not torch.equal(weight, original_data)
        logger.info(f"[SimpleRollout-{rank}] Data changed: {data_changed}")

        result_queue.put((f"rollout_{rank}_success", data_changed))

        barrier.wait()

    except Exception as e:
        logger.error(f"[SimpleRollout-{rank}] Error: {e}", exc_info=True)
        result_queue.put((f"rollout_{rank}_error", str(e)))


def simple_worker_process(
    rank: int,
    world_size: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    training_port: int = 50000,
):
    """Entry point for simple correctness test worker."""
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank == 0:
        simple_training_process(rank, world_size, result_queue, barrier, "127.0.0.1", training_port)
    else:
        simple_rollout_process(rank, world_size, result_queue, barrier, "127.0.0.1", training_port)


class TestP2PTransferCorrectness(unittest.TestCase):
    """Basic correctness tests with small weights."""

    def test_manager_correctness(self):
        """Test P2PTransferManager correctness with single small weight."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("Requires at least 2 CUDA devices")

        world_size = 2
        training_port = 50101

        logger.info("\n" + "="*80)
        logger.info("Testing P2PTransferManager Correctness (single 128x128 weight)")
        logger.info("="*80 + "\n")

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        context = mp.spawn(
            simple_worker_process,
            args=(world_size, result_queue, barrier, training_port),
            nprocs=world_size,
            join=False,
        )

        results = {}
        timeout = 30
        start_time = time.time()

        while len(results) < (world_size - 1):
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                key, value = result_queue.get(timeout=min(5, remaining_time))
                results[key] = value
            except Exception:
                if all(not p.is_alive() for p in context.processes):
                    break

        context.join()

        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        self.assertIn("rollout_1_success", results)
        self.assertTrue(results["rollout_1_success"], "Data should have changed after transfer")

        logger.info("✓ P2PTransferManager correctness test passed\n")

        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()


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
            "ln": self.hidden_size * 2,
        }

    def get_total_params(self) -> int:
        """Calculate total parameter count."""
        layer_params = self.get_layer_param_counts()
        params_per_layer = sum(layer_params.values())
        embedding_params = self.vocab_size * self.hidden_size
        lm_head_params = self.vocab_size * self.hidden_size
        total = embedding_params + (params_per_layer * self.num_layers) + lm_head_params
        return total

    def get_total_size_bytes(self, dtype=torch.float16) -> int:
        """Get total model size in bytes."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        return self.get_total_params() * element_size


def create_qwen3_layer_weights(config: Qwen3_32BConfig, device: torch.device, dtype=torch.float16) -> Dict[str, torch.Tensor]:
    """Create mock weights for a single Qwen3-32B transformer layer."""
    weights = {}
    qkv_size = config.hidden_size + 2 * (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads
    weights["qkv_proj"] = torch.randn(qkv_size, config.hidden_size, device=device, dtype=dtype)
    weights["o_proj"] = torch.randn(config.hidden_size, config.hidden_size, device=device, dtype=dtype)
    weights["gate_up_proj"] = torch.randn(config.intermediate_size * 2, config.hidden_size, device=device, dtype=dtype)
    weights["down_proj"] = torch.randn(config.hidden_size, config.intermediate_size, device=device, dtype=dtype)
    weights["ln1"] = torch.randn(config.hidden_size, device=device, dtype=dtype)
    weights["ln2"] = torch.randn(config.hidden_size, device=device, dtype=dtype)
    return weights


def create_qwen3_full_model(config: Qwen3_32BConfig, device: torch.device, dtype=torch.float16) -> Dict[str, torch.Tensor]:
    """Create full Qwen3-32B model weights (using 8 layers for testing)."""
    weights = {}
    weights["embedding"] = torch.randn(config.vocab_size, config.hidden_size, device=device, dtype=dtype)

    num_test_layers = min(config.num_layers, 8)
    for layer_idx in range(num_test_layers):
        layer_weights = create_qwen3_layer_weights(config, device, dtype)
        for name, weight in layer_weights.items():
            weights[f"layer_{layer_idx}_{name}"] = weight

    weights["lm_head"] = torch.randn(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
    return weights


class TrainingWeightSender:
    """Training-side weight sender that handles RDMA write operations."""

    def __init__(self, hostname: str, port: int, gpu_id: int):
        self.hostname = hostname
        self.port = port
        self.gpu_id = gpu_id
        self.running = False

        self.engine = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=None,
        )

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{hostname}:{port}")

        self.weight_buffers = {}

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

        try:
            # Find the weight with matching length
            weight_info = None
            weight_name = None
            for name, info in self.weight_buffers.items():
                if info['length'] == rollout_length:
                    weight_info = info
                    weight_name = name
                    break

            if weight_info is None:
                raise RuntimeError(f"No weight found with length {rollout_length}. Available lengths: {[info['length'] for info in self.weight_buffers.values()]}")

            src_ptr = weight_info['ptr']
            src_length = weight_info['length']

            if src_length != rollout_length:
                raise RuntimeError(f"Length mismatch: src={src_length}, dst={rollout_length}")

            status = self.engine.transfer_sync(
                session_id=rollout_transfer_session_id,
                buffer=src_ptr,
                peer_buffer_address=rollout_ptr,
                length=src_length,
            )

            if status != 0:
                raise RuntimeError(f"RDMA transfer failed with status {status}")

            logger.info(f"[TrainingSender] RDMA completed: task={task_id}, weight={weight_name}, size={src_length/1e6:.2f}MB")

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


def training_process(
    rank: int,
    world_size: int,
    config: Qwen3_32BConfig,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    hostname: str = "127.0.0.1",
    port: int = 50000,
):
    """Training process."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[Training] Initializing on GPU {rank}")

        sender = TrainingWeightSender(hostname=hostname, port=port, gpu_id=rank)

        logger.info(f"[Training] Creating Qwen3-32B model weights...")
        weights = create_qwen3_full_model(config, device, dtype=torch.float16)

        total_params = sum(w.numel() for w in weights.values())
        total_size_mb = sum(w.numel() * w.element_size() for w in weights.values()) / 1e6
        logger.info(f"[Training] Model created: {total_params/1e9:.2f}B params, {total_size_mb:.2f}MB")

        sender.register_weights(weights)
        sender.start()

        logger.info("[Training] Waiting for rollout processes...")
        barrier.wait()

        for update_idx in range(num_updates):
            update_start = time.time()

            for weight in weights.values():
                weight += torch.randn_like(weight) * 0.01
            torch.cuda.synchronize()

            update_end = time.time()
            logger.info(
                f"[Training] Update {update_idx + 1}/{num_updates} completed in "
                f"{(update_end - update_start)*1000:.2f}ms"
            )

            sender.register_weights(weights)
            barrier.wait()

        logger.info("[Training] All updates completed")
        sender.stop()

    except Exception as e:
        logger.error(f"[Training] Error: {e}", exc_info=True)
        result_queue.put(("training_error", str(e)))


def rollout_process(
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

        logger.info(f"[Rollout-{rank}] Initializing with Manager on GPU {rank}")

        transfer_impl = P2PTransferManager(
            hostname="127.0.0.1",
            gpu_id=rank,
            ib_device=None,
        )
        logger.info(f"[Rollout-{rank}] Using Manager with {transfer_impl.engine_pool_size} engines")

        weights = create_qwen3_full_model(config, device, dtype=torch.float16)

        total_params = sum(w.numel() for w in weights.values())
        total_size_mb = sum(w.numel() * w.element_size() for w in weights.values()) / 1e6
        logger.info(f"[Rollout-{rank}] Allocated buffers: {total_params/1e9:.2f}B params, {total_size_mb:.2f}MB")

        barrier.wait()

        all_transfer_times = []

        for update_idx in range(num_updates):
            update_start = time.time()

            session_id = f"{training_hostname}:{training_port}"
            handles = []

            submission_start = time.time()
            for weight_name, weight_tensor in weights.items():
                ptr = weight_tensor.data_ptr()
                length = weight_tensor.numel() * weight_tensor.element_size()

                handle = transfer_impl.submit_transfer_task(
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

            barrier.wait()

        avg_total_time = np.mean([t['total_time'] for t in all_transfer_times])
        avg_wait_time = np.mean([t['wait_time'] for t in all_transfer_times])
        total_bytes = all_transfer_times[0]['total_bytes']
        avg_bandwidth = (total_bytes * 8) / (avg_total_time * 1e9)

        stats = {
            'rank': rank,
            'impl_type': 'Manager',
            'num_updates': num_updates,
            'avg_total_time': avg_total_time,
            'avg_wait_time': avg_wait_time,
            'avg_bandwidth_gbps': avg_bandwidth,
            'total_bytes': total_bytes,
            'engine_pool_size': transfer_impl.engine_pool_size,
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


def worker_process(
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
        training_process(
            rank, world_size, config, num_updates,
            result_queue, barrier, "127.0.0.1", training_port
        )
    else:
        rollout_process(
            rank, world_size, config, num_updates,
            result_queue, barrier, "127.0.0.1", training_port
        )


def multi_training_worker_process(
    rank: int,
    world_size: int,
    config: Qwen3_32BConfig,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    num_training: int = 2,
    base_training_port: int = 50300,
):
    """
    Entry point for multi-training to single-rollout scenario.

    Args:
        rank: Process rank
        world_size: Total number of processes
        num_training: Number of training processes
        base_training_port: Base port for training processes (each uses base + rank)
    """
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank < num_training:
        # This is a training process
        training_port = base_training_port + rank
        training_process(
            rank, world_size, config, num_updates,
            result_queue, barrier, "127.0.0.1", training_port
        )
    else:
        # This is the rollout process
        multi_source_rollout_process(
            rank, world_size, config, num_updates,
            result_queue, barrier, num_training, base_training_port
        )


def multi_source_rollout_process(
    rank: int,
    world_size: int,
    config: Qwen3_32BConfig,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    num_training: int = 2,
    base_training_port: int = 50300,
):
    """
    Rollout process that receives weights from multiple training processes.

    This tests the Manager's ability to handle concurrent transfers from multiple sources.
    """
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[Rollout-{rank}] Initializing with Manager on GPU {rank}")
        logger.info(f"[Rollout-{rank}] Will receive from {num_training} training processes")

        transfer_impl = P2PTransferManager(
            hostname="127.0.0.1",
            gpu_id=rank,
            ib_device=None,
        )
        logger.info(f"[Rollout-{rank}] Using Manager with {transfer_impl.engine_pool_size} engines")

        # Create separate weight buffers for each training process
        weights_per_training = {}
        for training_idx in range(num_training):
            weights_per_training[training_idx] = create_qwen3_full_model(config, device, dtype=torch.float16)

        total_params = sum(w.numel() for w in weights_per_training[0].values()) * num_training
        total_size_mb = sum(w.numel() * w.element_size() for w in weights_per_training[0].values()) * num_training / 1e6
        logger.info(f"[Rollout-{rank}] Allocated buffers for {num_training} sources: {total_params/1e9:.2f}B params, {total_size_mb:.2f}MB")

        barrier.wait()

        all_transfer_times = []

        for update_idx in range(num_updates):
            update_start = time.time()

            # Submit tasks for all training sources concurrently
            all_handles = []
            submission_start = time.time()

            for training_idx in range(num_training):
                session_id = f"127.0.0.1:{base_training_port + training_idx}"
                weights = weights_per_training[training_idx]

                for weight_name, weight_tensor in weights.items():
                    ptr = weight_tensor.data_ptr()
                    length = weight_tensor.numel() * weight_tensor.element_size()

                    handle = transfer_impl.submit_transfer_task(
                        session_id=session_id,
                        ptr=ptr,
                        length=length,
                    )
                    all_handles.append((training_idx, weight_name, handle))

            submission_end = time.time()

            logger.info(
                f"[Rollout-{rank}] Submitted {len(all_handles)} transfer tasks from {num_training} sources in "
                f"{(submission_end - submission_start)*1000:.2f}ms"
            )

            # Wait for all transfers to complete
            wait_start = time.time()
            for training_idx, weight_name, handle in all_handles:
                handle.wait()

            torch.cuda.synchronize()
            wait_end = time.time()

            update_end = time.time()

            total_time = update_end - update_start
            wait_time = wait_end - wait_start
            total_bytes = sum(w.numel() * w.element_size() for w in weights_per_training[0].values()) * num_training

            all_transfer_times.append({
                'update_idx': update_idx,
                'total_time': total_time,
                'submission_time': submission_end - submission_start,
                'wait_time': wait_time,
                'num_tasks': len(all_handles),
                'num_sources': num_training,
                'total_bytes': total_bytes,
            })

            logger.info(
                f"[Rollout-{rank}] Update {update_idx + 1}/{num_updates} completed: "
                f"total={total_time*1000:.2f}ms, submission={(submission_end - submission_start)*1000:.2f}ms, "
                f"wait={wait_time*1000:.2f}ms, bandwidth={(total_bytes * 8) / (total_time * 1e9):.2f}Gbps, "
                f"tasks={len(all_handles)}, sources={num_training}"
            )

            barrier.wait()

        avg_total_time = np.mean([t['total_time'] for t in all_transfer_times])
        avg_wait_time = np.mean([t['wait_time'] for t in all_transfer_times])
        total_bytes = all_transfer_times[0]['total_bytes']
        avg_bandwidth = (total_bytes * 8) / (avg_total_time * 1e9)

        stats = {
            'rank': rank,
            'impl_type': 'Manager',
            'num_updates': num_updates,
            'num_sources': num_training,
            'avg_total_time': avg_total_time,
            'avg_wait_time': avg_wait_time,
            'avg_bandwidth_gbps': avg_bandwidth,
            'total_bytes': total_bytes,
            'engine_pool_size': transfer_impl.engine_pool_size,
            'all_transfers': all_transfer_times,
        }

        result_queue.put((f"rollout_{rank}_stats", stats))

        logger.info(
            f"[Rollout-{rank}] All updates completed. Avg time: {avg_total_time*1000:.2f}ms, "
            f"Avg bandwidth: {avg_bandwidth:.2f}Gbps, Sources: {num_training}"
        )

    except Exception as e:
        logger.error(f"[Rollout-{rank}] Error: {e}", exc_info=True)
        result_queue.put((f"rollout_{rank}_error", str(e)))


class TestP2PTransferPerformance(unittest.TestCase):
    """Performance tests for P2PTransferManager."""

    def test_manager_1v1_qwen3(self):
        """Test P2PTransferManager: 1 training + 1 rollout with Qwen3-32B."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("Requires at least 2 CUDA devices")

        world_size = 2
        config = Qwen3_32BConfig()
        num_updates = 3
        training_port = 50201

        logger.info(
            f"\n{'='*80}\n"
            f"Testing P2PTransferManager (1v1)\n"
            f"Config: {world_size} processes, {num_updates} updates\n"
            f"{'='*80}\n"
        )

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        context = mp.spawn(
            worker_process,
            args=(world_size, config, num_updates, result_queue, barrier, training_port),
            nprocs=world_size,
            join=False,
        )

        results = {}
        timeout = 120
        start_time = time.time()

        while len(results) < (world_size - 1):
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

        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        if "rollout_1_stats" in results:
            stats = results["rollout_1_stats"]
            logger.info(
                f"\n{'='*80}\n"
                f"P2PTransferManager (1v1) Results:\n"
                f"  Impl: {stats['impl_type']}\n"
                f"  Engine pool size: {stats['engine_pool_size']}\n"
                f"  Avg total time: {stats['avg_total_time']*1000:.2f}ms\n"
                f"  Avg wait time: {stats['avg_wait_time']*1000:.2f}ms\n"
                f"  Avg bandwidth: {stats['avg_bandwidth_gbps']:.2f}Gbps\n"
                f"  Total data: {stats['total_bytes']/1e6:.2f}MB\n"
                f"{'='*80}\n"
            )

        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()

    def test_manager_1v2_qwen3(self):
        """Test P2PTransferManager: 1 training + 2 rollouts with Qwen3-32B."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            self.skipTest("Requires at least 3 CUDA devices")

        world_size = 3
        config = Qwen3_32BConfig()
        num_updates = 2
        training_port = 50203

        logger.info(
            f"\n{'='*80}\n"
            f"Testing P2PTransferManager (1v2)\n"
            f"Config: {world_size} processes, {num_updates} updates\n"
            f"{'='*80}\n"
        )

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        context = mp.spawn(
            worker_process,
            args=(world_size, config, num_updates, result_queue, barrier, training_port),
            nprocs=world_size,
            join=False,
        )

        results = {}
        timeout = 120
        start_time = time.time()

        while len(results) < (world_size - 1):
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

        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        logger.info(f"\n{'='*80}\nP2PTransferManager (1v2) Results:\n")
        for rank in range(1, world_size):
            if f"rollout_{rank}_stats" in results:
                stats = results[f"rollout_{rank}_stats"]
                logger.info(
                    f"Rollout-{rank} ({stats['impl_type']}): "
                    f"avg_time={stats['avg_total_time']*1000:.2f}ms, "
                    f"bandwidth={stats['avg_bandwidth_gbps']:.2f}Gbps, "
                    f"pool_size={stats['engine_pool_size']}"
                )
        logger.info(f"{'='*80}\n")

        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()

    def test_manager_2v1_qwen3(self):
        """Test P2PTransferManager: 2 trainings + 1 rollout with Qwen3-32B (multi-to-one)."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            self.skipTest("Requires at least 3 CUDA devices")

        world_size = 3
        num_training = 2
        config = Qwen3_32BConfig()
        num_updates = 2
        base_training_port = 50300

        logger.info(
            f"\n{'='*80}\n"
            f"Testing P2PTransferManager (2v1 - multi-to-one)\n"
            f"Config: {world_size} processes ({num_training} training + 1 rollout), {num_updates} updates\n"
            f"{'='*80}\n"
        )

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        context = mp.spawn(
            multi_training_worker_process,
            args=(world_size, config, num_updates, result_queue, barrier, num_training, base_training_port),
            nprocs=world_size,
            join=False,
        )

        results = {}
        timeout = 120
        start_time = time.time()

        # Only waiting for rollout process (rank = num_training)
        while len(results) < 1:
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

        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        if f"rollout_{num_training}_stats" in results:
            stats = results[f"rollout_{num_training}_stats"]
            logger.info(
                f"\n{'='*80}\n"
                f"P2PTransferManager (2v1 - multi-to-one) Results:\n"
                f"  Impl: {stats['impl_type']}\n"
                f"  Engine pool size: {stats['engine_pool_size']}\n"
                f"  Number of sources: {stats['num_sources']}\n"
                f"  Avg total time: {stats['avg_total_time']*1000:.2f}ms\n"
                f"  Avg wait time: {stats['avg_wait_time']*1000:.2f}ms\n"
                f"  Avg bandwidth: {stats['avg_bandwidth_gbps']:.2f}Gbps\n"
                f"  Total data: {stats['total_bytes']/1e6:.2f}MB\n"
                f"{'='*80}\n"
            )

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
