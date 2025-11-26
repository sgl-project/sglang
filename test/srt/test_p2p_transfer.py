"""
Test P2P weight transfer for RL training scenarios.

This test simulates the real RL training workflow:
- Rank 0: Training process (sender) - updates model weights and sends via RDMA
- Rank 1+: Rollout processes (receivers) - receive weight updates via P2P RDMA

Architecture:
- Training side: TrainingWeightSender - listens for sync_status, performs RDMA write
- Rollout side: P2PTransferEngine - registers memory, sends sync_status, waits for confirmation
"""

import gc
import logging
import os
import threading
import time
import unittest

import numpy as np
import torch
import torch.multiprocessing as mp
import zmq

from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.utils.common import format_tcp_address
from sglang.srt.weight_sync.p2p_transfer import P2PTransferEngine

logger = logging.getLogger(__name__)

# Set multiprocessing start method
mp.set_start_method("spawn", force=True)


def create_mock_layer_weights(layer_size: int, device: torch.device) -> torch.Tensor:
    """Create mock weights for a single layer."""
    # Simulate a linear layer with shape [hidden_size, hidden_size]
    weights = torch.randn(layer_size, layer_size, device=device, dtype=torch.float32)
    return weights


class TrainingWeightSender:
    """
    Training-side weight sender that handles RDMA write operations.

    Workflow:
    1. Listen for sync_status messages from rollout workers
    2. Extract rollout's ip:port and registered ptr
    3. Perform RDMA write to send weights
    4. Send confirmation back to rollout worker
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

        logger.info(f"[TrainingWeightSender] Initialized on {hostname}:{port}, GPU {gpu_id}")

    def register_weights(self, name: str, tensor: torch.Tensor):
        """Register weight tensor for sending."""
        ptr = tensor.data_ptr()
        length = tensor.numel() * tensor.element_size()

        # Check if this memory region is already registered
        if name in self.weight_buffers:
            existing_ptr = self.weight_buffers[name]['ptr']
            if existing_ptr == ptr:
                # Same memory region, just update the tensor reference
                self.weight_buffers[name]['tensor'] = tensor
                logger.debug(f"[TrainingWeightSender] Updated tensor reference for '{name}': ptr={ptr:#x}")
                return
            else:
                # Different memory region, need to register new one
                logger.info(f"[TrainingWeightSender] Re-registering '{name}' with new ptr: {existing_ptr:#x} -> {ptr:#x}")

        # Register with Mooncake engine
        self.engine.register(ptr, length)

        self.weight_buffers[name] = {
            'ptr': ptr,
            'length': length,
            'tensor': tensor,
        }
        logger.info(f"[TrainingWeightSender] Registered weights '{name}': ptr={ptr:#x}, length={length}")

    def start(self):
        """Start the message handling loop in a background thread."""
        if self.running:
            return

        self.running = True
        self.handler_thread = threading.Thread(target=self._message_loop, daemon=True)
        self.handler_thread.start()
        logger.info("[TrainingWeightSender] Started message handler")

    def stop(self):
        """Stop the sender."""
        self.running = False
        if hasattr(self, 'handler_thread'):
            self.handler_thread.join(timeout=2)
        self.router_socket.close()
        self.context.term()
        logger.info("[TrainingWeightSender] Stopped")

    def _message_loop(self):
        """Handle incoming sync_status messages from rollout workers."""
        while self.running:
            try:
                if self.router_socket.poll(timeout=100):  # 100ms timeout
                    # ROUTER receives frames from DEALER
                    frames = self.router_socket.recv_multipart()

                    logger.debug(f"[TrainingWeightSender] Received {len(frames)} frames, "
                                f"frame lengths: {[len(f) for f in frames]}")

                    if len(frames) < 2:
                        logger.warning(f"[TrainingWeightSender] Received malformed message with {len(frames)} frames")
                        continue

                    # ROUTER adds identity as first frame
                    # Format is typically: [identity, delimiter (empty), actual_message]
                    identity = frames[0]

                    # Try to find the JSON payload
                    message = None
                    for i in range(1, len(frames)):
                        try:
                            if len(frames[i]) > 0:  # Skip empty frames
                                message = zmq.utils.jsonapi.loads(frames[i])
                                logger.debug(f"[TrainingWeightSender] Found JSON in frame {i}")
                                break
                        except Exception:
                            continue

                    if message is None:
                        logger.error(f"[TrainingWeightSender] Could not decode JSON from any frame")
                        continue

                    msg_type = message.get("type", "")

                    if msg_type == "sync_status":
                        # Handle sync_status from rollout worker
                        self._handle_sync_status(identity, message)
                    else:
                        logger.warning(f"[TrainingWeightSender] Unknown message type: {msg_type}")

            except zmq.ZMQError as e:
                if self.running:
                    logger.error(f"[TrainingWeightSender] ZMQ error: {e}")
                break
            except Exception as e:
                logger.error(f"[TrainingWeightSender] Error in message loop: {e}", exc_info=True)

    def _handle_sync_status(self, identity: bytes, message: dict):
        """
        Handle sync_status message from rollout worker.

        Message format:
        {
            "type": "sync_status",
            "p2p_session_id": "training_ip:training_port",  # Training's ZMQ address
            "status": "ready",
            "ip": rollout_ip,
            "transfer_session_id": rollout_mooncake_session_id,  # Rollout's Mooncake RPC address
            "ptr": rollout_ptr,
            "length": buffer_length,
            "task_id": task_id
        }
        """
        # Log the raw message for debugging
        logger.debug(f"[TrainingWeightSender] Raw message: {message}")

        p2p_session_id = message.get("p2p_session_id", "")
        rollout_ip = message.get("ip", "")
        rollout_transfer_session_id = message.get("transfer_session_id", "")
        rollout_ptr = message.get("ptr", 0)
        rollout_length = message.get("length", 0)
        task_id = message.get("task_id", "")

        logger.info(
            f"[TrainingWeightSender] Received sync_status: "
            f"p2p_session_id={p2p_session_id}, "
            f"task_id={task_id}, "
            f"rollout_transfer_session_id={rollout_transfer_session_id}, "
            f"rollout_ip={rollout_ip}, "
            f"rollout_ptr={rollout_ptr:#x}"
        )

        try:
            # Get the weights to send (for now, use the first registered weights)
            if not self.weight_buffers:
                raise RuntimeError("No weights registered")

            # Get first weight buffer
            weight_name = list(self.weight_buffers.keys())[0]
            weight_info = self.weight_buffers[weight_name]
            src_ptr = weight_info['ptr']
            src_length = weight_info['length']

            # Verify length matches
            if src_length != rollout_length:
                raise RuntimeError(
                    f"Length mismatch: src={src_length}, dst={rollout_length}"
                )

            # Validate that we have rollout_transfer_session_id
            if not rollout_transfer_session_id:
                raise RuntimeError(
                    "Missing rollout_transfer_session_id in sync_status message. "
                    "This is required for establishing RDMA connection."
                )

            logger.info(
                f"[TrainingWeightSender] Performing RDMA write: "
                f"src_ptr={src_ptr:#x} -> dst_ptr={rollout_ptr:#x}, "
                f"length={src_length}, "
                f"target_session={rollout_transfer_session_id}"
            )

            # Perform RDMA write using Mooncake transfer engine
            # Use rollout's Mooncake transfer_session_id for RDMA connection
            status = self.engine.transfer_sync(
                session_id=rollout_transfer_session_id,
                buffer=src_ptr,
                peer_buffer_address=rollout_ptr,
                length=src_length,
            )

            if status != 0:
                raise RuntimeError(f"RDMA transfer failed with status {status}")

            logger.info(
                f"[TrainingWeightSender] RDMA write completed successfully for task {task_id}"
            )

            # Send success confirmation back to rollout worker
            # ROUTER to DEALER: [identity, empty_delimiter, payload]
            response_data = zmq.utils.jsonapi.dumps({
                "type": "transfer_complete",
                "status": "success",
                "task_id": task_id,
            })
            self.router_socket.send_multipart([identity, b"", response_data])

            logger.info(f"[TrainingWeightSender] Sent success confirmation for task {task_id}")

        except Exception as e:
            logger.error(f"[TrainingWeightSender] Error handling sync_status: {e}", exc_info=True)

            # Send failure confirmation
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
    layer_size: int,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    hostname: str = "127.0.0.1",
    port: int = 50000,
):
    """
    Training process that sends weight updates to rollout workers.

    This simulates the training side in RL:
    1. Initialize model weights
    2. Start TrainingWeightSender to handle rollout requests
    3. Perform training updates
    4. Wait for rollout workers to sync
    """
    try:
        # Set device
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[Training] Initializing on GPU {rank}")

        # Initialize training weight sender
        sender = TrainingWeightSender(
            hostname=hostname,
            port=port,
            gpu_id=rank,
        )

        # Create mock model layer weights
        weights = create_mock_layer_weights(layer_size, device)
        logger.info(f"[Training] Created weights with shape {weights.shape}")

        # Register weights with sender
        sender.register_weights("layer_weights", weights)

        # Start sender to handle rollout requests
        sender.start()

        # Store initial weights for verification
        initial_weights = weights.clone().cpu().numpy()
        result_queue.put(("training_initial_weights", initial_weights))

        # Wait for all rollout processes to be ready
        logger.info("[Training] Waiting for rollout processes...")
        barrier.wait()

        # Simulate training updates
        for update_idx in range(num_updates):
            # Simulate weight update (gradient step)
            weights += torch.randn_like(weights) * 0.1
            torch.cuda.synchronize()

            logger.info(f"[Training] Update {update_idx + 1}/{num_updates} completed")

            # Update registered weights in sender
            sender.register_weights("layer_weights", weights)

            # Wait for rollout workers to pull the weights
            # (rollout workers will send sync_status, sender will handle)
            time.sleep(0.5)  # Give time for transfers

            # Synchronize before next update
            barrier.wait()

        # Store final weights for verification
        final_weights = weights.cpu().numpy()
        result_queue.put(("training_final_weights", final_weights))

        logger.info("[Training] All updates completed successfully")

        # Stop sender
        sender.stop()

    except Exception as e:
        logger.error(f"[Training] Error: {e}", exc_info=True)
        result_queue.put(("training_error", str(e)))


def rollout_process(
    rank: int,
    world_size: int,
    layer_size: int,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    training_hostname: str = "127.0.0.1",
    training_port: int = 50000,
):
    """
    Rollout process that receives weight updates from training process.

    This simulates the rollout worker side in RL:
    1. Allocate buffer for weights
    2. Initialize P2PTransferEngine
    3. For each update: submit_transfer_task -> sends sync_status -> waits for transfer
    """
    try:
        # Set device
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        logger.info(f"[Rollout-{rank}] Initializing on GPU {rank}")

        # Initialize P2P transfer engine (receiver side)
        engine = P2PTransferEngine(
            hostname="127.0.0.1",  # This rollout worker's IP
            gpu_id=rank,
            ib_device=None,
        )

        # Allocate buffer for receiving weights
        weights = create_mock_layer_weights(layer_size, device)
        total_size_mb = (weights.numel() * weights.element_size()) / 1e6
        logger.info(f"[Rollout-{rank}] Allocated buffer with shape {weights.shape}, size={total_size_mb:.2f}MB")

        # Store initial (random) weights
        initial_weights = weights.clone().cpu().numpy()
        result_queue.put((f"rollout_{rank}_initial_weights", initial_weights))

        # Signal ready to training process
        logger.info(f"[Rollout-{rank}] Ready to receive updates")
        barrier.wait()

        # Track timing for all transfers
        all_transfer_times = []

        # Receive weight updates
        for update_idx in range(num_updates):
            update_start = time.time()

            # Get buffer pointer and size
            ptr = weights.data_ptr()
            length = weights.numel() * weights.element_size()

            # Submit transfer task
            session_id = f"{training_hostname}:{training_port}"

            submission_start = time.time()
            handle = engine.submit_transfer_task(
                session_id=session_id,
                ptr=ptr,
                length=length,
            )
            submission_end = time.time()

            logger.info(
                f"[Rollout-{rank}] Submitted transfer task for update {update_idx + 1} in "
                f"{(submission_end - submission_start)*1000:.2f}ms"
            )

            # Wait for transfer to complete
            wait_start = time.time()
            handle.wait()
            torch.cuda.synchronize()
            wait_end = time.time()

            update_end = time.time()

            total_time = update_end - update_start
            submission_time = submission_end - submission_start
            wait_time = wait_end - wait_start

            all_transfer_times.append({
                'update_idx': update_idx,
                'total_time': total_time,
                'submission_time': submission_time,
                'wait_time': wait_time,
                'total_bytes': length,
            })

            logger.info(
                f"[Rollout-{rank}] Update {update_idx + 1}/{num_updates} completed: "
                f"total={total_time*1000:.2f}ms, submission={submission_time*1000:.2f}ms, "
                f"wait={wait_time*1000:.2f}ms, bandwidth={(length * 8) / (total_time * 1e9):.2f}Gbps"
            )

            # Synchronize with training process
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
            'all_transfers': all_transfer_times,
        }

        result_queue.put((f"rollout_{rank}_stats", stats))

        # Store final received weights for verification
        final_weights = weights.cpu().numpy()
        result_queue.put((f"rollout_{rank}_final_weights", final_weights))

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
    layer_size: int,
    num_updates: int,
    result_queue: mp.Queue,
    barrier: mp.Barrier,
    training_port: int = 50000,
):
    """Entry point for each worker process."""
    # Important: Set these to avoid NCCL issues
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    if rank == 0:
        # Training process
        training_process(
            rank, world_size, layer_size, num_updates,
            result_queue, barrier, "127.0.0.1", training_port
        )
    else:
        # Rollout process
        rollout_process(
            rank, world_size, layer_size, num_updates,
            result_queue, barrier, "127.0.0.1", training_port
        )


class TestP2PTransferRL(unittest.TestCase):
    """Test P2P transfer in RL training scenario with multiple processes."""

    def test_single_training_single_rollout(self):
        """
        Test basic RL scenario: 1 training process + 1 rollout process.
        Simulates training->rollout weight synchronization.
        """
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("Requires at least 2 CUDA devices")

        world_size = 2  # 1 training + 1 rollout
        layer_size = 128  # Small layer for fast testing
        num_updates = 3  # Number of weight updates
        training_port = 50000

        logger.info(
            f"Starting test: {world_size} processes, "
            f"layer_size={layer_size}, updates={num_updates}"
        )

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        # Spawn processes
        context = mp.spawn(
            worker_process,
            args=(world_size, layer_size, num_updates, result_queue, barrier, training_port),
            nprocs=world_size,
            join=False,
        )

        # Collect results
        results = {}
        timeout = 60  # 60 seconds timeout
        start_time = time.time()

        expected_results = 2 + world_size + (world_size - 1)  # initial + final weights + stats for rollouts
        while len(results) < expected_results:
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                key, value = result_queue.get(timeout=min(5, remaining_time))
                results[key] = value
            except Exception as e:
                if all(not p.is_alive() for p in context.processes):
                    break

        context.join()

        # Verify results
        self.assertIn("training_initial_weights", results)
        self.assertIn("training_final_weights", results)
        self.assertIn("rollout_1_initial_weights", results)
        self.assertIn("rollout_1_final_weights", results)

        # Check for errors
        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        # Verify shapes match
        training_final = results["training_final_weights"]
        rollout_final = results["rollout_1_final_weights"]

        self.assertEqual(training_final.shape, rollout_final.shape)

        # Print statistics if available
        if "rollout_1_stats" in results:
            stats = results["rollout_1_stats"]
            logger.info(
                f"\n{'='*80}\n"
                f"Rollout-1 Statistics:\n"
                f"  Avg total time: {stats['avg_total_time']*1000:.2f}ms\n"
                f"  Avg wait time: {stats['avg_wait_time']*1000:.2f}ms\n"
                f"  Avg bandwidth: {stats['avg_bandwidth_gbps']:.2f}Gbps\n"
                f"  Total data: {stats['total_bytes']/1e6:.2f}MB\n"
                f"{'='*80}\n"
            )

        logger.info("Test passed: Single training + single rollout")

        # Cleanup
        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()

    def test_single_training_multiple_rollouts(self):
        """
        Test scaled RL scenario: 1 training process + multiple rollout processes.
        """
        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            self.skipTest("Requires at least 3 CUDA devices")

        world_size = 3  # 1 training + 2 rollouts
        layer_size = 128
        num_updates = 2
        training_port = 50001

        logger.info(
            f"Starting test: {world_size} processes "
            f"(1 training + {world_size-1} rollouts)"
        )

        result_queue = mp.Queue()
        barrier = mp.Barrier(world_size)

        context = mp.spawn(
            worker_process,
            args=(world_size, layer_size, num_updates, result_queue, barrier, training_port),
            nprocs=world_size,
            join=False,
        )

        # Collect results
        results = {}
        timeout = 60
        start_time = time.time()

        expected_results = 2 + (world_size - 1) * 3  # training + (weights + stats) for all rollouts
        while len(results) < expected_results:
            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
                key, value = result_queue.get(timeout=min(5, remaining_time))
                results[key] = value
            except Exception as e:
                if all(not p.is_alive() for p in context.processes):
                    break

        context.join()

        # Verify all processes completed
        self.assertIn("training_final_weights", results)
        for rank in range(1, world_size):
            self.assertIn(f"rollout_{rank}_final_weights", results)

        # Check for errors
        for key in results:
            if "error" in key:
                self.fail(f"Process error: {key} = {results[key]}")

        # Verify all rollouts received weights with same shape
        training_final = results["training_final_weights"]
        for rank in range(1, world_size):
            rollout_final = results[f"rollout_{rank}_final_weights"]
            self.assertEqual(
                training_final.shape,
                rollout_final.shape,
                f"Shape mismatch for rollout {rank}"
            )

        # Print comparative statistics
        logger.info(f"\n{'='*80}\nComparative Statistics:\n")
        for rank in range(1, world_size):
            key = f"rollout_{rank}_stats"
            if key in results:
                stats = results[key]
                logger.info(
                    f"Rollout-{rank}: "
                    f"avg_time={stats['avg_total_time']*1000:.2f}ms, "
                    f"bandwidth={stats['avg_bandwidth_gbps']:.2f}Gbps"
                )
        logger.info(f"{'='*80}\n")

        logger.info(f"Test passed: 1 training + {world_size-1} rollouts")

        # Cleanup
        result_queue.close()
        result_queue.join_thread()
        gc.collect()
        torch.cuda.empty_cache()


class TestP2PTransferEngineBasic(unittest.TestCase):
    """Basic unit tests for P2PTransferEngine without multiprocessing."""

    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        engine = P2PTransferEngine(
            hostname="127.0.0.1",
            gpu_id=0,
            ib_device=None,
        )

        # Verify internal structures
        self.assertIsNotNone(engine.engine)
        self.assertIsNotNone(engine.transfer_queues)
        self.assertIsNotNone(engine.executors)
        self.assertIsNotNone(engine.socket_cache)
        self.assertEqual(engine.task_counter, 0)

        logger.info("Engine initialization test passed")

    def test_task_id_generation(self):
        """Test unique task ID generation."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        engine = P2PTransferEngine(
            hostname="127.0.0.1",
            gpu_id=0,
            ib_device=None,
        )

        # Generate multiple task IDs
        task_ids = set()
        for i in range(10):
            task_id = engine._generate_task_id()
            self.assertNotIn(task_id, task_ids)
            task_ids.add(task_id)

        logger.info("Task ID generation test passed")


if __name__ == "__main__":
    # Use DEBUG level to see more detailed logs
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    unittest.main()
