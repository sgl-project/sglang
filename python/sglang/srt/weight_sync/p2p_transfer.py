import concurrent.futures
import logging
import os
import threading
import zmq

from collections import defaultdict
from queue import Queue
from typing import Dict, Optional

from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.utils import get_int_env_var
from sglang.srt.utils.common import format_tcp_address

logger = logging.getLogger(__name__)


class TransferHandle:
    """A handle that mimics torch.distributed handle interface for compatibility"""

    def __init__(self, task_id: str, socket_cache: Dict, socket_lock: threading.Lock):
        self.completed = False
        self.success = False
        self._event = threading.Event()
        self.task_id = task_id
        self.socket_cache = socket_cache
        self.socket_lock = socket_lock

    def wait(self):
        """Wait for the transfer to complete"""
        self._event.wait()

        if not self.success:
            raise RuntimeError("P2P weight transfer failed")

    def _mark_done(self, success: bool):
        """Internal method to mark completion"""
        self.success = success
        self.completed = True
        self._event.set()


class TransferTask:
    """Represents a weight transfer task to be processed by workers"""

    def __init__(
        self,
        task_id: str,
        training_p2p_session_id: str,
        rollout_transfer_session_id: str,
        ptr: int,
        length: int,
        socket_cache: Dict,
        socket_lock: threading.Lock,
    ):
        self.task_id = task_id
        self.training_p2p_session_id = training_p2p_session_id  # ZMQ communication endpoint (training ip:port)
        self.rollout_transfer_session_id = rollout_transfer_session_id  # Local Mooncake session_id for RDMA
        self.ptr = ptr
        self.length = length
        self.handle = TransferHandle(task_id, socket_cache, socket_lock)

class P2PTransferEngine:

    def __init__(self, hostname: str, gpu_id: int, ib_device: Optional[str] = None):
        # Initialize the underlying Mooncake transfer engine
        self._init_transfer_engine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=ib_device,
        )

        # Initialize thread pool and transfer queues
        cpu_count = os.cpu_count() or 8
        transfer_thread_pool_size = get_int_env_var(
            "SGLANG_P2P_THREAD_POOL_SIZE",
            min(max(4, int(0.5 * cpu_count) // 8), 12),
        )
        transfer_queue_size = get_int_env_var("SGLANG_P2P_QUEUE_SIZE", 4)

        self.transfer_queues = [Queue() for _ in range(transfer_queue_size)]

        assert transfer_thread_pool_size >= transfer_queue_size, (
            f"SGLANG_P2P_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
            f"greater than or equal to SGLANG_P2P_QUEUE_SIZE={transfer_queue_size}."
        )

        self.executors = [
            concurrent.futures.ThreadPoolExecutor(
                transfer_thread_pool_size // transfer_queue_size
            )
            for _ in range(transfer_queue_size)
        ]

        # Track registered memory regions per session
        self.registered_ptrs: Dict[str, set] = defaultdict(set)
        self.registration_lock = threading.Lock()

        # Socket cache to avoid race conditions (each task gets its own socket)
        self.socket_cache: Dict[str, Dict] = {}
        self.socket_lock = threading.Lock()

        # Task counter for generating unique task IDs
        self.task_counter = 0
        self.task_counter_lock = threading.Lock()

        # Start worker threads for each queue
        for queue, executor in zip(self.transfer_queues, self.executors):
            threading.Thread(
                target=self.transfer_worker, args=(queue, executor), daemon=True
            ).start()

        logger.info(
            f"P2P transfer engine initialized with {transfer_thread_pool_size} workers "
            f"and {transfer_queue_size} queues"
        )

    def _generate_task_id(self) -> str:
        """Generate a unique task ID"""
        with self.task_counter_lock:
            task_id = f"task_{self.task_counter}"
            self.task_counter += 1
        return task_id

    def submit_transfer_task(
        self, session_id: str, ptr: int, length: int
    ) -> TransferHandle:
        """
        Submit a transfer task to the queue for processing.
        Returns a handle that can be used to wait for completion.

        The task will be picked up by an idle worker thread which will:
        1. Register the memory region
        2. Send sync_status to training side with ip:port and registered ptr

        Args:
            session_id: Remote training process address (ip:port) for ZMQ communication
            ptr: Local memory pointer to register
            length: Buffer length
        """
        task_id = self._generate_task_id()
        try:
            remote_ip, remote_port = session_id.split(":")[0], int(session_id.split(":")[1])
        except Exception as e:
            raise ValueError(f"Invalid session_id format: {session_id}") from e

        # Get local Mooncake transfer session_id
        local_transfer_session_id = self.engine.get_session_id()

        task = TransferTask(
            task_id=task_id,
            training_p2p_session_id=session_id,
            rollout_transfer_session_id=local_transfer_session_id,
            ptr=ptr,
            length=length,
            socket_cache=self.socket_cache,
            socket_lock=self.socket_lock
        )

        # Shard task to queue based on session_id
        queue_idx = hash(session_id) % len(self.transfer_queues)
        self.transfer_queues[queue_idx].put(task)

        logger.debug(
            f"Submitted transfer task {task_id} for session {session_id}, ptr={ptr:#x}, "
            f"length={length} to queue {queue_idx}"
        )

        return task.handle

    def transfer_worker(self, queue: Queue, executor: concurrent.futures.ThreadPoolExecutor):
        """
        Worker thread that processes transfer tasks from the queue.
        """
        while True:
            try:
                task: TransferTask = queue.get()

                # Process the task in the executor
                future = executor.submit(self._process_transfer_task, task)

                # Wait for completion and mark the handle
                try:
                    future.result()
                    task.handle._mark_done(True)
                except Exception as e:
                    logger.error(
                        f"Transfer task failed for session {task.training_p2p_session_id} : {e}"
                    )
                    task.handle._mark_done(False)

            except Exception as e:
                logger.error(f"Transfer worker failed: {e}")
                # Don't crash the worker thread, continue processing

    def _process_transfer_task(self, task: TransferTask):
        """
        Process a single transfer task:
        1. Register memory region if not already registered
        2. Send sync_status to training side with connection info and ptr
        3. Wait for success confirmation from training side
        """
        training_p2p_session_id = task.training_p2p_session_id
        rollout_transfer_session_id = task.rollout_transfer_session_id
        ptr = task.ptr
        length = task.length

        # Register memory region if not already registered
        with self.registration_lock:
            if ptr not in self.registered_ptrs[training_p2p_session_id]:
                logger.debug(
                    f"Registering memory region for session {training_p2p_session_id}: "
                    f"ptr={ptr:#x}, length={length}"
                )
                self.engine.register(ptr, length)
                self.registered_ptrs[training_p2p_session_id].add(ptr)
            else:
                logger.debug(
                    f"Memory region already registered for session {training_p2p_session_id}: ptr={ptr:#x}"
                )

        # Send sync_status to training side and wait for confirmation
        self._send_sync_status_and_wait(
            task_id=task.task_id,
            training_p2p_session_id=training_p2p_session_id,
            rollout_transfer_session_id=rollout_transfer_session_id,
            ptr=ptr,
            length=length
        )

    def _send_sync_status_and_wait(
        self,
        task_id: str,
        training_p2p_session_id: str,
        rollout_transfer_session_id: str,
        ptr: int,
        length: int
    ):
        """
        Send sync_status message to training side and wait for confirmation.

        Workflow:
        1. Send sync_status with local Mooncake session_id and registered ptr to training side
        2. Wait for training side to send back success/failure confirmation
        3. Socket is cleaned up after successful transfer

        Args:
            task_id: Unique task identifier
            remote_ip: Training process IP
            remote_port: Training process ZMQ port
            remote_p2p_session_id: Training process address for P2P communication (ip:port)
            local_transfer_session_id: Local Mooncake transfer engine session_id
            ptr: Registered memory pointer
            length: Buffer length
        """
        # Create a temporary socket and cache it for this task
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        remote_ip, remote_port = training_p2p_session_id.split(":")[0], int(training_p2p_session_id.split(":")[1])
        socket.connect(format_tcp_address(remote_ip, remote_port))

        # Cache the socket for later cleanup
        with self.socket_lock:
            self.socket_cache[task_id] = {
                'socket': socket,
                'context': context,
                'remote_ip': remote_ip,
                'remote_port': remote_port,
            }

        try:
            # Send sync_status to training side
            # Include our Mooncake session_id so training can connect to us for RDMA
            socket.send_json(
                {
                    "type": "sync_status",
                    "p2p_session_id": training_p2p_session_id,
                    "status": "ready",
                    "ip": self.engine.hostname,
                    "transfer_session_id": rollout_transfer_session_id,
                    "ptr": ptr,
                    "length": length,
                    "task_id": task_id,
                }
            )
            logger.info(
                f"Sent sync_status to {remote_ip}:{remote_port} for session {training_p2p_session_id}, "
                f"task_id={task_id}, rollout_transfer_session_id={rollout_transfer_session_id}, ptr={ptr:#x}"
            )

            # Wait for confirmation from training side
            # Set a timeout to avoid hanging forever
            socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 seconds timeout

            logger.debug(f"Waiting for confirmation from {remote_ip}:{remote_port} for task {task_id}")

            # DEALER receives from ROUTER: the message may have multiple frames
            # ROUTER sends: [identity, empty_delimiter, payload]
            # DEALER receives: [empty_delimiter, payload] (identity is automatically stripped)
            try:
                frames = socket.recv_multipart()
                logger.debug(f"Received {len(frames)} frames: {[len(f) for f in frames]}")

                # Find the JSON payload (skip empty frames)
                response = None
                for i, frame in enumerate(frames):
                    if len(frame) > 0:
                        try:
                            response = zmq.utils.jsonapi.loads(frame)
                            logger.debug(f"Parsed JSON from frame {i}: {response}")
                            break
                        except Exception:
                            logger.debug(f"Frame {i} is not JSON: {repr(frame[:100])}")
                            continue

                if response is None:
                    raise RuntimeError(f"No valid JSON found in {len(frames)} frames")

            except Exception as json_error:
                logger.error(f"Failed to receive/parse response: {json_error}")
                raise json_error

            # Check response status
            response_type = response.get("type", "")
            response_status = response.get("status", "")

            if response_type == "transfer_complete" and response_status == "success":
                logger.info(
                    f"Received success confirmation from {remote_ip}:{remote_port} "
                    f"for task {task_id}, transfer session {rollout_transfer_session_id}"
                )
                # Clean up the socket after successful transfer
                with self.socket_lock:
                    if task_id in self.socket_cache:
                        self.socket_cache.pop(task_id)
                socket.close()
                context.term()
                logger.debug(f"Released socket for task {task_id}")
            else:
                error_msg = response.get("error", "Unknown error")
                logger.error(
                    f"Received failure confirmation from {remote_ip}:{remote_port} "
                    f"for task {task_id}: {error_msg}"
                )
                raise RuntimeError(
                    f"Transfer failed for task {task_id}: {error_msg}"
                )

        except zmq.Again:
            # Timeout waiting for response
            logger.error(
                f"Timeout waiting for confirmation from {remote_ip}:{remote_port} "
                f"for task {task_id}"
            )
            # Clean up socket on timeout
            with self.socket_lock:
                if task_id in self.socket_cache:
                    self.socket_cache.pop(task_id)
            socket.close()
            context.term()
            raise RuntimeError(
                f"Timeout waiting for transfer confirmation for task {task_id}"
            )
        except Exception as e:
            # If sending or receiving fails, clean up the socket immediately
            logger.error(
                f"Error in sync_status_and_wait for task {task_id}: {e}"
            )
            with self.socket_lock:
                if task_id in self.socket_cache:
                    self.socket_cache.pop(task_id)
            socket.close()
            context.term()
            raise e

    def _init_transfer_engine(self, hostname: str, gpu_id: int, ib_device: Optional[str] = None):
        """Initialize the underlying Mooncake transfer engine"""
        self.engine = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=ib_device,
        )

    def get_session_id(self):
        """Get the session ID from the underlying transfer engine"""
        return self.engine.get_session_id()