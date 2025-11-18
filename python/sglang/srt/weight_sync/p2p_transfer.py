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
        """Wait for the transfer to complete and release the socket"""
        self._event.wait()

        # Clean up the socket after completion
        with self.socket_lock:
            if self.task_id in self.socket_cache:
                socket_info = self.socket_cache.pop(self.task_id)
                try:
                    socket_info['socket'].close()
                    socket_info['context'].term()
                    logger.debug(f"Released socket for task {self.task_id}")
                except Exception as e:
                    logger.warning(f"Error releasing socket for task {self.task_id}: {e}")

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
        session_id: str,
        ptr: int,
        length: int,
        remote_ip: str,
        remote_port: int,
        socket_cache: Dict,
        socket_lock: threading.Lock,
    ):
        self.task_id = task_id
        self.session_id = session_id
        self.ptr = ptr
        self.length = length
        self.remote_ip = remote_ip
        self.remote_port = remote_port
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
        """
        task_id = self._generate_task_id()
        try:
            remote_ip, remote_port = session_id.split(":")[0], int(session_id.split(":")[1])
        except Exception as e:
            raise ValueError(f"Invalid session_id format: {session_id}") from e
        task = TransferTask(
            task_id, session_id, ptr, length, remote_ip, remote_port,
            self.socket_cache, self.socket_lock
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
                        f"Transfer task failed for session {task.session_id}: {e}"
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
        session_id = task.session_id
        ptr = task.ptr
        length = task.length

        # Register memory region if not already registered
        with self.registration_lock:
            if ptr not in self.registered_ptrs[session_id]:
                logger.debug(
                    f"Registering memory region for session {session_id}: "
                    f"ptr={ptr:#x}, length={length}"
                )
                self.engine.register(ptr, length)
                self.registered_ptrs[session_id].add(ptr)
            else:
                logger.debug(
                    f"Memory region already registered for session {session_id}: ptr={ptr:#x}"
                )

        # Send sync_status to training side and wait for confirmation
        self._send_sync_status_and_wait(task.task_id, task.remote_ip, task.remote_port, session_id, ptr, length)

    def _send_sync_status_and_wait(
        self, task_id: str, remote_ip: str, remote_port: int, session_id: str, ptr: int, length: int
    ):
        """
        Send sync_status message to training side and wait for confirmation.

        Workflow:
        1. Send sync_status with ip:port and registered ptr to training side
        2. Wait for training side to send back success/failure confirmation
        3. Socket is cached and will be released when handle.wait() is called

        Args:
            task_id: Unique task identifier
            remote_ip: Training process IP
            remote_port: Training process port
            session_id: Session identifier
            ptr: Registered memory pointer
            length: Buffer length
        """
        # Create a temporary socket and cache it for this task
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
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
            socket.send_json(
                {
                    "type": "sync_status",
                    "session_id": session_id,
                    "status": "ready",
                    "ip": self.engine.hostname,
                    "ptr": ptr,
                    "length": length,
                    "task_id": task_id,
                }
            )
            logger.info(
                f"Sent sync_status to {remote_ip}:{remote_port} for session {session_id}, "
                f"task_id={task_id}, ptr={ptr:#x}"
            )

            # Wait for confirmation from training side
            # Set a timeout to avoid hanging forever
            socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 seconds timeout

            logger.debug(f"Waiting for confirmation from {remote_ip}:{remote_port} for task {task_id}")
            response = socket.recv_json()

            # Check response status
            response_type = response.get("type", "")
            response_status = response.get("status", "")

            if response_type == "transfer_complete" and response_status == "success":
                logger.info(
                    f"Received success confirmation from {remote_ip}:{remote_port} "
                    f"for task {task_id}, session {session_id}"
                )
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