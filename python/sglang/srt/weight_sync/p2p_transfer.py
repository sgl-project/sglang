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
        self.engine_idx: Optional[int] = None  # Engine index for P2PTransferManager


class P2PTransferManager:
    """
    Manager that maintains a pool of Mooncake TransferEngine instances.

    This class provides the same interface as P2PTransferEngine but uses multiple
    TransferEngine instances internally to improve parallelism and throughput.

    Key design difference from P2PTransferEngine:
    - Uses a pool of MooncakeTransferEngine instances instead of a single one
    - Task queue-based scheduling: selects idle engines for tasks
    - If all engines are busy, tasks wait in queue
    - Each engine has its own lock for thread-safe access
    """

    def __init__(self, hostname: str, gpu_id: int, ib_device: Optional[str] = None):
        """
        Initialize the P2P transfer manager with a pool of transfer engines.

        Args:
            hostname: Local hostname for RDMA connections
            gpu_id: GPU device ID
            ib_device: InfiniBand device name (optional)
        """
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.ib_device = ib_device

        # Initialize the engine pool
        cpu_count = os.cpu_count() or 8
        self.engine_pool_size = get_int_env_var(
            "SGLANG_TRANSFER_ENGINE_POOL_SIZE",
            min(max(2, int(0.25 * cpu_count)), 8),
        )

        self.engine_pool = []
        for _ in range(self.engine_pool_size):
            engine = MooncakeTransferEngine(
                hostname=hostname,
                gpu_id=gpu_id,
                ib_device=ib_device,
            )
            self.engine_pool.append(engine)

        # Track registered memory regions per engine
        self.registered_ptrs: Dict[int, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        self.registration_locks = [threading.Lock() for _ in range(self.engine_pool_size)]

        # Task counter for generating unique task IDs
        self.task_counter = 0
        self.task_counter_lock = threading.Lock()

        # Task queue and engine status tracking
        self.task_queue = Queue()
        self.engine_busy = [False] * self.engine_pool_size
        self.engine_status_lock = threading.Lock()

        # Start scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_worker,
            daemon=True
        )
        self.scheduler_thread.start()

        logger.info(
            f"P2PTransferManager initialized with {self.engine_pool_size} transfer engines "
            f"on {hostname}:{gpu_id} (task queue-based scheduling)"
        )

    def _get_engine_index(self) -> Optional[int]:
        """
        Get an idle engine index from the pool.

        Returns:
            Index of an idle engine, or None if all engines are busy
        """
        with self.engine_status_lock:
            for idx in range(self.engine_pool_size):
                if not self.engine_busy[idx]:
                    self.engine_busy[idx] = True
                    return idx
        return None

    def _release_engine(self, engine_idx: int):
        """
        Mark an engine as idle after task completion.

        Args:
            engine_idx: Index of the engine to release
        """
        with self.engine_status_lock:
            self.engine_busy[engine_idx] = False

    def _scheduler_worker(self):
        """
        Scheduler thread that continuously processes tasks from the queue.
        Waits for idle engines and assigns tasks to them.
        """
        while True:
            try:
                # Block until a task is available
                task_info = self.task_queue.get()

                # Wait for an idle engine
                engine_idx = None
                while engine_idx is None:
                    engine_idx = self._get_engine_index()
                    if engine_idx is None:
                        # All engines busy, wait a bit before retrying
                        threading.Event().wait(0.01)

                # Unpack task info
                task_id, session_id, ptr, length, handle, socket_cache, socket_lock = task_info

                logger.debug(
                    f"Scheduler assigned task {task_id} to engine {engine_idx} "
                    f"(session={session_id})"
                )

                # Start processing the task in a background thread
                thread = threading.Thread(
                    target=self._process_transfer_task_with_release,
                    args=(
                        task_id,
                        session_id,
                        ptr,
                        length,
                        engine_idx,
                        handle,
                        socket_cache,
                        socket_lock
                    ),
                    daemon=True
                )
                thread.start()

            except Exception as e:
                logger.error(f"Scheduler worker error: {e}", exc_info=True)

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

        The task will be queued and assigned to an idle engine by the scheduler.
        If all engines are busy, the task waits in the queue.

        Args:
            session_id: Remote training process address (ip:port) for ZMQ communication
            ptr: Local memory pointer to register
            length: Buffer length
        """
        task_id = self._generate_task_id()
        try:
            # Validate session_id format
            _ = session_id.split(":")[0]
            _ = int(session_id.split(":")[1])
        except Exception as e:
            raise ValueError(f"Invalid session_id format: {session_id}") from e

        # Create handle for this transfer
        socket_cache: Dict[str, Dict] = {}
        socket_lock = threading.Lock()
        handle = TransferHandle(task_id, socket_cache, socket_lock)

        # Put task into queue - scheduler will assign it to an idle engine
        task_info = (task_id, session_id, ptr, length, handle, socket_cache, socket_lock)
        self.task_queue.put(task_info)

        logger.debug(
            f"Queued transfer task {task_id} for session {session_id}, ptr={ptr:#x}, "
            f"length={length}"
        )

        return handle

    def _process_transfer_task_with_release(
        self,
        task_id: str,
        session_id: str,
        ptr: int,
        length: int,
        engine_idx: int,
        handle: TransferHandle,
        socket_cache: Dict,
        socket_lock: threading.Lock
    ):
        """
        Wrapper that processes a transfer task and releases the engine after memory registration.

        The engine is released as soon as memory region is registered, without waiting
        for the ZMQ sync communication to complete. This allows the engine to be reused
        for other registration tasks while the current task continues its sync process.
        """
        try:
            self._process_transfer_task(
                task_id, session_id, ptr, length, engine_idx,
                handle, socket_cache, socket_lock
            )
        except Exception as e:
            logger.error(
                f"Transfer task {task_id} failed for session {session_id}: {e}",
                exc_info=True
            )
            handle._mark_done(False)

    def _process_transfer_task(
        self,
        task_id: str,
        session_id: str,
        ptr: int,
        length: int,
        engine_idx: int,
        handle: TransferHandle,
        socket_cache: Dict,
        socket_lock: threading.Lock
    ):
        """
        Process a single transfer task in a background thread.

        Steps:
        1. Register memory region if not already registered
        2. Release engine (so it can be reused for other registrations)
        3. Send sync_status to training side
        4. Wait for confirmation
        5. Mark handle as done
        """
        # Get the engine's session_id
        local_transfer_session_id = self.engine_pool[engine_idx].get_session_id()

        # Register memory region if not already registered
        with self.registration_locks[engine_idx]:
            if ptr not in self.registered_ptrs[engine_idx][session_id]:
                logger.debug(
                    f"Registering memory region on engine {engine_idx} for session {session_id}: "
                    f"ptr={ptr:#x}, length={length}"
                )
                self.engine_pool[engine_idx].register(ptr, length)
                self.registered_ptrs[engine_idx][session_id].add(ptr)
            else:
                logger.debug(
                    f"Memory region already registered on engine {engine_idx} for session {session_id}: ptr={ptr:#x}"
                )

        # Release engine immediately after registration is done
        # The engine can now be reused for other registration tasks
        self._release_engine(engine_idx)
        logger.debug(f"Released engine {engine_idx} after memory registration for task {task_id}")

        # Send sync_status and wait for confirmation
        # This can take time, but doesn't need the engine anymore
        self._send_sync_status_and_wait(
            task_id=task_id,
            training_p2p_session_id=session_id,
            rollout_transfer_session_id=local_transfer_session_id,
            ptr=ptr,
            length=length,
            engine_idx=engine_idx,
            socket_cache=socket_cache,
            socket_lock=socket_lock
        )

        # Mark as successfully completed
        handle._mark_done(True)

        # Deregiter memory region
        with self.registration_locks[engine_idx]:
            if ptr in self.registered_ptrs[engine_idx][session_id]:
                logger.debug(
                    f"Deregistering memory region on engine {engine_idx} for session {session_id}: ptr={ptr:#x}"
                )
                self.engine_pool[engine_idx].deregister(ptr)
                self.registered_ptrs[engine_idx][session_id].remove(ptr)


    def _send_sync_status_and_wait(
        self,
        task_id: str,
        training_p2p_session_id: str,
        rollout_transfer_session_id: str,
        ptr: int,
        length: int,
        engine_idx: int,
        socket_cache: Dict,
        socket_lock: threading.Lock
    ):
        """
        Send sync_status message to training side and wait for confirmation.
        """
        # Create a temporary socket
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        remote_ip, remote_port = training_p2p_session_id.split(":")[0], int(training_p2p_session_id.split(":")[1])
        socket.connect(format_tcp_address(remote_ip, remote_port))

        # Cache the socket for cleanup
        with socket_lock:
            socket_cache[task_id] = {
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
                    "p2p_session_id": training_p2p_session_id,
                    "status": "ready",
                    "ip": self.hostname,
                    "transfer_session_id": rollout_transfer_session_id,
                    "ptr": ptr,
                    "length": length,
                    "task_id": task_id,
                }
            )
            logger.info(
                f"Sent sync_status to {remote_ip}:{remote_port} for session {training_p2p_session_id}, "
                f"task_id={task_id}, engine={engine_idx}, rollout_transfer_session_id={rollout_transfer_session_id}, ptr={ptr:#x}"
            )

            # Wait for confirmation from training side
            socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 seconds timeout

            logger.debug(f"Waiting for confirmation from {remote_ip}:{remote_port} for task {task_id}")

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
                    f"for task {task_id}, transfer session {rollout_transfer_session_id}, engine {engine_idx}"
                )
                # Clean up the socket after successful transfer
                with socket_lock:
                    if task_id in socket_cache:
                        socket_cache.pop(task_id)
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
            with socket_lock:
                if task_id in socket_cache:
                    socket_cache.pop(task_id)
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
            with socket_lock:
                if task_id in socket_cache:
                    socket_cache.pop(task_id)
            socket.close()
            context.term()
            raise e

    def get_all_session_ids(self):
        """Get all session IDs from all engines in the pool."""
        return [engine.get_session_id() for engine in self.engine_pool]
