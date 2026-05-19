"""Credit-based KV Transfer Pacing for PD Disaggregation.

This module implements a flow control mechanism that coordinates RDMA KV
transfers with the decode forward pass to minimize HBM bandwidth contention.

Architecture:
    Decode side (DecodeForwardPacer):
        - Tracks forward pass state (running vs idle)
        - Sends CREDITS messages to prefill before/after each forward

    Prefill side (PrefillTransferGate):
        - Gates transfer_worker threads via a semaphore
        - Receives credits from decode to release waiting workers
"""

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KVPacingConfig:
    enabled: bool = False
    burst_credits: int = 8
    throttle_credits: int = 1
    max_credits: int = 16


class DecodeForwardPacer:
    """Decode-side: sends credits to prefill aligned with forward timing.

    Usage in decode event loop:
        pacer.on_forward_end()    # after process_batch_result
        ...scheduling...
        pacer.on_forward_start()  # before run_batch
    """

    def __init__(self, config: KVPacingConfig, send_fn):
        """
        Args:
            config: Pacing parameters.
            send_fn: Callable(num_credits: int) that sends a CREDITS message
                     to the prefill bootstrap thread via ZMQ.
        """
        self.config = config
        self._send_fn = send_fn
        self._forward_count = 0

    def on_forward_start(self):
        if not self.config.enabled:
            return
        self._forward_count += 1
        if self.config.throttle_credits > 0:
            self._send_fn(self.config.throttle_credits)

    def on_forward_end(self):
        if not self.config.enabled:
            return
        self._send_fn(self.config.burst_credits)


class PrefillTransferGate:
    """Prefill-side: gates RDMA transfers based on credits from decode.

    Usage in transfer_worker:
        gate.acquire()  # blocks if no credits
        engine.send_kvcache(...)
    """

    def __init__(self, config: KVPacingConfig):
        self.config = config
        self._sem = threading.Semaphore(config.burst_credits)
        self._lock = threading.Lock()
        self._credit_count = config.burst_credits

    def acquire(self, timeout: float = 5.0) -> bool:
        """Block until credit available. Returns False on timeout."""
        if not self.config.enabled:
            return True
        acquired = self._sem.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self._credit_count -= 1
        return acquired

    def receive_credits(self, num_credits: int):
        """Called when CREDITS message arrives from decode."""
        with self._lock:
            for _ in range(num_credits):
                if self._credit_count >= self.config.max_credits:
                    break
                self._sem.release()
                self._credit_count += 1
