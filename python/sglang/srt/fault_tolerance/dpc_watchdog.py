from __future__ import annotations

import logging
import threading
from multiprocessing import Process

import zmq
from sglang.srt.managers.io_struct import (
    FaultToleranceDPCShutdownReqInput,
    ProcessActiveRanksOutput,
    WatchdogHeartbeatOutput,
    sock_recv,
    sock_send,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket_on_host,
)
from sglang.srt.utils.watchdog import SubprocessWatchdog

logger = logging.getLogger(__name__)

FT_WATCHDOG_POLL_INTERVAL = 3.0
FT_WATCHDOG_SEND_TIMEOUT_MS = 60_000


class DPCFaultToleranceWatchdog(SubprocessWatchdog):
    """Own DPC fault-tolerance control sockets in the watchdog thread."""

    def __init__(
        self,
        *,
        context: zmq.Context,
        tokenizer_endpoint: str,
        node_rank: int,
        processes: list[Process],
        process_dp_ranks: list[int],
        process_global_ranks: list[int],
    ) -> None:
        self._context = context
        self._tokenizer_endpoint = tokenizer_endpoint
        self._node_rank = node_rank
        self._process_dp_ranks = process_dp_ranks
        self._process_global_ranks = process_global_ranks
        self._shutdown_receiver = None
        self._heartbeat_sender = None
        self._ready = threading.Event()
        self._start_error: Exception | None = None
        self.control_endpoint: str | None = None
        super().__init__(
            processes=processes,
            process_names=[f"scheduler_rank_{rank}" for rank in process_global_ranks],
            on_exit=self._report_process_exit,
            on_poll=self._poll_control,
            on_thread_stop=self._close_sockets,
            interval=FT_WATCHDOG_POLL_INTERVAL,
            fail_stop_on_exit=False,
            report_clean_exit=True,
        )

    def start(self) -> None:
        super().start()
        self._ready.wait()
        if self._start_error is not None:
            raise self._start_error

    def _on_thread_start(self) -> None:
        try:
            host = get_local_ip_auto(fallback="127.0.0.1")
            port, self._shutdown_receiver = get_zmq_socket_on_host(
                self._context, zmq.PULL, host=host
            )
            self.control_endpoint = NetworkAddress(host, port).to_tcp()

            self._heartbeat_sender = self._context.socket(zmq.PUSH)
            self._heartbeat_sender.setsockopt(zmq.LINGER, 0)
            self._heartbeat_sender.setsockopt(zmq.SNDHWM, 1)
            self._heartbeat_sender.setsockopt(zmq.IMMEDIATE, 1)
            self._heartbeat_sender.setsockopt(
                zmq.SNDTIMEO, FT_WATCHDOG_SEND_TIMEOUT_MS
            )
            if "[" in self._tokenizer_endpoint:
                self._heartbeat_sender.setsockopt(zmq.IPV6, 1)
            self._heartbeat_sender.connect(self._tokenizer_endpoint)
        except Exception as error:
            self._start_error = error
            raise
        finally:
            self._ready.set()

    def heartbeat(self) -> WatchdogHeartbeatOutput:
        return WatchdogHeartbeatOutput(
            node_rank=self._node_rank,
            ranks=sorted(self._process_global_ranks),
            control_endpoint=self.control_endpoint,
        )

    def _report_process_exit(self, index, proc, name) -> None:
        dp_rank = self._process_dp_ranks[index]
        global_rank = self._process_global_ranks[index]
        logger.warning(
            "Scheduler global rank %s for DP rank %s exited (pid=%s)",
            global_rank,
            dp_rank,
            proc.pid,
        )
        sock_send(
            self._heartbeat_sender,
            ProcessActiveRanksOutput(ranks=[global_rank], active=False),
        )

    def _poll_control(self) -> None:
        try:
            request = sock_recv(self._shutdown_receiver, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        else:
            self._shutdown_dp(request)

        try:
            sock_send(self._heartbeat_sender, self.heartbeat(), flags=zmq.NOBLOCK)
        except zmq.Again:
            logger.debug("Dropping watchdog heartbeat because tokenizer is unavailable")

    def _shutdown_dp(self, request: FaultToleranceDPCShutdownReqInput) -> None:
        targets = set(request.target_dp_ranks)
        for proc, dp_rank in zip(self._processes, self._process_dp_ranks):
            if dp_rank in targets and proc.is_alive():
                proc.kill()

    def _close_sockets(self) -> None:
        if self._heartbeat_sender is not None:
            self._heartbeat_sender.close(linger=0)
        if self._shutdown_receiver is not None:
            self._shutdown_receiver.close(linger=0)
