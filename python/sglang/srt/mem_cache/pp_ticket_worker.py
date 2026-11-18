"""Ordered prefetch commands across pipeline stages."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from queue import Empty, Queue
from typing import Generic, TypeVar

import torch
import torch.distributed as dist

from sglang.srt.utils import broadcast_pyobj

logger = logging.getLogger(__name__)
_Ticket = TypeVar("_Ticket")


def run_pp_ticket_loop(
    group: dist.ProcessGroup,
    queue: Queue[_Ticket | None],
    handle_ticket: Callable[[_Ticket], None],
) -> None:
    """Broadcast PP0's FIFO, including idle heartbeats and the stop sentinel."""
    rank = dist.get_rank()
    source = dist.get_process_group_ranks(group)[0]
    while True:
        commands = []
        if rank == source:
            try:
                commands = [queue.get(timeout=60)]
            except Empty:
                pass  # Complete idle broadcasts before the group times out.
        try:
            received = broadcast_pyobj(commands, rank, group, src=source)
            if received:
                ticket = received[0]
                if ticket is None:
                    return
                handle_ticket(ticket)
        finally:
            if commands:
                queue.task_done()


class PPTicketWorker(Generic[_Ticket]):
    """A FIFO worker with its own Gloo groups for command and vote ordering."""

    def __init__(
        self,
        *,
        pp_group: dist.ProcessGroup,
        attn_cp_group: dist.ProcessGroup | None,
        attn_tp_group: dist.ProcessGroup | None,
        tp_group: dist.ProcessGroup,
        handle_ticket: Callable[[_Ticket], None],
        finish_ticket: Callable[[_Ticket], None] | None = None,
    ) -> None:
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        self._queue: Queue[_Ticket | None] = Queue()
        self._failure: Exception | None = None
        self._groups: list[dist.ProcessGroup] = []
        self._completion_groups: list[dist.ProcessGroup] = []
        self._completion_queue: Queue[_Ticket | None] | None = None
        self._completion_thread: threading.Thread | None = None
        pp_ranks = dist.get_process_group_ranks(pp_group)
        self._is_source = dist.get_rank() == pp_ranks[0]
        attention_groups = (
            (attn_cp_group, attn_tp_group)
            if attn_cp_group is not None or attn_tp_group is not None
            else (tp_group,)
        )
        rank_sets = []
        for group in attention_groups:
            if group is not None:
                ranks = dist.get_process_group_ranks(group)
                if len(ranks) > 1 and ranks != pp_ranks and ranks not in rank_sets:
                    rank_sets.append(ranks)
        rank_sets.append(pp_ranks)
        group_lists = [self._groups]
        if finish_ticket is not None:
            group_lists.append(self._completion_groups)
        try:
            for groups in group_lists:
                for ranks in rank_sets:
                    groups.append(
                        create_custom_parallel_group(group_ranks=ranks, backend="gloo")
                    )
            if finish_ticket is not None:
                # An outstanding GET can complete while the command worker
                # votes on a newer ticket. Its ACK has its own collective order.
                self._completion_queue = Queue()
                self._completion_thread = threading.Thread(
                    target=self._run_completion,
                    args=(self._completion_queue, finish_ticket),
                    name="linker-pp-completion",
                    daemon=True,
                )
                self._completion_thread.start()
            self._thread = threading.Thread(
                target=self._run,
                args=(self._groups[-1], handle_ticket),
                name="linker-pp-ticket",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            self._stop_completion()
            self._destroy_groups()
            raise

    def _run(self, group, handle_ticket) -> None:
        def prepare(ticket):
            handle_ticket(ticket)
            if self._completion_queue is not None:
                self._completion_queue.put(ticket)

        try:
            run_pp_ticket_loop(group, self._queue, prepare)
        except Exception as error:
            self._failure = error
            logger.exception("PP linker ticket processing failed.")

    def _run_completion(self, queue, finish_ticket) -> None:
        try:
            while (ticket := queue.get()) is not None:
                finish_ticket(ticket)
        except Exception as error:
            self._failure = error
            logger.exception("PP linker ticket completion failed.")

    def submit(self, ticket: _Ticket) -> None:
        if not self._is_source:
            raise RuntimeError("Only PP0 can submit a prefetch ticket.")
        self.check_alive()
        self._queue.put(ticket)

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor:
        """Intersect the current ticket's votes on the FIFO worker."""
        for group in self._groups:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
        return tensor

    def reduce_completion(self, tensor: torch.Tensor) -> torch.Tensor:
        """Agree on completion in issued-ticket order, independently of new votes."""
        for group in self._completion_groups:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
        return tensor

    def check_alive(self) -> None:
        completion = self._completion_thread
        if not self._thread.is_alive() or (completion and not completion.is_alive()):
            raise RuntimeError("PP linker ticket worker exited.") from self._failure

    def close(self) -> None:
        """Drain issued tickets before releasing the worker's process groups."""
        if not self._groups:
            return
        if self._is_source and self._thread.is_alive():
            self._queue.put(None)
        self._thread.join()
        self._stop_completion()
        self._destroy_groups()
        if self._failure is not None:
            raise RuntimeError("PP linker ticket worker failed.") from self._failure

    def _stop_completion(self) -> None:
        if self._completion_thread is not None and self._completion_thread.is_alive():
            assert self._completion_queue is not None
            self._completion_queue.put(None)
            self._completion_thread.join()

    def _destroy_groups(self) -> None:
        for group in reversed(self._completion_groups):
            dist.destroy_process_group(group)
        self._completion_groups.clear()
        for group in reversed(self._groups):
            dist.destroy_process_group(group)
        self._groups.clear()
