"""Fixed-TP FIFO policy, independent of request admission and model execution."""

from collections import deque
from dataclasses import dataclass

from sglang.srt.distributed.vpp_layout import VPPLayout


@dataclass(frozen=True)
class VPPTask:
    batch_seq: int
    stage_id: int


class VPPFixedSchedule:
    """Multi-pass interleaving with the same task order on every TP lane.

    Rank zero fills a first-pass wave, then visits each remaining virtual stage
    in FIFO order. Wave slots are released only on the final local pass.
    An empty admission queue closes a partial wave. Other ranks follow activation
    receive order and never bypass its head for a faster TP lane.
    max_inflight bounds pending tasks plus retained wave batches; it does not
    count unfinished GPU operations.

    Callers must enqueue identical admissions/receive identities across TP lanes.
    Selection does not inspect local tensor readiness or issue TP collectives;
    after selection, the caller waits for that task's input before dispatching.
    """

    def __init__(self, layout: VPPLayout, physical_rank: int, max_inflight: int):
        layout.stages_for_rank(physical_rank)
        if type(max_inflight) is not int or max_inflight <= 0:
            raise ValueError("max_inflight must be a positive integer")
        self.layout = layout
        self.physical_rank = physical_rank
        self.max_inflight = max_inflight
        # Admitted stage-zero tasks on rank zero; received tasks on other ranks.
        self._pending_tasks: deque[VPPTask] = deque()
        # Rank-zero batch identities retained until their final local dispatch.
        # The head rotates during intermediate passes to preserve FIFO per pass.
        self._wave_batch_seqs: deque[int] = deque()
        # Zero-based local pass: 0 fills the wave; 1..V-1 drain later stages.
        self._wave_pass_index = 0
        # Batches not yet dispatched in the current drain pass, not GPU work.
        self._batches_left_to_dispatch_in_pass = 0
        self._last_enqueued_batch_seq_by_stage: dict[int, int] = {}

    def enqueue(self, task: VPPTask) -> None:
        """Enqueue an admitted stage-zero task or a remote activation identity."""
        stage = self.layout.stage(task.stage_id)
        if type(task.batch_seq) is not int or task.batch_seq < 0:
            raise ValueError("batch_seq must be a nonnegative integer")
        if stage.physical_rank != self.physical_rank:
            raise ValueError("Task belongs to another physical rank")
        if self.physical_rank == 0 and task.stage_id != 0:
            raise ValueError("Rank zero's later passes are selected from its wave")
        if task.batch_seq <= self._last_enqueued_batch_seq_by_stage.get(
            task.stage_id, -1
        ):
            raise ValueError("Task sequences must increase within each stage")
        if len(self._pending_tasks) + len(self._wave_batch_seqs) >= self.max_inflight:
            raise RuntimeError("Fixed VPP task window is full")
        self._last_enqueued_batch_seq_by_stage[task.stage_id] = task.batch_seq
        self._pending_tasks.append(task)

    def next_task(self) -> VPPTask | None:
        """Choose an identity before waiting for lane-local activation data.

        Selecting a later pass closes a partial wave until its final dispatch.
        """
        if self.physical_rank != 0:
            return self._pending_tasks[0] if self._pending_tasks else None
        if self._wave_batch_seqs and (
            self._wave_pass_index > 0 or not self._pending_tasks
        ):
            if self._wave_pass_index == 0:
                self._start_next_wave_pass()
            return VPPTask(
                self._wave_batch_seqs[0],
                self._wave_pass_index * self.layout.physical_size,
            )
        return self._pending_tasks[0] if self._pending_tasks else None

    def _start_next_wave_pass(self) -> None:
        """Advance the local pass and reset its batch dispatch count."""
        self._wave_pass_index += 1
        self._batches_left_to_dispatch_in_pass = len(self._wave_batch_seqs)

    def record_dispatch(self, task: VPPTask) -> None:
        """Advance only after the selected task has actually launched."""
        if task != self.next_task():
            raise RuntimeError("Fixed VPP dispatch is out of FIFO order")
        if self.physical_rank == 0 and task.stage_id != 0:
            if self._wave_pass_index == self.layout.virtual_stages - 1:
                self._wave_batch_seqs.popleft()
            else:
                # A full rotation preserves the batch order for the next pass
                # without releasing slots or indexing into the middle of a deque.
                self._wave_batch_seqs.rotate(-1)
            self._batches_left_to_dispatch_in_pass -= 1
            if not self._wave_batch_seqs:
                self._wave_pass_index = 0
            elif self._batches_left_to_dispatch_in_pass == 0:
                self._start_next_wave_pass()
        else:
            self._pending_tasks.popleft()
            if self.physical_rank == 0 and self.layout.virtual_stages > 1:
                self._wave_batch_seqs.append(task.batch_seq)
                if len(self._wave_batch_seqs) == self.max_inflight:
                    self._start_next_wave_pass()
