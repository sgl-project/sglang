import torch

from sglang.srt.runtime_context import get_parallel


class PollBasedBarrier:
    def __init__(self, noop: bool = False):
        self._noop = noop
        self._local_arrived = False
        self._local_pending = False
        self._local_polling = False
        self.global_pending = False
        self.global_polling = False

    def local_arrive(self):
        assert not self._local_arrived
        self._local_arrived = True

    def poll_global_arrived(
        self, *, pending: bool = False, polling: bool = False
    ) -> bool:
        self._local_pending = pending
        self._local_polling = polling
        global_arrived = self._compute_global_arrived()
        output = self._local_arrived and global_arrived
        if output:
            self._local_arrived = False
        return output

    def _compute_global_arrived(self) -> bool:
        local_arrived = self._noop or self._local_arrived
        votes = torch.tensor(
            [local_arrived, not self._local_pending, not self._local_polling]
        )
        # Can optimize if bottleneck
        torch.distributed.all_reduce(
            votes,
            torch.distributed.ReduceOp.MIN,
            group=get_parallel().world_group.cpu_group,
        )
        global_arrived, globally_idle, globally_sleepable = votes.tolist()
        self.global_pending = not globally_idle
        self.global_polling = not globally_sleepable
        return global_arrived
