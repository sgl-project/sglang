import torch


class PollBasedBarrier:
    def __init__(self, noop: bool = False):
        self._noop = noop
        self._local_arrived = False

    def local_arrive(self):
        assert not self._local_arrived
        self._local_arrived = True

    def poll_global_arrived(self) -> bool:
        global_arrived = self._compute_global_arrived()
        output = self._local_arrived and global_arrived
        if output:
            self._local_arrived = False
        return output

    def _compute_global_arrived(self) -> bool:
        local_arrived = self._noop or self._local_arrived
        global_arrived = torch.tensor(local_arrived).cuda()
        # Can optimize if bottleneck
        torch.distributed.all_reduce(global_arrived, torch.distributed.ReduceOp.MIN)
        return global_arrived.cpu().item()

    def _change_state(self, original: "_State", target: "_State"):
        assert self._state == original, f"{self._state=} {original=} {target=}"
        self._state = target
