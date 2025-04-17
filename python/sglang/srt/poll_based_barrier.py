class PollBasedBarrier:
    def __init__(self, noop: bool = False):
        self._noop = noop

    def local_arrive(self):
        TODO

    def poll_global_arrive(self) -> bool:
        TODO
