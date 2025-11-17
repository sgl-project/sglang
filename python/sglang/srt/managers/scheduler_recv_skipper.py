from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs


class SchedulerRecvSkipper:
    @staticmethod
    def maybe_create(server_args: ServerArgs):
        if server_args.scheduler_recv_interval <= 1:
            return None
        return SchedulerRecvSkipper(server_args)

    def __init__(self, server_args: ServerArgs):
        # Can be supported if needed, but may need e.g. `global_forward_mode`
        assert not server_args.enable_dp_attention
        self._counter = 0
        self._threshold = server_args.scheduler_recv_interval

    def handle(self, last_forward_mode: ForwardMode):
        should_recv = False

        last_weight = _WEIGHT_OF_FORWARD_MODE.get(last_forward_mode, _DEFAULT_WEIGHT)
        self._counter += last_weight

        if self._counter >= self._threshold:
            self._counter = 0
            should_recv = True

        return should_recv


# All can be tuned if needed
_DEFAULT_WEIGHT = 1000
_WEIGHT_OF_FORWARD_MODE = {
    ForwardMode.DECODE: 1,
    ForwardMode.TARGET_VERIFY: 1,
    None: 1,
}
