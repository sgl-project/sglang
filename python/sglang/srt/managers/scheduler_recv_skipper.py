from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs


class SchedulerRecvSkipper:
    @staticmethod
    def maybe_create(server_args: ServerArgs):
        if server_args.scheduler_recv_interval <= 1:
            return None
        return SchedulerRecvSkipper(server_args)

    def __init__(self, server_args: ServerArgs):
        self._enable_dp_attention = server_args.enable_dp_attention
        self._counter = 0
        self._threshold = server_args.scheduler_recv_interval
        # All can be tuned if needed
        self._default_weight = envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT.get()
        self._weight_of_forward_mode = {
            ForwardMode.DECODE: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE.get(),
            ForwardMode.TARGET_VERIFY: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY.get(),
            None: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE.get(),
        }

    def handle(
        self,
        last_forward_mode: ForwardMode,
        global_forward_mode: Optional[int] = None,
    ):
        """
        Determine whether to receive requests based on forward mode.

        Args:
            last_forward_mode: The local forward mode from the last batch.
            global_forward_mode: The global forward mode synchronized across DP ranks.
                                 Used when enable_dp_attention is True
        Returns:
            bool: True if should receive requests, False otherwise.
        """
        if self._enable_dp_attention:
            if global_forward_mode is None:
                # First round or no previous global_forward_mode, don't skip
                return True
            effective_mode = ForwardMode(global_forward_mode)
        else:
            effective_mode = last_forward_mode

        last_weight = self._weight_of_forward_mode.get(
            effective_mode, self._default_weight
        )
        self._counter += last_weight

        if self._counter >= self._threshold:
            self._counter = 0
            return True

        return False
