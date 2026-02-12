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
        self._threshold = server_args.scheduler_recv_interval
        self._counter = self._threshold
        # All can be tuned if needed
        self._default_weight = envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT.get()
        self._weight_of_forward_mode = {
            ForwardMode.DECODE: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE.get(),
            ForwardMode.TARGET_VERIFY: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY.get(),
            None: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE.get(),
        }

    def update_counter_dp(self, global_forward_mode: Optional[ForwardMode]) -> None:
        """
        Update counter based on global forward mode (for dp_attention mode).
        This should be called AFTER the sync point (prepare_mlp_sync_batch) to ensure
        all DP ranks have the same counter value.

        Args:
            global_forward_mode: The global forward mode synchronized across DP ranks.
        """
        self._counter += self._weight_of_forward_mode.get(
            global_forward_mode, self._default_weight
        )

    def handle(
        self,
        last_forward_mode: ForwardMode,
    ):
        """
        Determine whether to receive requests based on forward mode.

        For dp_attention mode:
            - This method only CHECKS if counter >= threshold (does not update counter)
            - Counter is updated separately via update_counter_dp() after sync point
            - This ensures all DP ranks make the same skip/recv decision

        For non-dp_attention mode:
            - check and update counter in one call

        Args:
            last_forward_mode: The local forward mode from the last batch.
        Returns:
            bool: True if should receive requests, False otherwise.
        """
        if self._enable_dp_attention:
            if self._counter >= self._threshold:
                self._counter = 0
                return True
            return False
        else:
            last_weight = self._weight_of_forward_mode.get(
                last_forward_mode, self._default_weight
            )
            self._counter += last_weight

            if self._counter >= self._threshold:
                self._counter = 0
                return True

            return False
