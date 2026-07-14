from typing import List, Optional

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs

# Modes whose weight keeps the counter accumulating slowly (see the weight
# table below); every other mode maps to the default weight, which forces a
# recv on the next iteration.
_SLOW_RECV_MODE_VALUES = (ForwardMode.DECODE.value, ForwardMode.TARGET_VERIFY.value)


class SchedulerRecvSkipper:
    @staticmethod
    def maybe_create(server_args: ServerArgs):
        if server_args.scheduler_recv_interval <= 1:
            return None
        return SchedulerRecvSkipper(server_args)

    @staticmethod
    def derive_global_forward_mode(
        global_forward_modes: List[int],
    ) -> Optional[ForwardMode]:
        """Collapse per-DP-rank forward modes into one weight-table bucket.

        The input comes from the MLP sync all-gather, so it is identical on
        every rank and the derived mode (hence the recv decision) is too.
        """
        active = [
            m
            for m in global_forward_modes
            if m != ForwardMode.IDLE.value and m != ForwardMode.PREBUILT.value
        ]
        if not active:
            # Globally idle. Use the same bucket as "no last batch" so ranks
            # holding only an IDLE/PREBUILT batch agree with ranks whose
            # last_batch is None.
            return None
        if any(m not in _SLOW_RECV_MODE_VALUES for m in active):
            # At least one rank ran an extend-like step; bucket to EXTEND so
            # the default weight forces a prompt recv for new requests.
            return ForwardMode.EXTEND
        if all(m == ForwardMode.TARGET_VERIFY.value for m in active):
            return ForwardMode.TARGET_VERIFY
        return ForwardMode.DECODE

    def __init__(self, server_args: ServerArgs):
        # Safe under DP-attention: the scheduler feeds ``handle`` a mode
        # derived from the gathered per-rank forward modes (identical on
        # every rank), so all ranks accumulate the same counter and make the
        # same ``should_recv`` decision. This keeps the request-broadcast
        # collectives consistent across ranks without extra communication.
        self._counter = 0
        self._threshold = server_args.scheduler_recv_interval
        # All can be tuned if needed
        self._default_weight = envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT.get()
        self._weight_of_forward_mode = {
            ForwardMode.DECODE: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE.get(),
            ForwardMode.TARGET_VERIFY: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY.get(),
            None: envs.SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE.get(),
        }

    def handle(self, last_forward_mode: ForwardMode):
        should_recv = False

        last_weight = self._weight_of_forward_mode.get(
            last_forward_mode, self._default_weight
        )
        self._counter += last_weight

        if self._counter >= self._threshold:
            self._counter = 0
            should_recv = True

        return should_recv
