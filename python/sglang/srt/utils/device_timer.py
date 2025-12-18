from contextlib import contextmanager

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode

_ENABLE_METRICS_DEVICE_TIMER = envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.get()


@contextmanager
def time_device_forward_pass(forward_mode: ForwardMode):
    TODO


class DeviceTimer:
    TODO
