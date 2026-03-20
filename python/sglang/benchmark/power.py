"""Power recording utilities for benchmarking."""

import threading
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch

# NOTE: If using `from sglang.srt.utils import is_hip`, we later always get empty cpu socket handles.
if torch.version.hip:
    try:
        import amdsmi
        amdsmi.amdsmi_init(amdsmi.AmdSmiInitFlags.INIT_ALL_PROCESSORS)
    except Exception as e:
        amdsmi.amdsmi_init(amdsmi.AmdSmiInitFlags.INIT_AMD_GPUS)

class PowerRecorderMixin(ABC):
    """Abstract base class for power recorders."""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.accelerator_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._accelerator_handles = None
        self._cpu_handles = None
        self.init_accelerator()

    @abstractmethod
    def init_accelerator(self) -> None:
        """Initialize accelerator handles into self._accelerator_handles."""

    @abstractmethod
    def _record_accelerator(self) -> float:
        """Sample and return total accelerator power in watts."""

    @abstractmethod
    def _record_cpu(self) -> float:
        """Sample and return total CPU power in watts."""

    def start(self) -> None:
        self._stop_event.clear()
        self.accelerator_samples = []
        self.cpu_samples = []
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def _record_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._accelerator_handles:
                self.accelerator_samples.append(self._record_accelerator())
            if self._cpu_handles:
                self.cpu_samples.append(self._record_cpu())
            self._stop_event.wait(timeout=self.interval)

    def get_stats(self) -> dict:
        """Return a dict of power statistics collected since start()."""
        assert self._accelerator_handles is not None
        assert self._cpu_handles is not None

        if not self.accelerator_samples:
            raise RuntimeError("No accelerator power samples collected.")

        arr = np.array(self.accelerator_samples)
        num_devices = len(self._accelerator_handles)
        stats = {
            "p25_power_w": float(np.percentile(arr, 25)),
            "mean_power_w": float(np.mean(arr)),
            "median_power_w": float(np.median(arr)),
            "median_power_per_accelerator_w": float(np.median(arr) / num_devices),
            "p75_power_w": float(np.percentile(arr, 75)),
            "num_samples": len(self.accelerator_samples),
        }

        if self.cpu_samples and len(self._cpu_handles) > 0:
            cpu_arr = np.array(self.cpu_samples)
            num_sockets = len(self._cpu_handles)
            stats.update(
                {
                    "cpu_p25_power_w": float(np.percentile(cpu_arr, 25)),
                    "cpu_mean_power_w": float(np.mean(cpu_arr)),
                    "cpu_median_power_w": float(np.median(cpu_arr)),
                    "cpu_median_power_per_socket_w": float(
                        np.median(cpu_arr) / num_sockets
                    ),
                    "cpu_p75_power_w": float(np.percentile(cpu_arr, 75)),
                    "cpu_num_samples": len(self.cpu_samples),
                }
            )
        else:
            stats.update(
                {
                    "cpu_p25_power_w": None,
                    "cpu_mean_power_w": None,
                    "cpu_median_power_w": None,
                    "cpu_median_power_per_socket_w": None,
                    "cpu_p75_power_w": None,
                    "cpu_num_samples": None,
                }
            )
        return stats

class PowerRecorderAMD(PowerRecorderMixin):
    """Records GPU and CPU power usage in a background thread using amdsmi."""

    def init_accelerator(self) -> None:
        self.amdsmi = amdsmi
        self._accelerator_handles = amdsmi.amdsmi_get_processor_handles()
        # Will be an empty list if amdsmi CPU dependencies are not met.
        self._cpu_handles = amdsmi.amdsmi_get_cpusocket_handles()
        
        if len(self._cpu_handles) == 0:
            print("WARNING: could not find CPU handles for power recording, amdsmi dependencies are likely not met. Recording GPU power only.")


    def _record_accelerator(self) -> float:
        """Sample total GPU power across all devices. Returns watts."""
        total_power = 0.0
        for handle in self._accelerator_handles:
            power_info = self.amdsmi.amdsmi_get_power_info(handle)
            total_power += power_info["socket_power"]
        return total_power

    def _record_cpu(self) -> float:
        """Sample total CPU socket power across all sockets. Returns watts."""
        total_power_mw = 0.0
        for handle in self._cpu_handles:
            # amdsmi_get_cpu_socket_power returns a string like "121650 mW"
            power_str = self.amdsmi.amdsmi_get_cpu_socket_power(handle)
            mw_value = float(power_str.split()[0])
            total_power_mw += mw_value
        return total_power_mw / 1000.0


def get_power_recorder(interval: float = 5.0) -> PowerRecorderMixin:
    """Return the appropriate PowerRecorder for the current hardware."""
    from sglang.srt.utils import is_hip

    if is_hip():
        return PowerRecorderAMD(interval=interval)
    raise NotImplementedError(
        "Power recording is only implemented on AMD GPUs (is_hip()=False)."
    )

