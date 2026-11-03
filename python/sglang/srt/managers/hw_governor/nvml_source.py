"""In-process NVML telemetry source, the default for NVIDIA GPUs."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Optional, Sequence, TypeVar

import msgspec

from sglang.srt.managers.hw_governor.telemetry import DeviceTelemetry, TelemetrySource

logger = logging.getLogger(__name__)

T = TypeVar("T")


class NvmlTelemetrySource(TelemetrySource):
    """Reads power, temperature, clocks and clock-event reasons through NVML.

    NVML addresses physical GPUs, so devices are matched by UUID and
    ``CUDA_VISIBLE_DEVICES`` reordering does not matter. A device NVML cannot
    see (another node's GPU) is skipped.
    """

    def __init__(self, *, device_uuids: Sequence[str]) -> None:
        super().__init__(device_uuids=device_uuids)
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._devices = []
        for uuid in self.device_uuids:
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(_to_nvml_uuid(uuid))
            except pynvml.NVMLError:
                logger.info(
                    "Hardware governor: device %s is not visible to NVML here; "
                    "not watching it.",
                    uuid,
                )
                continue
            self._devices.append(
                _NvmlDevice(
                    uuid=uuid,
                    handle=handle,
                    sm_clock_max_mhz=self._read(
                        pynvml.nvmlDeviceGetMaxClockInfo, handle, pynvml.NVML_CLOCK_SM
                    ),
                    max_operating_temp_c=self._max_operating_temp_c(handle),
                )
            )
        if not self._devices:
            pynvml.nvmlShutdown()
            raise RuntimeError(
                f"NVML can read none of the devices {list(self.device_uuids)}."
            )

    def sample(self) -> List[DeviceTelemetry]:
        return [self._sample_device(device) for device in self._devices]

    def close(self) -> None:
        self._nvml.nvmlShutdown()

    def _sample_device(self, device: _NvmlDevice) -> DeviceTelemetry:
        nvml, handle = self._nvml, device.handle
        gpu_temp_c = self._read(
            nvml.nvmlDeviceGetTemperature, handle, nvml.NVML_TEMPERATURE_GPU
        )
        power_mw = self._read(nvml.nvmlDeviceGetPowerUsage, handle)
        power_limit_mw = self._read(nvml.nvmlDeviceGetEnforcedPowerLimit, handle)
        return DeviceTelemetry(
            device_uuid=device.uuid,
            timestamp=time.monotonic(),
            gpu_temp_c=gpu_temp_c,
            temp_margin_c=self._temp_margin_c(device, gpu_temp_c),
            power_w=None if power_mw is None else power_mw / 1000.0,
            power_limit_w=None if power_limit_mw is None else power_limit_mw / 1000.0,
            sm_clock_mhz=self._read(
                nvml.nvmlDeviceGetClockInfo, handle, nvml.NVML_CLOCK_SM
            ),
            sm_clock_max_mhz=device.sm_clock_max_mhz,
            clock_event_reasons=self._read(
                nvml.nvmlDeviceGetCurrentClocksEventReasons, handle
            )
            or 0,
            energy_mj=self._read(nvml.nvmlDeviceGetTotalEnergyConsumption, handle),
        )

    def _temp_margin_c(
        self, device: _NvmlDevice, gpu_temp_c: Optional[float]
    ) -> Optional[float]:
        margin = self._read(self._nvml.nvmlDeviceGetMarginTemperature, device.handle)
        if margin is not None:
            return margin
        if gpu_temp_c is None or device.max_operating_temp_c is None:
            return None
        return device.max_operating_temp_c - gpu_temp_c

    def _max_operating_temp_c(self, handle) -> Optional[float]:
        # NVML's margin is measured to GPU_MAX (88 C on L40S, slowdown at 95 C),
        # so the fallback uses the same threshold when the driver reports it.
        for threshold in (
            self._nvml.NVML_TEMPERATURE_THRESHOLD_GPU_MAX,
            self._nvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN,
        ):
            value = self._read(
                self._nvml.nvmlDeviceGetTemperatureThreshold, handle, threshold
            )
            if value is not None:
                return value
        return None

    def _read(self, fn: Callable[..., T], *args) -> Optional[T]:
        # Unsupported and transient failures both read as "not reported".
        try:
            return fn(*args)
        except self._nvml.NVMLError:
            return None


class _NvmlDevice(msgspec.Struct, frozen=True, kw_only=True):
    uuid: str
    # pynvml c_nvmlDevice_t.
    handle: Any
    sm_clock_max_mhz: Optional[float]
    max_operating_temp_c: Optional[float]


def _to_nvml_uuid(uuid: str) -> str:
    # torch reports the bare UUID; NVML expects the "GPU-" (or "MIG-") prefix.
    return uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"
