"""Hardware telemetry: per-device readings and the pluggable source that produces them."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Type

import msgspec

from sglang.srt.managers.hw_governor.registry import resolve_class


class ClockEventReason(enum.IntFlag):
    """Why clocks are below maximum; bit values of NVML's ``nvmlClocksEventReason*``."""

    GPU_IDLE = 0x1
    APPLICATIONS_CLOCKS_SETTING = 0x2
    SW_POWER_CAP = 0x4
    HW_SLOWDOWN = 0x8
    SYNC_BOOST = 0x10
    SW_THERMAL_SLOWDOWN = 0x20
    HW_THERMAL_SLOWDOWN = 0x40
    HW_POWER_BRAKE_SLOWDOWN = 0x80
    DISPLAY_CLOCK_SETTING = 0x100


class DeviceTelemetry(msgspec.Struct, frozen=True, kw_only=True):
    """One reading of one device; a field is ``None`` when the device does not report it."""

    device_uuid: str
    # time.monotonic() seconds.
    timestamp: float
    gpu_temp_c: Optional[float] = None
    # Degrees C below the max operating temperature, where clocks start to drop.
    temp_margin_c: Optional[float] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    sm_clock_mhz: Optional[float] = None
    sm_clock_max_mhz: Optional[float] = None
    # ClockEventReason bitmask.
    clock_event_reasons: int = 0
    # Cumulative since the driver loaded, in millijoules.
    energy_mj: Optional[int] = None

    @property
    def power_ratio(self) -> Optional[float]:
        if self.power_w is None or not self.power_limit_w:
            return None
        return self.power_w / self.power_limit_w

    @property
    def sm_clock_ratio(self) -> Optional[float]:
        if self.sm_clock_mhz is None or not self.sm_clock_max_mhz:
            return None
        return self.sm_clock_mhz / self.sm_clock_max_mhz


class TelemetrySource(ABC):
    """Reads telemetry for a fixed set of devices, identified by UUID.

    Subclass to add a backend (DCGM, a BMC, a remote exporter) and select it
    with ``--hw-telemetry-source package.module:ClassName``. ``sample`` runs on
    the governor's background thread, never on the scheduler loop.
    """

    def __init__(self, *, device_uuids: Sequence[str]) -> None:
        self.device_uuids = tuple(device_uuids)

    @abstractmethod
    def sample(self) -> List[DeviceTelemetry]:
        """Return one reading per device this source can read; omit the rest."""

    def close(self) -> None:
        pass


_BUILTIN_SOURCES = {
    "nvml": "sglang.srt.managers.hw_governor.nvml_source:NvmlTelemetrySource",
}


def resolve_telemetry_source_cls(spec: str) -> Type[TelemetrySource]:
    return resolve_class(
        spec, builtins=_BUILTIN_SOURCES, base=TelemetrySource, kind="telemetry source"
    )
