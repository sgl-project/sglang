"""Hardware governor: scale existing scheduler knobs from GPU power, temperature and clock telemetry.

``TelemetrySource`` reads the hardware, ``GovernorPolicy`` turns readings into
``KnobOverrides``, and ``HardwareGovernor`` applies them identically on every
rank. Sources and policies are pluggable through ``package.module:ClassName``
specs on ``--hw-telemetry-source`` and ``--hw-governor-policy``.
"""

from sglang.srt.managers.hw_governor.knobs import KnobOverrides
from sglang.srt.managers.hw_governor.policy import (
    GovernorDecision,
    GovernorPolicy,
    StepwisePolicy,
)
from sglang.srt.managers.hw_governor.telemetry import (
    ClockEventReason,
    DeviceTelemetry,
    TelemetrySource,
)

__all__ = [
    "ClockEventReason",
    "DeviceTelemetry",
    "GovernorDecision",
    "GovernorPolicy",
    "KnobOverrides",
    "StepwisePolicy",
    "TelemetrySource",
]
