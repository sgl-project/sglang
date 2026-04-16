"""PHANTOM-SOK: Self-Optimizing Kernel subsystem.

Cumulative kernel cache, shape profiling, and (future) safe autotuning
for PHANTOM speculative decoding on AMD RDNA2.

Phase F1: Cache provenance + fingerprint + profile-guided prewarm
Phase F2: Shape profiling + telemetry
Phase F3: Replay + correctness harness (future)
Phase F4: Static kernel families + selection (future)
Phase F5: Offline conservative autotuning (future)
"""

from sglang.srt.speculative.sok.config import SOKConfig
from sglang.srt.speculative.sok.fingerprint import KernelFingerprint
from sglang.srt.speculative.sok.kernel_cache import KernelCache
from sglang.srt.speculative.sok.shape_profile import ShapeKey, ShapeProfile
from sglang.srt.speculative.sok.telemetry import DispatchTelemetry

__all__ = [
    "SOKConfig",
    "KernelFingerprint",
    "KernelCache",
    "ShapeKey",
    "ShapeProfile",
    "DispatchTelemetry",
]
