"""PHANTOM-SOK: Self-Optimizing Kernel subsystem.

Cumulative kernel cache, shape profiling, and (future) safe autotuning
for PHANTOM speculative decoding on AMD RDNA2.

Phase F1: Cache provenance + fingerprint + profile-guided prewarm
Phase F2: Shape profiling + telemetry (future)
Phase F3: Replay + correctness harness (future)
Phase F4: Static kernel families + selection (future)
Phase F5: Offline conservative autotuning (future)
"""

from sglang.srt.speculative.sok.config import SOKConfig
from sglang.srt.speculative.sok.fingerprint import KernelFingerprint
from sglang.srt.speculative.sok.kernel_cache import KernelCache

__all__ = ["SOKConfig", "KernelFingerprint", "KernelCache"]
