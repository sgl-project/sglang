"""Environment + model fingerprinting for safe kernel cache reuse.

A KernelFingerprint captures everything that could affect compiled kernel
correctness or performance. Any field change → cache miss → recompile.
Stale winners from different runtime environments are NEVER reused.
"""

import hashlib
import json
import logging
import os
import platform
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KernelFingerprint:
    """Immutable environment identity for kernel cache keying."""

    # Hardware
    gpu_arch: str           # e.g. "gfx1030"
    wave_size: int          # 32 or 64

    # Software stack
    rocm_version: str       # e.g. "6.4.43800"
    triton_version: str     # e.g. "3.6.0"
    python_version: str     # e.g. "3.12.9"

    # Model structure (affects kernel compile params)
    model_family: str       # e.g. "qwen3-4b"
    head_dim: int
    n_heads: int
    n_kv_heads: int
    dtype_mode: str         # "fp16", "bf16", "int8", "mixed"

    # Quantization
    quant_mode: str         # e.g. "q4_k_m", "q1_0_g128", "fp16"

    # PHANTOM version (our fork tag)
    phantom_version: str

    @property
    def hex_digest(self) -> str:
        """Short hex digest for cache directory naming."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "KernelFingerprint":
        return cls(**json.loads(s))


def detect_fingerprint(
    model_family: str = "unknown",
    head_dim: int = 0,
    n_heads: int = 0,
    n_kv_heads: int = 0,
    dtype_mode: str = "fp16",
    quant_mode: str = "unknown",
) -> KernelFingerprint:
    """Auto-detect runtime environment and build a KernelFingerprint.

    Model-specific fields must be passed in; hardware/software fields
    are auto-detected from the environment.
    """
    import sys

    # GPU arch
    gpu_arch = "unknown"
    wave_size = 32
    try:
        import torch
        if hasattr(torch.version, "hip") and torch.version.hip:
            props = torch.cuda.get_device_properties(0)
            gcn = getattr(props, "gcnArchName", "")
            if gcn:
                gpu_arch = gcn.split(":")[0]  # "gfx1030:sramecc+..." → "gfx1030"
            # RDNA2/3 = wave32, CDNA = wave64
            wave_size = 32 if "gfx10" in gpu_arch or "gfx11" in gpu_arch else 64
    except Exception:
        pass

    # ROCm version
    rocm_version = "unknown"
    try:
        import torch
        if hasattr(torch.version, "hip") and torch.version.hip:
            rocm_version = torch.version.hip
    except Exception:
        pass

    # Triton version
    triton_version = "unknown"
    try:
        import triton
        triton_version = getattr(triton, "__version__", "unknown")
    except ImportError:
        pass

    # Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # PHANTOM version from git or env
    phantom_version = os.environ.get("PHANTOM_VERSION", "dev")

    fp = KernelFingerprint(
        gpu_arch=gpu_arch,
        wave_size=wave_size,
        rocm_version=rocm_version,
        triton_version=triton_version,
        python_version=python_version,
        model_family=model_family,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dtype_mode=dtype_mode,
        quant_mode=quant_mode,
        phantom_version=phantom_version,
    )
    logger.info("PHANTOM-SOK: fingerprint %s (arch=%s, rocm=%s, triton=%s)",
                fp.hex_digest, gpu_arch, rocm_version, triton_version)
    return fp
