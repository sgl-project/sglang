"""Device identity — the coarse key's hardware half.

Reuses SGLang's canonical device key (``get_device_name().replace(" ", "_")``) so
Attune's per-device files are byte-compatible with the existing MoE tuned-config
ecosystem. In ``--mock`` mode the device is supplied by flag/env so the whole flow
runs with no GPU.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Optional


def _driver_version() -> str:  # pragma: no cover - real GPU only
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlSystemGetDriverVersion()
    except Exception:
        return "unknown"


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    name: str  # canonical, e.g. "NVIDIA_H100_80GB_HBM3"
    sm: int  # major*10 + minor, e.g. 90 for Hopper, 100 Blackwell, 120 RTX 5090
    is_hip: bool = False
    is_mps: bool = False
    # --- extended fingerprint inputs (guardrail #1) --------------------------------
    # A thermal-/power-limited H20 in a dense rack crosses over at a different
    # batch/seqlen point than a fully-powered one, and (per #31310) CUDA/driver combos
    # drastically alter kernel performance. These feed the LOCAL cache fingerprint hash,
    # NOT the committed-corpus filename key (which stays coarse and human-readable).
    cuda_version: str = "unknown"
    driver_version: str = "unknown"
    sm_clock_max_mhz: int = 0  # max supported SM clock
    sm_clock_limit_mhz: int = (
        0  # currently-applied/locked SM clock (changes when throttled)
    )
    power_limit_mw: int = (
        0  # applied power cap in milliwatts (changes with `nvidia-smi -pl`)
    )
    pcie_gen: int = 0
    pcie_width: int = 0  # lanes, e.g. 16
    l2_cache_bytes: int = 0

    @property
    def sm_tag(self) -> str:
        return f"sm{self.sm}"

    def fingerprint_inputs(self) -> dict:
        """The hardware half of the local-cache fingerprint. Kept separate from the
        coarse filename key so the same committed corpus can serve many racked units,
        while a throttled or CUDA-bumped box gets its own local re-tune."""
        return {
            "name": self.name,
            "sm": self.sm,
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
            "sm_clock_max_mhz": self.sm_clock_max_mhz,
            "sm_clock_limit_mhz": self.sm_clock_limit_mhz,
            "power_limit_mw": self.power_limit_mw,
            "pcie_gen": self.pcie_gen,
            "pcie_width": self.pcie_width,
        }


def _canonical(name: str) -> str:
    # Mirror SGLang platform_utils get_device_name_as_file_name: collapse whitespace/slashes.
    return re.sub(r"[\s/]+", "_", name.strip())


def detect_device(
    mock_name: Optional[str] = None, mock_sm: Optional[int] = None
) -> DeviceInfo:
    """Return the current device, or a mock one for CPU-only runs.

    Real path (guarded): torch.cuda.get_device_name / get_device_capability, plus
    torch.version.hip for ROCm. Mock path: env SGLANG_ATTUNE_MOCK_DEVICE / _SM or args.
    """
    mock_name = mock_name or os.environ.get("SGLANG_ATTUNE_MOCK_DEVICE")
    if mock_name is not None:
        sm = (
            mock_sm
            if mock_sm is not None
            else int(os.environ.get("SGLANG_ATTUNE_MOCK_SM", "90"))
        )
        return DeviceInfo(
            name=_canonical(mock_name),
            sm=sm,
            is_hip=False,
            cuda_version=os.environ.get("SGLANG_ATTUNE_MOCK_CUDA", "12.4"),
            driver_version=os.environ.get("SGLANG_ATTUNE_MOCK_DRIVER", "550.00"),
            sm_clock_max_mhz=int(os.environ.get("SGLANG_ATTUNE_MOCK_SMCLK", "1980")),
            pcie_gen=int(os.environ.get("SGLANG_ATTUNE_MOCK_PCIE_GEN", "5")),
            pcie_width=int(os.environ.get("SGLANG_ATTUNE_MOCK_PCIE_WIDTH", "16")),
            l2_cache_bytes=50 * 1024 * 1024,
        )

    try:  # pragma: no cover - exercised only on real GPUs
        import torch

        if getattr(torch.version, "hip", None):
            name = torch.cuda.get_device_name(0)
            return DeviceInfo(name=_canonical(name), sm=0, is_hip=True)
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        # Best-effort extended fingerprint; pynvml gives clock/pcie when present.
        sm_clk = sm_clk_limit = pwr_limit = pcie_gen = pcie_w = 0
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            sm_clk = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
            # Applied/locked clock and power cap — these MOVE when the GPU is throttled,
            # so the fingerprint re-tunes a power-limited box (guardrail #1).
            try:
                mn, mx = (
                    pynvml.nvmlDeviceGetGpcClkVfOffset(h),
                    None,
                )  # best-effort; not all drivers
            except Exception:
                pass
            try:
                sm_clk_limit = pynvml.nvmlDeviceGetApplicationsClock(
                    h, pynvml.NVML_CLOCK_SM
                )
            except Exception:
                sm_clk_limit = sm_clk
            try:
                pwr_limit = pynvml.nvmlDeviceGetPowerManagementLimit(h)  # milliwatts
            except Exception:
                pass
            pcie_gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(h)
            pcie_w = pynvml.nvmlDeviceGetMaxPcieLinkWidth(h)
        except Exception:
            pass
        return DeviceInfo(
            name=_canonical(name),
            sm=major * 10 + minor,
            is_hip=False,
            cuda_version=str(torch.version.cuda),
            driver_version=_driver_version(),
            sm_clock_max_mhz=int(sm_clk),
            sm_clock_limit_mhz=int(sm_clk_limit),
            power_limit_mw=int(pwr_limit),
            pcie_gen=int(pcie_gen),
            pcie_width=int(pcie_w),
            l2_cache_bytes=int(getattr(props, "l2_cache_size", 0)),
        )
    except Exception as e:  # no torch / no CUDA -> force mock
        raise RuntimeError(
            "No CUDA device detected. Pass --mock-device NAME (and --mock-sm N) to run "
            f"the Attune flow without a GPU. Underlying error: {e}"
        )
