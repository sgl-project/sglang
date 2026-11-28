"""ROCm part of the inkling override declarations: the gfx950 and MegaMoE guards.
inkling calls it on HIP; it is pure Python and never imports aiter."""

from typing import Any

from sglang.srt.runtime_context import get_platform


def validate(cfg: Any) -> None:
    """The Inkling2 checks of ROCm."""
    # its kernels are sized and tuned for gfx950
    if not get_platform().is_gfx95:
        raise ValueError("Inkling2 on ROCm requires gfx950 (MI350X / MI355X)")
    if "megamoe" in (cfg.moe_a2a_backend, cfg.moe_runner_backend):
        raise ValueError(
            "Inkling2's MegaMoE runs DeepGEMM kernels over CUDA symmetric "
            "memory; on ROCm its experts run their own kernels with an "
            "expert-parallel all-reduce. Drop --moe-a2a-backend / "
            "--moe-runner-backend megamoe."
        )
