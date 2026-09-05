# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 sampling params: the 8-NFE t2va grid on the MiniMax-H3 request surface."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams


@dataclass
class VDNH3SamplingParams(MiniMaxH3SamplingParams):
    """VDN-H3: nine sigma grid points, i.e. the eight distilled DiT forwards
    (VDN counts NFEs; SGLang counts sigma grid points). The turbo adapter is
    only valid at 8 NFE with video shift 12 / audio shift 3 (the defaults)."""

    num_inference_steps: int = 9

    def _validate(self) -> None:
        super()._validate()
        if self.num_inference_steps != 9:
            raise ValueError(
                "VDN-H3 is distilled for exactly nine sigma grid points (eight DiT "
                f"forwards); got num_inference_steps={self.num_inference_steps}. "
                "Use MiniMaxAI/MiniMax-H3 for other schedules."
            )
        if self.task is not None and self.task.strip().lower() != "t2va":
            raise ValueError(
                "VDN-H3 is trained for t2va only; fl2va and ref2va were not "
                f"trained (got task={self.task!r}). Use MiniMaxAI/MiniMax-H3 "
                "for those tasks."
            )


__all__ = ["VDNH3SamplingParams"]
