# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 sampling params: the 8-NFE grid on the MiniMax-H3 request surface."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams


@dataclass
class VDNH3SamplingParams(MiniMaxH3SamplingParams):
    """VDN-H3: nine sigma grid points, i.e. the eight distilled DiT forwards
    The turbo adapter is only valid at 8 NFE with video shift 12 / audio
    shift 3 (the defaults)."""

    num_inference_steps: int = 8

    def _validate(self) -> None:
        super()._validate()
        if self.num_inference_steps != 8:
            raise ValueError(
                "VDN-H3 is distilled for exactly eight inference steps (nine sigma "
                f"grid points); got num_inference_steps={self.num_inference_steps}. "
                "Use MiniMaxAI/MiniMax-H3 for other schedules."
            )
        if self.task is not None and self.task.strip().lower() not in (
            "t2va",
            "fl2va",
        ):
            raise ValueError(
                "VDN-H3 serves t2va and fl2va; ref2va was not trained (got "
                f"task={self.task!r}). Use MiniMaxAI/MiniMax-H3 --model-variant "
                "ref2va for that task."
            )


__all__ = ["VDNH3SamplingParams"]
