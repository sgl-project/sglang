# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class Yue2SamplingParams(SamplingParams):
    """YuE2 lyrics-to-audio request parameters."""

    data_type: DataType = DataType.IMAGE  # placeholder until T2A exists upstream
    ode_steps: int = 32
    vae_core_frames: int = 1024
    vae_halo_frames: int = 16
    abc_max_tokens: int = 4096
    semantic_max_tokens: int = 9000
    cot: str = "full"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 100
    repetition_penalty: float = 1.2
    style: str | None = field(default=None, metadata={"batch_sig_exclude": True})
    lyrics: str | None = field(default=None, metadata={"batch_sig_exclude": True})
    request_abc: str | None = None
    output_sample_rate: int | None = 48000

    def __post_init__(self):
        super().__post_init__()
        if self.num_inference_steps is not None:
            self.ode_steps = int(self.num_inference_steps)
