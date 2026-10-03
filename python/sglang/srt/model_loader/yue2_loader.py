# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loader for the YuE2 AR-NAR music model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from sglang.srt.models.yue2.modeling_vae import YuE2VAE, YuE2VAEConfig
from sglang.srt.models.yue2.modeling_yue2 import YuE2ForCausalLM
from sglang.srt.models.yue2.tokenization_yue2 import YuE2TextTokenizer


@dataclass
class Yue2Runtime:
    model_dir: Path
    vae_dir: Path
    device: torch.device
    model: YuE2ForCausalLM
    vae: YuE2VAE
    tokenizer: YuE2TextTokenizer

    @classmethod
    def from_paths(
        cls,
        model_dir: str | Path,
        vae_dir: str | Path,
        device: torch.device | None = None,
    ) -> "Yue2Runtime":
        model_dir, vae_dir = Path(model_dir), Path(vae_dir)
        device = device or torch.device("cuda", torch.cuda.current_device())

        model = YuE2ForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()

        vae_config = YuE2VAEConfig.from_pretrained(str(vae_dir), local_files_only=True)
        vae = YuE2VAE(vae_config)
        from safetensors.torch import load_file

        state = load_file(str(vae_dir / "model.safetensors"))
        vae.load_state_dict(state, strict=True)
        vae.to(device=device, dtype=torch.float32).eval()

        tokenizer = YuE2TextTokenizer(model_dir / "qwen.tiktoken")
        return cls(model_dir, vae_dir, device, model, vae, tokenizer)
