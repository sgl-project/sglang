from dataclasses import dataclass

import torch


@dataclass
class QwenImageEditPlusParams:
    prompt: str
    generator: torch.Generator = torch.manual_seed(0)
    true_cfg_scale: float = 4.0
    negative_prompt: str = " "
    num_inference_steps: int = 40
    guidance_scale: float = 1.0
    num_images_per_prompt: int = 1

    max_sequence_length: int = 512
    output_type: str = None
