# SPDX-License-Identifier: Apache-2.0
"""ComfyUI Qwen-Image checkpoint specs (text-to-image and edit)."""

from collections.abc import Callable

from sglang.multimodal_gen.configs.models.dits.qwenimage import (
    QwenImageArchConfig,
    QwenImageDitConfig,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoint import (
    ComfyUICheckpointSpec,
    register_comfyui_checkpoint,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _dit_config_builder(zero_cond_t: bool) -> Callable[[ServerArgs], QwenImageDitConfig]:
    def build(server_args: ServerArgs) -> QwenImageDitConfig:
        # ComfyUI checkpoints carry no config, so the architecture is pinned here.
        dit_config = QwenImageDitConfig(
            arch_config=QwenImageArchConfig(
                patch_size=2,
                in_channels=64,
                out_channels=16,
                num_layers=60,
                attention_head_dim=128,
                num_attention_heads=24,
                joint_attention_dim=3584,
                pooled_projection_dim=768,
                guidance_embeds=False,
                axes_dims_rope=(16, 56, 56),
                zero_cond_t=zero_cond_t,
            )
        )
        server_args.pipeline_config.dit_config = dit_config
        return dit_config

    return build


_PARAM_NAMES_MAPPING = {r"^model\.diffusion_model\.(.*)$": (r"\1", None, None)}


for _pipeline_name, _zero_cond_t in (
    ("QwenImagePipeline", False),
    ("QwenImageEditPlusPipeline", True),
):
    register_comfyui_checkpoint(
        _pipeline_name,
        ComfyUICheckpointSpec(
            dit_cls_name="QwenImageTransformer2DModel",
            build_dit_config=_dit_config_builder(_zero_cond_t),
            param_names_mapping=_PARAM_NAMES_MAPPING,
        ),
    )
