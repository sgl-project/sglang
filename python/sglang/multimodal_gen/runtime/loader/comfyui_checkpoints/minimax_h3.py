# SPDX-License-Identifier: Apache-2.0
"""ComfyUI MiniMax-H3 checkpoint spec.

The Comfy-Org BF16 FL2VA file uses ComfyUI parameter names and already stores
fused QKV as ``[q_all, k_all, v_all]``. Identity name mapping is enough, but
the DiT's default HF weight loader must not run the grouped→concat reorder.
"""

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTConfig
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec import (
    ComfyUICheckpointSpec,
    register_comfyui_checkpoint,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _build_dit_config(server_args: ServerArgs) -> MiniMaxH3DiTConfig:
    dit_config = getattr(server_args.pipeline_config, "dit_config", None)
    if not isinstance(dit_config, MiniMaxH3DiTConfig):
        dit_config = MiniMaxH3DiTConfig()
        server_args.pipeline_config.dit_config = dit_config
    # ComfyUI / Comfy-Org files match the runtime QKV layout. Native
    # ``sglang serve`` keeps the default grouped HF reorder.
    dit_config.arch_config.qkv_checkpoint_grouped = False
    return dit_config


register_comfyui_checkpoint(
    "MiniMaxH3Pipeline",
    ComfyUICheckpointSpec(
        dit_cls_name="MiniMaxH3DiTModel",
        build_dit_config=_build_dit_config,
        inherit_config_mapping=True,
    ),
)
