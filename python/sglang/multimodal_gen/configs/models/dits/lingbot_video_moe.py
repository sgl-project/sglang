# Adapted from: https://github.com/Robbyant/lingbot-video
# Reference (upstream): /vllm-workspace/lingbot-video/lingbot_video

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def _is_lingbot_video_block(name: str, module) -> bool:
    """FSDP shard predicate: shard each transformer block (``blocks.<digit>``).

    Module-level (not a lambda) so it is picklable across the multiprocessing
    spawn that launches GPU workers.
    """
    return "blocks" in name and str.isdigit(name.split(".")[-1])


@dataclass
class LingBotVideoMoEArchConfig(DiTArchConfig):
    """Architecture configuration for the LingBot-Video MoE 30B DiT.

    Mirrors the upstream ``transformer/config.json`` of
    ``LingBotVideoTransformer3DModel`` (the real 30B MoE text-to-video model).
    The DiT reads these fields directly (``config.<field>``), the same way
    ``LingBotWorldTransformer3DModel`` reads its config, so the field names must
    match the upstream config keys exactly.

    DeepSeek-V3-style MoE: 128 experts, top-8, 1 shared expert, every layer
    sparse (``decoder_sparse_step=1``). Single-GPU MVP scope: T2V base-only,
    batch size 1, structured-JSON captions.
    """

    # FSDP: shard each transformer block (the 48 MoE blocks dominate memory).
    # Module-level predicate (picklable across multiprocessing spawn).
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_lingbot_video_block]
    )

    # Identity mapping: the checkpoint keys already match the SGLang module
    # names verbatim (``blocks.<i>.ffn.experts.w1/w2/w3``,
    # ``blocks.<i>.ffn.router.e_score_correction_bias``,
    # ``blocks.<i>.attn.to_q/to_k/to_v/to_out``, ``patch_embedder``, ...).
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    # --- patch / channel dims ---
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 16
    out_channels: int = 16

    # --- transformer dims ---
    hidden_size: int = 2048
    num_attention_heads: int = 16
    depth: int = 48
    intermediate_size: int = 6144
    text_dim: int = 2560
    freq_dim: int = 256
    norm_eps: float = 1e-6
    rope_theta: float = 256.0
    axes_dims: tuple[int, ...] = (32, 48, 48)
    axes_lens: tuple[int, ...] = (4096, 512, 512)

    # --- projection biases (match upstream config.json) ---
    qkv_bias: bool = False
    out_bias: bool = True
    patch_embed_bias: bool = True
    timestep_mlp_bias: bool = True

    # --- MoE (DeepSeek-V3 style) ---
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    decoder_sparse_step: int = 1
    n_shared_experts: int = 1
    score_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 4
    topk_group: int = 2
    routed_scaling_factor: float = 2.5

    def __post_init__(self):
        super().__post_init__()
        # latent channels == out channels (VAE z_dim == 16)
        self.num_channels_latents = self.out_channels


@dataclass
class LingBotVideoMoEConfig(DiTConfig):
    """DiT config for the LingBot-Video MoE 30B model.

    ``prefix="LingBotVideo"`` matches the upstream ``model_index.json``
    transformer class prefix so weight loading resolves correctly.
    """

    arch_config: DiTArchConfig = field(default_factory=LingBotVideoMoEArchConfig)
    prefix: str = "LingBotVideo"
