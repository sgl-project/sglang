# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 (TI2VA) joint video+audio DiT architecture config.

Field names mirror the diffusers reference's ``Kandinsky6Transformer3DModel``
``transformer/config.json`` 1:1. The official Kandinsky-6.0-Pro-sft-5s and
Kandinsky-6.0-Pro-distill-5s ``-Diffusers`` checkpoints differ in the pi-Flow DX
head width: ``out_visual_dim`` / ``out_audio_dim`` are 16 / 40 for Pro-sft and
160 / 400 (x ``PiflowScheduler.n_grid`` = 10) for Pro-distill.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class Kandinsky6ArchConfig(DiTArchConfig):
    """Static architecture metadata for ``Kandinsky6Transformer3DModel``.

    ``__post_init__`` derives the ``DiTArchConfig`` contract fields
    (``hidden_size``, ``num_attention_heads``, ``num_channels_latents``,
    plus this model's own ``in_channels``/``out_channels``) and resolves the
    audio-tower dims that default to matching their video-tower
    counterparts, mirroring the diffusers reference's inline ``x or
    default`` resolution.

    The runtime DiT model class (``runtime/models/dits/kandinsky6.py``, not
    part of this config-layer port) is where the FSDP shard-condition
    predicate belongs, matching the ``MiniMaxH3``/``MOVA`` convention of
    keeping ``_fsdp_shard_conditions`` on the model class rather than the
    arch config: this model has two block stacks, ``text_transformer_blocks``
    (x ``num_text_blocks``) and ``visual_transformer_blocks`` (x
    ``num_visual_blocks``), matched together via
    ``sglang.multimodal_gen.configs.models.fsdp.is_module_list_entry_in(
    name, ("text_transformer_blocks", "visual_transformer_blocks"))``.
    """

    # Chained layer renames apply to checkpoint tensors and LoRA A/B/alpha keys.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^(visual_transformer_blocks\.\d+)\.video_dec_block\.(.*)$": r"\1.videoT.\2",
            r"^(visual_transformer_blocks\.\d+)\.audio_dec_block\.(.*)$": r"\1.audioT.\2",
            r"^(.*feed_forward)\.net\.0\.proj\.(.*)$": r"\1.mlp.fc_in.\2",
            r"^(.*feed_forward)\.net\.2\.(.*)$": r"\1.mlp.fc_out.\2",
            r"^(video_time_embeddings|audio_time_embeddings)\.timestep_embedder\.linear_1\.(.*)$": r"\1.in_layer.\2",
            r"^(video_time_embeddings|audio_time_embeddings)\.timestep_embedder\.linear_2\.(.*)$": r"\1.out_layer.\2",
            r"^((?:video|audio)_text_transformer_blocks\.\d+)\.attn_norm\.(.*)$": r"\1.self_attention_norm.\2",
            r"^((?:video|audio)_text_transformer_blocks\.\d+)\.attn\.(.*)$": r"\1.self_attention.\2",
        }
    )
    lora_param_names_mapping: dict = field(
        default_factory=lambda: {r"^transformer\.(.*)$": r"\1"}
    )

    # Diffusers Kandinsky6Transformer3DModel config fields (mirror
    # transformer/config.json 1:1).
    in_visual_dim: int = 16
    out_visual_dim: int = 16
    in_text_dim: int = 3584  # Qwen2.5-7B hidden size (Reason1 text encoder)
    in_text_dim2: int = 768  # CLIP pooled dim
    time_dim: int = 1024
    patch_size: tuple[int, int, int] = (1, 2, 2)
    # Per-axis (T, H, W) RoPE frequency scaling, read once from the
    # checkpoint's transformer/config.json (both real Pro checkpoints ship
    # [1.0, 2.0, 2.0]) -- matches the diffusers reference's
    # ``Kandinsky6TI2VAPipeline.__init__``, which resolves this the same way
    # (``transformer_config.get("scale_factor", (1.0, 2.0, 2.0))``) and reuses
    # it for every request regardless of the request's own height/width.
    scale_factor: tuple[float, float, float] = (1.0, 2.0, 2.0)
    model_dim: int = 4096
    ff_dim: int = 16384
    num_text_blocks: int = 4
    num_visual_blocks: int = 60
    axes_dims: tuple[int, int, int] = (32, 48, 48)  # 3D RoPE T/H/W split
    visual_cond: bool = True

    # T2VA/IT2VA joint video+audio generation. False reproduces the plain
    # Kandinsky5-parity T2V/I2V architecture (single text/time tower, no
    # audio stream) -- the TI2VA pipeline always sets this True.
    is_multimodal: bool = True
    out_audio_dim: int | None = None
    in_audio_dim: int = 20
    model_dim_a: int | None = None
    time_dim_a: int | None = None
    ff_dim_a: int | None = None
    axes_dims_a: tuple[int, int, int] | None = None
    audio_freqs_scaling: float = 1.0

    # Real diffusers Kandinsky6Transformer3DModel field name (mirrors
    # transformer/config.json's "attention_engine" key exactly, e.g. "auto"
    # or "sdpa"; the official Pro configs leave it out, so it stays "auto").
    # Kept as a plain string (not an attention-backend enum) for
    # config-file/HF checkpoint round-tripping. Only "nabla" is treated
    # specially: it selects NABLA block-sparse attention for the video
    # self-attention sub-layer only (audio self-attention and the
    # video<->audio cross-attention always stay dense). The runtime DiT has
    # no NABLA backend and raises NotImplementedError for it at construction;
    # dense/SDPA attention is the only supported path.
    attention_engine: str = "auto"
    attention_causal: bool | None = None
    attention_local: bool | None = None
    attention_glob: bool | None = None
    attention_window: int | None = None
    attention_P: float | None = None
    attention_wT: int | None = None
    attention_wW: int | None = None
    attention_wH: int | None = None
    attention_add_sta: bool | None = None
    attention_method: str | None = None

    # Real diffusers field; recognized here so `update_model_arch` doesn't
    # silently drop it (both official Pro checkpoints set this True). NOT
    # currently implemented by the runtime port: no attention-mask plumbing
    # exists for this DiT (matching Kandinsky5's convention of only ever
    # feeding already-unpadded/gathered text tokens), so a checkpoint that
    # actually relies on padded-with-mask text cross-attention would get
    # silently wrong behavior, not an error.
    text_token_padding: bool = False

    # Video<->audio fused block knobs (Kandinsky6FusedTransformerDecoderBlock).
    ca_rope: bool = False
    cross_gates: bool = False
    fix_modulation: bool = False
    # >0 adds a learned embedding distinguishing generated vs. reference
    # visual tokens, consumed by IT2VA's tail_cond_first_frame conditioning.
    # Defaults to 2 (generated/reference) rather than diffusers' bare-DiT
    # default of 0: one merged pipeline serves both T2VA (no image -- layer
    # exists but unused) and IT2VA (image -- tail_cond_first_frame needs it)
    # from the same checkpoint/config.
    visual_token_type_num_embeddings: int = 2

    # Derived, DiTArchConfig-contract fields (set in __post_init__ below;
    # not part of the diffusers config.json).
    in_channels: int = 0
    out_channels: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have 3 values, got {self.patch_size}.")
        if isinstance(self.scale_factor, list):
            self.scale_factor = tuple(self.scale_factor)
        if len(self.scale_factor) != 3:
            raise ValueError(
                f"scale_factor must have 3 values, got {self.scale_factor}."
            )
        self.scale_factor = tuple(float(value) for value in self.scale_factor)

        head_dim = sum(self.axes_dims)
        if self.model_dim % head_dim != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by head_dim "
                f"({head_dim})"
            )
        self.hidden_size = self.model_dim
        self.num_attention_heads = self.model_dim // head_dim
        self.in_channels = self.in_visual_dim
        self.out_channels = self.out_visual_dim
        self.num_channels_latents = self.in_visual_dim

        # Resolve the audio-tower dims once here (mirrors the `x or default`
        # resolution the diffusers reference's Kandinsky6Transformer3DModel
        # .__init__ does inline) so the model constructor can read them
        # unconditionally.
        self.model_dim_a = self.model_dim_a or self.model_dim
        self.time_dim_a = self.time_dim_a or self.time_dim
        self.ff_dim_a = self.ff_dim_a or self.ff_dim
        self.axes_dims_a = self.axes_dims_a or self.axes_dims
        head_dim_a = sum(self.axes_dims_a)
        if self.model_dim_a % head_dim_a != 0:
            raise ValueError(
                f"model_dim_a ({self.model_dim_a}) must be divisible by "
                f"head_dim_a ({head_dim_a})"
            )

        # Kandinsky6FusedTransformerDecoderBlock's cross-modal modulation is
        # driven by the *other* modality's time embedding by default
        # (fix_modulation=False, matching the diffusers reference): the
        # video-conditioning-on-audio modulation is built from time_dim but
        # invoked with the audio time embedding, and vice versa. That only
        # type-checks when the two time embeddings are the same width, so
        # reject the mismatched, fix_modulation=False combination here with
        # a clear message instead of a cryptic matmul shape error deep
        # inside the fused block.
        if (
            self.is_multimodal
            and not self.fix_modulation
            and self.time_dim_a != self.time_dim
        ):
            raise ValueError(
                f"time_dim_a ({self.time_dim_a}) must equal time_dim "
                f"({self.time_dim}) unless fix_modulation=True "
                "(Kandinsky6FusedTransformerDecoderBlock's cross-modal "
                "modulation is driven by the other modality's time "
                "embedding by default)."
            )


@dataclass
class Kandinsky6VideoAudioConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=Kandinsky6ArchConfig)
    prefix: str = "Kandinsky6"


__all__ = ["Kandinsky6ArchConfig", "Kandinsky6VideoAudioConfig"]
