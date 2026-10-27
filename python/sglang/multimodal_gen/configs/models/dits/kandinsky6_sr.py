# SPDX-License-Identifier: Apache-2.0
"""SR checkpoint architecture and sampling metadata.

Official out_visual_dim is the total DX head width; n_grid lives in the scheduler.
Unsupported architecture options and unknown fields are rejected."""

from dataclasses import dataclass, field, fields
from typing import Any

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig

WIDE_INPUT_INSTRUCT_TYPES = ("channel", "hybrid", "hybrid_anchor")
SUPPORTED_INSTRUCT_TYPES = ("noise", *WIDE_INPUT_INSTRUCT_TYPES)
# Post-load behaviour overrides the port accepts (no blind ``setattr``).
ATTRIBUTE_OVERRIDE_WHITELIST = ("instruct_type", "visual_cond", "attention_params")
# Bookkeeping keys the loader leaves in the component config.
_TOLERATED_EXTRA_KEYS = frozenset({"_class_name", "_diffusers_version"})
_SPARSE_ATTENTION_TYPES = ("nabla", "nabla_framewise_causal")
_LQ_NOISE_TYPES = ("ddpm", "linear")


@dataclass
class Kandinsky6SRArchConfig(DiTArchConfig):
    """Static architecture metadata of ``Kandinsky6SRTransformer3DModel``."""

    # Map the current Diffusers checkpoint names to the native SGLang port.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^(.*feed_forward)\.net\.0\.proj\.(weight|bias)$": r"\1.mlp.fc_in.\2",
            r"^(.*feed_forward)\.net\.2\.(weight|bias)$": r"\1.mlp.fc_out.\2",
            r"^time_embeddings\.timestep_embedder\.linear_1\.(weight|bias)$": r"time_embeddings.in_layer.\1",
            r"^time_embeddings\.timestep_embedder\.linear_2\.(weight|bias)$": r"time_embeddings.out_layer.\1",
        }
    )

    # trained architecture, before inference overrides
    in_visual_dim: int = 4
    in_text_dim: int = 3584
    in_text_dim2: int = 768
    time_dim: int = 512
    out_visual_dim: int = 4
    patch_size: tuple[int, int, int] = (1, 2, 2)
    model_dim: int = 2048
    ff_dim: int = 5120
    num_text_blocks: int = 2
    num_visual_blocks: int = 32
    axes_dims: tuple[int, int, int] = (16, 24, 24)
    visual_cond: bool = False
    instruct_type: str | None = None
    attention_params: dict[str, Any] | None = None
    use_motion_score: bool = False
    use_lq_modulation: bool = False
    lq_modulation_sublayers: dict[str, bool] | None = None
    zero_lq_in_main_path: bool = False
    use_adapter: bool = False
    adapter_gamma: float = 0.0
    use_text: bool = False
    use_lq_noise_cond: bool = False

    # inference-only overrides must not change the trained layer dimensions
    attribute_overrides: dict[str, Any] | None = field(default_factory=dict)

    # checkpoint sr_params expands into the flat sr_* fields
    sr_params: dict[str, Any] | None = None
    sr_visual_size: list[int] = field(default_factory=lambda: [512])
    sr_scale_factor: dict[str, list[float]] | None = None
    sr_scheduler_scale: float = 5.0
    sr_lq_noise_scale: float = 0.7
    sr_lq_noise_type: str = "ddpm"
    sr_lq_channel_noise_scale: float = 0.0
    sr_cap_noise_timestep: bool = False
    sr_fps: int = 24

    # DiTArchConfig contract fields, derived in __post_init__.
    in_channels: int = 0
    out_channels: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attribute_overrides is None:
            self.attribute_overrides = {}
        if self.sr_params is not None:
            known = {
                item.name.removeprefix("sr_")
                for item in fields(self)
                if item.name.startswith("sr_") and item.name != "sr_params"
            }
            unknown = sorted(set(self.sr_params) - known)
            if unknown:
                raise ValueError(
                    f"Kandinsky6SR sr_params has unsupported keys {unknown}; accepted keys: {sorted(known)}"
                )
            for key, value in self.sr_params.items():
                setattr(self, f"sr_{key}", value)
        unknown = sorted(set(self.extra_attrs) - _TOLERATED_EXTRA_KEYS)
        if unknown:
            raise ValueError(
                "Kandinsky6SR transformer config has unknown keys "
                f"{unknown}; declare them on Kandinsky6SRArchConfig or fix the checkpoint config."
            )

        for name, enabled in (
            ("use_text", self.use_text),
            ("use_adapter", self.use_adapter),
            ("use_lq_modulation", self.use_lq_modulation),
        ):
            if enabled:
                raise NotImplementedError(
                    f"Kandinsky6SR: {name}=True is not supported."
                )
        self.patch_size = tuple(self.patch_size)
        self.axes_dims = tuple(self.axes_dims)
        if len(self.patch_size) != 3 or self.patch_size[0] != 1:
            raise ValueError(
                "Kandinsky6SR patch_size must be (1, ph, pw) with a unit temporal "
                f"patch, got {self.patch_size}"
            )
        if len(self.axes_dims) != 3 or any(dim % 2 for dim in self.axes_dims):
            raise ValueError(
                f"axes_dims must be three even sizes, got {self.axes_dims}"
            )
        head_dim = sum(self.axes_dims)
        if self.model_dim % head_dim != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by head_dim "
                f"({head_dim}) = sum(axes_dims)"
            )
        if self.model_dim % 2 != 0:
            raise ValueError(f"model_dim must be even, got {self.model_dim}")

        unknown = sorted(
            set(self.attribute_overrides) - set(ATTRIBUTE_OVERRIDE_WHITELIST)
        )
        if unknown:
            raise ValueError(
                f"attribute_overrides keys {unknown} are not allowed; "
                f"supported keys: {list(ATTRIBUTE_OVERRIDE_WHITELIST)}"
            )
        instruct_type = self.effective_override("instruct_type")
        if instruct_type not in (None, *SUPPORTED_INSTRUCT_TYPES):
            raise ValueError(
                f"instruct_type must be one of {list(SUPPORTED_INSTRUCT_TYPES)}, "
                f"got {instruct_type!r}"
            )
        visual_cond = self.effective_override("visual_cond")
        if not isinstance(visual_cond, bool):
            raise ValueError(f"visual_cond must be a bool, got {visual_cond!r}")
        if self.sr_lq_noise_type not in _LQ_NOISE_TYPES:
            raise ValueError(
                f"sr_lq_noise_type must be one of {list(_LQ_NOISE_TYPES)}, "
                f"got {self.sr_lq_noise_type!r}"
            )
        self.hidden_size = self.model_dim
        self.num_attention_heads = self.model_dim // head_dim
        self.in_channels = self.in_visual_dim
        self.out_channels = self.out_visual_dim
        self.num_channels_latents = self.in_visual_dim

    @property
    def trained_wide_input(self) -> bool:
        """Whether the input layer was trained on ``[x | cond | mask]`` channels."""
        return bool(self.visual_cond or self.instruct_type in WIDE_INPUT_INSTRUCT_TYPES)

    def effective_override(self, name: str) -> Any:
        """Post-load value of ``name``: the override if present, else as trained."""
        return self.attribute_overrides.get(name, self.__dict__[name])

    def requested_sparse_attention(self) -> str | None:
        """Sparse attention type asked for by ``attention_params``, if any."""
        params = self.effective_override("attention_params")
        if isinstance(params, dict):
            for value in params.values():
                if (
                    isinstance(value, dict)
                    and value.get("type") in _SPARSE_ATTENTION_TYPES
                ):
                    return value["type"]
        return None


@dataclass
class Kandinsky6SRDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=Kandinsky6SRArchConfig)
    prefix: str = "Kandinsky6SR"


__all__ = [
    "ATTRIBUTE_OVERRIDE_WHITELIST",
    "Kandinsky6SRArchConfig",
    "Kandinsky6SRDitConfig",
    "SUPPORTED_INSTRUCT_TYPES",
    "WIDE_INPUT_INSTRUCT_TYPES",
]
