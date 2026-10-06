# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 video super-resolution (VSR) DiT architecture config.

Field names mirror the ``transformer/config.json`` of the official Diffusers
``Kandinsky6SRTransformer3DModel`` (for example ``kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers``):
every keyword of the constructor sits at the top level, next to ``attribute_overrides``
(post-load behaviour overrides) and the nested ``sr_params`` (SR sampling parameters,
expanded into the ``sr_*`` fields below).  ``out_visual_dim`` is the *total* head width:
a flow-matching checkpoint stores the latent width there, a DX / pi-Flow (distilled)
checkpoint stores ``base_dim * n_grid`` and keeps ``n_grid`` and the other pi-Flow values
in ``scheduler/scheduler_config.json`` (``PiflowScheduler``).

Unknown keys are rejected: ``ModelConfig.update_model_arch`` would otherwise park
them silently in ``arch_config.extra_attrs``.  Configurations that this port does
not implement (text-conditioned DiT, video adapter, LQ modulation) raise
``NotImplementedError`` naming the flag instead of building a model that would
load only partially.
"""

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig

WIDE_INPUT_INSTRUCT_TYPES = ("channel", "hybrid", "hybrid_anchor")
SUPPORTED_INSTRUCT_TYPES = ("noise", *WIDE_INPUT_INSTRUCT_TYPES)
# Post-load behaviour overrides the port accepts (no blind ``setattr``).
ATTRIBUTE_OVERRIDE_WHITELIST = ("instruct_type", "visual_cond", "attention_params")
# Bookkeeping keys the loader leaves in the component config.
_TOLERATED_EXTRA_KEYS = frozenset({"_class_name", "_diffusers_version"})
_SPARSE_ATTENTION_TYPES = ("nabla", "nabla_framewise_causal")
_PIFLOW_FIELDS = (
    "piflow_nfe",
    "piflow_num_policy_substeps",
    "piflow_final_step_size_scale",
    "piflow_shift",
    "piflow_eps",
)
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

    # --- DiffusionTransformer3D constructor keys (trained architecture) ---
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
    # The reference default is True; the SR release is text-free and its config says
    # so explicitly.
    use_text: bool = False
    use_lq_noise_cond: bool = False

    # --- optional legacy / test-only DX head / pi-Flow fields ---
    # An official checkpoint stores the TOTAL head width in ``out_visual_dim`` and the
    # pi-Flow values in its scheduler config; with ``n_grid > 1`` here the head is
    # ``out_visual_dim * n_grid`` wide instead (``out_visual_dim`` being the base dim).
    n_grid: int = 1
    piflow_nfe: int | None = None
    piflow_num_policy_substeps: int | None = None
    piflow_final_step_size_scale: float | None = None
    piflow_shift: float | None = None
    piflow_eps: float | None = None

    # Post-load attribute overrides (whitelist: instruct_type, visual_cond,
    # attention_params); the architecture above stays as trained.  The official
    # config writes ``null`` when there are none.
    attribute_overrides: dict[str, Any] | None = field(default_factory=dict)

    # --- SR sampling parameters of the training config ---
    # As stored in ``transformer/config.json`` (``sr_params``); expanded into the
    # flat ``sr_*`` fields below in ``__post_init__``.
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
            self._expand_sr_params(self.sr_params)
        self._reject_unknown_keys()
        self._reject_unsupported_flags()
        self._normalize_and_validate_shapes()
        self._validate_overrides()
        self._validate_sampler_fields()
        self.hidden_size = self.model_dim
        self.num_attention_heads = self.model_dim // sum(self.axes_dims)
        self.in_channels = self.in_visual_dim
        self.out_channels = self.out_visual_dim
        self.num_channels_latents = self.in_visual_dim

    @property
    def is_piflow(self) -> bool:
        return self.piflow_nfe is not None

    @property
    def head_width(self) -> int:
        """Channels the DiT head emits per latent position (all grids of a DX head)."""
        return self.out_visual_dim * self.n_grid

    @property
    def base_out_visual_dim(self) -> int:
        """Channels of one sampled latent grid.

        An official checkpoint stores the TOTAL head width in ``out_visual_dim``
        (``n_grid`` comes from the scheduler config), so the sampled width is the input
        dim; with a legacy ``n_grid > 1`` the base dim is ``out_visual_dim``.
        """
        return self.out_visual_dim if self.n_grid > 1 else self.in_visual_dim

    @property
    def trained_wide_input(self) -> bool:
        """Whether the input layer was trained on ``[x | cond | mask]`` channels."""
        return bool(self.visual_cond or self.instruct_type in WIDE_INPUT_INSTRUCT_TYPES)

    def effective_override(self, name: str) -> Any:
        """Post-load value of ``name``: the override if present, else as trained."""
        if name in self.attribute_overrides:
            return self.attribute_overrides[name]
        return getattr(self, name)

    def requested_sparse_attention(self) -> str | None:
        """Sparse attention type asked for by ``attention_params``, if any."""
        params = self.effective_override("attention_params")
        if not isinstance(params, dict):
            return None
        for value in params.values():
            attention_type = value.get("type") if isinstance(value, dict) else None
            if attention_type in _SPARSE_ATTENTION_TYPES:
                return attention_type
        return None

    _SR_PARAM_FIELDS = {
        "visual_size": "sr_visual_size",
        "scale_factor": "sr_scale_factor",
        "scheduler_scale": "sr_scheduler_scale",
        "lq_noise_scale": "sr_lq_noise_scale",
        "lq_noise_type": "sr_lq_noise_type",
        "lq_channel_noise_scale": "sr_lq_channel_noise_scale",
        "cap_noise_timestep": "sr_cap_noise_timestep",
        "fps": "sr_fps",
    }

    def _expand_sr_params(self, sr_params: dict[str, Any]) -> None:
        """Nested ``sr_params`` of ``transformer/config.json`` -> the flat ``sr_*`` fields."""
        unknown = sorted(set(sr_params) - set(self._SR_PARAM_FIELDS))
        if unknown:
            raise ValueError(
                f"Kandinsky6SR sr_params has unsupported keys {unknown}; accepted "
                f"keys: {sorted(self._SR_PARAM_FIELDS)}"
            )
        for key, value in sr_params.items():
            setattr(self, self._SR_PARAM_FIELDS[key], value)

    def _reject_unknown_keys(self) -> None:
        unknown = sorted(set(self.extra_attrs) - _TOLERATED_EXTRA_KEYS)
        if unknown:
            raise ValueError(
                "Kandinsky6SR transformer config has unknown keys "
                f"{unknown}; declare them on Kandinsky6SRArchConfig or fix the checkpoint config."
            )

    def _reject_unsupported_flags(self) -> None:
        if self.use_text:
            raise NotImplementedError(
                "Kandinsky6SR: use_text=True (text-conditioned SR DiT) is not "
                "supported; only the text-free SR DiT (use_text=False) is implemented."
            )
        if self.use_adapter:
            raise NotImplementedError(
                "Kandinsky6SR: use_adapter=True (VideoAdapter) is not supported."
            )
        if self.use_lq_modulation:
            raise NotImplementedError(
                "Kandinsky6SR: use_lq_modulation=True (per-block LQ modulation) is "
                "not supported."
            )

    def _normalize_and_validate_shapes(self) -> None:
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
        if self.n_grid < 1:
            raise ValueError(f"n_grid must be >= 1, got {self.n_grid}")

    def _validate_overrides(self) -> None:
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

    def _validate_sampler_fields(self) -> None:
        present = [name for name in _PIFLOW_FIELDS if getattr(self, name) is not None]
        if present and len(present) != len(_PIFLOW_FIELDS):
            missing = sorted(set(_PIFLOW_FIELDS) - set(present))
            raise ValueError(
                f"pi-Flow config is incomplete: {missing} missing next to {present}"
            )
        if self.sr_lq_noise_type not in _LQ_NOISE_TYPES:
            raise ValueError(
                f"sr_lq_noise_type must be one of {list(_LQ_NOISE_TYPES)}, "
                f"got {self.sr_lq_noise_type!r}"
            )


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
