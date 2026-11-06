# SPDX-License-Identifier: Apache-2.0
"""Architecture configuration for the BRIA Fibo 2.0 diffusion transformer.

Fibo 2.0 keeps the Z-Image transformer block and adds three things on top of it:

* a Perceiver resampler that squeezes the text encoder features into a variable
  number of "gist" tokens,
* five injection blocks that also cross-attend to their own text bundle,
* per-token (noisy/clean) adaLN modulation, which is how editing conditions the
  generation on the clean latents of the images to edit.

The parameter names emitted here are the checkpoint's own diffusers names
(``attention.to_q``, ``attention.to_out.0``, ``cross_attn_gate``,
``perceiver.latent_queries``, ...). The runtime model mirrors them one to one, so
``param_names_mapping`` stays empty and no weight reshuffle happens at load time.
In particular q/k/v are kept separate: a cross-attention block reads a
``cap_feat_dim`` (7680) wide text bundle while its query comes from a ``dim``
(3584) wide stream, so the two cannot be fused into one stacked projection.
"""

from dataclasses import dataclass, field
from typing import Any, Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class BriaFibo2ArchConfig(DiTArchConfig):
    # 2x2 latent patches become one token, features ordered (channel, dy, dx)
    all_patch_size: Tuple[int, ...] = (2,)
    patch_size: int = 2
    in_channels: int = 32
    out_channels: int | None = None

    dim: int = 3584
    num_layers: int = 44
    n_refiner_layers: int = 2
    num_attention_heads: int = 28
    norm_eps: float = 1e-5

    # Text conditioning: 3 Qwen3-VL layers of 2560 channels concatenated
    cap_feat_dim: int = 7680
    # Blocks that cross-attend to their own text bundle. Bundle 0 feeds the
    # Perceiver, bundle k + 1 feeds injection_layer_ids[k].
    injection_layer_ids: Tuple[int, ...] = (6, 14, 22, 30, 38)
    perceiver_num_layers: int = 2

    # Gist token count: interpolated from the text length, quantized to a step
    min_num_gist_tokens: int = 256
    max_num_gist_tokens: int = 512
    gist_step: int = 64
    gist_min_text_len: int = 256
    gist_max_text_len: int = 2048

    rope_theta: float = 256.0
    t_scale: float = 1000.0
    # The diffusion stages hand the transformer the scheduler's own timesteps
    # (0..num_train_timesteps), while the reference transformer takes the noise
    # level in [0, 1] and multiplies by `t_scale` itself. Dividing by the
    # scheduler's unit count bridges the two conventions without touching the
    # embedding domain the checkpoint was trained in.
    scheduler_timesteps: float = 1000.0
    # Three axes (plane, row, column); the sum is the attention head dim
    axes_dims: Tuple[int, int, int] = (32, 48, 48)
    axes_lens: Tuple[int, int, int] = (2048, 512, 512)

    # Fibo keeps q/k/v separate (see module docstring), so nothing is stacked
    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)

    # diffusers names are used as-is by the runtime model
    param_names_mapping: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.num_channels_latents = self.in_channels
        self.hidden_size = self.dim
        self.num_attention_heads = self.num_attention_heads
        if sum(self.axes_dims) != self.dim // self.num_attention_heads:
            raise ValueError(
                "axes_dims must sum to the attention head dim: "
                f"sum({self.axes_dims}) != {self.dim} // {self.num_attention_heads}"
            )


@dataclass
class BriaFibo2DitConfig(DiTConfig):
    arch_config: BriaFibo2ArchConfig = field(default_factory=BriaFibo2ArchConfig)

    prefix: str = "bria_fibo2"

    # One block serves the gist tokens (hundreds) and the image plus context
    # tokens (thousands); the gist branch is the short one, so its GEMMs should
    # not be autotuned for the long shape alone.
    torch_compile_inductor_config: dict[str, Any] = field(
        default_factory=lambda: {"multi_kernel_hints": [512, 4096]}
    )


__all__ = ["BriaFibo2ArchConfig", "BriaFibo2DitConfig"]
