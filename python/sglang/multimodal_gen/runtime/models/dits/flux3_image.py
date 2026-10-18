# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 image DiT on SGLang's native diffusion framework.

Builds on the FLUX 3 transformer in :mod:`.flux3` (shared with FLUX.3 Action):
the stream / text embedders, timestep embedding, modulation layout, RoPE table
and output heads are reused as they are. Only the blocks and the modulation
projections are swapped for implementations that serve the image checkpoints:

  * packed q/k/v -> :class:`MergedColumnParallelLinear` (``to_qkv``): one GEMM
    then fused QK-norm+RoPE.
  * gated ``mlp_in`` -> :class:`MergedColumnParallelLinear` (``[gate, value]``),
    fused SwiGLU, and native output projections. Keeping ``to_qkv`` and
    ``mlp_in`` separate is what lets the FP8 / NVFP4 PrimsTS epilogues own them.
  * packed variable-length attention -> :class:`USPAttention`, preserving CFG
    sample boundaries through the selected backend's varlen implementation.
  * fused LayerNorm modulation and residual-gate branch merge.

Tensor parallel shards heads and MLP channels with one all-reduce per block.
Ulysses sequence parallel shards each CFG sample's joint sequence (and long
content streams in the mode blocks), as the FLUX 3 transformer does.
Native fused kernels are bit-close, not bit-exact, to the torch reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sglang.kernels.ops.diffusion import residual_gate_add, residual_gate_add2
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_world_size,
    get_tp_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    gather_seq,
    shard_like,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.flux3_fused_ops import (
    flux3_gated_mlp,
    flux3_linear,
    flux3_qkv_qknorm_rope,
    flux3_shared_activation,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    RMSNorm,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.models.dits.flux3 import (
    SP_MIN_STREAM_TOKENS,
    TEXT_STREAM,
    Flux3MLPEmbedder,
    Flux3SequenceShard,
    Flux3Transformer,
    _compute_dtype,
    _sequence_shard,
    rope_cos_sin,
    timestep_embedding,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import set_mixed_precision_policy
from sglang.srt.environ import envs
from sglang.srt.utils import add_prefix

logger = init_logger(__name__)

# (cos_sin, positions): fp32 ``(L, head_dim)`` ``[cos | sin]`` rows and the
# int64 row index of each token, as the fused QK-norm + RoPE kernels consume.
Flux3Rope = tuple[torch.Tensor, torch.Tensor]
# Per-token (shift, scale, gate), each ``(L, hidden)``.
TokenModulation = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

_CONTENT_STACK_PREFIX = "content_mode_blocks."


TEXT = "text"


@dataclass
class StaticStreams:
    """Request-owned work for streams whose tokens and timestep stay fixed."""

    hidden: dict[str, Tensor] = field(default_factory=dict)
    modulation: dict[str, TokenModulation] = field(default_factory=dict)
    rope: dict[str, Flux3Rope] = field(default_factory=dict)
    joint_rope: Flux3Rope | None = None


@dataclass
class Packed:
    """Mode-major tokens with per-sample bounds for segmented attention.

    A dataclass, not a msgspec.Struct: the breakable CUDA graph runner flattens
    dataclass inputs into their tensor leaves.
    """

    data: dict[str, Tensor]
    ids: dict[str, Tensor]
    # Either one shared timestep or one per token, independently for each mode.
    timesteps: dict[str, Tensor]
    mode_indices: dict[str, list[tuple[int, int]]]
    # Flow map only: the time each sample jumps to, ``(num_samples,)``. A tensor,
    # not a float, so a captured CUDA graph reads the live value.
    timesteps_ref: Tensor | None = None
    # Request-owned positional tables; the ids do not change during denoising.
    rope: dict[str, Tensor] | None = None
    # Eager-only reuse of invariant text/reference computation.
    static: StaticStreams | None = None

    @property
    def num_samples(self) -> int:
        return len(next(iter(self.mode_indices.values())))

    def sample(self, index: int) -> Packed:
        """Return one sample with mode-local bounds rebased to zero."""
        return self.slice_samples(index, index + 1)

    def slice_samples(self, first: int, stop: int) -> Packed:
        if not 0 <= first < stop <= self.num_samples:
            raise IndexError(f"packed sample range [{first}, {stop}) out of range")
        data = {}
        ids = {}
        timesteps = {}
        mode_indices = {}
        rope = None if self.rope is None else {}
        static = None if self.static is None else StaticStreams()
        for mode in self.data:
            bounds = self.mode_indices[mode][first:stop]
            start, end = bounds[0][0], bounds[-1][1]
            data[mode] = self.data[mode][start:end]
            ids[mode] = self.ids[mode][start:end]
            time = self.timesteps[mode]
            timesteps[mode] = time if time.numel() == 1 else time[start:end]
            mode_indices[mode] = [(lo - start, hi - start) for lo, hi in bounds]
            if rope is not None:
                rope[mode] = self.rope[mode][start:end]
            if static is not None and mode in self.static.hidden:
                static.hidden[mode] = self.static.hidden[mode][start:end]
                static.modulation[mode] = tuple(
                    value[start:end] for value in self.static.modulation[mode]
                )
        timesteps_ref = (
            None if self.timesteps_ref is None else self.timesteps_ref[first:stop]
        )
        return Packed(data, ids, timesteps, mode_indices, timesteps_ref, rope, static)


def pack(
    samples: list[dict[str, tuple[Tensor, Tensor] | None]],
    channels: dict[str, int],
) -> Packed:
    modes = sorted(channels)
    data = {mode: [] for mode in modes}
    ids = {mode: [] for mode in modes}
    mode_indices = {mode: [] for mode in modes}
    counts = dict.fromkeys(modes, 0)
    device = None
    for sample in samples:
        unknown = set(sample) - set(modes)
        if unknown:
            raise ValueError(
                f"unknown packed modes {sorted(unknown)}; expected {modes}"
            )
        for mode in modes:
            item = sample.get(mode)
            start = counts[mode]
            if item is not None:
                data[mode].append(item[0])
                ids[mode].append(item[1])
                counts[mode] += item[0].shape[0]
                device = device or item[0].device
            mode_indices[mode].append((start, counts[mode]))

    def cat(parts, width, dtype):
        return (
            torch.cat(parts)
            if parts
            else torch.zeros((0, width), device=device, dtype=dtype)
        )

    data = {mode: cat(data[mode], channels[mode], torch.bfloat16) for mode in modes}
    ids = {mode: cat(ids[mode], 4, torch.float32) for mode in modes}
    return Packed(
        data=data,
        ids=ids,
        timesteps={
            mode: torch.zeros(data[mode].shape[0], device=device, dtype=torch.bfloat16)
            for mode in modes
        },
        mode_indices=mode_indices,
    )


def pack_cfg(
    text: list[tuple[Tensor, Tensor]],
    condition: tuple[Tensor, Tensor] | None,
    image: tuple[Tensor, Tensor],
    channels: dict[str, int],
) -> Packed:
    return pack(
        [
            {TEXT: text_tokens, "image_cond": condition, "image": image}
            for text_tokens in text
        ],
        channels,
    )


def flux3_dit_layer_names(module: nn.Module) -> list[str]:
    """Layerwise ModuleList paths. Expand each ModuleDict group per key."""
    names = ["single_blocks", "txt_mode_blocks"]
    content = module.content_mode_blocks
    if isinstance(content, nn.ModuleDict):
        names.extend(f"content_mode_blocks.{key}" for key in content)
    return names


def flux3_content_stack_is_active(
    layers_attr: str, active_mods: frozenset[str] | None
) -> bool:
    """Whether this layerwise group is an active content stack. None = all."""
    if active_mods is None or not layers_attr.startswith(_CONTENT_STACK_PREFIX):
        return True
    return layers_attr[len(_CONTENT_STACK_PREFIX) :] in active_mods


def flux3_cross_group_prefetch_target(
    layers_attr: str,
    layer_idx: int,
    active_mods: list[str],
    depth: int,
    depth_single_blocks: int,
) -> tuple[str, int] | None:
    """Next stack after ``layers_attr[layer_idx]`` in the interleaved forward."""
    if not active_mods:
        return None
    if layers_attr.startswith(_CONTENT_STACK_PREFIX):
        mod = layers_attr[len(_CONTENT_STACK_PREFIX) :]
        if mod not in active_mods:
            return None
        pos = active_mods.index(mod)
        if pos + 1 < len(active_mods):
            return f"{_CONTENT_STACK_PREFIX}{active_mods[pos + 1]}", layer_idx
        return "txt_mode_blocks", layer_idx
    if layers_attr == "txt_mode_blocks":
        if layer_idx + 1 < depth:
            return f"{_CONTENT_STACK_PREFIX}{active_mods[0]}", layer_idx + 1
        if depth_single_blocks:
            return "single_blocks", 0
    return None


# Checkpoint / LoRA q_proj|k_proj|v_proj -> packed ``to_qkv``. The generic
# loader cats the three shards on dim 0 once all arrive (see
# ``hf_to_custom_state_dict``); LoRA stacks them the same way. Scalar
# per-projection scales are promoted to ``[1]`` first and cat into ``[3]``.
_QKV_PARAM_NAMES_MAPPING: dict[str, str | tuple[str, int, int]] = {
    r"(.*)\.q_proj\.weight$": (r"\1.to_qkv.weight", 0, 3),
    r"(.*)\.k_proj\.weight$": (r"\1.to_qkv.weight", 1, 3),
    r"(.*)\.v_proj\.weight$": (r"\1.to_qkv.weight", 2, 3),
    r"(.*)\.q_proj\.weight_scale$": (r"\1.to_qkv.weight_scale", 0, 3),
    r"(.*)\.k_proj\.weight_scale$": (r"\1.to_qkv.weight_scale", 1, 3),
    r"(.*)\.v_proj\.weight_scale$": (r"\1.to_qkv.weight_scale", 2, 3),
    # NVFP4 additionally serializes one tensorwise scale per Q/K/V projection.
    # The packed native linear owns the corresponding three logical partitions.
    r"(.*)\.q_proj\.weight_scale_2$": (r"\1.to_qkv.weight_scale_2", 0, 3),
    r"(.*)\.k_proj\.weight_scale_2$": (r"\1.to_qkv.weight_scale_2", 1, 3),
    r"(.*)\.v_proj\.weight_scale_2$": (r"\1.to_qkv.weight_scale_2", 2, 3),
    r"(.*)\.q_proj\.input_scale$": (r"\1.to_qkv.input_scale", 0, 3),
    r"(.*)\.k_proj\.input_scale$": (r"\1.to_qkv.input_scale", 1, 3),
    r"(.*)\.v_proj\.input_scale$": (r"\1.to_qkv.input_scale", 2, 3),
    r"(.*)\.q_proj\.(lora_A|lora_B)$": (r"\1.to_qkv.\2", 0, 3),
    r"(.*)\.k_proj\.(lora_A|lora_B)$": (r"\1.to_qkv.\2", 1, 3),
    r"(.*)\.v_proj\.(lora_A|lora_B)$": (r"\1.to_qkv.\2", 2, 3),
}


# Backends the FLUX.3 native DiT supports; the platform selector auto-picks per
# GPU (Hopper -> FA3, Blackwell -> FA4, sm120 -> Torch SDPA, with Torch-SDPA
# safety nets for missing flash_attn / unsupported head sizes). The current
# FLUX.3 architecture uses head_dim 128, which FA3/FA4 and Sol-Attn support.
# Sage / Sol are opt-in via ``--attention-backend`` (lossy).
_SUPPORTED_ATTENTION_BACKENDS = frozenset(
    {
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.FA,
        AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.SAGE_ATTN_3,
        AttentionBackendEnum.SOL_ATTN,
    }
)


class Flux3ImageAttention(nn.Module):
    """Attention over this rank's heads through :class:`USPAttention`.

    q/k/v arrive as ``(1, L, H_local, D)`` with QK-norm + RoPE applied.
    backend selection must succeed before inference; a local SDPA fallback
    would drop the collectives required by sequence parallelism
    """

    def __init__(self, local_heads: int, head_dim: int, prefix: str = ""):
        super().__init__()
        self._usp = USPAttention(
            num_heads=local_heads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=False,
            supported_attention_backends=set(_SUPPORTED_ATTENTION_BACKENDS),
            prefix=prefix,
            allow_cudnn_sdp=True,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seq_lens: list[int] | None,
        sp: Flux3SequenceShard | None,
    ) -> torch.Tensor:
        usp = self._usp
        if sp is None:
            # Replicated input: every rank holds the whole packed sequence.
            out = usp(q, k, v, skip_sequence_parallel_override=True, seq_lens=seq_lens)
        else:
            # q/k/v are strided views of the fused projection; the SP exchanges
            # need dense tensors.
            out = usp(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                attn_mask_meta=sp.attn_mask_meta,
            )
        return out.flatten(2) if out.dim() == 4 else out


class Flux3ImageQKNorm(nn.Module):
    """Per-head RMSNorm for q/k using the framework's fused :class:`RMSNorm`.

    Normalizes over ``head_dim`` (the last axis of a ``(B, L, H, D)`` tensor),
    which is TP-invariant, so no cross-shard reduction is needed."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.query_norm = RMSNorm(head_dim, eps=1e-6)
        self.key_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, q, k, v):
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class Flux3ImageModulation(nn.Module):
    """Quantizable per-token shift / scale / gate projection."""

    def __init__(
        self,
        dim: int,
        *,
        prefix: str,
        quant_config: QuantizationConfig | None,
    ):
        super().__init__()
        self.lin = ReplicatedLinear(
            dim,
            3 * dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.lin",
        )

    def forward(self, vec: torch.Tensor) -> TokenModulation:
        projected = flux3_linear(self.lin, F.silu(vec))
        shift, scale, gate = projected.chunk(3, dim=-1)
        return shift.contiguous(), scale.contiguous(), gate


class Flux3ImageBlock(nn.Module):
    """FLUX 3 parallel attention + SwiGLU block on native parallel linears.

    Same math as :class:`.flux3.Flux3Block`; serves as both a mode block and a
    joint block (the modulation is per token either way).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        prefix: str,
        quant_config: QuantizationConfig | None,
    ):
        super().__init__()
        self.tp_size = get_tp_world_size()
        self.local_heads = divide(num_heads, self.tp_size)
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.to_qkv = MergedColumnParallelLinear(
            hidden_size,
            [hidden_size] * 3,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("to_qkv", prefix),
        )
        self.mlp_in = MergedColumnParallelLinear(
            hidden_size,
            [mlp_hidden_dim] * 2,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("mlp_in", prefix),
        )
        self.attn_out = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("attn_out", prefix),
        )
        self.mlp_out = RowParallelLinear(
            mlp_hidden_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("mlp_out", prefix),
        )
        self.norm = Flux3ImageQKNorm(head_dim)
        self.norm_scale_shift = LayerNormScaleShift(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.attention = Flux3ImageAttention(
            self.local_heads, head_dim, prefix=add_prefix("attention", prefix)
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: Flux3Rope,
        mod: TokenModulation,
        seq_lens: list[int] | None = None,
        sp: Flux3SequenceShard | None = None,
    ) -> torch.Tensor:
        """``x`` is ``(L, hidden)``: packed samples split by ``seq_lens``, or
        this rank's shard of one sample under ``sp``.

        Blocks are always entered through ``__call__`` so that forward hooks
        (layerwise offload) see every block.
        """
        shift, scale, gate = mod
        x_mod = self.norm_scale_shift(
            x.unsqueeze(0), shift.unsqueeze(0), scale.unsqueeze(0)
        )
        quantized = flux3_shared_activation(x_mod, self.to_qkv, self.mlp_in)
        cos_sin, positions = rope
        q, k, v = flux3_qkv_qknorm_rope(
            self.to_qkv,
            x_mod,
            self.norm,
            cos_sin,
            positions,
            self.local_heads,
            quantized,
        )
        if sp is None and sum(seq_lens) == 0:
            # A content mode without tokens runs on one placeholder row.
            attn = torch.zeros_like(v).flatten(2)
        else:
            attn = self.attention(q, k, v, seq_lens=seq_lens, sp=sp)
        attn_out = flux3_linear(self.attn_out, attn.squeeze(0))
        mlp_out = flux3_gated_mlp(self.mlp_in, self.mlp_out, x_mod, quantized)
        mlp_out = mlp_out.squeeze(0)
        if self.tp_size == 1:
            return residual_gate_add2(x, attn_out, mlp_out, gate)
        # One reduce of the summed partials, not one per branch.
        update = tensor_model_parallel_all_reduce(attn_out + mlp_out)
        return residual_gate_add(x, update, gate)


class _SpStream:
    """One CFG sample's stream, sharded across the SP group or replicated."""

    def __init__(
        self,
        x: torch.Tensor,
        rope_rows: torch.Tensor,
        mod: TokenModulation,
        min_tokens: int,
    ):
        self.length = x.shape[0]
        self.sp = _sequence_shard(self.length, x[None], min_tokens=min_tokens)
        if self.sp is not None:
            shard = self.sp.shard
            x = shard_like(x, shard, dim=0)
            rope_rows = shard_like(rope_rows, shard, dim=0)
            # Zero modulation on the tail pad leaves those rows unchanged.
            mod = tuple(shard_like(m, shard, dim=0) for m in mod)
        self.x = x
        self.rope = _with_positions(rope_rows)
        self.mod = mod
        self.seq_lens = None if self.sp is not None else [self.length]

    def run(self, block: nn.Module) -> None:
        self.x = block(self.x, self.rope, self.mod, seq_lens=self.seq_lens, sp=self.sp)

    def gather(self) -> torch.Tensor:
        if self.sp is None:
            return self.x
        return gather_seq(self.x, self.length, dim=0)


def _with_positions(cos_sin: torch.Tensor) -> Flux3Rope:
    positions = torch.arange(cos_sin.shape[0], device=cos_sin.device)
    return cos_sin.contiguous(), positions


def _seq_lens(bounds: list[tuple[int, int]]) -> list[int]:
    return [end - start for start, end in bounds]


class _JointLayout:
    """Where each mode's packed rows sit in the joint sequence.

    The joint order is sample-major, and within a sample the modes in sorted
    order (``image``, ``image_cond``, ``text``).
    """

    def __init__(self, packed: Packed, modes: tuple[str, ...]):
        self.placement: list[tuple[str, int, int]] = []
        self.sample_bounds: list[tuple[int, int]] = []
        offset = 0
        for sample in range(packed.num_samples):
            start = offset
            for mode in modes:
                left, right = packed.mode_indices[mode][sample]
                if right > left:
                    self.placement.append((mode, left, right))
                    offset += right - left
            self.sample_bounds.append((start, offset))
        self.sample_lengths = _seq_lens(self.sample_bounds)

    def pack(self, per_mode: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [per_mode[mode][left:right] for mode, left, right in self.placement]
        )

    def unpack(
        self, joint: torch.Tensor, mode: str, like: torch.Tensor
    ) -> torch.Tensor:
        tokens = torch.empty_like(like)
        offset = 0
        for placed_mode, left, right in self.placement:
            if placed_mode == mode:
                tokens[left:right] = joint[offset : offset + right - left]
            offset += right - left
        return tokens


class Flux3ImageDiT(Flux3Transformer):
    """FLUX.3 image DiT: the FLUX 3 transformer with native blocks and a
    packed (variable-length, sample-isolated) CFG forward."""

    _fsdp_shard_conditions = [
        lambda name, module: isinstance(module, Flux3ImageBlock),
    ]
    # Whole-module compile; the packed forward is not split into regions.
    _compile_conditions: list = []
    _fsdp_forward_methods = ()

    # Base-DiT contract: backends the selector may pick for this model (the
    # actual USPAttention calls use the same set, see Flux3ImageAttention).
    _supported_attention_backends = _SUPPORTED_ATTENTION_BACKENDS

    _aliases = ["Flux3ImageDiT"]

    # Packed QKV remap (q_proj/k_proj/v_proj -> to_qkv); ``mlp_in`` stays its
    # own module so PrimsTS can own the SwiGLU epilogue.
    param_names_mapping = _QKV_PARAM_NAMES_MAPPING

    # Checkpoint keys that must be renamed to the native module tree.
    _param_renames = {
        ".query_norm.scale": ".query_norm.weight",
        ".key_norm.scale": ".key_norm.weight",
    }
    _ignored_checkpoint_prefixes = ("repa_projectors.",)
    pin_parked_non_layer_weights = True

    def __init__(self, config, hf_config=None, quant_config=None, **kwargs):
        super().__init__(config, hf_config=hf_config, quant_config=quant_config)
        # The base installs the arch's fused ``qkv_mlp`` mapping.
        self.param_names_mapping = _QKV_PARAM_NAMES_MAPPING
        self.depth_single_blocks = len(self.single_blocks)
        self.layer_names = flux3_dit_layer_names(self)
        self._layerwise_active_content_mods: frozenset[str] | None = None
        self.flow_map = config.arch_config.flow_map
        self.ref_time_in = (
            Flux3MLPEmbedder(256, self.hidden_size) if self.flow_map else None
        )

    def _make_modulation(self, prefix: str) -> nn.Module:
        return Flux3ImageModulation(
            self.hidden_size, prefix=prefix, quant_config=self.quant_config
        )

    def _make_block(self, prefix: str) -> nn.Module:
        return Flux3ImageBlock(
            self.hidden_size,
            self.num_attention_heads,
            self.arch.mlp_ratio,
            prefix=prefix,
            quant_config=self.quant_config,
        )

    # ------------------------------------------------------------------ offload
    def configure_layerwise_offload(self, server_args, **kwargs):
        super().configure_layerwise_offload(server_args, **kwargs)
        self.park_non_layer_weights_between_uses = (
            not server_args.pipeline_config.flux3_dit_keep_non_layer
        )

    def should_prefetch_layerwise_manager(self, manager) -> bool:
        return flux3_content_stack_is_active(
            manager.layers_attr_str, self._layerwise_active_content_mods
        )

    def layerwise_prefetch_successor(self, manager, layer_idx: int) -> None:
        active = self._layerwise_active_content_mods
        if not active:
            return
        mods = []
        for name in self.sequence.values():
            if name in active and name not in mods:
                mods.append(name)
        target = flux3_cross_group_prefetch_target(
            manager.layers_attr_str,
            layer_idx,
            mods,
            self.depth,
            self.depth_single_blocks,
        )
        if target is not None:
            self.prefetch_layerwise_attr(*target)

    def park_non_layer_weights(self, *, force: bool = True) -> None:
        if not self.park_non_layer_weights_between_uses:
            return
        super().park_non_layer_weights(force=force)

    # ------------------------------------------------------------------ loading
    def _remap(self, name: str) -> str:
        for src, dst in self._param_renames.items():
            if name.endswith(src):
                return name[: -len(src)] + dst
        return name

    def preprocess_loaded_state_dict(self, weights):
        """Adapt the reference checkpoint before generic streaming load."""
        for name, weight in weights:
            if name.startswith(self._ignored_checkpoint_prefixes):
                continue
            if weight.ndim == 0 and any(
                name.endswith(f".{projection}.{suffix}")
                for projection in ("q_proj", "k_proj", "v_proj")
                for suffix in ("weight_scale_2", "input_scale")
            ):
                weight = weight.reshape(1)
            yield self._remap(name), weight

    def post_load_weights(self) -> None:
        """Generic-loader hook; quant methods prepare their own weights."""
        return None

    # ------------------------------------------------------------------ forward
    def forward(self, packed: Packed) -> dict[str, torch.Tensor]:
        # USPAttention reads its compute dtype from the mixed-precision policy;
        # pin bf16 so the backend selector can pick FlashAttention.
        set_mixed_precision_policy(torch.bfloat16, torch.bfloat16)
        # Native attention resolves its impl from a runtime ForwardContext; keep
        # an outer timestep (set by the denoise loop) so Sol-Attn can keep its
        # first N steps dense.
        try:
            ctx = get_forward_context()
            timestep = int(ctx.current_timestep)
            attn_metadata = ctx.attn_metadata
        except AssertionError:
            timestep = 0
            attn_metadata = None
        with set_forward_context(
            current_timestep=timestep, attn_metadata=attn_metadata
        ):
            self._prepare_trtllm_gen_sage_metadata(packed)
            return self._forward_packed(packed)

    @staticmethod
    def _prepare_trtllm_gen_sage_metadata(packed: Packed) -> None:
        """Build the TRT-LLM Gen Sage ragged metadata for every sequence length
        this forward attends over, once, before the blocks run."""
        if not (
            envs.SGLANG_USE_SAGE_ATTENTION_IMPL.get().lower()
            in {"trtllm_gen", "trtllm-gen"}
            and get_global_server_args().attention_backend == "sage_attn"
            and get_sp_world_size() == 1
        ):
            return
        tensors = [tensor for tensor in packed.data.values() if tensor.numel()]
        if not tensors:
            return
        device_index = tensors[0].device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        # Mode blocks attend per sample within a mode; single blocks attend over
        # each sample's joint sequence.
        lengths = {
            end - start
            for bounds in packed.mode_indices.values()
            for start, end in bounds
            if end > start
        }
        for sample in range(packed.num_samples):
            joint_length = sum(
                end - start
                for bounds in packed.mode_indices.values()
                for start, end in [bounds[sample]]
            )
            if joint_length:
                lengths.add(joint_length)
        context = get_forward_context()
        for length in lengths:
            context.get_trtllm_gen_sequence_metadata(device_index, 1, length, length)

    def _token_vector(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Per-token timesteps ``(L,)`` -> timestep vectors ``(L, hidden)``."""
        return self._timestep_vector(timesteps[None])[0]

    def _add_reference_time(
        self, packed: Packed, modes: tuple[str, ...], vec: dict[str, torch.Tensor]
    ) -> None:
        """Flow map: add each sample's embedded jump target to all of its tokens."""
        if (packed.timesteps_ref is not None) != self.flow_map:
            raise ValueError(
                "timesteps_ref is required by, and only by, a flow-map DiT"
            )
        if not self.flow_map:
            return
        # The target is rounded to the timestep dtype before embedding, as in
        # the reference.
        target = packed.timesteps_ref.to(next(iter(packed.timesteps.values())).dtype)
        weight_dtype = _compute_dtype(self.ref_time_in.in_layer)
        with torch.autocast(device_type=target.device.type, enabled=False):
            ref = self.ref_time_in(timestep_embedding(target).to(weight_dtype))
        ref = ref.to(self.dtype)
        for mode in modes:
            rows = [
                ref[sample].expand(end - start, -1)
                for sample, (start, end) in enumerate(packed.mode_indices[mode])
                if end > start
            ]
            # A phantom token of an empty mode gets no target, as in the reference.
            if rows:
                vec[mode] = vec[mode] + torch.cat(rows)

    def _forward_packed(self, packed: Packed) -> dict[str, torch.Tensor]:
        # A content mode without tokens (image_cond in t2i) feeds neither the
        # joint stream nor an output, so its mode blocks are skipped.
        content = [
            mode
            for mode in sorted(self.in_channels)
            if packed.data[mode].shape[0] > 0 or mode in self.final_layer
        ]
        modes = tuple(sorted([*content, TEXT]))
        self._layerwise_active_content_mods = frozenset(content)
        data = dict(packed.data)
        ids = dict(packed.ids)
        timesteps = dict(packed.timesteps)
        for mode in modes:
            if data[mode].shape[0] == 0:
                data[mode] = data[mode].new_zeros(1, data[mode].shape[1])
                ids[mode] = ids[mode].new_zeros(1, 4)
                timesteps[mode] = timesteps[mode].new_zeros(1)

        def mod_key(mode: str) -> str:
            return TEXT_STREAM if mode == TEXT else mode

        cached = {} if packed.static is None else packed.static.hidden
        active = tuple(mode for mode in modes if mode not in cached)
        vec = {mode: self._token_vector(timesteps[mode]) for mode in active}
        self._add_reference_time(packed, active, vec)

        def broadcast_modulation(values, mode):
            length = data[mode].shape[0]
            return tuple(
                value
                if value.shape[0] == length
                else value.expand(length, -1).contiguous()
                for value in values
            )

        early = {
            mode: broadcast_modulation(
                self.early_stream_modulations[mod_key(mode)](vec[mode]), mode
            )
            for mode in active
        }
        single = {
            mode: broadcast_modulation(
                self.single_stream_modulations[mod_key(mode)](vec[mode]), mode
            )
            for mode in active
        }
        if packed.static is not None:
            single.update(packed.static.modulation)
        x = {
            mode: (
                self.txt_in(data[mode].to(self.dtype))
                if mode == TEXT
                else self.emb_in[mode](data[mode].to(self.dtype))
            )
            for mode in active
        }
        rope = {
            mode: (
                packed.rope[mode]
                if packed.rope is not None and packed.ids[mode].shape[0]
                else rope_cos_sin(ids[mode], self.axes_dim, self.theta)
            )
            for mode in modes
        }
        layout = _JointLayout(packed, modes)
        if get_sp_world_size() == 1:
            hidden = self._packed_blocks(
                packed, content, x, rope, early, single, layout
            )
        else:
            hidden = self._sp_blocks(packed, content, x, rope, early, single, layout)

        out = {}
        for mode in self.final_layer:
            tokens = layout.unpack(hidden, mode, like=x[mode])
            out[mode] = self.final_layer[mode](tokens[None], vec[mode][None])[0]
        return out

    def _packed_blocks(self, packed, content, x, rope, early, single, layout):
        """All samples in one varlen sequence per block (no sequence parallel)."""
        cached = packed.static.hidden if packed.static is not None else {}
        x.update(cached)
        if packed.static is not None and packed.static.rope:
            mode_rope = packed.static.rope
        else:
            mode_rope = {mode: _with_positions(rows) for mode, rows in rope.items()}
            if packed.static is not None:
                packed.static.rope = mode_rope
        for depth in range(self.depth):
            for mode in content:
                if mode in cached:
                    continue
                x[mode] = self.content_mode_blocks[mode][depth](
                    x[mode],
                    mode_rope[mode],
                    early[mode],
                    seq_lens=_seq_lens(packed.mode_indices[mode]),
                )
            if TEXT not in cached:
                x[TEXT] = self.txt_mode_blocks[depth](
                    x[TEXT],
                    mode_rope[TEXT],
                    early[TEXT],
                    seq_lens=_seq_lens(packed.mode_indices[TEXT]),
                )
        if packed.static is not None:
            packed.static.hidden.update(
                (mode, x[mode]) for mode in (TEXT, "image_cond") if mode in x
            )
            packed.static.modulation.update(
                (mode, single[mode]) for mode in packed.static.hidden
            )
        hidden = layout.pack(x)
        if packed.static is not None and packed.static.joint_rope is not None:
            joint_rope = packed.static.joint_rope
        else:
            joint_rope = _with_positions(layout.pack(rope))
            if packed.static is not None:
                packed.static.joint_rope = joint_rope
        joint_mods = tuple(
            layout.pack({mode: single[mode][index] for mode in single})
            for index in range(3)
        )
        for block in self.single_blocks:
            hidden = block(
                hidden, joint_rope, joint_mods, seq_lens=layout.sample_lengths
            )
        return hidden

    def can_reuse_mode_blocks(self) -> bool:
        return get_sp_world_size() == 1 and all(
            block.attention._usp.backend
            in (
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.TORCH_CUDNN_SDPA,
                AttentionBackendEnum.DYNAMIC_CUDNN_SDPA,
            )
            for blocks in [self.txt_mode_blocks, *self.content_mode_blocks.values()]
            for block in blocks
        )

    def _sp_blocks(self, packed, content, x, rope, early, single, layout):
        """One sequence per CFG sample, each sharded across the SP group.

        Ulysses attention runs dense over the gathered sequence, so samples
        cannot share a varlen pack. The loops keep the packed path's block
        order (every sample per block) so layerwise offload streams each block
        once per step.
        """

        def streams(tokens, rows, mod, bounds, min_tokens):
            return [
                _SpStream(
                    tokens[left:right],
                    rows[left:right],
                    tuple(m[left:right] for m in mod),
                    min_tokens,
                )
                for left, right in bounds
                if right > left
            ]

        mode_streams = {
            mode: streams(
                x[mode],
                rope[mode],
                early[mode],
                packed.mode_indices[mode],
                SP_MIN_STREAM_TOKENS,
            )
            for mode in (*content, TEXT)
        }
        for depth in range(self.depth):
            for mode in content:
                for stream in mode_streams[mode]:
                    stream.run(self.content_mode_blocks[mode][depth])
            for stream in mode_streams[TEXT]:
                stream.run(self.txt_mode_blocks[depth])
        for mode, mode_stream in mode_streams.items():
            if mode_stream:
                x[mode] = torch.cat([stream.gather() for stream in mode_stream])

        hidden = layout.pack(x)
        joint_rows = layout.pack(rope)
        joint_mods = tuple(
            layout.pack({mode: single[mode][index] for mode in single})
            for index in range(3)
        )
        joint = streams(hidden, joint_rows, joint_mods, layout.sample_bounds, 0)
        for block in self.single_blocks:
            for stream in joint:
                stream.run(block)
        return torch.cat([stream.gather() for stream in joint])


EntryClass = Flux3ImageDiT
