"""Native MLX implementation of Qwen3.5 MoE (Qwen3.6-35B-A3B).

Text-only model extracted from
`Qwen3_5MoeForConditionalGeneration`.  The class layout mirrors
upstream ``mlx_lm.models.qwen3_5`` and ``qwen3_next`` (architecture,
weight names, hybrid-attention interval, MoE-with-shared-expert block)
so token outputs match ``mlx_lm.generate`` for the same weights.

Architecture (Qwen3.6-35B-A3B):
  40 hidden layers, hidden_size 2048, head_dim 256
  16 attention heads, 2 KV heads (heavy GQA)
  Hybrid attention 3:1 — ``GatedDeltaNet`` (linear) for 3 of every 4
    layers, ``Attention`` (full) every 4th
  MoE: 256 experts, 8 active per token; SwitchGLU per expert
  ``attn_output_gate`` on full attention (sigmoid gate on the o_proj input)
  ``partial_rotary_factor=0.25`` RoPE; ``rope_type=default`` so the
    ``mrope_section`` field is informational only
  Shared expert path (single MLP, not routed)
  ``tie_word_embeddings=False``

Only the low-level MLX primitives — ``gated_delta_update`` (the delta
rule compute), ``swiglu``, mask helpers, and cache containers — are
imported from ``mlx_lm.models``.  Everything architecture-specific is
implemented here so sglang fully owns the model definition, weight
sanitization, and quantization predicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.switch_layers import SwitchGLU as _QuantizedSwitchGLU


@dataclass
class TextModelArgs(BaseModelArgs):
    """Subset of fields needed to construct a Qwen3.5 MoE text model.

    Defaults match Qwen3.6-35B-A3B.  Use :meth:`from_dict` with the HF
    ``text_config`` block to construct from a real ``config.json``.
    """

    model_type: str = "qwen3_5_moe_text"
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    num_key_value_heads: int = 2
    max_position_embeddings: int = 262144
    head_dim: int = 256
    full_attention_interval: int = 4

    linear_num_value_heads: int = 32
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attn_output_gate: bool = True

    num_experts: int = 256
    num_experts_per_tok: int = 8
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 512
    moe_intermediate_size: int = 512
    norm_topk_prob: bool = True
    mlp_only_layers: Optional[List[int]] = None

    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_parameters: Optional[Dict[str, Any]] = None
    layer_types: Optional[List[str]] = None

    @classmethod
    def from_hf_config(cls, text_config: Dict[str, Any]) -> "TextModelArgs":
        """Build from an HF ``text_config`` block.

        Pre-extracts ``rope_theta``/``partial_rotary_factor`` from the
        ``rope_parameters`` dict (when present) and flattens
        ``layer_types`` so the dataclass can be constructed via
        :meth:`BaseModelArgs.from_dict`.
        """
        cfg = dict(text_config)
        rope_params = cfg.get("rope_parameters") or {}
        if "rope_theta" in rope_params and "rope_theta" not in cfg:
            cfg["rope_theta"] = rope_params["rope_theta"]
        if "partial_rotary_factor" in rope_params and "partial_rotary_factor" not in cfg:
            cfg["partial_rotary_factor"] = rope_params["partial_rotary_factor"]
        if "rope_scaling" in rope_params and "rope_scaling" not in cfg:
            cfg["rope_scaling"] = rope_params
        if rope_params and "rope_parameters" not in cfg:
            cfg["rope_parameters"] = rope_params
        return cls.from_dict(cfg)


@partial(mx.compile, shapeless=True)
def _precise_swiglu(h, gate, x):
    """SwiGLU activation computed in fp32 for numerical stability."""
    gate = nn.silu(gate.astype(mx.float32))
    x = x.astype(mx.float32)
    return (gate * x).astype(h.dtype)


class RMSNormGated(nn.Module):
    """RMSNorm with an optional SwiGLU gate (used by GatedDeltaNet output)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(
        self, hidden_states: mx.array, gate: Optional[mx.array] = None
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return _precise_swiglu(hidden_states, gate, x)
        return x.astype(hidden_states.dtype)


class Attention(nn.Module):
    """Full (softmax) attention with ``attn_output_gate`` and per-head RMSNorm.

    Exposes the attention-contract attrs (``q_proj``/``k_proj``/``v_proj``/
    ``o_proj``/``rope``/``scale`` plus ``q_norm``/``k_norm``) so the
    sglang :class:`MLXAttentionWrapper` patches it transparently.
    """

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        rope_dims = int(self.head_dim * args.partial_rotary_factor)
        self.rope = nn.RoPE(
            rope_dims,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class MLP(nn.Module):
    """Standard SwiGLU MLP — used for the shared expert path."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SwitchGLU(_QuantizedSwitchGLU):
    """sglang-named alias for the mlx_lm SwitchGLU.

    Uses the upstream class verbatim so 4-bit ``QuantizedSwitchLinear``
    weights load directly via :meth:`Model.load_weights` and the
    sglang Path B fusion (``SGLANG_MLX_FUSE_SWIGLU=1``) can patch it.
    """


class GatedDeltaNet(nn.Module):
    """Linear attention block — Q/K/V from a depthwise conv, delta-rule update.

    Has no ``q_proj``/``k_proj``/``v_proj``/``o_proj``/``rope``/``scale``
    attrs so the sglang attention contract skips it (no KV pool needed;
    cache state is a 2-slot ``ArraysCache``).
    """

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by "
                f"num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.eps = args.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            args.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(args.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)
        a_init = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(a_init)

        self.norm = RMSNormGated(self.head_v_dim, eps=self.eps)
        self.out_proj = nn.Linear(self.value_dim, args.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = x.shape

        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=x.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class SparseMoeBlock(nn.Module):
    """Routed MoE + shared expert with sigmoid gate on the shared path."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(args.hidden_size, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, self.num_experts
        )
        self.shared_expert = MLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        return y + shared_y


class DecoderLayer(nn.Module):
    """One transformer block: hybrid attention + MoE."""

    def __init__(self, args: TextModelArgs, layer_idx: int, is_linear: bool):
        super().__init__()
        self.is_linear = is_linear
        self.layer_idx = layer_idx
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        self.mlp = SparseMoeBlock(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class TextModel(nn.Module):
    """Qwen3.5 MoE text-only backbone."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(
                args=args,
                layer_idx=i,
                is_linear=self._is_linear_layer(
                    args.layer_types, i, args.full_attention_interval
                ),
            )
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = self._pick_fa_idx(args)

    @staticmethod
    def _pick_fa_idx(args: TextModelArgs) -> int:
        """Index of a full-attention layer to source the causal mask from.

        Prefers the explicit ``layer_types`` list; falls back to the
        ``full_attention_interval`` formula.  Clamps to the last layer
        so small test configs (num_hidden_layers < interval) still build.
        """
        if args.layer_types is not None:
            for i, kind in enumerate(args.layer_types):
                if kind == "full_attention":
                    return min(i, args.num_hidden_layers - 1)
        interval = max(1, args.full_attention_interval)
        candidate = interval - 1
        return min(candidate, args.num_hidden_layers - 1)

    @staticmethod
    def _is_linear_layer(
        layer_types: Optional[List[str]],
        layer_idx: int,
        full_attention_interval: int,
    ) -> bool:
        """Resolve per-layer attention kind from ``layer_types`` or the
        ``full_attention_interval`` formula (one full-attn layer per N)."""
        if layer_types is not None and layer_idx < len(layer_types):
            return layer_types[layer_idx] != "full_attention"
        interval = max(1, full_attention_interval)
        return (layer_idx + 1) % interval != 0

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm(hidden_states)


class Model(nn.Module):
    """Top-level: text-only Qwen3.5 MoE with LM head."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self) -> List[DecoderLayer]:
        return self.model.layers

    def make_cache(self) -> List[Any]:
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Remap HF weight names to the sglang native module layout.

        * Drop ``mtp.*`` (Phase 3) and ``model.visual.*`` (Phase 4).
        * Transpose ``conv1d.weight`` from HF's (out, in, k) to MLX's
          (out, k, in) layout.
        * Shift RMSNorm weights by ``+1`` when the file also contains
          ``mtp.*`` or unsanitised conv1d — these signals mark a fresh
          HF export whose RMSNorm weights use the (1 - gamma) convention.
        """
        has_mtp = any("mtp." in k for k in weights)
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k
            and isinstance(v, mx.array)
            and v.ndim == 3
            and v.shape[-1] != 1
            for k, v in weights.items()
        )
        shift_norms = has_mtp or has_unsanitized_conv1d

        weights = {
            k: v
            for k, v in weights.items()
            if "mtp." not in k and "model.visual" not in k
        }

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if shift_norms and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        """Exclude the router and shared-expert gate from 4-bit quant.

        Matches the upstream Qwen3.5 MoE behaviour: ``mlp.gate`` and
        ``shared_expert_gate`` stay at 8-bit so routing quality is
        preserved while the bulk of the model is 4-bit.
        """

        def predicate(path: str, _: Any) -> Any:
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str) -> bool:
            if path.endswith("A_log"):
                return False
            return True

        return predicate


def load(path: str) -> "Model":
    """Load a Qwen3.5 MoE checkpoint from a local directory.

    Mirrors ``mlx_lm.utils.load_model`` for the Qwen3.5 MoE family:
    reads ``config.json``, merges all ``model*.safetensors`` shards,
    applies :meth:`Model.sanitize`, then calls ``load_weights`` so the
    model's random-init buffers are replaced in place.  The HF/MLX
    4-bit repos already store the routed experts stacked, so no
    per-expert stacking is required.
    """
    import glob
    import json
    from pathlib import Path

    from mlx.utils import tree_flatten

    model_path = Path(path)
    with (model_path / "config.json").open() as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)

    args = TextModelArgs.from_hf_config(text_cfg)
    model = Model(args)

    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights: Dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    weights = model.sanitize(weights)

    quant_cfg = cfg.get("quantization_config") or cfg.get("quantization")
    if quant_cfg is not None and isinstance(quant_cfg, dict):
        _quantize_modules(model, weights, quant_cfg)

    n_loaded, n_expected = model.load_weights(list(weights.items()))
    if n_loaded == 0:
        raise RuntimeError(f"No weights loaded from {model_path}")

    mx.eval(tree_flatten(model.parameters()))
    del weights

    return model


def _quantize_modules(
    model: "Model",
    weights: Dict[str, mx.array],
    quant_cfg: Dict[str, Any],
) -> None:
    """Convert ``nn.Linear`` / ``SwitchLinear`` modules to their
    quantized counterparts when the loaded weights carry ``.scales``.

    Honors per-layer overrides from ``quantization_config``: any path
    whose dict specifies a different ``bits`` value is skipped or
    quantized at that value via :func:`mlx.nn.quantize`.
    """
    group_size = int(quant_cfg.get("group_size", 64))
    bits = int(quant_cfg.get("bits", 4))
    mode = quant_cfg.get("mode", "affine")

    per_layer = {
        k: v for k, v in quant_cfg.items() if isinstance(v, dict) and "bits" in v
    }

    def class_predicate(path: str, module: Any) -> bool:
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        if any(path.endswith(sfx) or sfx.endswith(path) for sfx in ("A_log",)):
            return False
        return f"{path}.scales" in weights

    nn.quantize(model, group_size=group_size, bits=bits, mode=mode, class_predicate=class_predicate)

    # Re-quantize any per-layer override that landed on a different bits
    # value (e.g. mlp.gate is 8bit while the model default is 4bit).
    for path, override in per_layer.items():
        ov_bits = int(override.get("bits", bits))
        ov_group = int(override.get("group_size", group_size))
        ov_mode = override.get("mode", mode)
        if ov_bits == bits and ov_group == group_size and ov_mode == mode:
            continue
        leaf = model
        for part in path.split("."):
            leaf = getattr(leaf, part, None) if leaf is not None else None
            if leaf is None:
                break
        if leaf is None or not hasattr(leaf, "to_quantized"):
            continue
        if isinstance(leaf, _QuantizedSwitchGLU):
            continue
        leaf.update(
            leaf.to_quantized(group_size=ov_group, bits=ov_bits, mode=ov_mode)
        )
