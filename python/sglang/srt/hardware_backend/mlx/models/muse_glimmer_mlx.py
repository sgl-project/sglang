# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Muse Glimmer (dense, text-only) for mlx-lm.

Loaded via mlx-lm's custom-architecture path: ship this file in the checkpoint
directory as ``muse_glimmer_mlx.py``, set ``"model_file": "muse_glimmer_mlx.py"`` in
``config.json``.  This copy under ``sglang/srt/hardware_backend/mlx/models/``
is the maintained source; artifacts ship a byte-identical copy.  It must stay
importable standalone (mlx / mlx-lm imports only — no sglang imports),
because mlx-lm executes it from the checkpoint directory.

Ported from the vendor reference implementation (and cross-checked against the
SGLang CUDA port in ``python/sglang/srt/models/muse_glimmer.py``).  Deviations from a
llama-style decoder, and how each is mapped:

* **Sandwich norms.** ``h = h + post_norm(branch(pre_norm(h)))`` — the post
  norm applies to the branch output.  Pre-norms use ``rms_norm_eps`` (1e-5),
  post-norms ``post_norm_eps`` (1e-8).
* **Norm weights are offsets from 1.0.** The four per-layer norms compute
  ``rms_norm(x, weight + 1.0)``; ``sanitize`` folds the +1 in at load time so
  plain ``nn.RMSNorm`` is exact.  The final ``model.norm`` uses its weight
  directly and is NOT offset.
* **Non-parametric QK-norm** over ``head_dim`` (no learnable scale), applied
  BEFORE RoPE.  Exposed as ``q_norm``/``k_norm`` so the SGLang MLX batched
  decode wrapper applies them at the same point.
* **Folded attention scale.** The reference multiplies q by
  ``qk_scale_factor / sqrt(head_dim)`` after the QK-norm and SDPA then applies
  its default ``1/sqrt(head_dim)``.  RoPE is orthogonal and softmax scale is
  linear in q, so both fold into ``scale = qk_scale_factor / head_dim``.
* **Attention output gate.** ``sigmoid(output_gate_proj(pre_normed_x))`` is
  applied elementwise to the attention output before ``o_proj``.  The gate
  reads the same input as ``q_proj``, so ``sanitize`` fuses it into ``q_proj``
  per-head-interleaved (``[q_head; gate_head]``) — the exact layout the SGLang
  ``MLXAttentionWrapper`` gate path splits back out during batched decode.
* **iRoPE.** ``no_rope_layers[i] == 0`` marks NoPE layers (also the
  ``full_attention`` layers); they get a ``NoPE`` identity that still
  satisfies the wrapper's ``rope(x, offset=...)`` call.  RoPE layers use the
  interleaved GPT-J convention (``nn.RoPE(traditional=True)``), which the AOT
  Metal RoPE kernel does not support — Muse Glimmer always takes the
  ``mx.fast.rope`` fallback.
* **Sliding window.** ``layer_types`` marks the non-NoPE layers as
  ``sliding_attention`` (window 2048, including the query position — the same
  band as HF's ``create_sliding_window_causal_mask``, so no off-by-one).  The
  container exposes ``layer_types`` + ``sliding_window`` per the gpt-oss
  convention that both mlx-lm and the SGLang MLX backend read; windowing is
  done by banded masks over full-history KV, never a per-module
  ``is_sliding`` flag.
* **Full-history caches.** ``make_cache`` returns a plain ``KVCache`` for
  every layer, including sliding ones (unlike gpt-oss's ``RotatingKVCache``).
  Banded masks provide the window semantics; keeping full history makes
  greedy output exactly reproducible across prefill chunkings and matches how
  the SGLang MLX KV pool stores history.
* **Embedding norm** (scaleless RMS) when ``normalize_tok_embeddings``.
* **Logit head.** ``cap * tanh(lm_head(h) * output_multiplier / cap)``,
  computed in float32 like the reference (``None`` cap leaves just the
  multiplier).

Checkpoint formats.  ``sanitize`` accepts exactly three weight layouts and
rejects everything else with an actionable error:

* **Raw HF export** (output of the vendor's HF converter): carries
  ``output_gate_proj`` and offset-form norm weights.  Recognized by the
  complete raw key schema; transformed on load.
* **RC multimodal export** (``transformers >= 5.15``
  vendor schema): text weights under ``model.language_model.`` with
  HF-canonical names.  Normalized to the raw schema first (see the rename
  table at ``_RC_SUFFIX_RENAMES`` — the norm renames are POSITIONAL: the
  RC ``post_attention_layernorm`` is the post-attn sandwich norm, i.e. the
  raw ``post_attn_norm``, while the raw ``post_attention_layernorm`` is the
  pre-MLP norm, i.e. the RC ``pre_feedforward_layernorm``), vision tower /
  adapter / projection dropped, then transformed like a raw export.  Two
  converter generations ship this layout: the older one bakes the scaleless
  embedding RMS-norm into ``embed_tokens.weight`` and keeps q/k in the native
  interleaved layout; the current one ships the raw table and permutes q/k
  into the NeoX rotary layout.  The RC path reads the current
  conventions (``rope_is_neox_style`` pinned True) and leaves
  ``normalize_tok_embeddings`` at its default True — the norm is
  idempotent on a baked table, so always-on covers both generations,
  but an older-generation export served through this path gets the wrong
  rope layout (their configs are byte-identical; prefer repackaging).
* **Packaged MLX artifact**: already fused/folded, marked by
  ``"muse_glimmer_mlx_format": 1`` in ``config.json`` (stamped at packaging time
  only, never present on raw HF exports).  Passed through untouched.

Config schemas.  ``ModelArgs.from_dict`` accepts the flat schema written at
packaging time and the RC nested schema (``text_config`` present).
The RC schema differs in two conventions beyond field names:
``qk_scale_factor`` is expressed against SDPA's standard ``1/sqrt(head_dim)``
(flat-schema value = RC value * sqrt(head_dim); both fold to the same
``scale = flat_qk_scale / head_dim``), and NoPE layers are marked by zeros
in ``layer_rope_theta`` rather than ``no_rope_layers``.
"""

import math
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache

# Version of the packaged (fused/folded) weight layout this file understands.
MUSE_GLIMMER_MLX_FORMAT_VERSION = 1

# The four per-layer norms whose checkpoint weight is an offset from 1.0.
# model.norm (MuseGlimmerFinalRMSNorm) is NOT in this list and must not be offset.
_OFFSET_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attn_norm.weight",
    "post_attention_layernorm.weight",
    "post_ffn_norm.weight",
)

# Text-only port: the vision tower/projector are not built.
_VISION_KEY_MARKERS = (
    "vision_encoder",
    "vision_adapter",
    "vision_projection",
    "vision_tower",
    "perception_emb_norm",
)

# Keys that only appear in a raw HF export, never in a packaged artifact.
# "language_model" catches RC-layout strays (text weights live under
# model.language_model. there).
_RAW_ONLY_KEY_MARKERS = (
    "output_gate_proj",
    "rotary_emb",
    "language_model",
) + _VISION_KEY_MARKERS

# RC (transformers >= 5.15 vendor schema) -> raw-schema key renames, applied
# per key after stripping the "model.language_model." prefix.  The norm
# renames are positional, not textual: RC's post_attention_layernorm is the
# post-attn sandwich norm (raw post_attn_norm, eps=post_norm_eps) and RC's
# pre_feedforward_layernorm is the pre-MLP norm (raw post_attention_layernorm,
# eps=rms_norm_eps).  self_attn.gate_proj is the attention output gate
# (mlp.gate_proj is untouched: the suffixes below carry the self_attn./
# module context).
_RC_SUFFIX_RENAMES = (
    ("self_attn.gate_proj.weight", "self_attn.output_gate_proj.weight"),
    ("post_attention_layernorm.weight", "post_attn_norm.weight"),
    ("pre_feedforward_layernorm.weight", "post_attention_layernorm.weight"),
    ("post_feedforward_layernorm.weight", "post_ffn_norm.weight"),
)

_RC_PREFIX = "model.language_model."


def flatten_rc_config(config: dict) -> dict:
    """Translate the RC nested config schema into this file's flat schema.

    Field mapping plus three convention conversions (see module docstring):
    qk_scale_factor gains the sqrt(head_dim) that the RC schema leaves to
    SDPA, NoPE layers come from zeros in layer_rope_theta, and the vendor
    export permutes q/k into the NeoX rotary layout (``_permute_for_rope``)
    so rope_is_neox_style is pinned True -- ``nn.RoPE(traditional=True)``
    on those weights emits garbled text rather than raising.

    normalize_tok_embeddings is left at its default. Older vendor exports baked
    the embedding norm into embed_tokens.weight and needed it off; the current
    export ships the native table instead.
    """
    text = config["text_config"]

    activation = text.get("hidden_activation", "silu")
    if activation != "silu":
        raise ValueError(
            f"RC config has hidden_activation={activation!r}; this port hardcodes silu"
        )

    head_dim = int(text.get("head_dim", 128))
    rope_params = text.get("rope_parameters") or {}
    layer_rope_theta = text.get("layer_rope_theta")

    flat = {
        "model_type": "muse_glimmer",
        "hidden_size": text["hidden_size"],
        "num_hidden_layers": text["num_hidden_layers"],
        "num_attention_heads": text["num_attention_heads"],
        "num_key_value_heads": text["num_key_value_heads"],
        "head_dim": head_dim,
        "intermediate_size": text["intermediate_size"],
        "vocab_size": text["vocab_size"],
        "rms_norm_eps": text["rms_norm_eps"],
        "post_norm_eps": text["post_norm_eps"],
        "rope_theta": rope_params.get("rope_theta", text.get("rope_theta", 500_000.0)),
        "max_position_embeddings": text["max_position_embeddings"],
        "qk_scale_factor": text["qk_scale_factor"] * math.sqrt(head_dim),
        "output_multiplier": text["output_multiplier"],
        "output_soft_cap_temp": text.get("final_logit_softcapping"),
        "rope_is_neox_style": True,
        "sliding_window": text["sliding_window"],
    }
    if "layer_types" in text:
        flat["layer_types"] = list(text["layer_types"])
    if layer_rope_theta is not None:
        flat["no_rope_layers"] = [0 if not theta else 1 for theta in layer_rope_theta]
    return flat


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "muse_glimmer"
    hidden_size: int = 6656
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 19968
    vocab_size: int = 202048
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    rope_theta: float = 500_000.0
    max_position_embeddings: int = 16384
    use_qk_norm: bool = True
    qk_scale_factor: float = 43.7840518911
    use_attn_output_gate: bool = True
    output_multiplier: float = 0.19611613513818404
    output_soft_cap_temp: Optional[float] = 20.0
    rope_is_neox_style: bool = False
    normalize_tok_embeddings: bool = True
    sliding_window: int = 2048
    every_n_layers_nope: int = 4
    no_rope_layers: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None
    # Set on saved MLX artifacts at packaging time (never on raw HF
    # exports); tells sanitize() the weights are already fused/folded.
    muse_glimmer_mlx_format: Optional[int] = None

    @classmethod
    def from_dict(cls, params):
        # RC multimodal schema: text fields nested under text_config, with
        # convention differences handled by flatten_rc_config.
        if "text_config" in params:
            params = flatten_rc_config(params)
        return super().from_dict(params)

    def __post_init__(self):
        # Mirror the vendor config's derivations so a config.json that
        # omits the explicit lists still builds the right architecture.
        if self.every_n_layers_nope <= 0:
            raise ValueError(
                f"every_n_layers_nope must be positive, got {self.every_n_layers_nope}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a "
                f"multiple of num_key_value_heads ({self.num_key_value_heads})"
            )

        derived_no_rope = [
            0 if (self.num_hidden_layers - i - 1) % self.every_n_layers_nope == 0 else 1
            for i in range(self.num_hidden_layers)
        ]
        if self.no_rope_layers is None:
            self.no_rope_layers = derived_no_rope
        else:
            if len(self.no_rope_layers) != self.num_hidden_layers:
                raise ValueError(
                    f"no_rope_layers has {len(self.no_rope_layers)} entries but "
                    f"num_hidden_layers is {self.num_hidden_layers}"
                )
            bad_flags = sorted(set(self.no_rope_layers) - {0, 1})
            if bad_flags:
                raise ValueError(
                    f"no_rope_layers contains non-binary entries {bad_flags}; "
                    "each entry must be 0 (NoPE) or 1 (RoPE)"
                )

        # NoPE layers are the full-attention layers; the rest slide.
        derived_layer_types = [
            "full_attention" if rope_flag == 0 else "sliding_attention"
            for rope_flag in self.no_rope_layers
        ]
        if self.layer_types is None:
            self.layer_types = derived_layer_types
        else:
            if len(self.layer_types) != self.num_hidden_layers:
                raise ValueError(
                    f"layer_types has {len(self.layer_types)} entries but "
                    f"num_hidden_layers is {self.num_hidden_layers}"
                )
            bad = sorted(
                set(self.layer_types) - {"full_attention", "sliding_attention"}
            )
            if bad:
                raise ValueError(
                    f"layer_types contains unknown entries {bad}; expected only "
                    "'full_attention' or 'sliding_attention'"
                )
            if self.layer_types != derived_layer_types:
                mismatches = [
                    i
                    for i, (got, want) in enumerate(
                        zip(self.layer_types, derived_layer_types)
                    )
                    if got != want
                ]
                raise ValueError(
                    "layer_types disagrees with no_rope_layers (NoPE layers "
                    "must be the full_attention layers) at layer indices "
                    f"{mismatches}"
                )

        if self.muse_glimmer_mlx_format is not None and (
            self.muse_glimmer_mlx_format != MUSE_GLIMMER_MLX_FORMAT_VERSION
        ):
            raise ValueError(
                f"muse_glimmer_mlx_format {self.muse_glimmer_mlx_format} is not supported by "
                f"this model file (expected {MUSE_GLIMMER_MLX_FORMAT_VERSION}); "
                "regenerate the artifact with a matching packager"
            )


class ScalelessRMSNorm(nn.Module):
    """RMS norm with no learnable scale (reference MuseGlimmerScalelessRMSNorm)."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.dims = dims
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class NoPE(nn.Module):
    """Identity standing in for RoPE on NoPE layers.

    Accepts the ``offset`` kwarg so both this file's forward and the SGLang
    ``MLXAttentionWrapper`` (which calls ``rope(x, offset=offsets)``
    unconditionally) can treat every layer uniformly.  ``dims = 0`` keeps the
    AOT Metal RoPE kernel gating disabled for these layers.
    """

    dims = 0
    traditional = True

    def __call__(self, x: mx.array, offset: Any = 0) -> mx.array:
        return x


class MuseGlimmerAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.use_attn_output_gate = args.use_attn_output_gate

        q_dim = args.num_attention_heads * args.head_dim
        kv_dim = args.num_key_value_heads * args.head_dim

        # With the output gate, q_proj holds the per-head-interleaved
        # [q_head; gate_head] fusion produced by sanitize(): output width
        # 2 * q_dim, split back out in the forward pass.
        self.q_proj = nn.Linear(
            args.hidden_size,
            2 * q_dim if self.use_attn_output_gate else q_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(args.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, args.hidden_size, bias=False)

        # Bool flag for the forward-pass branch; q_norm/k_norm stay ABSENT
        # (not None) when unused — the SGLang batched-decode wrapper
        # duck-types them via hasattr.
        self.use_qk_norm = args.use_qk_norm
        if args.use_qk_norm:
            self.q_norm = ScalelessRMSNorm(args.head_dim, args.rms_norm_eps)
            self.k_norm = ScalelessRMSNorm(args.head_dim, args.rms_norm_eps)
            # Reference: q *= qk_scale_factor / sqrt(head_dim) after the
            # QK-norm, then SDPA scales by 1/sqrt(head_dim); folded here.
            self.scale = args.qk_scale_factor / args.head_dim
        else:
            self.scale = args.head_dim**-0.5

        use_rope = args.no_rope_layers[layer_idx] == 1
        self.rope = (
            nn.RoPE(
                args.head_dim,
                traditional=not args.rope_is_neox_style,
                base=args.rope_theta,
            )
            if use_rope
            else NoPE()
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        H, Hk, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim

        q = self.q_proj(x)
        gate = None
        if self.use_attn_output_gate:
            # Per-head layout [q_head; gate_head]: same split the SGLang MLX
            # batched-decode wrapper performs.
            q, gate = mx.split(q.reshape(B, L, H, 2 * D), 2, axis=-1)
        else:
            q = q.reshape(B, L, H, D)
        k = self.k_proj(x).reshape(B, L, Hk, D)
        v = self.v_proj(x).reshape(B, L, Hk, D)

        # QK-norm BEFORE RoPE, matching the reference.
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(q, k, v, cache, scale=self.scale, mask=mask)

        out = out.transpose(0, 2, 1, 3)
        if gate is not None:
            out = mx.sigmoid(gate) * out
        return self.o_proj(out.reshape(B, L, -1))


class MuseGlimmerMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.self_attn = MuseGlimmerAttention(args, layer_idx)
        self.post_attn_norm = nn.RMSNorm(args.hidden_size, args.post_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = MuseGlimmerMLP(args)
        self.post_ffn_norm = nn.RMSNorm(args.hidden_size, args.post_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Sandwich norms: the post-norm normalizes the branch output before
        # the residual add.
        x = x + self.post_attn_norm(
            self.self_attn(self.input_layernorm(x), mask, cache)
        )
        return x + self.post_ffn_norm(self.mlp(self.post_attention_layernorm(x)))


class MuseGlimmerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_norm = (
            ScalelessRMSNorm(args.hidden_size, args.rms_norm_eps)
            if args.normalize_tok_embeddings
            else None
        )
        self.layers = [
            MuseGlimmerDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        # Reference MuseGlimmerFinalRMSNorm: weight is the scale, not an offset.
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

        # Container-level window declaration (gpt-oss convention), read by
        # both this forward and the SGLang MLX backend's
        # get_layer_window_sizes(); per-module ``is_sliding`` flags would
        # instead trip the backend's uniform-KV-pool check.
        self.layer_types = list(args.layer_types)
        self.sliding_window = args.sliding_window

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        x = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        if self.embed_norm is not None:
            x = self.embed_norm(x)

        if cache is None:
            cache = [None] * len(self.layers)

        # One mask per layer type present, anchored to the first cache of
        # that type (all caches of a type share the same offset).
        masks = {}
        for layer_type in ("full_attention", "sliding_attention"):
            try:
                idx = self.layer_types.index(layer_type)
            except ValueError:
                continue
            window = self.sliding_window if layer_type == "sliding_attention" else None
            if window is not None:
                masks[layer_type] = create_attention_mask(
                    x, cache[idx], window_size=window
                )
            else:
                masks[layer_type] = create_attention_mask(x, cache[idx])

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            x = layer(x, masks[layer_type], c)

        return self.norm(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseGlimmerModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> List[Any]:
        # Full-history caches for every layer, sliding ones included: banded
        # masks provide the window, and full history keeps greedy output
        # exactly reproducible across prefill chunkings (a RotatingKVCache
        # would diverge once the prompt exceeds the window).
        return [KVCache() for _ in range(len(self.model.layers))]

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        hidden = self.model(inputs, cache, input_embeddings)
        # Reference computes the logit head in float32.
        logits = self.lm_head(hidden).astype(mx.float32)
        if self.args.output_soft_cap_temp is not None:
            cap = self.args.output_soft_cap_temp
            logits = cap * mx.tanh(logits * self.args.output_multiplier / cap)
        else:
            logits = logits * self.args.output_multiplier
        return logits

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _expected_raw_keys(self) -> set:
        """The complete key schema of a raw HF export (text path only)."""
        keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        for i in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{i}."
            keys.update(
                prefix + suffix
                for suffix in (
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "input_layernorm.weight",
                    "post_attn_norm.weight",
                    "post_attention_layernorm.weight",
                    "post_ffn_norm.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                )
            )
            if self.args.use_attn_output_gate:
                keys.add(prefix + "self_attn.output_gate_proj.weight")
        return keys

    def sanitize(self, weights: dict) -> dict:
        if self.args.muse_glimmer_mlx_format == MUSE_GLIMMER_MLX_FORMAT_VERSION:
            # Packaged artifact: weights are already fused/folded.  A raw-only
            # key here means the marker was stamped on the wrong directory.
            stray = sorted(
                k
                for k in weights
                if any(marker in k for marker in _RAW_ONLY_KEY_MARKERS)
            )
            if stray:
                raise ValueError(
                    "config.json claims a packaged Muse Glimmer MLX artifact "
                    f"(muse_glimmer_mlx_format={MUSE_GLIMMER_MLX_FORMAT_VERSION}) but the "
                    f"weights contain raw-checkpoint keys {stray[:4]}"
                    f"{'...' if len(stray) > 4 else ''}; the marker belongs "
                    "on packaged artifacts only — repackage from the raw HF export"
                )
            return weights

        # No marker: a raw HF export, possibly in the RC multimodal layout —
        # normalize that to the raw schema first.
        if any(k.startswith(_RC_PREFIX) for k in weights):
            weights = _normalize_rc_layout(weights)

        text_keys = {
            k
            for k in weights
            if not any(marker in k for marker in _VISION_KEY_MARKERS)
            and "rotary_emb" not in k
        }
        expected = self._expected_raw_keys()
        missing = sorted(expected - text_keys)
        unexpected = sorted(text_keys - expected)
        if missing or unexpected:
            hint = ""
            gate_missing = all("output_gate_proj" in k for k in missing) and missing
            if gate_missing and not unexpected:
                hint = (
                    " (weights look already fused: if this is a packaged "
                    'artifact, its config.json must carry "muse_glimmer_mlx_format": '
                    f"{MUSE_GLIMMER_MLX_FORMAT_VERSION})"
                )
            raise ValueError(
                "not a complete raw Muse Glimmer HF checkpoint: "
                f"{len(missing)} missing keys {missing[:4]}"
                f"{'...' if len(missing) > 4 else ''}, "
                f"{len(unexpected)} unexpected keys {unexpected[:4]}"
                f"{'...' if len(unexpected) > 4 else ''}{hint}"
            )

        H = self.args.num_attention_heads
        D = self.args.head_dim
        hidden = self.args.hidden_size

        embed_shape = tuple(weights["model.embed_tokens.weight"].shape)
        if embed_shape != (self.args.vocab_size, hidden):
            raise ValueError(
                f"embed_tokens.weight has shape {embed_shape} but config says "
                f"(vocab_size, hidden_size) = ({self.args.vocab_size}, {hidden})"
            )
        raw_q_shape = tuple(weights["model.layers.0.self_attn.q_proj.weight"].shape)
        if raw_q_shape != (H * D, hidden):
            raise ValueError(
                f"raw q_proj.weight has shape {raw_q_shape}, expected "
                f"({H * D}, {hidden}); a width of {2 * H * D} means the gate "
                "is already fused — such artifacts must carry "
                f'"muse_glimmer_mlx_format": {MUSE_GLIMMER_MLX_FORMAT_VERSION} in config.json'
            )

        new_weights = {}
        for name, w in weights.items():
            # mlx derives RoPE itself; drop cached buffers.
            if "rotary_emb" in name:
                continue
            if any(marker in name for marker in _VISION_KEY_MARKERS):
                continue
            # Consumed below when its q_proj comes up.
            if name.endswith("output_gate_proj.weight"):
                continue

            # The reference computes rms_norm(x, weight + 1.0) for these four
            # norms; fold the +1 so plain nn.RMSNorm is exact.  model.norm
            # (MuseGlimmerFinalRMSNorm) is deliberately not offset.
            if name.endswith(_OFFSET_NORM_SUFFIXES):
                w = w + 1.0

            if name.endswith("q_proj.weight") and self.args.use_attn_output_gate:
                gate_name = name.replace("q_proj.weight", "output_gate_proj.weight")
                g = weights[gate_name]
                if tuple(g.shape) != (H * D, hidden):
                    raise ValueError(
                        f"{gate_name} has shape {tuple(g.shape)}, expected "
                        f"({H * D}, {hidden})"
                    )
                # Per-head interleave [q_head; gate_head]: (H*D, hidden) x2
                # -> (H, 2D, hidden) -> (2*H*D, hidden).
                w = mx.concatenate(
                    [w.reshape(H, D, hidden), g.reshape(H, D, hidden)], axis=1
                ).reshape(2 * H * D, hidden)

            new_weights[name] = w

        return new_weights


def _normalize_rc_layout(weights: dict) -> dict:
    """Rewrite RC multimodal keys to the raw text-only schema.

    Drops the vision tower/adapter/projection, strips the
    ``model.language_model.`` prefix, and applies the positional norm/gate
    renames from ``_RC_SUFFIX_RENAMES``.  Suffix matching happens per key in
    one pass, so the post_attention_layernorm name swap cannot cascade.
    """
    out = {}
    for name, w in weights.items():
        if any(marker in name for marker in _VISION_KEY_MARKERS):
            continue
        if name.startswith(_RC_PREFIX):
            name = "model." + name[len(_RC_PREFIX) :]
        for rc_suffix, raw_suffix in _RC_SUFFIX_RENAMES:
            if name.endswith(rc_suffix):
                name = name[: -len(rc_suffix)] + raw_suffix
                break
        out[name] = w
    return out


EntryClass = Model
