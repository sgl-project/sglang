"""HY4 DFly draft: block-parallel draft layers bundled in the target checkpoint.

One forward runs ``N + 1`` queries per request (the bonus token, then mask
tokens). Draft layer ``i`` keeps its KV in target layer ``target_kv_layers[i]``
and attends over externally built sparse indices; the drafts then come from a
sequential hidden-correction chain over the target LM head.
"""

import copy
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.configs.hy_v4 import HYV4VLConfig
from sglang.srt.configs.hy_v4_dfly import (
    DFLY_CHECKPOINT_PREFIX,
    HYV4DFlySpec,
    build_hyv4_dfly_draft_config,
    check_hyv4_dfly_quant_config,
    resolve_hyv4_dfly_spec,
)
from sglang.srt.configs.model_config import dsa_layer_skips_topk
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
    get_attn_tp_context,
    layer_stack,
)
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match,
    should_ignore_layer,
)
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, MoEGate
from sglang.srt.models.hunyuan_v4 import (
    HYV4_PACKED_MODULES_MAPPING,
    HYV4Attention,
    hyv4_mrope_rows_are_equal,
)
from sglang.srt.models.hunyuan_v4_vl import HYV4VLForConditionalGeneration
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.dfly_utils import (
    dfly_full_width_lengths,
    dfly_sparse_attention_metadata,
)
from sglang.srt.utils import BumpAllocator

_IGNORED_CHECKPOINT_PREFIXES = ("confidence_head.",)
_FUSED_CHECKPOINT_NAMES = {
    ".fused_qkv_a_proj_with_mqa.": (".q_a_proj.", ".kv_a_proj_with_mqa."),
    ".gate_up_proj.": (".gate_proj.", ".up_proj."),
}
# Attention KV-cache scales come from the quant config, not the checkpoint.
_RUNTIME_ONLY_PARAM_SUFFIXES = (".k_scale", ".v_scale")


def _dfly_checkpoint_layer_name(name: str) -> str:
    # model.layers.<i>.X -> model.dfly_layers.<i>.X; model.<module>.X likewise.
    head, found, rest = name.partition("model.")
    if not found:
        return name
    return f"{head}{DFLY_CHECKPOINT_PREFIX}{rest.removeprefix('layers.')}"


def _dfly_quant_config(quant_config, quantization_config: Optional[dict]):
    check_hyv4_dfly_quant_config(quantization_config)
    if quant_config is None:
        return None
    if not isinstance(quant_config, CompressedTensorsConfig):
        raise ValueError(
            "HYV4 DFly drafts follow a BF16 or compressed-tensors checkpoint; "
            f"got quantization {quant_config.get_name()!r}"
        )
    quant_config = copy.deepcopy(quant_config)
    # Every draft linear must be targeted or ignored by the checkpoint, even
    # when the source config was adapted for the target's lenient lookup.
    quant_config.ignore_unmatched_layers = False
    quant_config.checkpoint_layer_name = _dfly_checkpoint_layer_name
    return quant_config


def _check_draft_linear_schemes(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if (
            isinstance(module, LinearBase)
            and isinstance(module.quant_method, CompressedTensorsLinearMethod)
            and not isinstance(module.scheme, CompressedTensorsW8A8Fp8)
        ):
            raise ValueError(
                f"HYV4 DFly draft linear {name} must be BF16 or block FP8; the "
                f"checkpoint quantizes it as {type(module.scheme).__name__}"
            )


def check_hyv4_dfly_mxfp4_placement(
    quantization_config: dict, *, hf_config, spec: HYV4DFlySpec
) -> None:
    """Weight-only MXFP4 may quantize only the target's routed experts; every
    other target and draft projection must resolve to BF16 or block FP8."""
    is_vl = isinstance(hf_config, HYV4VLConfig)
    text_config = hf_config.text_config if is_vl else hf_config
    selected = _mxfp4_projections(
        quantization_config,
        _hyv4_target_projections(
            text_config, prefix="language_model." if is_vl else ""
        ),
        mapper=HYV4VLForConditionalGeneration.hf_to_sglang_mapper if is_vl else None,
    ) + _mxfp4_projections(
        quantization_config,
        [
            (_dfly_checkpoint_layer_name(name), module_cls)
            for name, module_cls in _dfly_draft_projections(spec)
        ],
        mapper=None,
    )
    if selected:
        more = f" and {len(selected) - 4} more" if len(selected) > 4 else ""
        raise ValueError(
            "HYV4 DFly admits weight-only MXFP4 only on the target's routed "
            f"experts; the checkpoint also selects it for {', '.join(selected[:4])}"
            f"{more}"
        )


def _mxfp4_projections(quantization_config, projections, *, mapper) -> list[str]:
    checkpoint_config = _compressed_tensors_config(quantization_config, mapper=None)
    loader_config = (
        checkpoint_config
        if mapper is None
        else _compressed_tensors_config(quantization_config, mapper=mapper)
    )
    selected = []
    for name, module_cls in _with_checkpoint_shards(projections):
        # The loader matches its own names against the mapped config; it never
        # quantizes a MoEGate, so only the checkpoint decides the router.
        loader_selects = module_cls is not MoEGate and _is_mxfp4(
            _loader_scheme(loader_config, name, module_cls)
        )
        checkpoint_names = (
            [name] if mapper is None else _checkpoint_names(name, mapper=mapper)
        )
        if loader_selects or _checkpoint_selects_mxfp4(
            checkpoint_config, checkpoint_names
        ):
            selected.append(name)
    return selected


def _checkpoint_selects_mxfp4(quant_config, names) -> bool:
    # The checkpoint stores the module under one of `names`. A target or ignore
    # entry naming any of them is taken to name the stored one, so only named
    # selections count; module-type targets decide a module the config never names.
    named = [
        quant_config.get_scheme_dict(object.__new__(nn.Module), name) for name in names
    ]
    if any(scheme is not None for scheme in named) or any(
        should_ignore_layer(
            name,
            ignore=quant_config.ignore,
            fused_mapping=quant_config.packed_modules_mapping,
        )
        for name in names
    ):
        return any(_is_mxfp4(scheme) for scheme in named)
    # Every checkpoint projection, the router and head included, is a Linear.
    linear = object.__new__(nn.Linear)
    return any(_is_mxfp4(quant_config.get_scheme_dict(linear, name)) for name in names)


def _compressed_tensors_config(quantization_config, *, mapper):
    # Built as the model loader builds the target's or the draft's config.
    quant_config = CompressedTensorsConfig.from_config(
        {**quantization_config, "packed_modules_mapping": HYV4_PACKED_MODULES_MAPPING}
    )
    if mapper is not None:
        quant_config.apply_weight_name_mapper(mapper)
    # Unmatched modules select no scheme here; construction decides whether
    # that is an error.
    quant_config.ignore_unmatched_layers = True
    return quant_config


def _checkpoint_names(name, *, mapper) -> list[str]:
    # The mapper rewrites at most one substring, then one prefix, then one
    # suffix, so undoing any one of each in reverse order and keeping the exact
    # round trips yields every checkpoint name it maps onto `name`.
    weight = f"{name}.weight"
    names = {weight}
    names |= {
        candidate[: len(candidate) - len(new)] + orig
        for candidate in names
        for orig, new in mapper.orig_to_new_suffix.items()
        if new is not None and candidate.endswith(new)
    }
    names |= {
        orig + candidate[len(new) :]
        for candidate in names
        for orig, new in mapper.orig_to_new_prefix.items()
        if new is not None and candidate.startswith(new)
    }
    names |= {
        candidate[:start] + orig + candidate[start + len(new) :]
        for candidate in names
        for orig, new in mapper.orig_to_new_substr.items()
        if new is not None
        for start in range(len(candidate) + 1)
        if candidate.startswith(new, start)
    }
    checkpoint_names = sorted(
        candidate.removesuffix(".weight")
        for candidate in names
        if candidate.endswith(".weight") and mapper.apply_list([candidate]) == [weight]
    )
    if not checkpoint_names:
        raise RuntimeError(f"No checkpoint name maps onto {name}")
    return checkpoint_names


def _with_checkpoint_shards(projections):
    # A fused module is checked as itself and as each checkpoint projection it loads.
    for name, module_cls in projections:
        yield name, module_cls
        head, _, proj = name.rpartition(".")
        for shard in HYV4_PACKED_MODULES_MAPPING.get(proj, ()):
            yield f"{head}.{shard}", module_cls


def _loader_scheme(quant_config, name, module_cls) -> Optional[dict]:
    if module_cls is ParallelLMHead:
        # Heads match by name only, as in get_lm_head_scheme.
        target = next(
            (
                target
                for target in quant_config.target_scheme_map
                if check_equal_or_regex_match(layer_name=name, targets=[target])
            ),
            None,
        )
        if target is None:
            return None
        return quant_config.get_scheme_dict(
            object.__new__(module_cls), name, matched_target=target
        )
    # compressed-tensors matches module-type targets such as "Linear" on the class name.
    return quant_config.get_scheme_dict(object.__new__(module_cls), name)


def _is_mxfp4(scheme: Optional[dict]) -> bool:
    # The checkpoint contract admits 8-bit FP8 and 4-bit MXFP4 weights only.
    return (
        scheme is not None
        and scheme["weights"] is not None
        and scheme["weights"].num_bits != 8
    )


def _hyv4_target_projections(text_config, *, prefix):
    projections = []
    for layer_id in range(text_config.num_hidden_layers):
        layer = f"{prefix}model.layers.{layer_id}"
        projections += _attention_projections(f"{layer}.self_attn")
        if not dsa_layer_skips_topk(text_config, layer_id):
            projections += [
                (f"{layer}.self_attn.indexer.{name}", ReplicatedLinear)
                for name in ("wq_b", "wk", "weights_proj")
            ]
        projections += [
            (f"{layer}.{hc}.hc_pre.hc_fn", ReplicatedLinear)
            for hc in ("hc_attn_layer", "hc_mlp_layer")
        ]
        if text_config.mlp_layer_types[layer_id] == "dense":
            projections += _mlp_projections(f"{layer}.mlp")
            continue
        # Routed experts are the admitted MXFP4 placement and stay unlisted.
        projections.append((f"{layer}.mlp.gate", MoEGate))
        if text_config.n_shared_experts:
            projections += _mlp_projections(f"{layer}.mlp.shared_experts")
    return projections + [
        (f"{prefix}model.hc_head.hc_head_fn", ReplicatedLinear),
        (f"{prefix}lm_head", ParallelLMHead),
    ]


def _dfly_draft_projections(spec):
    projections = []
    for layer_id in range(spec.num_hidden_layers):
        layer = f"model.layers.{layer_id}"
        projections += _attention_projections(f"{layer}.self_attn")
        projections += _mlp_projections(f"{layer}.mlp")
    return (
        projections
        + [("model.mtp_fusion.eh_proj", ReplicatedLinear)]
        + _mlp_projections("model.hidden_correction")
    )


def _attention_projections(attn):
    return [
        (f"{attn}.fused_qkv_a_proj_with_mqa", ReplicatedLinear),
        (f"{attn}.q_b_proj", ColumnParallelLinear),
        (f"{attn}.kv_b_proj", ColumnParallelLinear),
        (f"{attn}.o_proj", RowParallelLinear),
        (f"{attn}.linear_gate", ColumnParallelLinear),
    ]


def _mlp_projections(mlp):
    return [
        (f"{mlp}.gate_up_proj", MergedColumnParallelLinear),
        (f"{mlp}.down_proj", RowParallelLinear),
    ]


class HYV4DFlyDecoderLayer(nn.Module):
    def __init__(
        self, config, layer_id, quant_config=None, prefix="", *, use_mrope=False
    ):
        super().__init__()
        self.self_attn = HYV4Attention(
            config,
            layer_id,
            quant_config,
            f"{prefix}.self_attn",
            use_mrope=use_mrope,
            attn_post_proj_norm=config.attn_post_proj_norm is not False,
        )
        if self.self_attn.indexer is not None or not self.self_attn.skip_topk:
            raise ValueError("HYV4 DFly draft attention must take external top-k")
        # Mirrors the reference draft, which keeps the gate on the unfused path.
        self.self_attn._gate_backend = self.self_attn._gate_fallback_backend
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DeepseekV2MLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            reduce_results=False,
            allow_fused_down=False,
        )
        self.init_boundaries(config.attn_post_proj_norm is True)

    def init_boundaries(self, post_proj_norm: bool):
        self.attn_boundary, self.ffn_boundary = append_stages(
            (
                declare_attn(
                    output_transform=(
                        OutputTransform(self.self_attn.post_proj_norm)
                        if post_proj_norm
                        else None
                    )
                ),
                self.input_layernorm,
                {"qkv_latent_func": self.self_attn.prepare_qkv_latent},
            ),
            (declare_ffn(sparse=False), self.post_attention_layernorm),
        )

    def bind_target_kv_layer(self, target_layer_id: int) -> None:
        # The draft writes and reads its KV in the target layer's pool slots.
        attn = self.self_attn
        if (
            attn.attn_mqa_for_dsa_token_shard is not None
            or getattr(attn, "attn_mqa_for_dcp_decode", None) is not None
        ):
            raise ValueError(
                "HYV4 DFly draft attention does not support DSA token sharding or DCP"
            )
        attn.attn_mqa.layer_id = target_layer_id
        attn.attn_mha.layer_id = target_layer_id

    def forward(self, positions, hidden_states, forward_batch, zero_allocator, topk):
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        try:
            hidden_states = self.self_attn(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                prev_topk_indices=topk,
            )
        finally:
            get_attn_tp_context().clear_attn_inputs()
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states)
        return self.ffn_boundary.finish(hidden_states, forward_batch)


class HYV4DFlyFusion(nn.Module):
    """Slot 0 mixes the bonus-token embedding with the target's final hidden."""

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        hidden_size = config.hidden_size
        self.enorm = RMSNorm(hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden_size, config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.eh_proj",
        )
        self.gate = nn.Parameter(torch.empty(hidden_size), requires_grad=False)

    def forward(self, embeds, target_hidden):
        fused, _ = self.eh_proj(
            torch.cat((self.enorm(embeds), self.hnorm(target_hidden)), dim=-1)
        )
        return embeds + self.gate * fused


class HYV4DFlyHiddenCorrection(nn.Module):
    """Corrects a draft hidden state with the embedding of the previous draft."""

    def __init__(self, config, intermediate_size, quant_config=None, prefix=""):
        super().__init__()
        hidden_size = config.hidden_size
        self.hidden_norm = RMSNorm(hidden_size, config.rms_norm_eps)
        self.embed_norm = RMSNorm(hidden_size, config.rms_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            2 * hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden, previous_embeds):
        gate_up, _ = self.gate_up_proj(
            torch.cat((self.hidden_norm(hidden), self.embed_norm(previous_embeds)), -1)
        )
        delta, _ = self.down_proj(self.act(gate_up))
        return hidden + delta


class HYV4DFlyModel(nn.Module):
    def __init__(self, config, spec, quant_config=None, prefix="", *, use_mrope=False):
        super().__init__()
        self.query_width = spec.query_width
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
            **get_embedding_tp_kwargs(),
        )
        with layer_stack():
            self.layers = nn.ModuleList(
                HYV4DFlyDecoderLayer(
                    config,
                    layer_id,
                    quant_config,
                    f"{prefix}.layers.{layer_id}",
                    use_mrope=use_mrope,
                )
                for layer_id in range(spec.num_hidden_layers)
            )
        for layer, target_layer_id in zip(self.layers, spec.target_kv_layers):
            layer.bind_target_kv_layer(target_layer_id)
        self.start_layer, self.end_layer = 0, spec.num_hidden_layers
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mtp_fusion = HYV4DFlyFusion(config, quant_config, f"{prefix}.mtp_fusion")
        self.hidden_correction = HYV4DFlyHiddenCorrection(
            config,
            spec.hidden_correction_intermediate_size,
            quant_config,
            f"{prefix}.hidden_correction",
        )

    def forward(self, input_ids, positions, forward_batch, target_hidden, topk_indices):
        hidden_states = self.embed_tokens(input_ids)
        anchors = slice(None, None, self.query_width)
        hidden_states[anchors] = self.mtp_fusion(hidden_states[anchors], target_hidden)
        zero_allocator = BumpAllocator(
            buffer_size=2 * len(self.layers),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        residual_batch.start(forward_batch)
        for layer, topk in zip(self.layers, topk_indices, strict=True):
            hidden_states = layer(
                positions, hidden_states, forward_batch, zero_allocator, topk
            )
        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
        return residual_batch.final_norm(hidden_states, forward_batch, self.norm)


class HYV4ForCausalLMDFly(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = HYV4_PACKED_MODULES_MAPPING

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        text_config = config.text_config if isinstance(config, HYV4VLConfig) else config
        self.spec = resolve_hyv4_dfly_spec(
            text_config,
            speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
        )
        self.config = build_hyv4_dfly_draft_config(text_config, self.spec)
        self.quant_config = _dfly_quant_config(
            quant_config, getattr(config, "quantization_config", None)
        )
        self.num_fused_shared_experts = 0
        self.pp_group = get_parallel().pp_group
        self.is_mrope_enabled = "mrope_section" in text_config.rope_parameters and not (
            getattr(config, "language_model_only", False)
        )
        self.model = HYV4DFlyModel(
            self.config,
            self.spec,
            self.quant_config,
            add_prefix("model", prefix),
            use_mrope=self.is_mrope_enabled,
        )
        _check_draft_linear_schemes(self.model)
        # The draft drafts through the target's BF16 head.
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.has_own_embed_tokens = False
        self._block_inputs = None

    def _check_block_inputs(self, *, target_hidden, topk_indices) -> None:
        num_rows = target_hidden.shape[0] * self.spec.query_width
        expected = (num_rows, self.spec.draft_index_topk)
        shapes = [(tuple(t.shape), t.dtype) for t in topk_indices]
        if len(shapes) != self.spec.num_hidden_layers or any(
            shape != expected or dtype != torch.int32 for shape, dtype in shapes
        ):
            raise ValueError(
                f"HYV4 DFly needs {self.spec.num_hidden_layers} int32 sparse-index "
                f"tensors of shape {expected}, one per draft layer; got {shapes}"
            )

    @torch.no_grad()
    def forward_block(
        self, *, input_ids, positions, forward_batch, target_hidden, topk_indices
    ):
        """Hidden states ``[bs * (N + 1), hidden]`` of one draft block.

        ``target_hidden`` is ``[bs, hidden]``; ``topk_indices`` holds one
        ``[bs * (N + 1), draft_index_topk]`` int32 sparse-id tensor per draft
        layer. ``forward_batch`` is a TARGET_VERIFY batch over the target KV pool
        whose ``out_cache_loc`` holds the draft write slots, and the attention
        backend's metadata must scan every sparse row in full
        (``dfly_sparse_attention_metadata``).
        """
        self._check_block_inputs(target_hidden=target_hidden, topk_indices=topk_indices)
        if input_ids.shape[0] != target_hidden.shape[0] * self.spec.query_width:
            raise ValueError(
                f"HYV4 DFly got {target_hidden.shape[0]} requests for "
                f"{input_ids.shape[0]} query rows"
            )
        if self.is_mrope_enabled and forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
            if hyv4_mrope_rows_are_equal(forward_batch.forward_mode):
                positions = positions[0]
        return self.model(
            input_ids, positions, forward_batch, target_hidden, list(topk_indices)
        )

    def stage_block(self, *, target_hidden, topk_indices) -> None:
        """Hand the next runner forward its per-request target state."""
        self._check_block_inputs(target_hidden=target_hidden, topk_indices=topk_indices)
        self._block_inputs = (target_hidden, list(topk_indices))

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, pp_proxy_tensors=None):
        """The draft runner's entry: run the staged block over the runner's
        attention backend, whose metadata it widens to the full sparse rows."""
        if self._block_inputs is None:
            raise RuntimeError("HYV4 DFly forward runs only on a staged draft block")
        target_hidden, topk_indices = self._block_inputs
        self._block_inputs = None
        backend = get_attn_backend()
        lengths, cu_lengths = dfly_full_width_lengths(
            num_rows=input_ids.shape[0],
            index_topk=self.spec.draft_index_topk,
            device=input_ids.device,
        )
        backend.forward_metadata = dfly_sparse_attention_metadata(
            backend.forward_metadata, lengths=lengths, cu_lengths=cu_lengths
        )
        hidden_states = self.forward_block(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            target_hidden=target_hidden,
            topk_indices=topk_indices,
        )
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    @torch.no_grad()
    def propose(self, hidden_states, bonus_tokens, active):
        """Greedy ``[bs, N]`` drafts from one block's hidden states; -1 when inactive.

        Draft ``k`` reads query row ``k + 1`` corrected by the embedding of draft
        ``k - 1`` (the bonus token for ``k = 0``), so the chain is sequential.
        """
        query_width = self.spec.query_width
        hidden_states = hidden_states.view(-1, query_width, hidden_states.shape[-1])
        drafts = torch.empty(
            (hidden_states.shape[0], query_width - 1),
            dtype=torch.int64,
            device=hidden_states.device,
        )
        previous = bonus_tokens.to(torch.int64)
        for step in range(query_width - 1):
            corrected = self.model.hidden_correction(
                hidden_states[:, step + 1], self.model.embed_tokens(previous)
            )
            previous = self._greedy_tokens(corrected)
            drafts[:, step] = previous
        return drafts.masked_fill_(~active[:, None], -1)

    def _greedy_tokens(self, hidden_states):
        weight = self.lm_head.weight
        if (
            hidden_states.is_cuda
            and hidden_states.dtype == weight.dtype == torch.bfloat16
        ):
            logits = torch.mm(hidden_states, weight.T, out_dtype=torch.float32)
        else:
            logits = hidden_states.float() @ weight.float().T
        shard = self.lm_head.shard_indices
        local_max, local_ids = logits[:, : shard.num_org_elements].max(dim=-1)
        tp_group = get_parallel().tp_group
        if tp_group.world_size == 1:
            return local_ids
        global_ids = local_ids + shard.org_vocab_start_index
        maxes = tp_group.all_gather(local_max[None], dim=0)
        ids = tp_group.all_gather(global_ids[None], dim=0)
        # argmax keeps the lowest rank on ties, i.e. the lowest vocab id.
        return ids.gather(0, maxes.argmax(dim=0, keepdim=True))[0]

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        if not self.has_own_embed_tokens:
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        del self.lm_head.weight
        self.lm_head.weight = head

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        expected = self._checkpoint_manifest()
        ffn_shapes = self._ffn_checkpoint_shapes()
        seen: Set[str] = set()

        def draft_weights():
            for name, loaded_weight in weights:
                if not name.startswith(DFLY_CHECKPOINT_PREFIX):
                    continue
                local_name = name.removeprefix(DFLY_CHECKPOINT_PREFIX)
                if local_name.startswith(_IGNORED_CHECKPOINT_PREFIXES):
                    continue
                _check_checkpoint_weight(local_name, loaded_weight, expected, seen)
                if local_name.endswith(".self_attn.post_proj_norm.weight") and tuple(
                    loaded_weight.shape
                ) != (self.config.hidden_size,):
                    raise ValueError(
                        f"Invalid HYV4 DFly post-projection norm shape: {name}"
                    )
                shape = ffn_shapes.get(local_name)
                if shape is not None and tuple(loaded_weight.shape) != shape:
                    raise ValueError(
                        f"HYV4 DFly checkpoint weight {name} has shape "
                        f"{tuple(loaded_weight.shape)}, but the draft FFN is "
                        f"{shape}; dfly_intermediate_size must declare the "
                        "checkpoint's FFN width"
                    )
                seen.add(local_name)
                yield _sglang_param_name(local_name), loaded_weight
            missing = expected.keys() - seen - {"embed_tokens.weight"}
            norm_names = {
                f"{i}.self_attn.post_proj_norm.weight"
                for i in range(self.spec.num_hidden_layers)
            }
            loaded_norms = norm_names & seen
            if loaded_norms and loaded_norms != norm_names:
                raise ValueError(
                    "Inconsistent HYV4 DFly attention post-projection norms"
                )
            if not loaded_norms and self.config.attn_post_proj_norm is not True:
                missing -= norm_names
                for layer in self.model.layers:
                    layer.self_attn.post_proj_norm = None
            if missing:
                raise ValueError(
                    f"HYV4 DFly checkpoint is missing draft weights: {sorted(missing)}"
                )

        self.do_load_weights(draft_weights())
        enabled = self.model.layers[0].self_attn.post_proj_norm is not None
        if enabled and self.config.attn_post_proj_norm is None:
            # Resolve the optional norm before binding its post-sum transform.
            with layer_stack():
                for layer in self.model.layers:
                    layer.init_boundaries(True)
        self.config.attn_post_proj_norm = enabled
        self.has_own_embed_tokens = "embed_tokens.weight" in seen

    def _checkpoint_manifest(self) -> dict[str, torch.dtype]:
        """Checkpoint names under ``model.dfly_layers.`` and their stored dtypes."""
        manifest = {}
        for name, param in self.model.named_parameters():
            if name.endswith(_RUNTIME_ONLY_PARAM_SUFFIXES):
                continue
            dtype = (
                param.dtype
                if param.dtype == torch.float8_e4m3fn or name.endswith(".weight_scale")
                else torch.bfloat16
            )
            if name.startswith("layers."):
                name = name.removeprefix("layers.")
            elif name == "norm.weight":
                name = "final_norm.weight"
            for fused, parts in _FUSED_CHECKPOINT_NAMES.items():
                if fused in name:
                    manifest.update(
                        (name.replace(fused, part), dtype) for part in parts
                    )
                    break
            else:
                manifest[name] = dtype
        return manifest

    def _ffn_checkpoint_shapes(self) -> dict[str, tuple[int, int]]:
        hidden, width = self.config.hidden_size, self.config.intermediate_size
        shapes = {}
        for i in range(self.spec.num_hidden_layers):
            shapes[f"{i}.mlp.gate_proj.weight"] = (width, hidden)
            shapes[f"{i}.mlp.up_proj.weight"] = (width, hidden)
            shapes[f"{i}.mlp.down_proj.weight"] = (hidden, width)
        return shapes

    def _check_loaded_weights(
        self, unloaded_params: Set[str], unresolved_indexer_pairs: Set[str]
    ) -> None:
        optional_norms = {
            f"model.layers.{i}.self_attn.post_proj_norm.weight"
            for i, layer in enumerate(self.model.layers)
            if layer.self_attn.post_proj_norm is None
        }
        unloaded = {
            name
            for name in unloaded_params - optional_norms
            if not name.endswith(_RUNTIME_ONLY_PARAM_SUFFIXES)
            and name not in ("model.embed_tokens.weight", "lm_head.weight")
        }
        if unloaded:
            raise ValueError(
                f"HYV4 DFly draft parameters left unloaded: {sorted(unloaded)}"
            )


def _sglang_param_name(local_name: str) -> str:
    if local_name.split(".", 1)[0].isdigit():
        return f"model.layers.{local_name}"
    if local_name == "final_norm.weight":
        return "model.norm.weight"
    return f"model.{local_name}"


def _check_checkpoint_weight(name, weight, expected, seen) -> None:
    if name not in expected:
        raise ValueError(
            f"Unexpected HYV4 DFly checkpoint weight {DFLY_CHECKPOINT_PREFIX}{name} "
            f"with shape {tuple(weight.shape)}"
        )
    if name in seen:
        raise ValueError(f"Duplicate HYV4 DFly checkpoint weight {name}")
    if weight.dtype != expected[name]:
        raise ValueError(
            f"HYV4 DFly checkpoint weight {name} must be {expected[name]}, "
            f"got {weight.dtype}"
        )


EntryClass = [HYV4ForCausalLMDFly]
