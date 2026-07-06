"""Qwen3-family DSpark draft model.

Served through `DSparkWorkerV2` (python/sglang/srt/speculative/dspark_worker_v2.py),
the same worker that already drives `DeepseekV4ForCausalLMDSpark`
(python/sglang/srt/models/deepseek_v4_dspark.py). This file is the Qwen3-family
analog of that V4 model.

Design: subclass DFlash, not DeepSeek-V4
-----------------------------------------
A DSpark draft block for a Qwen3-family target is architecturally the *same*
parallel, non-causal, context-KV-fed block drafter that
`python/sglang/srt/models/dflash.py` (`DFlashDraftModel`) already implements
for Qwen3-family targets: plain multi-head (GQA) attention with per-head
q/k RMSNorm + RoPE + `RadixAttention` over a paged KV cache, where the KV
cache is pre-seeded, once per committed token, with the projected target
context feature (`fc` -> `hidden_norm`), and the block's own "noise" positions
are computed by an ordinary forward pass reading the *same* cache. This is
verified against the DeepSpec reference (`deepspec/modeling/dspark/qwen3/modeling.py`):
`Qwen3DSparkAttention.forward` there concatenates `k_ctx = k_proj(target_hidden_states)`
(the fused context feature, shared by every layer) with `k_noise = k_proj(hidden_states)`
before attending; SGLang's paged-KV-cache serving path realizes that same
concatenation by writing the context feature into the cache ahead of time
(see `Qwen3DSparkAttention.kv_from_hidden` below) instead of concatenating
tensors at every attention call. DeepSeek-V4's DSpark decoder layer
(`DeepseekV4DecoderLayer`) is MLA-specific (compressed KV, fp8 packing, DSA)
and does not generalize to a plain-MHA/GQA Qwen3 checkpoint, so this file
subclasses DFlash's attention/decoder-layer machinery instead of V4's.

This file deliberately imports nothing from `deepseek_v4_dspark.py` /
`deepseek_v4.py`. `deepseek_v4.py` transitively imports `deepseek_v2.py`'s DSA
indexer, which imports `deep_gemm` at module level; on a host without
DeepSeek-V4's exact CUDA/deep_gemm build available, that import raises (e.g.
`RuntimeError: ... libcudart.so.13`), independently observed while writing
this file. That is exactly the "unconditional DSpark import crashes non-V4
targets" bug this phase's model_runner.py fix addresses for the *target*
model-runner path -- so this *draft* model file must not reintroduce the same
fragility via a convenience import for non-V4 (Qwen3) deployments. The
Markov head below is therefore a small, self-contained copy of
`deepseek_v4_dspark.py`'s `DSparkMarkovHead` (generic vocab-parallel
Embedding + LM-head-shaped Linear, no V4-specific assumptions in either
version) rather than an import.

What DSpark adds on top of a plain DFlash block, gated by config so ONE class
serves every released Qwen3 DSpark checkpoint:
  - The draft owns its OWN trained `embed_tokens` / `lm_head` (checkpoint has
    `tie_word_embeddings=False` and ships both tensors) -- NOT tied to the
    target, unlike `DeepseekV4ForCausalLMDSpark`. See `owns_vocab_weights`
    below and the corresponding conditional in `DSparkWorkerV2.__init__`.
  - An optional low-rank Markov refine head (`markov_head`, built iff
    `config.markov_rank > 0`).
  - An optional confidence ("accept-rate") head (`confidence_head`, built iff
    `config.enable_confidence_head`).

Two checkpoints are released under this exact architecture string today:
`dspark_qwen3_4b_block7` (markov_rank=256, enable_confidence_head=true) and
`dflash_qwen3_4b_block7` (markov_rank=0, enable_confidence_head=false, meant
to run under the DFLASH algorithm/`DFlashWorkerV2`). This file only targets
the `DSparkWorkerV2` contract; see the handoff notes for the open question
around `DFlashWorkerV2` compatibility (that worker expects a flat
`.layers`/`.fc`/`.project_target_hidden` shape, not this file's nested
`.model.layers` shape).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple, Union

import msgspec
import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashAttention, DFlashDecoderLayer
from sglang.srt.runtime_context import get_flags
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen3DSparkAttention(DFlashAttention):
    """`DFlashAttention` plus context-KV materialization.

    `DSparkWorkerV2._materialize_main_hidden_to_draft_kv` seeds the draft's
    paged KV cache with the projected target context feature once per
    committed token, ahead of the block forward pass; the block forward's own
    self-attention (inherited unmodified from `DFlashAttention`) then reads
    that context transparently out of the same cache alongside the block's
    own "noise" positions -- see the module docstring.

    This mirrors `DeepseekV4Attention.kv_from_hidden` (deepseek_v4.py), which
    is the MLA-specific version of the same idea. It is also exactly the math
    DFlash's own worker already runs inline (`dflash_worker_v2.py`
    `_append_target_hidden_sequential`: `kv_proj_only` -> `apply_k_norm` ->
    `apply_k_rope` -> `set_kv_buffer`); it is exposed here as a method because
    `DSparkWorkerV2` calls `layer.self_attn.kv_from_hidden(...)` directly.
    """

    def kv_from_hidden(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        attn_backend,
    ) -> None:
        k, v = self.kv_proj_only(x)
        k = self.apply_k_norm(k)
        k = self.apply_k_rope(positions, k)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        attn_backend.token_to_kv_pool.set_kv_buffer(
            self.attn, cache_loc, k, v, self.attn.k_scale, self.attn.v_scale
        )


class Qwen3DSparkDecoderLayer(DFlashDecoderLayer):
    attention_cls = Qwen3DSparkAttention


class Qwen3DSparkMarkovHead(nn.Module):
    """Low-rank additive Markov refine head over the vocab ("VanillaMarkov").

    Ground truth: `deepspec/modeling/dspark/markov_head.py` `VanillaMarkov`
    (`markov_w1 = nn.Embedding(vocab, rank)`, `markov_w2 = nn.Linear(rank,
    vocab, bias=False)`, additive logit bias `logits + markov_w2(markov_w1(
    prev_token))`). `markov_w1` is replicated (`enable_tp=False`) so
    `DSparkWorkerV2`'s per-step token lookup is a local gather; `markov_w2`
    is vocab-sharded via `ParallelLMHead` so its bias aligns column-for-column
    with the tied/sharded `lm_head` base logits it gets added to -- see
    `DSparkWorkerV2._build_refine_refs`'s explicit shard-alignment check.
    Shape- and semantics-identical to `deepseek_v4_dspark.py`'s
    `DSparkMarkovHead`; kept as a separate copy rather than an import -- see
    the module docstring.
    """

    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size,
            markov_rank,
            enable_tp=False,
            prefix=add_prefix("markov_w1", prefix),
        )
        self.markov_w2 = ParallelLMHead(
            vocab_size,
            markov_rank,
            quant_config=quant_config,
            prefix=add_prefix("markov_w2", prefix),
        )


class Qwen3DSparkConfidenceHead(nn.Module):
    """Accept-rate predictor over `[hidden_state ; markov_prev_embedding]`.

    Ground truth: `deepspec/modeling/dspark/common.py` `AcceptRatePredictor`
    (`nn.Linear(input_dim, 1)`, default `bias=True`) plus
    `deepspec/modeling/dspark/qwen3/modeling.py` `predict_confidence_step`
    (`features = cat([hidden_states, prev_embeddings], dim=-1)`, then
    `.float()`). Mirrors `DSparkConfidenceHead` (deepseek_v4_dspark.py) --
    same fp32 proj for numeric stability of the calibration signal -- except
    this checkpoint's proj HAS a bias (`confidence_head.proj.bias` is a real
    tensor here; V4's confidence head hardcodes `bias=False`).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1, bias=True, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        features = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(features.float()).squeeze(-1)


class Qwen3DSparkBackbone(nn.Module):
    """Backbone exposed to `DSparkWorkerV2` as `draft_model.model`.

    Qwen3-family analog of `DeepseekV4DSparkModel`. Attribute names
    (`embed_tokens`, `layers`, `markov_head`, `confidence_head`, `vocab_size`,
    `noise_token_id`, `markov_rank`, `shared_head.norm`) are exactly what
    `DSparkWorkerV2` reads off `self._draft_inner` -- see
    `dspark_worker_v2.py` `_build_refine_refs` / `__init__`.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.hidden_size)
        self.rms_norm_eps = float(config.rms_norm_eps)

        # head_dim MUST come from config, never derived: for this checkpoint
        # family hidden_size // num_attention_heads (2560 // 32 == 80) is NOT
        # the real head_dim (128). DFlashAttention (subclassed below) already
        # reads config.head_dim first via getattr and only derives as a
        # fallback, so this checkpoint loads correctly either way -- this
        # assertion just makes the requirement explicit and fails loudly
        # instead of silently deriving 80 if a future config regresses it.
        head_dim = getattr(config, "head_dim", None)
        if not head_dim or int(head_dim) <= 0:
            raise ValueError(
                "Qwen3DSparkModel requires a positive config.head_dim (must not "
                "be derived from hidden_size // num_attention_heads for this "
                f"checkpoint family: {self.hidden_size}//{config.num_attention_heads} "
                "!= head_dim)."
            )

        target_layer_ids = getattr(config, "target_layer_ids", None)
        if not target_layer_ids:
            raise ValueError(
                "Qwen3DSparkModel requires a non-empty config.target_layer_ids."
            )
        self.target_layer_ids = [int(x) for x in target_layer_ids]

        block_size = getattr(config, "block_size", None)
        if not block_size or int(block_size) <= 0:
            raise ValueError("Qwen3DSparkModel requires a positive config.block_size.")
        self.block_size = int(block_size)

        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if not num_hidden_layers or int(num_hidden_layers) <= 0:
            raise ValueError(
                "Qwen3DSparkModel requires a positive config.num_hidden_layers."
            )
        self.num_dspark_layers = int(num_hidden_layers)

        # DSparkWorkerV2 fills undrafted block positions with this id (the
        # released checkpoints spell it `mask_token_id`, matching the
        # DeepSpec reference's training-time `create_noise_embed`; V4's own
        # config instead spells it `dspark_noise_token_id`).
        mask_token_id = getattr(config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("Qwen3DSparkModel requires config.mask_token_id.")
        self.noise_token_id = int(mask_token_id)

        self.markov_rank = int(getattr(config, "markov_rank", 0) or 0)
        self.enable_confidence_head = bool(
            getattr(config, "enable_confidence_head", False)
        )
        # DSparkWorkerV2._refine_block_markov_sharded always calls the
        # confidence head with the Markov embedding concatenated in; there is
        # no served path for a hidden-only confidence head.
        if self.enable_confidence_head and not getattr(
            config, "confidence_head_with_markov", True
        ):
            raise ValueError(
                "Qwen3DSparkModel/DSparkWorkerV2 only support "
                "confidence_head_with_markov=True."
            )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.layers = nn.ModuleList(
            [
                Qwen3DSparkDecoderLayer(
                    config=config, layer_id=i, quant_config=quant_config
                )
                for i in range(self.num_dspark_layers)
            ]
        )

        # Combine step for the per-token target context feature: fc THEN
        # hidden_norm, in that order (see module docstring / project_main_hidden).
        # Do not skip hidden_norm and do not reorder -- DFlash's own
        # `project_target_hidden` (dflash.py) does this correctly; copying
        # the layer stack without also copying this exact order/composition
        # would silently produce wrong context features.
        self.fc = nn.Linear(
            len(self.target_layer_ids) * self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # Final backbone norm. Applied by
        # DSparkWorkerV2._refine_block_markov_sharded (as `shared_head.norm`,
        # matching DeepseekV4DSparkModel's own attribute path), NOT here:
        # forward() must return the RAW block hidden state so the worker's
        # fused Markov-refine loop can norm + lm_head-project it itself (and
        # fold that whole step into one cuda-graph capture). Applying it
        # again inside forward() would double-normalize.
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        self.markov_head: Optional[Qwen3DSparkMarkovHead] = None
        if self.markov_rank > 0:
            self.markov_head = Qwen3DSparkMarkovHead(
                self.vocab_size,
                self.markov_rank,
                quant_config=quant_config,
                prefix=add_prefix("markov_head", prefix),
            )

        self.confidence_head: Optional[Qwen3DSparkConfidenceHead] = None
        if self.enable_confidence_head:
            if self.markov_head is None:
                raise ValueError(
                    "Qwen3DSparkModel: enable_confidence_head requires "
                    "markov_rank > 0 (DSparkWorkerV2 always calls the "
                    "confidence head with the Markov embedding concatenated in)."
                )
            self.confidence_head = Qwen3DSparkConfidenceHead(
                self.hidden_size + self.markov_rank
            )

    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """fc THEN hidden_norm -- see the comment on `self.fc` above."""
        return self.hidden_norm(self.fc(main_hidden))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual: Optional[torch.Tensor] = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        if hidden_states.numel() != 0 and residual is not None:
            # Fold in the last layer's pending residual WITHOUT applying the
            # final norm's weight scaling -- see the shared_head.norm comment
            # in __init__. (DFlashDraftModel.forward instead calls
            # `self.norm(hidden_states, residual)` and keeps the *normed*
            # output here, which would double-normalize for this model.)
            hidden_states = hidden_states + residual
        return hidden_states


class Qwen3DSparkWeightMapping(msgspec.Struct, omit_defaults=True):
    """Resolution of one checkpoint tensor name for `Qwen3DSparkModel`.

    Either `dest_param` (a `named_parameters()` key, plus `shard_id` for the
    qkv_proj/gate_up_proj stacked params) is set, or `drop_reason` is set
    (the tensor is legitimately absent from this config's parameter set, e.g.
    a Markov/confidence tensor when that head is gated off). Exactly one of
    the two is ever set. Deliberately dependency-light (no torch distributed,
    no model instantiation) so it can be exercised standalone -- see the
    phase-3 parity script.
    """

    checkpoint_name: str
    dest_param: Optional[str] = None
    shard_id: Optional[Union[str, int]] = None
    drop_reason: Optional[str] = None


# (param_name, weight_name, shard_id), identical in spirit to DFlashDraftModel's
# own stacked_params_mapping (dflash.py) -- both checkpoints fuse q/k/v and
# gate/up the same way.
_STACKED_PARAMS: Tuple[Tuple[str, str, Union[str, int]], ...] = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)

# Top-level (non-"layers.*") checkpoint tensor -> destination parameter, for
# tensors that are always present regardless of the markov/confidence gates.
_TOP_LEVEL_PARAM_MAP = {
    "embed_tokens.weight": "model.embed_tokens.weight",
    "fc.weight": "model.fc.weight",
    "hidden_norm.weight": "model.hidden_norm.weight",
    "norm.weight": "model.shared_head.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# Present only when config.markov_rank > 0.
_MARKOV_PARAM_MAP = {
    "markov_head.markov_w1.weight": "model.markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight": "model.markov_head.markov_w2.weight",
}

# Present only when config.enable_confidence_head.
_CONFIDENCE_PARAM_MAP = {
    "confidence_head.proj.weight": "model.confidence_head.proj.weight",
    "confidence_head.proj.bias": "model.confidence_head.proj.bias",
}


def resolve_qwen3_dspark_weight(
    name: str,
    *,
    has_markov_head: bool,
    has_confidence_head: bool,
) -> Qwen3DSparkWeightMapping:
    """Map one raw checkpoint tensor name to its destination parameter.

    Pure function (no `self`, no torch distributed) so both
    `Qwen3DSparkModel.load_weights` and the standalone CPU parity/mapping
    script can share exactly one source of truth for the mapping table.
    Raises for a genuinely unrecognized name (a real mapping gap); returns a
    `drop_reason` only for the specific, expected case of a Markov/confidence
    tensor present in the checkpoint while this config gates that head off.
    """
    for param_name, weight_name, shard_id in _STACKED_PARAMS:
        if f".{weight_name}." not in name:
            continue
        mapped = "model." + name.replace(weight_name, param_name)
        return Qwen3DSparkWeightMapping(
            checkpoint_name=name, dest_param=mapped, shard_id=shard_id
        )

    if name.startswith("layers."):
        # Direct 1:1 renames under the backbone: input_layernorm,
        # post_attention_layernorm, self_attn.{q,k}_norm, self_attn.o_proj.
        return Qwen3DSparkWeightMapping(
            checkpoint_name=name, dest_param="model." + name
        )

    if name in _TOP_LEVEL_PARAM_MAP:
        return Qwen3DSparkWeightMapping(
            checkpoint_name=name, dest_param=_TOP_LEVEL_PARAM_MAP[name]
        )

    if name in _MARKOV_PARAM_MAP:
        if not has_markov_head:
            return Qwen3DSparkWeightMapping(
                checkpoint_name=name,
                drop_reason=(
                    "markov_head is not built for this checkpoint "
                    "(config.markov_rank == 0)"
                ),
            )
        return Qwen3DSparkWeightMapping(
            checkpoint_name=name, dest_param=_MARKOV_PARAM_MAP[name]
        )

    if name in _CONFIDENCE_PARAM_MAP:
        if not has_confidence_head:
            return Qwen3DSparkWeightMapping(
                checkpoint_name=name,
                drop_reason=(
                    "confidence_head is not built for this checkpoint "
                    "(config.enable_confidence_head is false)"
                ),
            )
        return Qwen3DSparkWeightMapping(
            checkpoint_name=name, dest_param=_CONFIDENCE_PARAM_MAP[name]
        )

    raise ValueError(
        f"Qwen3DSparkModel: checkpoint tensor {name!r} has no known destination "
        "parameter and no stated drop reason -- add it to the mapping."
    )


class Qwen3DSparkModel(nn.Module):
    """EntryClass.

    The class name matches the checkpoint's `architectures` string exactly
    (the model registry keys on `cls.__name__`, see
    `python/sglang/srt/models/registry.py` `import_model_classes`), so this
    resolves with zero config rewriting in
    `ModelConfig._config_draft_model`. Do not rename this class: upstream
    PR #29917 broke registration by adding a `Draft` suffix to the outer
    class, which no longer matched `architectures[0]` == "Qwen3DSparkModel".

    Serves both released Qwen3 DSpark checkpoints from one class via config
    gates -- see `Qwen3DSparkBackbone`. `dspark_qwen3_4b_block7` has both
    `markov_head` and `confidence_head`; the sibling `dflash_qwen3_4b_block7`
    (markov_rank=0, enable_confidence_head=false) has neither and is meant to
    run under the DFLASH algorithm/`DFlashWorkerV2` -- this class targets the
    `DSparkWorkerV2` contract only (see the module docstring).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        # Unlike DeepseekV4ForCausalLMDSpark (which ties embed_tokens/lm_head
        # to the target unconditionally), this checkpoint trains and ships
        # its own embed_tokens/lm_head (config.tie_word_embeddings=False).
        # DSparkWorkerV2.__init__ reads this flag to skip its target-weight
        # tying for models that own real trained vocab weights.
        self.owns_vocab_weights = True

        self.model = Qwen3DSparkBackbone(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_flags().enable_dp_lm_head,
        )
        # Unused by forward() below: DSparkWorkerV2 computes logits itself
        # (norm + lm_head + Markov refine fused into its own cuda-graph-
        # capturable step, see _refine_block_markov_sharded). Constructed
        # anyway for parity with every other CausalLM model class, matching
        # DeepseekV4ForCausalLMDSpark's own unused self.logits_processor.
        self.logits_processor = LogitsProcessor(config)

    @property
    def block_size(self) -> int:
        return self.model.block_size

    @property
    def num_dspark_layers(self) -> int:
        return self.model.num_dspark_layers

    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        return self.model.project_main_hidden(main_hidden)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        block_hidden = self.model(input_ids, positions, forward_batch)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=block_hidden)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        has_markov_head = self.model.markov_head is not None
        has_confidence_head = self.model.confidence_head is not None
        params_dict = dict(self.named_parameters())
        loaded_params: set = set()

        for name, loaded_weight in weights:
            mapping = resolve_qwen3_dspark_weight(
                name,
                has_markov_head=has_markov_head,
                has_confidence_head=has_confidence_head,
            )
            if mapping.dest_param is None:
                logger.info(
                    "Qwen3DSparkModel: dropping checkpoint tensor %s (%s).",
                    name,
                    mapping.drop_reason,
                )
                continue
            if mapping.dest_param not in params_dict:
                raise ValueError(
                    f"Qwen3DSparkModel: checkpoint tensor {name!r} mapped to "
                    f"{mapping.dest_param!r}, which is not a model parameter."
                )
            param = params_dict[mapping.dest_param]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if mapping.shard_id is not None:
                weight_loader(param, loaded_weight, mapping.shard_id)
            else:
                weight_loader(param, loaded_weight)
            loaded_params.add(mapping.dest_param)

        missing = sorted(set(params_dict) - loaded_params)
        if missing:
            raise ValueError(
                f"Qwen3DSparkModel: {len(missing)} parameter(s) never received "
                f"a checkpoint weight: {missing}"
            )


EntryClass = [Qwen3DSparkModel]
