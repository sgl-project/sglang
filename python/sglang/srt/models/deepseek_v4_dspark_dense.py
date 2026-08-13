from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v4_dspark import DSparkAttention, DSparkV4MarkovHead
from sglang.srt.models.dspark import DSparkConfidenceHead, gather_and_crop_vocab
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_draft_model import (
    CommitKvProj,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

_DEFAULT_DSPARK_DENSE_NUM_LAYERS = 3


def get_dspark_num_layers(config: DeepSeekV4Config) -> int:
    num_layers = getattr(config, "dspark_num_layers", None)
    return int(num_layers) if num_layers else _DEFAULT_DSPARK_DENSE_NUM_LAYERS


class DeepseekV4DSparkDenseMLP(nn.Module):
    """Dense SwiGLU MLP for the DSpark dense draft (no MoE, no Hyper-Connection)."""

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        intermediate_size = int(config.moe_intermediate_size)
        self.gate_proj = ReplicatedLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = ReplicatedLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = ReplicatedLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        down, _ = self.down_proj(self.act_fn(gate) * up)
        return down


class DeepseekV4DSparkDenseLayer(nn.Module):
    """Pre-norm residual block: DSpark MLA attention (plain RoPE, no YaRN)
    followed by a dense SwiGLU MLP. No Hyper-Connection streams."""

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__()
        self.self_attn = DSparkAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
        )
        self.mlp = DeepseekV4DSparkDenseMLP(
            config, quant_config=quant_config, prefix=add_prefix("mlp", prefix)
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, prev_residual)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def build_dense_confidence_head(
    *, config: DeepSeekV4Config, markov_rank: int
) -> Optional[DSparkConfidenceHead]:
    """Build the dense draft's confidence head.

    Unlike the MoE draft (``bias=False``), the dense draft was trained with a
    bias term on the confidence projection, so the head is built with
    ``bias=True`` to stay weight-compatible with the checkpoint.
    """
    if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
        return None
    if markov_rank <= 0:
        raise ValueError(
            "DSpark dense confidence head requires markov_rank > 0, "
            f"got markov_rank={markov_rank}."
        )
    return DSparkConfidenceHead(
        hidden_size=int(config.hidden_size),
        markov_rank=int(markov_rank),
        with_markov=True,
        bias=True,
    )


class DeepseekV4ForCausalLMDSparkDense(nn.Module):
    """Dense DSpark self-speculative draft for DeepSeek-V4.

    Implements the same worker-facing interface as the MoE draft
    (``DeepseekV4ForCausalLMDSpark``) — ``attach_shared_modules`` /
    ``forward_embed`` / ``forward`` / ``project_target_hidden`` /
    ``write_target_hidden_kv`` / ``compute_base_logits`` / ``markov_head`` /
    ``confidence_head`` — but uses a plain dense backbone: its own token
    embedding, MLA split-Q attention with plain RoPE (no YaRN), a dense SwiGLU
    MLP, and no Hyper-Connection streams.

    The ``lm_head`` is shared with the target (attached at runtime); the token
    embedding is owned by the draft and loaded from the draft checkpoint.
    """

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark dense draft requires markov_rank > 0, "
                f"got markov_rank={dspark_config.markov_rank}."
            )
        self.markov_rank = int(dspark_config.markov_rank)
        self.gamma = int(
            dspark_config.resolve_gamma(default=int(config.num_hidden_layers))
        )
        self.block_size = self.gamma
        self.num_dspark_layers = get_dspark_num_layers(config)
        self.start_layer = 0
        self.end_layer = self.num_dspark_layers

        if dspark_config.target_layer_ids is not None:
            self.num_target_features = len(dspark_config.target_layer_ids)
        elif dspark_config.num_target_layers is not None:
            self.num_target_features = int(dspark_config.num_target_layers)
        else:
            self.num_target_features = int(config.num_hidden_layers)

        use_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and envs.SGLANG_DSPARK_ENABLE_MULTI_STREAM.get()
            and torch.cuda.is_available()
        )
        self.alt_streams: Optional[List[torch.cuda.Stream]] = (
            [torch.cuda.Stream()] if use_multi_stream else None
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.fc = ReplicatedLinear(
            self.num_target_features * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                DeepseekV4DSparkDenseLayer(
                    config=config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_streams=self.alt_streams,
                )
                for layer_id in range(self.num_dspark_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.markov_head = DSparkV4MarkovHead(
            vocab_size=int(config.vocab_size),
            markov_rank=self.markov_rank,
        )
        self.confidence_head = build_dense_confidence_head(
            config=config, markov_rank=self.markov_rank
        )

        self.lm_head: Optional[nn.Module] = None
        self._use_fp32_lm_head = envs.SGLANG_DSPARK_FP32_LM_HEAD.get()
        self._opt_markov_w2_tp_shard = envs.SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD.get()

    @property
    def enable_confidence_head(self) -> bool:
        return self.confidence_head is not None

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        # The dense draft owns its token embedding (loaded from the draft
        # checkpoint); only the lm_head is shared with the target.
        del embed_tokens
        self.lm_head = lm_head
        self.markov_head.configure_tp_shard(lm_head=lm_head)

    def project_target_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        projected, _ = self.fc(main_hidden)
        return self.hidden_norm(projected)

    def write_target_hidden_kv(
        self,
        *,
        main_hidden: torch.Tensor,
        swa_loc: torch.Tensor,
        positions: torch.Tensor,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        main_x = self.project_target_hidden(main_hidden)
        swa_loc = swa_loc.to(torch.int32)
        kvs = CommitKvProj.execute(
            main_x=main_x,
            wkv_linears=[layer.self_attn.wkv for layer in self.layers],
        )
        for layer, kv in zip(self.layers, kvs):
            attn = layer.self_attn
            pool.set_swa_key_buffer_radix_fused_norm_rope(
                layer_id=attn.layer_id,
                swa_loc=swa_loc,
                kv=kv,
                kv_weight=attn.kv_norm.weight.data,
                eps=attn.eps,
                freqs_cis=attn.freqs_cis,
                positions=positions,
            )

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        del get_embedding, pp_proxy_tensors
        if input_embeds is None:
            hidden_states = self.forward_embed(input_ids)
        else:
            hidden_states = input_embeds
        prev_residual: Optional[torch.Tensor] = None
        for layer in self.layers:
            hidden_states, prev_residual = layer(
                positions, hidden_states, forward_batch, prev_residual
            )
        hidden_states, _ = self.norm(hidden_states, prev_residual)
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # The dense draft returns a post-norm hidden state, so the confidence
        # head consumes it directly (via the planner's generic path); there is
        # no separate pre-norm tap, hence the second return value is None.
        if self.lm_head is None:
            raise ValueError(
                "DeepseekV4ForCausalLMDSparkDense requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        weight = self.lm_head.weight
        if self._use_fp32_lm_head:
            local_logits = F.linear(hidden.float(), weight.float())
        else:
            local_logits = torch.matmul(hidden.to(weight.dtype), weight.T)
        if self._opt_markov_w2_tp_shard:
            return local_logits, None
        return gather_and_crop_vocab(local_logits, self.lm_head), None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            # The draft checkpoint stores the attention sink as
            # `...sinks_param(.weight)`; split-Q attention registers it as the
            # bare parameter `...attn_sink`.
            name = name.replace(".sinks_param.weight", ".attn_sink")
            name = name.replace(".sinks_param", ".attn_sink")
            # lm_head is shared with the target (attach_shared_modules); skip
            # any bundled copy in the draft checkpoint.
            if name.startswith("lm_head."):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if name not in params_dict:
                logger.warning(
                    "DSpark dense draft: unexpected checkpoint weight %r (no "
                    "matching parameter); skipping.",
                    name,
                )
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        self._warn_unloaded(params_dict=params_dict, loaded_params=loaded_params)
        self._assert_confidence_head_loaded(
            params_dict=params_dict, loaded_params=loaded_params
        )

    def _warn_unloaded(self, *, params_dict: dict, loaded_params: set) -> None:
        # attn k/v scales only exist for an fp8 KV cache; a bf16 draft has none.
        skip_patterns = (".k_scale", ".v_scale")
        unloaded = {
            name
            for name in (params_dict.keys() - loaded_params)
            if not any(s in name for s in skip_patterns)
        }
        if unloaded:
            logger.warning(
                "DSpark dense draft: parameters not initialized from the "
                "checkpoint: %s",
                sorted(unloaded),
            )

    def _assert_confidence_head_loaded(
        self, *, params_dict: dict, loaded_params: set
    ) -> None:
        if self.confidence_head is None:
            return
        confidence_param_names = {
            name for name in params_dict if name.startswith("confidence_head.")
        }
        missing = confidence_param_names - loaded_params
        if missing:
            raise ValueError(
                "DSpark dense confidence head is enabled but the checkpoint is "
                f"missing {sorted(missing)}. Provide a checkpoint with trained "
                "confidence weights, or run in STATIC ragged-verify mode."
            )


EntryClass = [DeepseekV4ForCausalLMDSparkDense]
