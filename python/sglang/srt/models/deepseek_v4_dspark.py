import logging
import os
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_rank
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    dp_gather_replicate,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

COMPRESS_RATIO_DSPARK_LAYER = 0


class DSparkMarkovHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size,
            markov_rank,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("markov_w1", prefix),
        )
        self.markov_w2 = ParallelLMHead(
            vocab_size,
            markov_rank,
            prefix=add_prefix("markov_w2", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def project_bias(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(embeddings, self.markov_w2.weight)


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        hidden_dim = hidden.shape[-1]
        weight = self.proj.weight
        hidden_score = F.linear(hidden.float(), weight[:, :hidden_dim])
        markov_score = F.linear(markov_embed.float(), weight[:, hidden_dim:])
        return (hidden_score + markov_score).squeeze(-1)


class DeepseekV4DSparkModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.block_size = config.dspark_block_size
        self.markov_rank = config.dspark_markov_rank
        self.noise_token_id = config.dspark_noise_token_id
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = num_dspark_layers = get_dspark_num_layers(config)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.main_proj = ReplicatedLinear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("main_proj", prefix),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    is_nextn=True,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_streams=None,
                    compress_ratio_override=COMPRESS_RATIO_DSPARK_LAYER,
                )
                for layer_id in range(num_dspark_layers)
            ]
        )

        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=add_prefix("markov_head", prefix),
        )
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size + config.dspark_markov_rank
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() > 0:
            from sglang.srt.layers.mhc_head import fused_hc_head

            return fused_hc_head(
                x.contiguous(),
                hc_fn,
                hc_scale,
                hc_base,
                norm_eps=self.rms_norm_eps,
                hc_eps=self.hc_eps,
            )
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        projected, _ = self.main_proj(main_hidden)
        return self.main_norm(projected)

    def forward_backbone(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        if get_parallel().attn_dp_size > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (_DpGatheredBufferWrapper._global_dp_buffer_len, 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            dp_gather_replicate(input_ids_global, input_ids[:, None], forward_batch)
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        prev_residual, prev_post, prev_comb = None, None, None
        last_layer = None
        for layer in self.layers:
            last_layer = layer
            hidden_states, prev_residual, prev_post, prev_comb = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
                prev_residual=prev_residual,
                prev_post=prev_post,
                prev_comb=prev_comb,
            )
        if (
            last_layer is not None
            and last_layer.use_fused_mhc_post_pre
            and prev_residual is not None
        ):
            hidden_states = last_layer.hc_post(
                hidden_states, prev_residual, prev_post, prev_comb
            )
        return hidden_states

    def collapse_block_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        # Match vLLM DeepSeek-V4 DSpark: draft forward returns the pre-norm
        # hc_head hidden. The shared output norm is applied only when computing
        # base LM logits.
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        spec_info = getattr(forward_batch, "spec_info", None)
        main_hidden = getattr(spec_info, "hidden_states", None)
        spec_name = None
        if getattr(spec_info, "spec_input_type", None) is not None:
            spec_name = getattr(spec_info.spec_input_type, "name", None)
        if spec_name == "DSPARK_DRAFT_EXTEND" and main_hidden is not None:
            raise RuntimeError(
                "DSpark DRAFT_EXTEND with target hidden is not DeepSpec-aligned. "
                "Target hidden must materialize context KV; draft forward should "
                "run from noise/block token embeddings."
            )
        hidden_states = self.forward_backbone(input_ids, positions, forward_batch)
        return self.collapse_block_hidden(hidden_states)


def get_dspark_num_layers(config: PretrainedConfig) -> int:
    return int(getattr(config, "dspark_num_layers", 0) or 3)


class DeepseekV4ForCausalLMDSpark(DeepseekV4ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()

        self.model = DeepseekV4DSparkModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False

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
    ) -> torch.Tensor:
        return self.model(input_ids, positions, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = list(weights)
        self.has_own_embed_tokens = any(
            name.startswith("mtp.")
            and (
                ".emb.tok_emb." in name
                or ".embed_tokens." in name
            )
            for name, _ in weights
        )
        self.has_own_lm_head = any(
            name.startswith("mtp.")
            and (
                ".head." in name
                or ".lm_head." in name
                or ".shared_head.head." in name
            )
            for name, _ in weights
        )
        super().load_weights(weights, is_nextn=False, is_dspark=True)
        self._maybe_use_absolute_moe_layer_ids()

    def _maybe_use_absolute_moe_layer_ids(self) -> None:
        if os.getenv("SGLANG_DSPARK_USE_ABSOLUTE_MOE_LAYER_ID", "1") != "1":
            return
        if get_global_server_args().enable_eplb:
            if get_tensor_model_parallel_rank() == 0:
                logger.warning(
                    "Skip SGLANG_DSPARK_USE_ABSOLUTE_MOE_LAYER_ID because "
                    "enable_eplb is set; absolute draft MoE ids can index past "
                    "the target expert-location table."
                )
            return

        base_layer_id = int(getattr(self.config, "num_hidden_layers", 0))
        mapping = []
        for local_layer_id, layer in enumerate(self.model.layers):
            absolute_layer_id = base_layer_id + int(local_layer_id)
            mapping.append((int(local_layer_id), int(absolute_layer_id)))
            layer.layer_id = absolute_layer_id
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            mlp.layer_id = absolute_layer_id
            topk = getattr(mlp, "topk", None)
            if topk is not None:
                topk.layer_id = absolute_layer_id
            experts = getattr(mlp, "experts", None)
            if experts is not None:
                experts.layer_id = absolute_layer_id
                runner_config = getattr(experts, "moe_runner_config", None)
                if runner_config is not None:
                    runner_config.layer_id = absolute_layer_id
        if get_tensor_model_parallel_rank() == 0:
            logger.warning(
                "DSpark absolute draft layer ids enabled "
                "(KV/cache layer ids remain local): %s",
                mapping,
            )

    def post_load_weights(self, is_nextn=False, is_dspark=False, weight_names=None):
        super().post_load_weights(is_dspark=True, weight_names=weight_names)


EntryClass = [DeepseekV4ForCausalLMDSpark]
