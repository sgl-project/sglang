# Copyright 2023-2024 SGLang Team
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

"""Inference-only DeepSeek NextN Speculative Decoding."""

import logging
import os
from contextlib import ExitStack
from typing import Iterable, Optional, Tuple

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import PretrainedConfig

from sglang.kernels.ops.layernorm.fused_eh_norm import fused_eh_norm
from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_cp_split,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_round_robin_split,
)
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.cp.utils import is_cp_v2_active
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import Fp8Config
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils.cp_utils import (
    can_cp_split,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_mla_prefill_cp_enabled,
    mla_use_prefill_cp,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_common.utils import enable_nextn_moe_bf16_cast_to_fp8
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV3ForCausalLM
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.runtime_context import (
    get_model,
    get_parallel,
    get_spec,
)
from sglang.srt.utils import BumpAllocator, add_prefix, is_cuda, is_npu


def _gather_dsa_topk_indices_for_cp(
    topk_indices: torch.Tensor,
    local_num_tokens: int,
    cp_size: int,
    forward_batch: ForwardBatch,
    stream,
) -> torch.Tensor:
    if (
        is_dsa_prefill_cp_round_robin_split()
        and topk_indices.shape[0] < local_num_tokens
    ):
        pad_rows = local_num_tokens - topk_indices.shape[0]
        topk_indices = torch.cat(
            [
                topk_indices,
                topk_indices.new_full((pad_rows, topk_indices.shape[1]), -1),
            ],
            dim=0,
        )
    return cp_all_gather_rerange_output(
        topk_indices,
        cp_size,
        forward_batch,
        stream,
    )


logger = logging.getLogger(__name__)


_is_cuda = is_cuda()
_is_npu = is_npu()


class DeepseekModelNextN(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if enable_nextn_moe_bf16_cast_to_fp8(quant_config):
            # refer to real DeepSeek V3 quant config
            moe_quant_config_override = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[128, 128],
            )
        else:
            moe_quant_config_override = None

        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            logger.warning(
                "Overriding DeepseekV3ForCausalLMNextN quant config for modelopt_fp4 Deepseek model."
            )
            quant_config = None

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
            **get_embedding_tp_kwargs(),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if quant_config is not None and quant_config.get_name() == "quark":
            self.eh_proj = ReplicatedLinear(
                2 * config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("eh_proj", prefix),
            )
        else:
            self.eh_proj = nn.Linear(
                2 * config.hidden_size, config.hidden_size, bias=False
            )

        self.rot_weight = None
        if _is_npu:
            rot_weight_path = get_model().model_path + "/rot.safetensors"
            if os.path.isfile(rot_weight_path):
                self.rot_weight = load_file(rot_weight_path)
                self.rot_weight = self.rot_weight["rot.weight"].npu()

        self.alt_stream = (
            torch.cuda.Stream()
            if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            else None
        )

        layer_name = "decoder"
        if _is_npu and (
            get_spec().speculative_draft_model_path == get_model().model_path
        ):
            layer_name = "layers." + str(config.num_hidden_layers)

        self.quant_config = quant_config
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.mla_enable_prefill_cp = (
            is_mla_prefill_cp_enabled() and not is_deepseek_dsa(config)
        )
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_size = None
        self.decoder = DeepseekV2DecoderLayer(
            config,
            0,
            quant_config=quant_config,
            moe_quant_config_override=moe_quant_config_override,
            is_nextn=True,
            prefix=add_prefix(layer_name, prefix),
            alt_stream=self.alt_stream,
            dsa_enable_prefill_cp=self.dsa_enable_prefill_cp,
            mla_enable_prefill_cp=self.mla_enable_prefill_cp,
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        exit_stack = ExitStack()
        if (
            _is_npu
            and self.quant_config is None
            and get_model().quantization is not None
        ):
            # ascend mtp unquant
            exit_stack.enter_context(envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True))
            exit_stack.enter_context(
                envs.DEEP_NORMAL_MODE_USE_INT8_QUANT.override(False)
            )

        try:
            zero_allocator = BumpAllocator(
                buffer_size=2,
                dtype=torch.float32,
                device=(
                    input_embeds.device
                    if input_embeds is not None
                    else input_ids.device
                ),
            )

            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds

            if hidden_states.shape[0] > 0:
                previous_hidden_states = forward_batch.spec_info.hidden_states
                if self.rot_weight is not None:
                    previous_hidden_states = torch.matmul(
                        previous_hidden_states, self.rot_weight
                    )
                if _is_cuda:
                    eh_input = fused_eh_norm(
                        hidden_states,
                        previous_hidden_states,
                        self.enorm.weight,
                        self.hnorm.weight,
                        self.enorm.variance_epsilon,
                    )
                else:
                    eh_input = torch.cat(
                        (
                            self.enorm(hidden_states),
                            self.hnorm(previous_hidden_states),
                        ),
                        dim=-1,
                    )
                if isinstance(self.eh_proj, ReplicatedLinear):
                    hidden_states, _ = self.eh_proj(eh_input)
                else:
                    hidden_states = self.eh_proj(eh_input)

            # CP-v2 shards/gathers at the eager-runner boundary instead.
            use_cp_v1 = (
                dsa_use_prefill_cp(forward_batch, self.dsa_enable_prefill_cp)
                or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp)
            ) and not is_cp_v2_active(forward_batch)
            if use_cp_v1:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
                positions = cp_split_and_rebuild_position(forward_batch, positions)
            residual = None
            index_topk_share = IndexTopKShareState(forward_batch)
            with get_global_expert_distribution_recorder().disable_this_region():
                hidden_states, residual, topk_indices = self.decoder(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    zero_allocator,
                    prev_topk_indices=index_topk_share.prev_topk_indices(),
                )
            if not forward_batch.forward_mode.is_idle():
                if residual is not None:
                    hidden_states, _ = self.shared_head.norm(hidden_states, residual)
                else:
                    hidden_states = self.shared_head.norm(hidden_states)

                if use_cp_v1:
                    local_num_tokens = hidden_states.shape[0]
                    hidden_states = cp_all_gather_rerange_output(
                        hidden_states,
                        self.cp_size,
                        forward_batch,
                        torch.cuda.current_stream(),
                    )
                    if index_topk_share.should_update and topk_indices is not None:
                        topk_indices = _gather_dsa_topk_indices_for_cp(
                            topk_indices,
                            local_num_tokens,
                            self.cp_size,
                            forward_batch,
                            torch.cuda.current_stream(),
                        )
            index_topk_share.store_topk_indices(topk_indices)
        finally:
            exit_stack.close()

        return hidden_states


class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):

    # Support amd/DeepSeek-R1-0528-MXFP4 renaming: model.layers.61*.
    # Ref: HF config.json for amd/DeepSeek-R1-0528-MXFP4
    # https://huggingface.co/amd/DeepSeek-R1-0528-MXFP4/blob/main/config.json
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "model.layers.61": "model.decoder",
        },
    )

    def _resolve_nextn_quant_config(self, config, quant_config):
        if quant_config is None or quant_config.get_name() != "quark":
            return quant_config

        from sglang.srt.layers.quantization.quark.utils import should_ignore_layer

        ckpt_prefix = f"model.layers.{config.num_hidden_layers}"
        mapped_prefix = self.hf_to_sglang_mapper._map_name(ckpt_prefix)
        if should_ignore_layer(mapped_prefix, quant_config.exclude_layers):
            return None
        return quant_config

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        # if not set, model load will be broken in DeepseekV3ForCausalLM load_weights()
        self.pp_group = get_pp_group()
        self.determine_num_fused_shared_experts("DeepseekV3ForCausalLMNextN")
        self.use_dsa = is_deepseek_dsa(config)
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.mla_enable_prefill_cp = is_mla_prefill_cp_enabled() and not self.use_dsa
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
            self.cp_rank = get_parallel().attn_cp_rank
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_rank = None
            self.cp_size = None

        nextn_quant_config = self._resolve_nextn_quant_config(config, quant_config)

        self.model = DeepseekModelNextN(
            config, nextn_quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        if self.dsa_enable_prefill_cp:
            if can_dsa_cp_split(
                len(input_ids), self.cp_size, self.use_dsa, forward_batch
            ):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
        elif self.mla_enable_prefill_cp:
            if can_cp_split(len(input_ids), self.cp_size, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)

    def post_load_weights(self, is_nextn=True, weight_names=None):
        # `is_nextn` is pinned to True for the NextN subclass; the parameter is kept
        # only because the mixin's `do_load_weights` calls `self.post_load_weights`
        # with `is_nextn=...` as a kwarg.
        super().post_load_weights(is_nextn=True, weight_names=weight_names)


EntryClass = [DeepseekV3ForCausalLMNextN]
