"""Inference-only Qwen4-Exp MTP speculative decoding."""

import copy
import logging
from contextlib import ExitStack
from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.utils import get_dynamic_override
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.models.qwen4_exp import Qwen4ExpModel
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)


def _is_mtp_dynamically_unquantized(quant_config, prefix: str) -> bool:
    dynamic = getattr(quant_config, "dynamic", None)
    return bool(dynamic) and get_dynamic_override(quant_config, prefix) is False


def _mtp_quant_config(quant_config):
    """The quantization the MTP module itself is built with.

    RadixArk's ModelOpt FP4 checkpoint excludes the complete MTP module and
    stores it in BF16. Inferact's variant excludes the dense MTP projections
    but provides the MTP experts in NVFP4. Preserve ModelOpt quantization for
    the latter so the per-layer exclusion rules can select the right methods.
    """
    if quant_config and quant_config.get_name() == "modelopt_fp4":
        if quant_config.is_layer_excluded("mtp.layers.0.mlp.experts"):
            return None
        return quant_config
    if (
        is_npu()
        and get_global_server_args().speculative_draft_model_quantization is None
    ):
        return None
    if quant_config and quant_config.get_name() == "quark":
        exclude_layers = getattr(quant_config, "exclude_layers", [])
        if any(
            isinstance(layer, str) and layer.startswith("mtp.")
            for layer in exclude_layers
        ):
            return None
    if quant_config and _is_mtp_dynamically_unquantized(
        quant_config, add_prefix("mtp", "")
    ):
        return None
    return quant_config


class Qwen4ExpForCausalLMMTP(Qwen3_5ForCausalLMMTP):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.is_multimodal = hasattr(config, "text_config")
        if self.is_multimodal:
            config = config.text_config

        # Deepcopy so MTP-only mutations below don't leak into the main model.
        config = copy.deepcopy(config)
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        config.full_attention_interval = 1
        config.ple_layer_ids = []

        quant_config = _mtp_quant_config(quant_config)

        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        # The draft passes the fused HC stream in the inputs_embeds position.
        # Register the replay slot at HC width, not the token embedding width.
        self.input_embeds_hidden_size = self.hidden_size * self.hc_count
        self._mtp_input_fusion = self._init_mtp_input_fusion(config)

        self.model = Qwen4ExpModel(
            config,
            quant_config,
            prefix=add_prefix("mtp", prefix),
            is_nextn=True,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=is_dp_attention_enabled(),
        )
        self.logits_processor = LogitsProcessor(config)

    def _init_pre_fc_norms(self, config: PretrainedConfig) -> None:
        self.pre_fc_norm_embedding = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        hidden_norm_size = (
            self.hc_count * config.hidden_size
            if self.hc_count > 1
            else config.hidden_size
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            hidden_norm_size, eps=config.rms_norm_eps
        )

    def _init_linear_projections(self, config: PretrainedConfig) -> None:
        self.fc_embedding = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.fc_hidden = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _init_standard_fusion(self, config: PretrainedConfig):
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self._init_pre_fc_norms(config)
        return self._fuse_standard

    def _init_mtp_input_fusion(self, config: PretrainedConfig):
        if self.hc_count <= 1:
            return self._init_standard_fusion(config)

        self._init_linear_projections(config)
        self._init_pre_fc_norms(config)
        return self._fuse_residual_linear_shared

    def _fuse_residual_linear_shared(
        self, input_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        input_embeds = self.fc_embedding(self.pre_fc_norm_embedding(input_embeds))
        orig_shape = hidden_states.shape
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        decoder_view = hidden_states.view(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        encoder_inputs = self.fc_hidden(decoder_view)
        return (input_embeds.unsqueeze(-2) + encoder_inputs).view(orig_shape)

    def _fuse_standard(
        self, input_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        input_embeds = self.pre_fc_norm_embedding(input_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        return self.fc(torch.cat((input_embeds, hidden_states), dim=-1))

    def _npu_quant_context(self):
        # NPU-only quant context; the V100 fork has no NPU backend.
        return ExitStack()

    def _prepare_input_embeds(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert input_embeds is None
        input_embeds = forward_batch.mm_input_embeds
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.contains_mm_inputs()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            assert input_embeds is not None
            last_indices = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            ).long()
            input_embeds[last_indices] = self.model.embed_tokens(
                input_ids[last_indices]
            )
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)
        return input_embeds

    def _set_hc_logits_hidden_states(
        self,
        logits_output,
        hc_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> None:
        if hc_hidden_states is None:
            return

        # EAGLE v2 stores one hidden state per request in the future map.
        # When draft extend emits a token-shaped HC tensor, keep only the
        # last token per request so the overlap cache sees [bs, hidden].
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and forward_batch.extend_seq_lens is not None
            and hc_hidden_states.shape[0] != forward_batch.extend_seq_lens.shape[0]
        ):
            last_index = (
                torch.cumsum(forward_batch.extend_seq_lens.to(torch.int64), dim=0) - 1
            )
            hc_hidden_states = hc_hidden_states[last_index]

        assert hc_hidden_states.shape[-1] == self.hc_count * self.hidden_size
        logits_output.hidden_states = hc_hidden_states

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        with self._npu_quant_context():
            input_embeds = self._prepare_input_embeds(
                input_ids, forward_batch, input_embeds
            )
            hidden_states = forward_batch.spec_info.hidden_states
            if not forward_batch.forward_mode.is_idle():
                hidden_states = self._mtp_input_fusion(input_embeds, hidden_states)

            with get_global_expert_distribution_recorder().disable_this_region():
                model_output = self.model(
                    input_ids,
                    positions,
                    forward_batch,
                    hidden_states,
                )

            hc_hidden_states = None
            if isinstance(model_output, tuple):
                hidden_states, hc_hidden_states = model_output
            else:
                hidden_states = model_output

        logits_output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        self._set_hc_logits_hidden_states(
            logits_output, hc_hidden_states, forward_batch
        )
        return logits_output


EntryClass = [Qwen4ExpForCausalLMMTP]
