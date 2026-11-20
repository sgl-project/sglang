# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
# Gate/up checkpoint conversion adapted from vLLM PR #41026.
# Copyright contributors to the vLLM project.
"""OpenAI Privacy Filter: bidirectional MoE token classification.

The native encode API returns raw logits for every input token. BIOES decoding
and span reconstruction belong to the caller, not to the serving model.
"""

from copy import copy

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.models.gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssForCausalLM,
    GptOssModel,
    GptOssSparseMoeBlock,
)
from sglang.srt.runtime_context import attention_backends, get_parallel


def _interleave_gate_up(weights):
    # HF stores [gate | up]; the GPT-OSS expert loader consumes gate/up pairs.
    for name, weight in weights:
        if name.endswith((".gate_up_proj", ".gate_up_proj_bias")):
            gate, up = weight.chunk(2, dim=-1)
            weight = torch.stack((gate, up), dim=-1).flatten(-2).contiguous()
        yield name, weight


class OpenAIPrivacyFilterAttention(GptOssAttention):
    rope_is_neox_style = False
    attn_type = AttentionType.ENCODER_ONLY

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve HF's FP32 sink parameters, including when the backbone is BF16.
        self.sinks = nn.Parameter(self.sinks.float(), requires_grad=False)


class _FP32Router(nn.Linear):
    def forward(self, hidden_states):
        return F.linear(hidden_states.float(), self.weight, self.bias), None


class OpenAIPrivacyFilterMoE(GptOssSparseMoeBlock):
    def __init__(self, layer_id, config, quant_config=None, prefix=""):
        super().__init__(layer_id, config, quant_config, prefix)
        self.router = _FP32Router(
            config.hidden_size, config.num_local_experts, dtype=torch.float32
        )


class OpenAIPrivacyFilterEncoderLayer(GptOssDecoderLayer):
    attention_cls = OpenAIPrivacyFilterAttention
    moe_cls = OpenAIPrivacyFilterMoE

    def __init__(self, config, **kwargs):
        # RadixAttention accepts the radius directly; GPT-OSS subtracts one.
        super().__init__(config, sliding_window_size=config.sliding_window, **kwargs)


class OpenAIPrivacyFilterForTokenClassification(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if quant_config is not None:
            raise ValueError(
                "OpenAI Privacy Filter currently requires unquantized weights"
            )
        if get_parallel().tp_size != 1 or get_parallel().pp_size != 1:
            raise ValueError("OpenAI Privacy Filter currently requires TP=1 and PP=1")
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise ValueError("OpenAI Privacy Filter currently requires NVIDIA CUDA")
        if attention_backends()[0] not in ("fa3", "fa4"):
            raise ValueError(
                "OpenAI Privacy Filter requires --attention-backend fa3 or fa4"
            )
        if (
            attention_backends()[0] == "fa3"
            and torch.cuda.get_device_capability()[0] < 9
        ):
            # The FA2 fallback does not implement attention sinks.
            raise ValueError("OpenAI Privacy Filter with FA3 requires SM90 or newer")
        if not get_moe_runner_backend().is_triton():
            raise ValueError(
                "OpenAI Privacy Filter requires --moe-runner-backend triton"
            )

        # Normalize the config for the shared GPT-OSS backbone without mutating
        # the HF config that other parts of the server inspect.
        self.config = copy(config)
        self.config.layer_types = ["sliding_attention"] * config.num_hidden_layers
        self.config.hidden_act = "silu"
        self.config.hidden_act_alpha = config.swiglu_alpha
        self.quant_config = None
        self.model = GptOssModel(
            self.config,
            prefix=add_prefix("model", prefix),
            decoder_layer_type=OpenAIPrivacyFilterEncoderLayer,
        )
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=True,
    ):
        if not get_embedding:
            raise ValueError("OpenAI Privacy Filter only supports token classification")
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        logits = self.score(hidden_states.to(self.score.weight.dtype)).float()
        lengths = forward_batch.extend_seq_lens_cpu or []
        # Graph execution may pad the token axis; padding is not a request.
        logits = logits[: sum(lengths)]
        return EmbeddingPoolerOutput(embeddings=list(logits.split(lengths)))

    def load_weights(self, weights):
        self._load_normal_weights(
            _interleave_gate_up(weights), is_nextn=False, weight_name_mapping=None
        )

    _load_normal_weights = GptOssForCausalLM._load_normal_weights
    _get_default_weight_mapping = GptOssForCausalLM._get_default_weight_mapping
    get_input_embeddings = GptOssForCausalLM.get_input_embeddings
    get_model_config_for_expert_location = (
        GptOssForCausalLM.get_model_config_for_expert_location
    )
    start_layer = GptOssForCausalLM.start_layer
    end_layer = GptOssForCausalLM.end_layer

    @property
    def routed_experts_weights_of_layer(self):
        return {
            i: self.model.layers[i].mlp.get_moe_weights()
            for i in range(self.start_layer, self.end_layer)
        }

    def get_attention_sliding_window_size(self):
        return self.config.sliding_window


EntryClass = OpenAIPrivacyFilterForTokenClassification
