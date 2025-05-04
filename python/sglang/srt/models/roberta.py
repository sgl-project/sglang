# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.pooler import CrossEncodingPooler, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bert import BertEncoder
from sglang.srt.utils import WeightsMapper

RobertaConfig = None


class RobertaEmbedding(nn.Module):

    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.position_ids = nn.Parameter(
            torch.empty((1, config.max_position_embeddings)),
        )

        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type != "absolute":
            raise ValueError(
                "Only 'absolute' position_embedding_type" + " is supported"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        position_ids: torch.Tensor = None,
        inputs_embeds=None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        inputs_embeds = self.word_embeddings(input_ids)

        # adpated from vllm: https://github.com/vllm-project/vllm/commit/4a18fd14ba4a349291c798a16bf62fa8a9af0b6b/vllm/model_executor/models/roberta.py

        pos_list = []
        token_list = []
        offset = 0
        for seq_len in seq_lens:
            pos_list.append(position_ids[offset : offset + seq_len])
            token_list.append(input_ids[offset : offset + seq_len])
            offset += seq_len

        new_pos_list = []
        for tokens in token_list:
            new_pos_list.append(
                create_position_ids_from_input_ids(tokens, self.padding_idx)
            )
        position_ids = torch.cat(new_pos_list)

        # Position embeddings.
        position_embeddings = self.position_embeddings(position_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=inputs_embeds.device
            )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class XLMRobertaModel(nn.Module):
    def __init__(
        self,
        *,
        config: RobertaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config
        self.embeddings = RobertaEmbedding(config)
        self.encoder = BertEncoder(config=config, quant_config=quant_config, prefix="")
        self.pooler = Pooler(pooling_type=PoolingType.CLS, normalize=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        return_hidden_states: bool = False,
    ) -> torch.Tensor:
        assert get_embedding == True
        # Your tokenized IDs

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=positions,
            seq_lens=forward_batch.seq_lens,
        )

        hidden_states = self.encoder(hidden_states, forward_batch=forward_batch)
        if return_hidden_states:
            return hidden_states

        pooler_out = self.pooler(hidden_states, forward_batch)
        return pooler_out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "query", "q"),
            ("qkv_proj", "key", "k"),
            ("qkv_proj", "value", "v"),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            name = name.replace("self", "self_attn")
            if "pooler" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:

                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


class XLMRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = linear_cls(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def roberta_task_weights_filter(
    all_weights: Iterable[Tuple[str, torch.Tensor]]
) -> Tuple[Iterable[Tuple[str, torch.Tensor]], Iterable[Tuple[str, torch.Tensor]]]:
    """
    Separate task-specific weights that are applied on top
    of the encoder-decoder bert base.
    To do so, return two generators over the original iterator.
    Also, remove the "roberta." prefix to make it loadable
    from vanilla BertModel.
    """
    # Copy of a lazy iterator without in-memory overhead so both
    # iterators can be iterated upon independently.
    all_weights1, all_weights2 = itertools.tee(all_weights)

    def encoder_decoder_weights():
        for name, weight in all_weights1:
            if name.startswith("roberta."):
                yield (name[len("roberta.") :], weight)

    return encoder_decoder_weights(), (
        (n, w) for n, w in all_weights2 if not n.startswith("roberta.")
    )


jina_to_sglang_mapper = WeightsMapper(
    orig_to_new_substr={
        "emb_ln": "embeddings.LayerNorm",
        "layers": "layer",
        "mixer.Wqkv": "attention.self.qkv_proj",
        "mixer.out_proj": "attention.output.dense",
        "norm1": "attention.output.LayerNorm",
        "mlp.fc1": "intermediate.dense",
        "mlp.fc2": "output.dense",
        "norm2": "output.LayerNorm",
    }
)


class XLMRobertaForSequenceClassification(nn.Module):

    def __init__(
        self,
        *,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_labels = config.num_labels
        self.roberta = XLMRobertaModel(
            config=config, quant_config=quant_config, prefix=prefix
        )
        self.classifier = XLMRobertaClassificationHead(config)
        self.pooler = CrossEncodingPooler(config, self.classifier)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        bert_weights, task_weights = roberta_task_weights_filter(weights)
        bert_weights = jina_to_sglang_mapper.apply(bert_weights)

        self.roberta.load_weights(bert_weights)

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in task_weights:
            if name.startswith("classifier"):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ):
        assert get_embedding == True
        hidden_states = self.roberta(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            get_embedding=get_embedding,
            return_hidden_states=True,
        )

        return self.pooler(hidden_states, forward_batch)


# Adapted from transformers
def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=0
):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (
        torch.cumsum(mask, dim=0).type_as(mask) + past_key_values_length
    ) * mask
    return incremental_indices.long() + padding_idx


EntryClass = [XLMRobertaModel, XLMRobertaForSequenceClassification]
