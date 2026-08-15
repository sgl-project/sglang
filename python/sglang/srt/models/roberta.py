# SPDX-License-Identifier: Apache-2.0

import os
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.pooler import CrossEncodingPooler, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.sparse_pooler import SparsePooler
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bert import BertEncoder
from sglang.srt.utils.hf_transformers_utils import download_from_hf

RobertaConfig = None


# Adapted from transformers
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


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
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        inputs_embeds = self.word_embeddings(input_ids)

        # Adapted from vllm: https://github.com/vllm-project/vllm/commit/4a18fd14ba4a349291c798a16bf62fa8a9af0b6b/vllm/model_executor/models/roberta.py

        pos_list = []
        token_list = []
        offset = 0
        for seq_len in seq_lens:
            pos_list.append(position_ids[offset : offset + seq_len])
            token_list.append(input_ids[offset : offset + seq_len])
            offset += seq_len

        new_pos_list = []
        for positions, tokens in zip(pos_list, token_list):
            # Verify assumption that incoming position are
            # always a sequence from 0 to N.
            expected_pos = torch.arange(
                positions.size()[0], dtype=torch.long, device=inputs_embeds.device
            )
            assert torch.equal(positions, expected_pos)
            new_pos_list.append(
                create_position_ids_from_input_ids(tokens, self.padding_idx)
            )
        position_ids = torch.cat(new_pos_list)

        # Position embeddings.
        position_embeddings = self.position_embeddings(position_ids)

        token_type_ids = forward_batch.token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=inputs_embeds.device
            )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class XLMRobertaBaseModel(nn.Module):
    def __init__(
        self,
        *,
        config: RobertaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        add_pooling_layer: bool = False,
    ):
        super().__init__()

        self.config = config
        self.embeddings = RobertaEmbedding(config)
        self.encoder = BertEncoder(config=config, quant_config=quant_config, prefix="")
        self.pooler = (
            Pooler(pooling_type=PoolingType.CLS, normalize=True)
            if add_pooling_layer
            else None
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        assert get_embedding == True
        # Your tokenized IDs

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=positions,
            seq_lens=forward_batch.seq_lens,
            forward_batch=forward_batch,
        )

        hidden_states = self.encoder(hidden_states, forward_batch=forward_batch)

        return hidden_states

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
            if self.pooler is None and "pooler" in name:
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


# Adapted from transformers
def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=0
):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (
        torch.cumsum(mask, dim=0).type_as(mask) + past_key_values_length
    ) * mask
    return incremental_indices.long() + padding_idx


class XLMRobertaModel(nn.Module):
    def __init__(
        self,
        *,
        config: RobertaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sparse_head: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__()
        self.roberta = XLMRobertaBaseModel(
            config=config, quant_config=quant_config, prefix=prefix
        )
        if sparse_head is not None:
            self._is_sparse = True
            self._model_path = model_path
            self._sparse_head = sparse_head
            self.pooler = SparsePooler(config=config)
            # Zero out special tokens
            self._special_tokens = [
                config.bos_token_id,
                config.eos_token_id,
                config.pad_token_id,
                # self.config.unk_token_id # not available in the XLMRobertaConfig
            ]
            self._special_tokens = [t for t in self._special_tokens if t is not None]
        else:
            self._is_sparse = False
            self.pooler = Pooler(pooling_type=PoolingType.CLS, normalize=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.roberta(
            input_ids, positions, forward_batch, input_embeds, get_embedding
        )
        embeddings = self.pooler(hidden_states, forward_batch)

        if self._is_sparse:
            for token_id in self._special_tokens:
                embeddings.embeddings[:, token_id] = 0.0
            embeddings.embeddings = embeddings.embeddings.to_sparse()

        return embeddings

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self.roberta.load_weights(weights)

        if self._is_sparse:
            sparse_dict = XLMRobertaModel._load_sparse_linear(
                self._model_path, self._sparse_head
            )
            self.pooler.load_weights(sparse_dict)

    @staticmethod
    def _load_sparse_linear(model_path_or_dir: str, sparse_head: str) -> dict:
        """
        Load sparse_head from local dir or HF Hub.
        Returns a state_dict suitable for nn.Linear.load_state_dict().
        """
        if os.path.isdir(model_path_or_dir):
            path = os.path.join(model_path_or_dir, sparse_head)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"'{sparse_head}' not found in {model_path_or_dir}"
                )
        else:
            # remote → use SGLang HF utility
            local_dir = download_from_hf(model_path_or_dir, allow_patterns=sparse_head)
            path = os.path.join(local_dir, sparse_head)

        state_dict = torch.load(path)
        return state_dict


class XLMRobertaForSequenceClassification(nn.Module):
    def __init__(
        self,
        *,
        config: RobertaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.roberta = XLMRobertaBaseModel(
            config=config, quant_config=quant_config, prefix=prefix
        )
        self.classifier = RobertaClassificationHead(config)
        self.pooler = CrossEncodingPooler(config, self.classifier, self.roberta.pooler)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> torch.Tensor:
        assert (
            get_embedding
        ), "XLMRobertaForSequenceClassification is only used for rerank"

        hidden_states = self.roberta(
            input_ids, positions, forward_batch, input_embeds, get_embedding
        )
        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self_weights = []

        def weight_filter():
            for name, weight in weights:
                if name.startswith("roberta."):
                    yield (name[len("roberta.") :], weight)
                else:
                    self_weights.append((name, weight))

        self.roberta.load_weights(weight_filter())

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in self_weights:
            if name.startswith("classifier"):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [XLMRobertaModel, XLMRobertaForSequenceClassification]
