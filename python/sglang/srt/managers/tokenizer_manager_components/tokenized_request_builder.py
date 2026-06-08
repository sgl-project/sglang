from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

import torch

from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    SessionParams,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState
from sglang.srt.sampling.sampling_params import SamplingParams


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizedRequestBuilderConfig:
    vocab_size: int
    preferred_sampling_params: Optional[dict]
    sampling_params_class: Type[SamplingParams]
    disaggregation_transfer_backend: str


@dataclass(slots=True, kw_only=True)
class TokenizedRequestBuilder:
    tokenizer: Optional[Any]
    config: TokenizedRequestBuilderConfig
    rid_to_state: Dict[str, ReqState]
    fake_bootstrap_room_counter: int = 0

    def build(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        input_text: str,
        input_ids: Optional[List[int]],
        input_embeds: Optional[Union[List[float], None]] = None,
        mm_inputs=None,
        token_type_ids: Optional[List[int]] = None,
    ) -> Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]:
        """Create a tokenized request object from common parameters."""
        input_ids_arr: Optional[array[int]] = (
            array("q", input_ids) if input_ids is not None else None
        )
        # Parse sampling parameters
        # Note: if there are preferred sampling params, we use them if they are not
        # explicitly passed in sampling_params
        if self.config.preferred_sampling_params:
            sampling_kwargs = {
                **self.config.preferred_sampling_params,
                **obj.sampling_params,
            }
        else:
            sampling_kwargs = obj.sampling_params
        sampling_params = self.config.sampling_params_class(**sampling_kwargs)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.config.vocab_size)

        # Build return object
        if isinstance(obj, GenerateReqInput):
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

            bootstrap_room = obj.bootstrap_room
            if (
                bootstrap_room is None
                and self.config.disaggregation_transfer_backend == "fake"
            ):
                bootstrap_room = self.fake_bootstrap_room_counter
                self.fake_bootstrap_room_counter += 1

            tokenized_obj = TokenizedGenerateReqInput(
                input_text,
                input_ids_arr,
                mm_inputs,
                sampling_params,
                obj.return_logprob,
                obj.logprob_start_len,
                obj.top_logprobs_num,
                obj.token_ids_logprob,
                obj.stream,
                rid=obj.rid,
                http_worker_ipc=obj.http_worker_ipc,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=bootstrap_room,
                lora_id=obj.lora_id,
                input_embeds=input_embeds,
                positional_embed_overrides=obj.positional_embed_overrides,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                require_reasoning=obj.require_reasoning,
                return_hidden_states=obj.return_hidden_states,
                return_routed_experts=obj.return_routed_experts,
                routed_experts_start_len=obj.routed_experts_start_len,
                return_indexer_topk=obj.return_indexer_topk,
                routed_dp_rank=obj.routed_dp_rank,
                disagg_prefill_dp_rank=obj.disagg_prefill_dp_rank,
                priority=obj.priority,
                extra_key=obj.extra_key,
                routing_key=obj.routing_key,
                token_type_ids=token_type_ids,
                need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
                num_items_assigned=obj.num_items_assigned,
                multi_item_delimiter_indices=obj.multi_item_delimiter_indices,
                mm_data_mooncake=obj.mm_data_mooncake,
            )
        elif isinstance(obj, EmbeddingReqInput):
            # Resolve unresolved embed overrides now that input_ids are available
            positional_embed_overrides = obj.positional_embed_overrides
            if (
                positional_embed_overrides is None
                and obj.embed_overrides is not None
                and obj.embed_override_token_id is not None
            ):
                positional_embed_overrides = self._resolve_embed_overrides(
                    input_ids_arr, obj.embed_override_token_id, obj.embed_overrides
                )

            tokenized_obj = TokenizedEmbeddingReqInput(
                input_text,
                input_ids_arr,
                mm_inputs,
                token_type_ids,
                sampling_params,
                positional_embed_overrides=positional_embed_overrides,
                rid=obj.rid,
                priority=obj.priority,
                dimensions=obj.dimensions,
                lora_id=obj.lora_id,
                http_worker_ipc=obj.http_worker_ipc,
                return_pooled_hidden_states=obj.return_pooled_hidden_states,
                multi_item_delimiter_indices=obj.multi_item_delimiter_indices,
            )

        tokenized_obj.time_stats = self.rid_to_state[obj.rid].time_stats
        self.rid_to_state[obj.rid].time_stats.set_tokenize_finish_time()

        return tokenized_obj

    @staticmethod
    def _resolve_embed_overrides(
        input_ids: array[int],
        token_id: int,
        embeds: List[torch.Tensor],
    ) -> PositionalEmbeds:
        """Resolve placeholder positions in input_ids and create PositionalEmbeds.

        Scans input_ids for occurrences of token_id and pairs them with the
        provided embedding tensors.
        """
        positions = [idx for idx, tok in enumerate(input_ids) if tok == token_id]
        if len(positions) != len(embeds):
            raise ValueError(
                f"input contains {len(positions)} occurrences of "
                f"embed_override_token_id={token_id}, "
                f"but embed_overrides has {len(embeds)} entries."
            )
        return PositionalEmbeds(embeds=embeds, positions=positions)
