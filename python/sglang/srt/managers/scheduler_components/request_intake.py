"""Turn a tokenized input into a request, or a rejection, and queue it."""

from __future__ import annotations

import dataclasses
import functools
import logging
import sys
from array import array
from contextlib import ExitStack
from http import HTTPStatus
from typing import TYPE_CHECKING, AbstractSet, Any, List, Optional

import torch
import torch.distributed

from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.disaggregation.kv_events import is_kv_publisher_rank
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    MMInputsProcessError,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.mm_utils import duplicate_pad_mm_input_ids
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    MultimodalInputs,
    MultimodalProcessorOutput,
    Req,
)
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.runtime_context import (
    get_device,
    get_disagg,
    get_exec,
    get_memory,
    get_mm,
    get_observability,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
    hicache_storage_enabled,
)
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_sampling_mask import (
    validate_spec_sampling_mask_request,
)
from sglang.srt.speculative.uno_validation import validate_uno_request

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.constrained.grammar_manager import GrammarManager
    from sglang.srt.managers.request_lists import CoreLists
    from sglang.srt.managers.scheduler_components.bundles import (
        KvCapacity,
        KvResources,
        RequestLimits,
        SchedulerServices,
        Workers,
    )
    from sglang.srt.managers.scheduler_components.engine_state import (
        LaunchCounters,
    )
    from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
    from sglang.srt.managers.scheduler_components.request_retirer import (
        RequestRetirer,
    )
    from sglang.srt.mem_cache.events import KvEventLoraNames
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _MultimodalInputBroadcast:
    inputs: Optional[MultimodalInputs] = None
    error: Optional[str] = None


class _MultimodalInputProcessingError(RuntimeError):
    pass


class RequestIntake:
    """Builds a request from each tokenized input, rejects the ones that
    cannot run, and puts the rest into the waiting queue or the PD owner,
    starting their HiCache storage prefetch."""

    def __init__(
        self,
        *,
        core_lists: CoreLists,
        counters: LaunchCounters,
        kv: KvResources,
        capacity: KvCapacity,
        services: SchedulerServices,
        beam_coordinator: BeamCoordinator,
        session_controller: SessionController,
        grammar_manager: GrammarManager,
        request_retirer: RequestRetirer,
        mode_owners: ModeOwners,
        model_config: ModelConfig,
        tokenizer: Any,
        mm_processor: Any,
        workers: Workers,
        dp_tp_group: Any,
        limits: RequestLimits,
        kv_event_lora_names: KvEventLoraNames,
    ):
        spec_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        transfer_backend = TransferBackend(get_disagg().disaggregation_transfer_backend)
        sampling_mask_max_tokens = get_exec().features.sampling_mask_max_tokens
        enable_kv_cache_events = is_kv_publisher_rank(
            get_observability().kv_events_config
        )
        self.lists = core_lists
        self.counters = counters
        self.tree_cache = kv.tree_cache
        self.token_to_kv_pool_allocator = kv.token_to_kv_pool_allocator
        self.capacity = capacity
        self.sliding_window_size = capacity.sliding_window_size
        self.metrics_sink = services.metrics_sink
        self.metrics_collector = services.metrics_sink.metrics_collector
        self.output_streamer = services.output_streamer
        self.beam_coordinator = beam_coordinator
        self.session_controller = session_controller
        self.grammar_manager = grammar_manager
        self.request_retirer = request_retirer
        # A PD role switch replaces the owners in place.
        self.mode_owners = mode_owners
        self.model_config = model_config
        self.spec_algorithm = spec_algorithm
        self.transfer_backend = transfer_backend
        self.tokenizer = tokenizer
        self._mm_processor = mm_processor
        self.pad_input_ids_func = workers.tp_worker.get_pad_input_ids_func()
        self.dp_tp_group = dp_tp_group
        self.dp_tp_cpu_group = dp_tp_group.cpu_group
        self.max_req_len = limits.max_req_len
        self.max_req_input_len = limits.max_req_input_len
        self.max_queued_requests = limits.max_queued_requests
        self.chunked_prefill_size = limits.chunked_prefill_size
        self.sampling_mask_max_tokens = sampling_mask_max_tokens
        self.kv_event_lora_names = kv_event_lora_names
        self.enable_kv_cache_events = enable_kv_cache_events
        self._last_logged_elastic_radix_namespace: Optional[str] = None

    @staticmethod
    def _tokenized_requests(recv_req):
        if isinstance(
            recv_req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
        ):
            return (recv_req,)
        if isinstance(
            recv_req,
            (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput),
        ):
            return tuple(recv_req)
        return ()

    def reconstruct_positional_embed_overrides(self, recv_req) -> None:
        """Rebuild each request's override embeds from their CUDA IPC handles,
        which releases the sender's pool slot."""
        for tokenized_req in self._tokenized_requests(recv_req):
            if tokenized_req.positional_embed_overrides is not None:
                tokenized_req.positional_embed_overrides.reconstruct(
                    get_device().gpu_id
                )

    def _gather_vmm_materialization_errors(
        self, local_error: Optional[str]
    ) -> List[Optional[str]]:
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.dp_tp_cpu_group is not None
        ):
            return [local_error]

        world_size = torch.distributed.get_world_size(group=self.dp_tp_cpu_group)
        errors = [None] * world_size
        torch.distributed.all_gather_object(
            errors,
            local_error,
            group=self.dp_tp_cpu_group,
        )
        return errors

    def materialize_cuda_vmm_inputs(self, recv_req) -> Optional[List[Optional[str]]]:
        """Materialize each request and agree on failures across TP ranks."""
        tokenized_reqs = self._tokenized_requests(recv_req)
        if not tokenized_reqs:
            return None

        request_errors = []
        for tokenized_req in tokenized_reqs:
            # The request broadcast makes this skip consistent across ranks.
            if tokenized_req.mm_inputs is None:
                request_errors.append(None)
                continue

            local_error = None
            try:
                if not isinstance(tokenized_req.mm_inputs, MultimodalInputs):
                    tokenized_req.mm_inputs = MultimodalInputs.from_processor_output(
                        tokenized_req.mm_inputs,
                        requires_mm_token_modalities=self.model_config.requires_mm_token_modalities,
                    )
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"

            rank_errors = self._gather_vmm_materialization_errors(local_error)
            failed_ranks = [
                rank for rank, error in enumerate(rank_errors) if error is not None
            ]
            if failed_ranks:
                details = "; ".join(
                    f"rank {rank}: {rank_errors[rank]}" for rank in failed_ranks
                )
                error_msg = f"Multimodal feature reconstruction failed ({details})"
                logger.error(error_msg)
                if isinstance(
                    tokenized_req.mm_inputs,
                    (MultimodalInputs, MultimodalProcessorOutput),
                ):
                    for item in tokenized_req.mm_inputs.mm_items:
                        item.release_transport_proxies()
                tokenized_req.mm_inputs = None
                request_errors.append(error_msg)
            else:
                request_errors.append(None)
        return request_errors

    def dispatch_tokenized_mm_requests(
        self, recv_req, errors: List[Optional[str]]
    ) -> None:
        tokenized_reqs = self._tokenized_requests(recv_req)
        if len(tokenized_reqs) != len(errors):
            raise RuntimeError("VMM materialization results do not match requests")
        for tokenized_req, error in zip(tokenized_reqs, errors, strict=True):
            if isinstance(tokenized_req, TokenizedGenerateReqInput):
                self.handle_generate_request(tokenized_req, mm_input_error=error)
            elif isinstance(tokenized_req, TokenizedEmbeddingReqInput):
                self.handle_embedding_request(tokenized_req, mm_input_error=error)
            else:
                raise TypeError(
                    f"Unsupported tokenized request type: {type(tokenized_req).__name__}"
                )

    def init_req_max_new_tokens(self, req):
        input_len = len(req.origin_input_ids)
        max_new_tokens = (
            req.sampling_params.max_new_tokens
            if req.sampling_params.max_new_tokens is not None
            else 1 << 30
        )
        if (
            envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get() is not None
            and envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get() > 0
        ):
            if max_new_tokens > envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get():
                logger.warning(
                    f"Capping max_new_tokens of request {req.rid} to "
                    f"SGLANG_MAX_NEW_TOKENS_LIMIT={envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get()} "
                    f"(requested: {req.sampling_params.max_new_tokens})."
                )
            max_new_tokens = min(max_new_tokens, envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get())

        # Keep this bound consistent with PrefillAdder's admission budget.
        max_new_tokens = max(
            0,
            min(
                max_new_tokens,
                self.max_req_len - input_len - 1,
            ),
        )
        # PrefillAdder reserves one page per shard; the allocator reserves one.
        # Subtract the other N - 1 pages to keep queue admission schedulable.
        token_capacity = (
            self.capacity.widened_max_total_num_tokens
            - get_schedule().page_size * (self.capacity.kv_shard_widening - 1)
        )
        max_new_tokens = self.token_to_kv_pool_allocator.max_new_tokens_for_memory(
            input_len,
            max_new_tokens,
            token_capacity=token_capacity,
            sliding_window_size=self.sliding_window_size,
            chunk_size=self.chunked_prefill_size,
        )
        if max_new_tokens is None:
            req.set_finish_with_abort(
                f"Request prompt exceeds the KV memory budget: input_len={input_len}."
            )
            max_new_tokens = 0

        req.sampling_params.max_new_tokens = max(0, max_new_tokens)
        # Clipping above can push max_new_tokens below min_new_tokens, which
        # would suppress EOS for the whole generation. Restore the invariant.
        if req.sampling_params.min_new_tokens > req.sampling_params.max_new_tokens:
            req.sampling_params.min_new_tokens = req.sampling_params.max_new_tokens

    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs,
    ):
        """Materialize MultimodalInputs once on the entry rank and broadcast to others.

        Entry rank:
        - constructs MultimodalInputs.from_processor_output() once
        - broadcasts to other ranks in self.cpu_group (if world_size > 1)

        Non-entry ranks:
        - receive the object via broadcast (if world_size > 1)
        - otherwise (single-rank / no group) fall back to local from_processor_output

        Returns:
            MultimodalInputs | None

        Raises:
            _MultimodalInputProcessingError: The entry rank could not build the
                request's multimodal inputs. The same error is broadcast to all
                ranks before it is raised.
        """
        if raw_mm_inputs is None:
            return None

        group_world_size = 1
        try:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.dp_tp_cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.dp_tp_cpu_group
                )
        except Exception as e:
            logger.warning(
                f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
            )

        # In case tp size > 1, all the Scheduler TP ranks runs the duplicated computing
        # process in CPU which occupies the main thread CPU cycle. This computing logic
        # merely needs to be run on TP0 and be broadcast to other TP ranks.
        # Since the Scheduler is single-threaded, any large CPU cost will impact
        # handling of other messages. For example, CPU hits 99.9% can significantly
        # increase the CUDA kernel launch time.
        result = None
        if self.dp_tp_group.rank_in_group == 0:
            try:
                result = _MultimodalInputBroadcast(
                    inputs=MultimodalInputs.from_processor_output(
                        raw_mm_inputs,
                        requires_mm_token_modalities=self.model_config.requires_mm_token_modalities,
                    )
                )
            except Exception as error:
                result = _MultimodalInputBroadcast(
                    error=(
                        "Multimodal input processing failed on the TP entry rank: "
                        f"{type(error).__name__}: {error}"
                    )
                )

            # Broadcast either the prepared inputs or the request-local error.
            if group_world_size > 1:
                obj_list = [result]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                result = obj_list[0]
        else:
            # Non-entry ranks: receive if group size > 1; otherwise materialize locally.
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                result = obj_list[0]
            else:
                result = _MultimodalInputBroadcast(
                    inputs=MultimodalInputs.from_processor_output(
                        raw_mm_inputs,
                        requires_mm_token_modalities=self.model_config.requires_mm_token_modalities,
                    )
                )

        if result.error is not None:
            raise _MultimodalInputProcessingError(result.error)
        return result.inputs

    def _get_multimodal_inputs(self, mm_inputs):
        if isinstance(mm_inputs, MMInputsProcessError):
            raise _MultimodalInputProcessingError(mm_inputs.message)
        if isinstance(mm_inputs, MultimodalInputs):
            return mm_inputs

        if get_mm().enable_broadcast_mm_inputs_process:
            return self._process_and_broadcast_mm_inputs(mm_inputs)
        return MultimodalInputs.from_processor_output(
            mm_inputs,
            requires_mm_token_modalities=self.model_config.requires_mm_token_modalities,
        )

    @staticmethod
    def _try_apply_padded_mm_input_ids(recv_req, req, image_inputs) -> bool:
        """setup origin_input_ids with trying to reuse existing MultimodalInputs.padded_input_ids first,
        if absent, call pad_input_ids_func"""
        padded_input_ids = image_inputs.padded_input_ids
        if padded_input_ids is None or recv_req.input_ids is None:
            return False

        recv_input_len = len(recv_req.input_ids)
        if len(padded_input_ids) != recv_input_len:
            return False

        prefix_len = len(req.origin_input_ids) - recv_input_len
        if prefix_len < 0:
            return False

        padded_input_ids = array("q", padded_input_ids)
        if prefix_len == 0:
            req.origin_input_ids = padded_input_ids
        else:
            req.origin_input_ids = req.origin_input_ids[:prefix_len] + padded_input_ids
        return True

    def _maybe_compute_mrope_positions(self, req) -> None:
        """Compute M-RoPE positions when they are missing (e.g. gRPC preprocessed path)."""
        if self._mm_processor is None:
            return
        mm = req.multimodal_inputs
        if mm is None or mm.mrope_positions is not None:
            return

        mrope_positions, mrope_position_delta = (
            self._mm_processor.compute_mrope_positions(
                req.origin_input_ids, mm.mm_items
            )
        )
        if mrope_positions is not None:
            mm.mrope_positions = mrope_positions
            mm.mrope_position_delta = mrope_position_delta

    def _maybe_namespace_elastic_radix_cache(self, req: Req) -> None:
        if (
            get_exec().moe.elastic_ep_backend is None
            or not self.tree_cache.supports_prefix_sharing()
        ):
            return

        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        inst = ElasticEPStateManager.instance()
        if inst is None:
            return

        namespace = f"elastic_ep_size={ElasticEPStateManager.get_effective_ep_size()}"
        if req.extra_key:
            req.extra_key = f"{req.extra_key}|{namespace}"
        else:
            req.extra_key = namespace

        if self._last_logged_elastic_radix_namespace != namespace:
            self._last_logged_elastic_radix_namespace = namespace
            logger.debug(
                "[Elastic EP][scale] radix cache namespace is now %s",
                namespace,
            )

    def _record_kv_event_lora_name(self, req: Req, lora_name: Optional[str]) -> None:
        # Keyed by the final extra_key, so it must run after elastic namespacing.
        # Only a cache whose events are taken prunes the table.
        if self.enable_kv_cache_events and self.tree_cache.supports_prefix_sharing():
            self.kv_event_lora_names.register(req, lora_name)

    def create_generation_req(
        self, recv_req: TokenizedGenerateReqInput, *args, **kwargs
    ) -> Req:
        """The request of a generation input on every path (plain, session,
        unknown session): `Req(*args, **kwargs)`. A subclass builds its own
        `Req` subclass, with request-local state taken from `recv_req`."""
        return Req(*args, **kwargs)

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
        *,
        mm_input_error: Optional[str] = None,
    ):
        with ExitStack() as mm_cleanup:
            if isinstance(
                recv_req.mm_inputs, (MultimodalInputs, MultimodalProcessorOutput)
            ):
                for item in recv_req.mm_inputs.mm_items:
                    mm_cleanup.callback(item.release_transport_proxies)
            self._handle_generate_request(
                recv_req, mm_cleanup, mm_input_error=mm_input_error
            )

    def _handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
        mm_cleanup: ExitStack,
        *,
        mm_input_error: Optional[str] = None,
    ):
        # Read at call time: a PD role switch changes it.
        disaggregation_mode = DisaggregationMode(get_disagg().disaggregation_mode)
        # Route: normal request / session request / session-not-found
        session_id = (
            recv_req.session_params.id if recv_req.session_params is not None else None
        )
        if recv_req.bootstrap_port is None:
            recv_req.bootstrap_port = get_disagg().disaggregation_bootstrap_port
        # Radix-native sessions use only the top-level session_id.
        radix_native_session = (
            recv_req.session_id is not None and get_memory().enable_session_radix_cache
        )

        if session_id is None or radix_native_session:
            # Normal non-session request, or a radix-native session request
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                recv_req.input_ids = array("q", [1]) * seq_length

            is_beam = BeamCoordinator.request_beam_width(recv_req) > 1
            req = self.create_generation_req(
                recv_req,
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                return_sampling_mask=recv_req.return_sampling_mask,
                sampling_logprobs_mode=recv_req.sampling_logprobs_mode,
                return_flat_raw_top_logprobs=recv_req.return_flat_raw_top_logprobs,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                session_id=recv_req.session_id,
                input_embeds=recv_req.input_embeds,
                positional_embed_overrides=recv_req.positional_embed_overrides,
                token_type_ids=recv_req.token_type_ids,
                custom_logit_processor=recv_req.custom_logit_processor,
                require_reasoning=recv_req.require_reasoning,
                return_hidden_states=recv_req.return_hidden_states,
                return_routed_experts=recv_req.return_routed_experts,
                routed_experts_start_len=recv_req.routed_experts_start_len,
                return_indexer_topk=recv_req.return_indexer_topk,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                disagg_mode=disaggregation_mode,
                routed_dp_rank=recv_req.routed_dp_rank,
                disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
                kv_hints=recv_req.kv_hints,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector if self.metrics_sink.enable_metrics else None
                ),
                routing_key=recv_req.routing_key,
                extra_key=recv_req.extra_key,
                cache_salt=recv_req.cache_salt,
                http_worker_ipc=recv_req.http_worker_ipc,
                time_stats=recv_req.time_stats,
                multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
                token_indices_to_pool=recv_req.token_indices_to_pool,
            )
            req.tokenizer = self.tokenizer

            if radix_native_session:
                req.session_generation = self.tree_cache.ensure_session_generation(
                    recv_req.session_id
                )

            if is_beam:
                error_msg = self.beam_coordinator.validate_and_init(req, recv_req)
                if error_msg:
                    logger.error(error_msg)
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.output_streamer.stream_output([req], req.return_logprob)
                    return

            if disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if (
                    recv_req.bootstrap_room is None
                    and self.transfer_backend != TransferBackend.FAKE
                ):
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"bootstrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    if not envs.SGLANG_RUST_SERVER.get():
                        recv_req.time_stats.trace_ctx.abort(
                            abort_info={"reason": error_msg}
                        )
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.output_streamer.stream_output([req], req.return_logprob)
                    return

        elif (
            session_id in self.session_controller
            and not self.session_controller.get(session_id).close_on_finish
        ):
            # Session exists and is not closing: create request from session
            session = self.session_controller.get(session_id)
            req = session.create_req(
                recv_req,
                self.tokenizer,
                self.model_config.vocab_size,
                eos_token_ids=self.model_config.hf_eos_token_id,
                disagg_mode=disaggregation_mode,
                req_factory=functools.partial(self.create_generation_req, recv_req),
            )
            if get_memory().enable_session_radix_cache:
                req.session_generation = self.tree_cache.ensure_session_generation(
                    session_id
                )
            # TODO: set trace context
            if self.metrics_sink.enable_metrics:
                req.time_stats.set_metrics_collector(self.metrics_collector)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self.add_request_to_queue(req)
                return

        else:
            # Session not found, or session is closing
            if session_id in self.session_controller:
                error_msg = (
                    f"Invalid request: close was requested for session {session_id}"
                )
            else:
                error_msg = f"Invalid request: session id {session_id} does not exist"
            req = self.create_generation_req(
                recv_req,
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                vocab_size=self.model_config.vocab_size,
                http_worker_ipc=recv_req.http_worker_ipc,
            )
            req.tokenizer = self.tokenizer
            self._reject(req, error_msg, init_max_new_tokens=True)
            return

        if recv_req.pp_prefetch_ticketed is True:
            self.tree_cache.bind_prefetch_ticket(req.rid)
        self._maybe_namespace_elastic_radix_cache(req)
        self._record_kv_event_lora_name(req, recv_req.lora_name)

        if mm_input_error is not None:
            self._reject(
                req,
                mm_input_error,
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                err_type="InternalServerError",
                init_max_new_tokens=True,
            )
            return

        if self.mode_owners.dllm is not None and (
            error_msg := self.mode_owners.dllm.validate_request(req)
        ):
            self._reject_request(req, HTTPStatus.BAD_REQUEST, error_msg)
            return

        if self.spec_algorithm.is_dflash_family():
            error_msg = validate_dflash_request(
                req, (not get_schedule().disable_overlap_schedule)
            )
            if error_msg is not None:
                self._reject(req, error_msg, init_max_new_tokens=True)
                return

        if self.spec_algorithm.is_uno():
            error_msg = validate_uno_request(req)
            if error_msg is not None:
                self._reject(req, error_msg, init_max_new_tokens=True)
                return

        if req.return_sampling_mask:
            if (
                disaggregation_mode != DisaggregationMode.NULL
                and not (
                    self.mode_owners.disagg_prefill or self.mode_owners.disagg_decode
                ).metadata_buffers.enable_sampling_mask
            ):
                self.request_retirer.reject_sampling_mask_request(
                    req,
                    "return_sampling_mask requires "
                    "SGLANG_ENABLE_DISAGG_SAMPLING_MASK=1 on both prefill and "
                    "decode servers when using disaggregated serving.",
                )
                return
            top_k = req.sampling_params.top_k
            sampling_mask_cap = self.sampling_mask_max_tokens
            if top_k != 1 and not (1 < top_k <= sampling_mask_cap):
                error_msg = (
                    "return_sampling_mask requires top_k=1 for greedy sampling "
                    f"or finite 1 < top_k <= {sampling_mask_cap}; got top_k="
                    f"{top_k}. Lower top_k or increase "
                    "--sampling-mask-max-tokens."
                )
                self.request_retirer.reject_sampling_mask_request(req, error_msg)
                return

        if req.return_sampling_mask and not self.spec_algorithm.is_none():
            error_msg = validate_spec_sampling_mask_request(req, self.spec_algorithm)
            if error_msg is not None:
                self.request_retirer.reject_sampling_mask_request(req, error_msg)
                return

        if req.return_sampling_mask and get_exec().kernel.sampling_backend == "ascend":
            # The ascend backend samples from logits directly and never builds the
            # top-k/top-p support, so it cannot produce a sampling mask.
            error_msg = (
                "return_sampling_mask is not supported with the ascend "
                "sampling backend."
            )
            self.request_retirer.reject_sampling_mask_request(req, error_msg)
            return

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            try:
                image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)
            except _MultimodalInputProcessingError as error:
                self._reject(
                    req,
                    str(error),
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    err_type="InternalServerError",
                    init_max_new_tokens=True,
                )
                return

            SessionController.adjust_mm_offsets(recv_req, req, image_inputs)

            # The following steps are already fast, execute locally on each rank.
            # Expand a single image token into multiple dummy tokens for receiving image embeddings.
            # The pad function is model-specific and can be None for some backends.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                req.origin_input_ids = duplicate_pad_mm_input_ids(
                    req.origin_input_ids, image_inputs, self.pad_input_ids_func
                )
            req.extend_image_inputs(image_inputs)
            # the request now owns cleanup, including rejection after padding
            mm_cleanup.pop_all()
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                self._reject(
                    req,
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    ),
                    init_max_new_tokens=True,
                )
                return

        # initialize before returning
        self.init_req_max_new_tokens(req)

        # Validate prompt length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            get_serving().allow_auto_truncate,
        )
        if error_msg:
            self._reject(req, error_msg)
            return

        if not recv_req.return_logprob and recv_req.logprob_start_len != -1:
            # When return_logprob is False, logprob_start_len should be ignored
            recv_req.logprob_start_len = -1

        if recv_req.logprob_start_len == -1:
            if recv_req.return_logprob and recv_req.token_ids_logprob is None:
                # If logprob is required but neither token_ids_logprob nor logprob_start_len is
                # set, return the logprobs for output tokens by default
                req.logprob_start_len = len(req.origin_input_ids)
            elif req.is_prefill_only:
                # For prefill-only requests with logprob_start_len == -1, set logprob_start_len
                # beyond input sequence to skip input logprob computation entirely
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # If return_logprob is False, only the last token requires logprob computation
                req.logprob_start_len = -1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len > len(req.origin_input_ids):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = -1
            self._reject(req, error_msg)
            return

        if recv_req.return_routed_experts:
            error_msg = None
            if recv_req.routed_experts_start_len < 0:
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is lower than 0. "
                    "Please use a non-negative routed_experts_start_len."
                )

            if recv_req.routed_experts_start_len > len(req.origin_input_ids):
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is higher than the "
                    f"number of input tokens {len(req.origin_input_ids)=}. Please "
                    f"use a smaller routed_experts_start_len."
                )

            if error_msg is not None:
                req.routed_experts_start_len = 0
                self._reject(req, error_msg)
                return

        if get_parallel().pp_rank == 0 and getattr(
            self.tree_cache.cache_controller, "pp_prefetch_command_group", None
        ):
            recv_req.pp_prefetch_ticketed = bool(self._prefetch_kvcache(req))
            self.tree_cache.bind_prefetch_ticket(req.rid, recv_req.pp_prefetch_ticketed)

        added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
        if not added_to_grammar_queue:
            self.add_request_to_queue(req)

    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """Handle optimized batch generate request."""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)

    def _prefetch_kvcache(self, req: Req, storage_hit_end: Optional[int] = None):
        if hicache_storage_enabled() or get_memory().enable_lmcache:
            req.init_next_round_input(self.tree_cache, cow_mamba=False)
            tree_cache = self.tree_cache
            buffer_mode = get_memory().hicache_host_memory_mode == "buffer_only"
            last_host_node = req.last_host_node
            # Buffer mode host-backups nothing, so match_prefix anchors at
            # root; re-anchor on the deepest device node. The anchor is only
            # read for hash/extra-key context here (never locked), so a device
            # node serves. Cache mode keeps the is_backuped gate below: its
            # write-through prefix is contiguous from root, so an unbacked
            # anchor means a guaranteed storage miss.
            if (
                buffer_mode
                and tree_cache.is_root(last_host_node)
                and not tree_cache.is_root(req.last_node)
            ):
                last_host_node = req.last_node

            matched_len = req.prefix_len + req.host_hit_length
            req.storage_prefetch_last_match_len = matched_len

            if (
                tree_cache.is_backuped(last_host_node)
                or tree_cache.is_root(last_host_node)
                or (
                    buffer_mode
                    and tree_cache.get_last_hash_value(last_host_node) is not None
                )
            ):
                match_end = req._compute_max_prefix_len(
                    len(req.full_untruncated_fill_ids)
                )
                new_input_tokens = req.full_untruncated_fill_ids[matched_len:match_end]
                prefix_keys = (
                    tree_cache.get_prefix_hash_values(last_host_node)
                    if tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                return tree_cache.prefetch_from_storage(
                    req.cache_request_handle,
                    last_host_node,
                    new_input_tokens,
                    tree_cache.get_last_hash_value(last_host_node),
                    prefix_keys,
                    matched_prefix_tokens=req.full_untruncated_fill_ids[:matched_len],
                    extra_key=req.extra_key,
                    cache_salt=req.cache_salt,
                    storage_hit_end=storage_hit_end,
                )

    def process_storage_prefetch_retries(self):
        """Issue due L3 attempts in the current waiting-queue order."""
        retries = self.tree_cache.storage_prefetch_retries
        if retries is None:
            return
        memory = get_memory()
        for req, storage_hit_end in retries.pop_ready(
            self.lists.waiting_queue,
            memory.hicache_storage_prefetch_retry_poll_interval,
            memory.hicache_storage_prefetch_retry_max_attempts,
        ):
            self._retry_storage_prefetch(req, storage_hit_end)

    def _retry_storage_prefetch(
        self, req: Req, storage_hit_end: Optional[int] = None
    ) -> None:
        req.storage_prefetch_retry_attempts += 1
        max_attempts = get_memory().hicache_storage_prefetch_retry_max_attempts
        if req.storage_prefetch_retry_attempts >= max_attempts:
            logger.warning(
                "HiCache storage prefetch reissue cap reached req=%s attempts=%d; "
                "the request is admitted without further L3 lookups",
                req.rid,
                req.storage_prefetch_retry_attempts,
            )
        else:
            logger.debug(
                "HiCache storage prefetch re-issue req=%s attempt=%d",
                req.rid,
                req.storage_prefetch_retry_attempts,
            )
        self._prefetch_kvcache(req, storage_hit_end)

    def prefetch_after_device_hit_loss(self, req: Req) -> bool:
        """Re-query an L3 range newly exposed by queue-time device eviction."""
        previous_match_len = req.storage_prefetch_last_match_len
        buffer_pipeline = self.tree_cache.buffer_pipeline
        if not previous_match_len or (
            buffer_pipeline is not None
            and buffer_pipeline.has_staged(req.cache_request_handle)
        ):
            return False
        current_match_len = req.prefix_len + req.host_hit_length
        if current_match_len >= previous_match_len:
            return False
        if (
            req.storage_prefetch_retry_attempts
            >= get_memory().hicache_storage_prefetch_retry_max_attempts
        ):
            # Past the re-issue cap the shorter live match is admitted as is.
            req.storage_prefetch_last_match_len = current_match_len
            return False
        logger.warning(
            "HiCache device prefix shrank before admission req=%s "
            "lookup_match=%d current_match=%d; reissuing storage lookup",
            req.rid,
            previous_match_len,
            current_match_len,
        )
        self._retry_storage_prefetch(req)
        return True

    def _reject(
        self,
        req: Req,
        error_msg: str,
        *,
        init_max_new_tokens: bool = False,
        **abort_kwargs,
    ) -> None:
        """Reject a request at the door: it is queued as a one-token stub
        whose commit finishes it with the abort. ``init_max_new_tokens`` sizes
        a generate stub the validation rejected before sizing it."""
        self.request_retirer.reject_at_door(req, error_msg, **abort_kwargs)
        if init_max_new_tokens:
            self.init_req_max_new_tokens(req)
        self.add_request_to_queue(req)

    def requeue_ready_grammars(self) -> None:
        """Queue the requests whose grammar finished compiling. Under PP the
        ranks agree on them over the pipeline, so every rank calls it at the
        same point of a schedule, before any rank-local skip."""
        if self.grammar_manager.has_waiting_grammars():
            for req in self.grammar_manager.get_ready_grammar_requests():
                self.add_request_to_queue(req)

    def add_request_to_queue(
        self,
        req: Req,
        is_retracted: bool = False,
        admitted: AbstractSet[Req] = frozenset(),
    ):
        if not self._set_or_validate_priority(req):
            self.request_retirer.retire_aborted_waiting(req)
            return
        if is_retracted:
            req.storage_prefetch_retry_attempts = 0
            req.storage_prefetch_last_match_len = None
            req.staged_prefetch_plan = None
        if self.mode_owners.disagg_prefill is not None:
            self._prefetch_kvcache(req)
            self.mode_owners.disagg_prefill.enqueue(req)
        elif self.mode_owners.disagg_decode is not None:
            self.mode_owners.disagg_decode.enqueue(req, is_retracted=is_retracted)
        else:
            if self._abort_on_queued_limit(req, admitted):
                self.request_retirer.retire_aborted_waiting(req)
                return
            self._prefetch_kvcache(req)
            self.lists.waiting_queue.append(req)
            req.time_stats.set_wait_queue_entry_time()
            req.arrival_processed_tokens = self.counters.processed_tokens_counter

    def _set_or_validate_priority(self, req: Req) -> bool:
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
        if get_schedule().enable_priority_scheduling and req.priority is None:
            if get_schedule().schedule_low_priority_values_first:
                req.priority = sys.maxsize
            else:
                req.priority = -sys.maxsize - 1
        elif (
            not get_schedule().enable_priority_scheduling
            and req.priority is not None
            and get_schedule().abort_on_priority_when_disabled
        ):
            self._reject_request(
                req,
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Using priority is disabled for this server. Please send a new request without a priority.",
            )
            return False
        return True

    def _reject_request(self, req: Req, status_code: HTTPStatus, message: str) -> None:
        """Abort a request before it is queued."""
        finished_reason = {
            "type": "abort",
            "status_code": status_code,
            "message": message,
        }
        req.time_stats.trace_ctx.abort(abort_info=finished_reason)
        self.request_retirer.send_abort_output(req, finished_reason=finished_reason)

    def _abort_on_queued_limit(
        self, recv_req: Req, admitted: AbstractSet[Req] = frozenset()
    ) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted.

        The requests in ``admitted`` belong to the step being built: they
        leave the queue at its launch, so they neither count nor are aborted.
        """
        waiting = self.lists.waiting_queue
        if admitted:
            waiting = [r for r in waiting if r not in admitted]
        if (
            self.max_queued_requests is None
            or len(waiting) + 1 <= self.max_queued_requests
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req
        message = "The request queue is full."
        if get_schedule().enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
            direction = 1 if get_schedule().schedule_low_priority_values_first else -1
            key_fn = lambda req: (
                direction * req.priority,
                req.time_stats.wait_queue_entry_time,
            )
            candidate_req = max(waiting, key=key_fn)
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                self.lists.waiting_queue.remove(candidate_req)
                self.request_retirer.retire_aborted_waiting(candidate_req)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.request_retirer.send_abort_output(
            req_to_abort,
            finished_reason={
                "type": "abort",
                "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                "message": message,
            },
        )
        req_to_abort.time_stats.trace_ctx.abort(abort_info={"reason": message})
        return req_to_abort.rid == recv_req.rid

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
        *,
        mm_input_error: Optional[str] = None,
    ):
        with ExitStack() as mm_cleanup:
            if isinstance(
                recv_req.mm_inputs, (MultimodalInputs, MultimodalProcessorOutput)
            ):
                for item in recv_req.mm_inputs.mm_items:
                    mm_cleanup.callback(item.release_transport_proxies)
            self._handle_embedding_request(
                recv_req, mm_cleanup, mm_input_error=mm_input_error
            )

    def _handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
        mm_cleanup: ExitStack,
        *,
        mm_input_error: Optional[str] = None,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            positional_embed_overrides=recv_req.positional_embed_overrides,
            token_type_ids=recv_req.token_type_ids,
            routed_dp_rank=recv_req.routed_dp_rank,
            priority=recv_req.priority,
            dimensions=recv_req.dimensions,
            lora_id=recv_req.lora_id,
            http_worker_ipc=recv_req.http_worker_ipc,
            time_stats=recv_req.time_stats,
            return_pooled_hidden_states=recv_req.return_pooled_hidden_states,
            multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
            token_indices_to_pool=recv_req.token_indices_to_pool,
            decision_layout=recv_req.decision_layout,
        )
        req.tokenizer = self.tokenizer
        self._maybe_namespace_elastic_radix_cache(req)
        self._record_kv_event_lora_name(req, recv_req.lora_name)

        if mm_input_error is not None:
            self._reject(
                req,
                mm_input_error,
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                err_type="InternalServerError",
            )
            return

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            try:
                image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)
            except _MultimodalInputProcessingError as error:
                self._reject(
                    req,
                    str(error),
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    err_type="InternalServerError",
                )
                return
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            # The `pad_input_ids_func` is model-specific and may be None for
            # embedding models or models not requiring special padding.
            # If None, `req.origin_input_ids` is expected to be correctly populated already.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                req.origin_input_ids = duplicate_pad_mm_input_ids(
                    req.origin_input_ids, image_inputs, self.pad_input_ids_func
                )

            req.extend_image_inputs(image_inputs)
            mm_cleanup.pop_all()
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                self._reject(
                    req,
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    ),
                )
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            get_serving().allow_auto_truncate,
        )
        if error_msg:
            self._reject(req, error_msg)
            return

        # Copy more attributes
        req.logprob_start_len = -1
        self.add_request_to_queue(req)

    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """Handle optimized batch embedding request."""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)
