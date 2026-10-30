"""Opt-in same-version hybrid decode snapshots and native request reconstruction.

Attention KV and GDN conv/temporal tensors are persisted with the actual token,
sampling and output state. A fresh /generate stream can reconstruct a request
without the source engine's CPU objects. Grammar, speculation and multimodal
state are intentionally rejected rather than silently re-prefilled.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import time
from array import array
from pathlib import Path
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.managers.io_struct import (
    AbortReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    TokenKVSnapshotReqOutput,
)
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.token_kv_snapshot_store import TokenKVSnapshotStore
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel, get_serving
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _resolved_sampling_params(params):
    result = {}
    for field in msgspec.structs.fields(params):
        value = getattr(params, field.name)
        if field.name == "custom_params" and value is not None:
            value = {key: item for key, item in value.items() if key != "__req__"}
        if isinstance(value, set):
            value = sorted(value)
        result[field.name] = value
    # Refuse CPU callbacks or other objects that cannot survive engine exit.
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


def sampling_params_from_snapshot(values):
    values = dict(values)
    if values.get("stop_token_ids") is not None:
        values["stop_token_ids"] = set(values["stop_token_ids"])
    if not values.get("is_normalized"):
        raise ValueError("Snapshot requires resolved, normalized sampling parameters")
    return SamplingParams(**values)


def _validate_portable_metadata(metadata, descriptor):
    if metadata.get("request_schema_version") != 2:
        raise ValueError("Snapshot has no portable request state")
    inputs, outputs = metadata["origin_input_ids"], metadata["output_ids"]
    if not inputs or not outputs or any(type(t) is not int for t in inputs + outputs):
        raise ValueError("Snapshot token history is invalid")
    if _json_digest(inputs) != metadata["origin_input_ids_sha256"]:
        raise ValueError("Snapshot original input digest differs")
    if metadata["computed_tokens"] != len(inputs) + len(outputs) - 1:
        raise ValueError("Snapshot computed prefix is not aligned")
    if metadata["pending_input_token"] != outputs[-1]:
        raise ValueError("Snapshot pending decode token is not aligned")
    params = sampling_params_from_snapshot(metadata["resolved_sampling_params"])
    if getattr(params, "watermark", None) is not None:
        raise ValueError("Portable restore does not support watermark state")
    if (
        metadata["request_flags"].get("sampling_logprobs_mode", "selected")
        != "selected"
    ):
        raise ValueError("Portable restore supports selected-token logprobs only")
    if (
        params.n != 1
        or (getattr(params, "beam_width", None) or 1) != 1
        or any((params.json_schema, params.regex, params.ebnf, params.structural_tag))
    ):
        raise ValueError("Portable restore does not support beam, n>1 or grammar state")
    if params.max_new_tokens - len(outputs) != metadata["remaining_generation_tokens"]:
        raise ValueError("Snapshot remaining generation budget differs")
    if metadata["remaining_generation_tokens"] <= 0:
        raise ValueError("Snapshot has no remaining generation budget")
    if descriptor.get("source_binding") != metadata.get("source_binding"):
        raise ValueError("Snapshot trajectory/source binding differs")
    flags = metadata["request_flags"]
    if flags["return_logprob"] and (
        len(metadata["output_logprobs"]) != len(outputs)
        or metadata["output_logprob_token_ids"] != outputs
    ):
        raise ValueError("Snapshot output logprobs are not aligned with output IDs")
    return params


def prepare_portable_generate_input(obj):
    """Populate a new native HTTP request only from committed disk material."""
    if os.environ.get("SGLANG_ENABLE_TOKEN_KV_SNAPSHOT") != "1":
        raise ValueError("SGLANG_ENABLE_TOKEN_KV_SNAPSHOT=1 is required")
    if getattr(obj, "sampling_logprobs_mode", None) is not None:
        raise ValueError(
            "Portable restore does not accept an explicit sampling_logprobs_mode"
        )
    if (
        getattr(obj, "watermark", None) is not None
        or (getattr(obj, "sampling_logprobs_mode", None) or "selected") != "selected"
        or getattr(obj, "kv_hints", None) is not None
    ):
        raise ValueError(
            "Portable restore does not support watermark, alternate logprobs or KV hints"
        )
    descriptor = obj.token_kv_restore
    identity = descriptor["identity"]
    if descriptor["expected_weight_version"] != identity["weight_version"]:
        raise ValueError("Descriptor weight version differs from snapshot identity")
    store = TokenKVSnapshotStore(descriptor["storage_dir"])
    commit = store.receipt(descriptor["snapshot_id"])
    if (
        descriptor.get("manifest_ref") is not None
        and descriptor["manifest_ref"] != commit["manifest_ref"]
    ):
        raise ValueError(
            "Descriptor manifest reference differs from committed snapshot"
        )
    rank, _ = store.read_rank(
        descriptor["snapshot_id"],
        tp_rank=0,
        tp_size=identity["tp_size"],
        expected_identity=identity,
    )
    metadata = rank["request_metadata"]
    _validate_portable_metadata(metadata, descriptor)
    if isinstance(obj.rid, list) or (
        obj.rid is not None and obj.rid == metadata["rid"]
    ):
        raise ValueError("Portable restore requires a new single target request ID")
    if not obj.stream:
        raise ValueError(
            "Portable restore requires a fresh streaming /generate request"
        )
    if (
        obj.input_embeds is not None
        or obj.contains_mm_input()
        or obj.session_params
        or obj.session_id
        or obj.lora_path
        or obj.lora_id
        or obj.custom_logit_processor
        or obj.positional_embed_overrides is not None
        or obj.return_hidden_states
        or obj.return_routed_experts
        or obj.return_indexer_topk
        or obj.return_sampling_mask
    ):
        raise ValueError("Portable restore accepts only a plain text decode request")
    obj.text = None
    obj.input_ids = metadata["origin_input_ids"]
    obj.sampling_params = metadata["resolved_sampling_params"]
    for name, value in metadata["request_flags"].items():
        setattr(obj, name, value)
    # The internal selected default is omitted at the public HTTP boundary.
    if getattr(obj, "sampling_logprobs_mode", None) == "selected":
        obj.sampling_logprobs_mode = None


def _tensor_inventory(value, prefix=""):
    result = []
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        result.append(
            {
                "name": prefix,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.extend(_tensor_inventory(item, f"{prefix}.{index}"))
    elif isinstance(value, dict):
        for key in sorted(value):
            result.extend(_tensor_inventory(value[key], f"{prefix}.{key}"))
    elif value is not None:
        raise ValueError(f"Unsupported tensor payload at {prefix}: {type(value)}")
    return result


def _state_receipt(payload):
    inventory = _tensor_inventory(payload)
    layout = [{k: v for k, v in item.items() if k != "sha256"} for item in inventory]
    return {
        "state_kinds": ["attention_kv", "mamba_conv", "mamba_temporal"],
        "state_inventory": inventory,
        "state_inventory_sha256": _json_digest(layout),
        "state_sha256": _json_digest(inventory),
        "state_bytes": sum(item["bytes"] for item in inventory),
    }


def _cuda_memory():
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {"free_bytes": free, "total_bytes": total}


class TokenKVSnapshotManager:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.records = {}
        self.resume_metrics = {}
        self.portable_restore_by_rid = {}
        self.engine_instance_id = f"{socket.gethostname()}:{os.getpid()}"

    @property
    def has_released_request(self):
        # A detached source request no longer owns scheduler state. It must not
        # block the engine while other requests continue decoding.
        return any(
            record["released"] and not record.get("detached", False)
            for record in self.records.values()
        )

    def _cache_layout(self, *, request_scope=False):
        scheduler = self.scheduler
        pool = scheduler.req_to_token_pool.mamba_pool
        config = scheduler.model_config.hf_config
        config_dict = config.to_dict()
        full_pool = getattr(scheduler.token_to_kv_pool_allocator, "_kvcache", None)
        full_layer_ids = sorted(
            getattr(full_pool, "full_attention_layer_id_mapping", {})
        )

        conv_layout = []
        for tensor in pool.mamba_cache.conv:
            shape = list(tensor.shape)
            if len(shape) < 2:
                raise ValueError("Mamba conv cache metadata has no slot axis")
            if (
                not isinstance(shape[1], int)
                or isinstance(shape[1], bool)
                or shape[1] <= 0
            ):
                raise ValueError(
                    "Mamba conv cache slot dimension is not a positive integer"
                )
            if request_scope:
                shape[1] = 1
            conv_layout.append({"shape": shape, "dtype": str(tensor.dtype)})

        temporal_shape = list(pool.mamba_cache.temporal.shape)
        if len(temporal_shape) < 2:
            raise ValueError("Mamba temporal cache metadata has no slot axis")
        if (
            not isinstance(temporal_shape[1], int)
            or isinstance(temporal_shape[1], bool)
            or temporal_shape[1] <= 0
        ):
            raise ValueError(
                "Mamba temporal cache slot dimension is not a positive integer"
            )
        if request_scope:
            temporal_shape[1] = 1

        return {
            "allocator": type(scheduler.token_to_kv_pool_allocator).__name__,
            "request_pool": type(scheduler.req_to_token_pool).__name__,
            "page_size": scheduler.token_to_kv_pool_allocator.page_size,
            "full_attention_layer_ids": full_layer_ids,
            "mamba_layer_ids": list(getattr(pool, "mamba_layer_ids", [])),
            "conv": conv_layout,
            "temporal": {
                "shape": temporal_shape,
                "dtype": str(pool.mamba_cache.temporal.dtype),
            },
        }

    def _identity(self):
        scheduler = self.scheduler
        pool = scheduler.req_to_token_pool.mamba_pool
        config = scheduler.model_config.hf_config
        config_dict = config.to_dict()
        text_config = config_dict.get("text_config", config_dict)
        layout = self._cache_layout(request_scope=True)
        source_files = [Path(__file__), Path(__file__).with_name("schedule_batch.py")]
        source_digest = hashlib.sha256()
        for source in source_files:
            source_digest.update(source.name.encode())
            source_digest.update(source.read_bytes())
        version = get_serving().weight_version
        config_file = Path(scheduler.server_args.model_path) / "config.json"
        return {
            "model_path": str(scheduler.server_args.model_path),
            "model_revision": getattr(config, "_commit_hash", None),
            "model_config_sha256": _json_digest(config_dict),
            "actual_model_config_file_sha256": (
                hashlib.sha256(config_file.read_bytes()).hexdigest()
                if config_file.is_file()
                else None
            ),
            "model_type": config_dict.get("model_type"),
            "architectures": config_dict.get("architectures", []),
            "text_model_type": text_config.get("model_type"),
            "num_hidden_layers": text_config.get("num_hidden_layers"),
            "layer_types": text_config.get("layer_types"),
            "num_experts": text_config.get("num_experts"),
            "linear_conv_kernel_dim": text_config.get("linear_conv_kernel_dim"),
            "weight_version": version,
            "actualweight_version": version,
            "tp_size": get_parallel().tp_size,
            "cache_layout_sha256": _json_digest(layout),
            "engine_source_sha256": source_digest.hexdigest(),
            "cache_layout": layout,
        }

    def _check_runtime(self):
        scheduler = self.scheduler
        failures = []
        if os.environ.get("SGLANG_ENABLE_TOKEN_KV_SNAPSHOT") != "1":
            failures.append("SGLANG_ENABLE_TOKEN_KV_SNAPSHOT=1 is required")
        mode = scheduler.disaggregation_mode
        if getattr(mode, "value", mode) not in (None, "null"):
            failures.append("PD disaggregation is unsupported")
        if (
            get_parallel().pp_size != 1
            or get_parallel().attn_cp_size != 1
            or get_parallel().dcp_size != 1
        ):
            failures.append("PP and CP are unsupported")
        if get_parallel().attn_dp_size != 1:
            failures.append("DP is unsupported")
        if not scheduler.spec_algorithm.is_none():
            failures.append("speculative decoding is unsupported")
        pool = getattr(scheduler.req_to_token_pool, "mamba_pool", None)
        if pool is None:
            failures.append("hybrid attention/GDN pools are required")
        elif getattr(pool, "replayssm_write_pos", None) is not None:
            failures.append("ReplaySSM must be disabled")
        elif getattr(pool, "_slot_siblings", []):
            failures.append("additional hybrid side-state pools are unsupported")
        if get_serving().weight_version is None:
            failures.append("a non-null engine weight_version is required")
        return failures

    def _validate_identity(self, request):
        identity = self._identity()
        if request.expected_weight_version != identity["weight_version"]:
            raise ValueError(
                "Engine weight version differs from expected_weight_version"
            )
        if request.identity != identity:
            raise ValueError(
                "Snapshot model/version/TP/layout identity differs from engine"
            )
        return identity

    def _drain_overlap_results(self):
        """Finish the native overlap result boundary before reading request state.

        ``pause_generation(in_place)`` intentionally leaves the scheduler state
        untouched. In overlap mode that can leave one completed forward in
        ``result_queue``. Processing that result here reuses the scheduler's
        normal synchronization and request-update path; it does not disable
        overlap or launch another forward.
        """
        scheduler = self.scheduler
        if not scheduler.enable_overlap:
            return 0
        result_queue = getattr(scheduler, "result_queue", None)
        if result_queue is None:
            return 0
        drained = 0
        while result_queue:
            batch, result = result_queue.popleft()
            scheduler.process_batch_result(batch, result)
            drained += 1
        # ``last_batch`` is the native overlap marker for a queued result. Once
        # the queue is fully processed here, keep the scheduler's next iteration
        # from treating a stale marker as another outstanding result.
        scheduler.last_batch = None
        return drained

    def _request(self, request):
        scheduler = self.scheduler
        rid = request.target_rid or request.rid
        batch = scheduler.running_batch
        matches = [
            (index, req) for index, req in enumerate(batch.reqs) if req.rid == rid
        ]
        if len(matches) != 1:
            raise ValueError("Snapshot requires one active request matching rid")
        index, req = matches[0]
        if req.finished() or not req.output_ids:
            raise ValueError("Request is finished or has no sampled output token")
        if (
            req.grammar is not None
            or getattr(req, "beam_group", None) is not None
            or req.multimodal_inputs is not None
            or req.input_embeds is not None
            or req.session is not None
            or req.lora_id is not None
            or req.custom_logit_processor is not None
        ):
            raise ValueError(
                "Grammar, beam, multimodal, embeddings, sessions and LoRA are unsupported"
            )
        computed = req.kv.kv_committed_len
        expected = len(req.origin_input_ids) + len(req.output_ids) - 1
        if computed != expected or int(batch.seq_lens_cpu[index]) != computed:
            raise ValueError(
                "Computed prefix is not aligned with the pending last output token"
            )
        # Native non-overlap decode relays the sampled token through FutureMap
        # and clears batch.input_ids. Reading the previous forward's input would
        # either dereference None or check the token before the saved boundary.
        pending = self.scheduler.future_map.output_tokens_buf[req.kv.req_pool_idx]
        if int(pending) != req.output_ids[-1]:
            raise ValueError("Decode input differs from pending last output token")
        if not req.kv.holds_kv:
            raise ValueError("Request has no active KV allocation")
        if not 0 <= req.kv.cache_protected_len <= computed:
            raise ValueError("Shared prefix length is outside the computed prefix")
        if req.kv.kv_allocated_len != computed:
            raise ValueError("Overallocated KV is unsupported")
        return req, batch, computed, index

    def _request_metadata(self, req, computed, source_binding=None):
        if (
            getattr(req.sampling_params, "watermark", None) is not None
            or getattr(req, "sampling_logprobs_mode", "selected") != "selected"
            or getattr(req, "return_flat_raw_top_logprobs", False)
        ):
            raise ValueError(
                "Only non-watermarked selected-token logprobs are portable"
            )
        if (
            req.return_sampling_mask
            or req.return_hidden_states
            or req.return_routed_experts
            or req.return_indexer_topk
            or req.logprob.top_logprobs_num
            or req.logprob.token_ids_logprob
            or (
                req.return_logprob and req.logprob_start_len < len(req.origin_input_ids)
            )
        ):
            raise ValueError("Only ordinary output-token logprobs are portable")
        spans = getattr(req, "_token_kv_output_version_spans", [])
        if not spans or spans[-1]["end"] != len(req.output_ids):
            raise ValueError(
                "Actual processed-output weight version coverage is missing"
            )
        if len(spans) != 1 or spans[0]["version"] != get_serving().weight_version:
            raise ValueError(
                "A request spanning multiple weight versions is unsupported"
            )
        output_logprobs = list(req.logprob.output_token_logprobs_val or [])
        output_logprob_ids = list(req.logprob.output_token_logprobs_idx or [])
        return {
            "request_schema_version": 2,
            "rid": req.rid,
            "origin_input_ids": list(req.origin_input_ids),
            "origin_input_ids_sha256": _json_digest(list(req.origin_input_ids)),
            "resolved_sampling_params": _resolved_sampling_params(req.sampling_params),
            "request_flags": {
                "return_logprob": req.return_logprob,
                "logprob_start_len": req.logprob_start_len,
                "top_logprobs_num": req.logprob.top_logprobs_num,
                "token_ids_logprob": req.logprob.token_ids_logprob,
                "require_reasoning": req.require_reasoning,
                "sampling_logprobs_mode": "selected",
            },
            "output_ids": list(req.output_ids),
            "output_logprobs": output_logprobs,
            "output_logprob_token_ids": output_logprob_ids,
            "output_weight_versions": [dict(span) for span in spans],
            "computed_tokens": computed,
            "pending_input_token": int(req.output_ids[-1]),
            "send_token_offset": req.send_token_offset,
            "source_stream_cursors": {
                "send_token_offset": req.send_token_offset,
                "send_decode_id_offset": req.send_decode_id_offset,
                "send_output_token_logprobs_offset": req.send_output_token_logprobs_offset,
                "surr_offset": req.surr_offset,
                "read_offset": req.read_offset,
            },
            "reasoning_state": {
                "is_reasoning_over": req._is_reasoning_over,
                "reasoning_tokens": req.reasoning_tokens,
                "think_end_match_len": req._think_end_match_len,
            },
            "decode_batch_idx": req.decode_batch_idx,
            "temp_scaled_logprobs": req.temp_scaled_logprobs,
            "top_p_normalized_logprobs": req.top_p_normalized_logprobs,
            "source_binding": source_binding,
            "remaining_generation_tokens": req.sampling_params.max_new_tokens
            - len(req.output_ids),
            "sampling_state_recovery": "PORTABLE_NORMALIZED_PARAMS_AND_TOKEN_HISTORY",
        }

    def _copy(self, req, computed):
        scheduler = self.scheduler
        indices = scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :computed
        ].long()
        kv_cpu, mamba_cpu = scheduler.token_to_kv_pool_allocator.get_cpu_copy(
            indices,
            mamba_indices=req.kv.mamba_pool_idx,
        )
        if mamba_cpu is None or len(mamba_cpu) != 2:
            raise ValueError("Expected ordinary GDN conv and temporal state")
        return {
            "attention_kv": kv_cpu,
            "mamba_conv": mamba_cpu[0],
            "mamba_temporal": mamba_cpu[1],
        }

    def _load(self, req, computed, payload):
        scheduler = self.scheduler
        indices = scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :computed
        ].long()
        scheduler.token_to_kv_pool_allocator.load_cpu_copy(
            (
                payload["attention_kv"],
                (payload["mamba_conv"], payload["mamba_temporal"]),
            ),
            indices,
            mamba_indices=req.kv.mamba_pool_idx,
        )

    def _slots(self, req):
        pool = self.scheduler.req_to_token_pool
        return {
            "request_row": req.kv.req_pool_idx,
            "request_row_generation": int(pool.req_generation[req.kv.req_pool_idx]),
            "mamba_slot": int(req.kv.mamba_pool_idx),
        }

    def _export(self, request, identity, store):
        if request.snapshot_id in self.records:
            raise ValueError("Snapshot ID already has a local execution state")
        req, batch, computed, batch_index = self._request(request)
        if request.source_binding is not None:
            if request.source_binding.get("request_id") != req.rid:
                raise ValueError(
                    "Source binding request ID differs from active request"
                )
            if (
                get_parallel().tp_rank == 0
                and request.source_binding.get("source_engine_instance_id")
                != self.engine_instance_id
            ):
                raise ValueError(
                    "Source binding engine instance differs from actual TP rank zero"
                )
        metadata = self._request_metadata(req, computed, request.source_binding)
        started = time.perf_counter()
        payload = self._copy(req, computed)
        copied = time.perf_counter()
        state = _state_receipt(payload)
        stream = io.BytesIO()
        torch.save(payload, stream)
        receipt = store.write_rank(
            request.snapshot_id,
            tp_rank=get_parallel().tp_rank,
            tp_size=get_parallel().tp_size,
            identity=identity,
            request_metadata=metadata,
            tensor_metadata=state,
            payload=stream.getvalue(),
        )
        self.records[request.snapshot_id] = {
            "req": req,
            "batch": batch,
            "batch_index": batch_index,
            "metadata": metadata,
            "identity": identity,
            "state": state,
            "slots": self._slots(req),
            "released": False,
            "memory_suspended": False,
        }
        return {
            **state,
            **metadata,
            "source_slots": self.records[request.snapshot_id]["slots"],
            "allocation_epoch": self.records[request.snapshot_id]["slots"][
                "request_row_generation"
            ],
            "old_slots_released": False,
            "loaded_from_snapshot": False,
            "allocated_new_slots": False,
            "save_receipt": receipt,
            "copy_to_host_s": copied - started,
            "persist_rank_s": time.perf_counter() - copied,
        }

    def _preempt(self, request, failures):
        """Classify while paused on all ranks, then export or cancel exact queued RID.

        A frontend which has not received a token cannot infer that the engine
        did no work. Only an exact waiting request without native output or any
        allocated slot is eligible for original-input replay. Unknown states
        fail closed, and no rank mutates its queue before TP agreement.
        """
        scheduler = self.scheduler
        req = None
        intent = {"status": "FAILED", "tp_rank": get_parallel().tp_rank}
        try:
            if failures:
                raise ValueError("; ".join(failures))
            if not scheduler._engine_paused:
                raise ValueError("pause_generation(in_place) must precede preemption")
            drained_overlap_results = self._drain_overlap_results()
            identity = self._validate_identity(request)
            if not request.snapshot_id or not request.storage_dir:
                raise ValueError("snapshot_id and storage_dir are required")
            rid = request.target_rid or request.rid
            binding = request.source_binding
            if not isinstance(binding, dict) or binding.get("request_id") != rid:
                raise ValueError("Source preemption request binding differs")
            if (
                get_parallel().tp_rank == 0
                and binding.get("source_engine_instance_id") != self.engine_instance_id
            ):
                raise ValueError("Source preemption engine instance differs")
            if (
                type(binding.get("source_execution_generation")) is not int
                or binding["source_execution_generation"] < 0
                or any(
                    not isinstance(binding.get(key), str) or not binding[key]
                    for key in (
                        "trajectory_id",
                        "source_owner_id",
                        "source_lease_id",
                        "policy_identity_sha256",
                        "origin_input_ids_sha256",
                    )
                )
            ):
                raise ValueError(
                    "Source preemption has an incomplete execution/input binding"
                )
            active = [item for item in scheduler.running_batch.reqs if item.rid == rid]
            waiting = [item for item in scheduler.waiting_queue if item.rid == rid]
            if len(active) == 1 and not waiting:
                req = active[0]
                if req.finished() or not req.output_ids:
                    raise ValueError(
                        "Active request has no portable sampled-output boundary"
                    )
                disposition = "ACTIVE_EXPORTED"
            elif len(waiting) == 1 and not active:
                req = waiting[0]
                if (
                    req.finished()
                    or req.output_ids
                    or req.kv.req_pool_idx is not None
                    or req.kv.mamba_pool_idx is not None
                    or req.kv.kv_committed_len != 0
                    or req.kv.holds_kv
                    or req.kv.retraction_backup is not None
                    or req.prefix_len != 0
                    or getattr(getattr(scheduler, "chunked_req", None), "rid", None)
                    == rid
                ):
                    raise ValueError(
                        "Queued request has sampled output, allocated KV, or in-flight prefill"
                    )
                disposition = "PENDING_CANCELLED"
            else:
                raise ValueError(
                    "Source request is absent or not uniquely active/queued"
                )
            custom = req.sampling_params.custom_params
            encoded = (
                custom.get("trajectory_source_request_v1")
                if isinstance(custom, dict)
                else None
            )
            if not isinstance(encoded, str):
                raise ValueError("Native request has no original dispatch provenance")
            original = json.loads(encoded)
            expected = {
                "schema_version": 1,
                "request_id": rid,
                "trajectory_id": binding["trajectory_id"],
                "source_execution_generation": binding["source_execution_generation"],
                "source_owner_id": binding["source_owner_id"],
                "source_lease_id": binding["source_lease_id"],
                "policy_version": identity["weight_version"],
                "policy_identity_sha256": binding["policy_identity_sha256"],
                "origin_input_ids_sha256": _json_digest(list(req.origin_input_ids)),
            }
            if (
                not isinstance(original, dict)
                or type(original.get("source_execution_generation")) is not int
                or original != expected
                or binding["origin_input_ids_sha256"]
                != expected["origin_input_ids_sha256"]
            ):
                raise ValueError(
                    "Native original request policy/generation/input provenance differs"
                )
            intent.update(
                status="READY",
                source_disposition=disposition,
                identity=identity,
                source_binding=dict(binding),
                origin_input_ids_sha256=_json_digest(list(req.origin_input_ids)),
                generated_tokens=len(req.output_ids),
            )
        except Exception as error:
            intent["error"] = str(error)
        intents = [intent]
        if get_parallel().tp_size > 1:
            intents = [None] * get_parallel().tp_size
            torch.distributed.all_gather_object(
                intents, intent, group=scheduler.tp_cpu_group
            )
        if any(row.get("status") != "READY" for row in intents):
            raise ValueError("Native source preemption rejected: " + str(intents))
        comparable = [
            {key: value for key, value in row.items() if key != "tp_rank"}
            for row in intents
        ]
        if any(row != comparable[0] for row in comparable):
            raise ValueError(
                "TP ranks disagree on exact source request preemption state"
            )
        if intent["source_disposition"] == "ACTIVE_EXPORTED":
            return {
                **self._export(
                    request, identity, TokenKVSnapshotStore(request.storage_dir)
                ),
                "source_disposition": "ACTIVE_EXPORTED",
                "identity": identity,
            }
        # Follow the original native waiting-queue abort cleanup, with exact
        # equality rather than abort_request's RID-prefix matching. Resource
        # owning/active requests were rejected before this mutation.
        scheduler._release_aborted_request(req)
        matches = [
            index for index, item in enumerate(scheduler.waiting_queue) if item is req
        ]
        if len(matches) != 1:
            raise ValueError(
                "Paused native waiting request changed before cancellation"
            )
        scheduler.waiting_queue.pop(matches[0])
        scheduler.ipc_channels.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
        return {
            **intent,
            "pending_removed": True,
            "kv_allocated": False,
            "prepared_while_paused": True,
            "overlap_results_drained": drained_overlap_results,
            "native_cancel_mode": "EXACT_WAITING_QUEUE_REMOVE_AFTER_TP_PREPARE",
        }

    def _read(self, request, identity, store, record):
        metadata, raw = store.read_rank(
            request.snapshot_id,
            tp_rank=get_parallel().tp_rank,
            tp_size=get_parallel().tp_size,
            expected_identity=identity,
            expected_request_metadata=record["metadata"],
        )
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if _state_receipt(payload) != record["state"]:
            raise ValueError("Serialized state digest/inventory differs from export")
        return payload

    def _release(self, request, identity, store, record):
        if record["released"]:
            raise ValueError("Request slots were already released")
        req, batch, computed, _ = self._request(request)
        if (
            self._request_metadata(req, computed, record["metadata"]["source_binding"])
            != record["metadata"]
        ):
            raise ValueError("Request advanced after snapshot export")
        if self._slots(req) != record["slots"]:
            raise ValueError(
                "Source allocator row/slot generation changed after export"
            )
        self._read(request, identity, store, record)
        if (
            request.release_full_gpu_pool
            and not self.scheduler.memory_saver_adapter.enabled
        ):
            raise ValueError("Full GPU pool release requires --enable-memory-saver")
        scheduler = self.scheduler
        other_active_rids = [item.rid for item in batch.reqs if item is not req]
        has_other_work = (
            bool(other_active_rids)
            or bool(scheduler.waiting_queue)
            or scheduler.chunked_req is not None
        )
        if request.release_full_gpu_pool and has_other_work:
            raise ValueError(
                "Full GPU pool release requires no other active or pending requests"
            )
        started = time.perf_counter()
        memory_before = _cuda_memory()
        allocator = self.scheduler.token_to_kv_pool_allocator
        before_free = allocator.available_size()
        before_rows = self.scheduler.req_to_token_pool.available_size()
        shared_prefix_len = req.kv.cache_protected_len
        scheduler.tree_cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)
        release_kv_cache(req, self.scheduler.tree_cache, checkpoint=False)
        keep_indices = [
            index for index, item in enumerate(batch.reqs) if item is not req
        ]
        batch.filter_batch(keep_indices=keep_indices)
        record["released"] = True
        record["detached"] = True
        record["batch_index"] = None
        # The target is absent from every scheduler batch tensor. Other members,
        # the waiting queue and shared radix prefixes remain native state.
        if request.release_full_gpu_pool:
            self.scheduler.weight_updater.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
            )
            record["memory_suspended"] = True
        memory_after = _cuda_memory()
        memory_delta = (
            memory_after["free_bytes"] - memory_before["free_bytes"]
            if memory_before is not None and memory_after is not None
            else None
        )
        record["full_gpu_pool_released"] = (
            record["memory_suspended"] and memory_delta is not None and memory_delta > 0
        )
        return {
            **record["state"],
            **record["metadata"],
            "source_slots": record["slots"],
            "allocation_epoch": record["slots"]["request_row_generation"],
            "old_slots_released": req.kv.req_pool_idx is None
            and req.kv.mamba_pool_idx is None,
            "destroyed_old_gpu_slots": False,
            "detached": True,
            "engine_generation_unblocked": not self.has_released_request,
            "old_gpu_state_zeroed": False,
            "native_release_mode": "release_kv_cache(checkpoint=False)",
            "shared_prefix_len": shared_prefix_len,
            "other_active_request_ids": other_active_rids,
            "allocator_free_tokens_before": before_free,
            "allocator_free_tokens_after": allocator.available_size(),
            "request_rows_free_before": before_rows,
            "request_rows_free_after": self.scheduler.req_to_token_pool.available_size(),
            "memory_saver_kv_pause_executed": record["memory_suspended"],
            "full_gpu_pool_released": record["full_gpu_pool_released"],
            "kv_memory_region_released": record["full_gpu_pool_released"],
            "weights_memory_region_released": False,
            "gpu_returned_for_training": False,
            "gpu_memory_before": memory_before,
            "gpu_memory_after": memory_after,
            "gpu_free_bytes_delta": memory_delta,
            "release_and_destroy_s": time.perf_counter() - started,
            "release_s": time.perf_counter() - started,
        }

    def _restore(self, request, identity, store, record):
        if not record["released"]:
            raise ValueError("Restore requires released source slots")
        payload = self._read(request, identity, store, record)
        started = time.perf_counter()
        req, batch = record["req"], record["batch"]
        computed = record["metadata"]["computed_tokens"]
        req.advance_cache_request_handle()
        if record["memory_suspended"]:
            self.scheduler.weight_updater.resume_memory_occupation(
                ResumeMemoryOccupationReqInput(tags=["kv_cache"])
            )
            record["memory_suspended"] = False
        try:
            rows, restored = self._allocate_and_load(
                req, computed, payload, record["state"]
            )
            imported = self._prepare_portable_decode_batch(
                req, rows, computed, record["metadata"]
            )
            batch = self._merge_portable_decode_batch(imported)
        except Exception:
            # Stay paused on any failure; no request may use unverifiable state.
            if req.kv.req_pool_idx is not None or req.kv.mamba_pool_idx is not None:
                release_kv_cache(req, self.scheduler.tree_cache, checkpoint=False)
            raise
        self.scheduler.running_batch = batch
        self.scheduler.cur_batch_for_debug = batch
        record["released"] = False
        record["detached"] = False
        self.resume_metrics[req.rid] = {
            "resumed_prefill_tokens": 0,
            "resumed_decode_input_tokens": 0,
            "resumed_forward_batches": 0,
            "resumed_prefill_duration_s": None,
        }
        destination = self._slots(req)
        result = {
            **restored,
            **record["metadata"],
            "source_slots": record["slots"],
            "destination_slots": destination,
            "allocation_epoch": destination["request_row_generation"],
            "old_slots_released": True,
            "detached": False,
            "loaded_from_snapshot": True,
            "allocated_new_slots": True,
            "digestmatch": True,
            "full_gpu_pool_released": record.get("full_gpu_pool_released", False),
            "kv_memory_region_released": record.get("full_gpu_pool_released", False),
            "weights_memory_region_released": False,
            "gpu_returned_for_training": False,
            "restore_and_readback_s": time.perf_counter() - started,
        }
        del self.records[request.snapshot_id]
        return result

    def _rank_errors(self, error):
        errors = [error]
        if get_parallel().tp_size > 1:
            errors = [None] * get_parallel().tp_size
            torch.distributed.all_gather_object(
                errors, error, group=self.scheduler.tp_cpu_group
            )
        failures = [
            f"rank {rank}: {message}" for rank, message in enumerate(errors) if message
        ]
        if failures:
            raise ValueError("; ".join(failures))

    def _allocate_and_load(self, req, computed, payload, expected_state):
        if req.kv.holds_kv or req.kv.holds_mamba:
            raise ValueError(
                "Portable allocation requires an unowned request resource record"
            )
        pool = self.scheduler.req_to_token_pool
        allocator = self.scheduler.token_to_kv_pool_allocator
        allocated = (
            (computed + allocator.page_size - 1) // allocator.page_size
        ) * allocator.page_size
        rows = pool.alloc([req])
        if rows is None:
            raise ValueError("No free request or GDN rows for portable restore")
        indices = allocator.alloc(allocated)
        if indices is None:
            pool.free_mamba_cache(req)
            pool.free(req)
            raise ValueError("No free full-attention KV slots for portable restore")
        pool.req_to_token[req.kv.req_pool_idx, :computed] = indices[:computed].to(
            torch.int32
        )
        req.kv.kv_committed_len = computed
        req.kv.kv_allocated_len = computed
        req.kv.cache_protected_len = 0
        req.kv.cache_inserted_len = 0
        req.prefix_len = computed
        req.refresh_fill_ids()
        req.extend_end = computed
        # Every imported token owns a new physical slot. Anchor its native
        # cache lifecycle at the empty root, so completion can release its
        # reference and insert the restored sequence into a live radix tree.
        tree_cache = self.scheduler.tree_cache
        if not tree_cache.disable:
            root = tree_cache.root_node_handle(extra_key=req.extra_key)
            req.last_node = root
            req.last_host_node = root
            req.best_match_node = root
            req.lock = tree_cache.lock(root)
        self._load(req, computed, payload)
        req.kv.mamba_needs_clear = False
        readback = _state_receipt(self._copy(req, computed))
        if readback != expected_state:
            raise ValueError("Portable import failed GPU tensor readback verification")
        return rows, readback

    def _prepare_portable_decode_batch(self, req, rows, computed, metadata):
        scheduler = self.scheduler
        batch = ScheduleBatch.init_new(
            [req],
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
            scheduler.tree_cache,
            scheduler.model_config,
            scheduler.enable_overlap,
            scheduler.spec_algorithm,
        )
        batch.forward_mode = ForwardMode.DECODE
        batch.req_pool_indices = torch.tensor(
            rows, dtype=torch.int64, device=batch.device
        )
        batch.req_pool_indices_cpu = torch.tensor(rows, dtype=torch.int64)
        batch.seq_lens = torch.tensor(
            [computed], dtype=torch.int64, device=batch.device
        )
        batch.seq_lens_cpu = torch.tensor([computed], dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(
            [computed], dtype=torch.int32, device=batch.device
        )
        batch.seq_lens_sum = computed
        batch.multimodal_inputs = [None]
        batch.top_logprobs_nums = [req.logprob.top_logprobs_num]
        batch.token_ids_logprobs = [req.logprob.token_ids_logprob]
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, scheduler.model_config.vocab_size
        )
        penalties = batch.sampling_info.penalizer_orchestrator
        if penalties.is_required:
            # Native prepare_for_decode adds the pending last token. Rebuild
            # only earlier accepted outputs so penalties count each token once.
            for token in metadata["output_ids"][:-1]:
                penalties.cumulate_output_tokens(
                    torch.tensor([token], dtype=torch.int64, device=batch.device)
                )
        scheduler.future_map.stash(
            batch.req_pool_indices,
            RelayPayload(
                bonus_tokens=torch.tensor(
                    [metadata["pending_input_token"]],
                    dtype=torch.int64,
                    device=batch.device,
                )
            ),
        )
        batch.input_ids = None  # Native resolve_forward_inputs consumes relay.
        return batch

    def _merge_portable_decode_batch(self, imported):
        scheduler = self.scheduler
        running = scheduler.running_batch
        if running is None or running.is_empty():
            running = imported
        else:
            # Native merge preserves existing members, penalty state and
            # FutureMap rows before appending the restored request.
            running.merge_batch(imported)
        scheduler.running_batch = running
        scheduler.cur_batch_for_debug = running
        return running

    def import_for_generation(self, req, descriptor):
        """Import into a newly registered native /generate Req, never records[]."""
        scheduler = self.scheduler
        started = time.perf_counter()
        error = None
        try:
            failures = self._check_runtime()
            if failures:
                raise ValueError("; ".join(failures))
            # Input handling is a synchronous scheduler boundary. Finish queued
            # overlap results first, then merge into the current native batch.
            self._drain_overlap_results()
            other_active_before_import = [
                item.rid
                for item in getattr(
                    getattr(scheduler, "running_batch", None), "reqs", ()
                )
                if item.rid != req.rid
            ]
            identity = self._validate_identity(SimpleNamespace(**descriptor))
            store = TokenKVSnapshotStore(descriptor["storage_dir"])
            rank, raw = store.read_rank(
                descriptor["snapshot_id"],
                tp_rank=get_parallel().tp_rank,
                tp_size=get_parallel().tp_size,
                expected_identity=identity,
            )
            metadata = rank["request_metadata"]
            _validate_portable_metadata(metadata, descriptor)
            if list(req.origin_input_ids) != metadata["origin_input_ids"]:
                raise ValueError(
                    "New native request input differs from committed source"
                )
            if (
                _resolved_sampling_params(req.sampling_params)
                != metadata["resolved_sampling_params"]
            ):
                raise ValueError(
                    "New native request sampling differs from committed source"
                )
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            state = _state_receipt(payload)
            if state != rank["tensor_metadata"]:
                raise ValueError("Portable snapshot tensor inventory/digest differs")
        except Exception as failure:
            error = str(failure)
        # All ranks validate the committed input before any rank mutates its pool.
        self._rank_errors(error)

        error = None
        try:
            req.output_ids = array("q", metadata["output_ids"])
            req._token_kv_output_version_spans = [
                dict(span) for span in metadata["output_weight_versions"]
            ]
            req.logprob.output_token_logprobs_val = list(metadata["output_logprobs"])
            req.logprob.output_token_logprobs_idx = list(
                metadata["output_logprob_token_ids"]
            )
            req.input_logprob_sent = True  # This scope only returns output logprobs.
            req.temp_scaled_logprobs = metadata["temp_scaled_logprobs"]
            req.top_p_normalized_logprobs = metadata["top_p_normalized_logprobs"]
            req.decode_batch_idx = metadata["decode_batch_idx"]
            reasoning = metadata["reasoning_state"]
            req._is_reasoning_over = reasoning["is_reasoning_over"]
            req.reasoning_tokens = reasoning["reasoning_tokens"]
            req._think_end_match_len = reasoning["think_end_match_len"]
            # The matcher is rebuilt from the same resolved stop/reasoning params.
            # A new detokenizer emits the complete saved prefix plus the tail once.
            req.send_token_offset = 0
            req.send_decode_id_offset = 0
            req.send_output_token_logprobs_offset = 0
            req.surr_offset = None
            req.read_offset = None
            req.decoded_text = ""
            computed = metadata["computed_tokens"]
            rows, readback = self._allocate_and_load(req, computed, payload, state)
            imported = self._prepare_portable_decode_batch(
                req, rows, computed, metadata
            )
            batch = self._merge_portable_decode_batch(imported)
        except Exception as failure:
            error = str(failure)
        try:
            self._rank_errors(error)
        except Exception:
            if req.kv.req_pool_idx is not None or req.kv.mamba_pool_idx is not None:
                release_kv_cache(req, scheduler.tree_cache, checkpoint=False)
            raise

        scheduler.running_batch = batch
        scheduler.cur_batch_for_debug = batch
        self.resume_metrics[req.rid] = {
            "committed_prefix_tokens": computed,
            "actual_input_counter_status": "ARMED_NOT_YET_EXECUTED",
            "actual_recomputed_tokens": 0,
            "actual_processed_input_tokens": 0,
            "actual_extend_input_tokens": 0,
            "actual_decode_input_tokens": 0,
            "actual_completed_forward_iters": [],
            "resumed_prefill_tokens": 0,
            "resumed_decode_input_tokens": 0,
            "resumed_forward_batches": 0,
            "resumed_prefill_duration_s": None,
        }
        destination = self._slots(req)
        self.portable_restore_by_rid[req.rid] = {
            **readback,
            "status": "PASS",
            "snapshot_id": descriptor["snapshot_id"],
            "identity": identity,
            "source_rid": metadata["rid"],
            "target_rid": req.rid,
            "source_binding": metadata["source_binding"],
            "engine_instance_id": self.engine_instance_id,
            "computed_tokens": computed,
            "pending_input_token": metadata["pending_input_token"],
            "remaining_generation_tokens": metadata["remaining_generation_tokens"],
            "output_ids": metadata["output_ids"],
            "output_logprobs": metadata["output_logprobs"],
            "source_stream_cursors": metadata["source_stream_cursors"],
            "target_stream_offset": 0,
            "stream_returns_full_output": True,
            "request_reconstructed": True,
            "other_active_request_ids_before_import": other_active_before_import,
            "active_request_ids_after_import": [item.rid for item in batch.reqs],
            "from_retained_cpu_record": False,
            "sampling_state_recovery": "NORMALIZED_PARAMS_AND_HISTORY_RECONSTRUCTED",
            "sampling_rng_recovery": "FUNCTIONAL_SAMPLING_EQUIVALENCE_NO_GLOBAL_RNG_REPLAY",
            "loaded_from_snapshot": True,
            "allocated_new_slots": True,
            "digestmatch": True,
            "destination_slots": destination,
            "allocation_epoch": destination["request_row_generation"],
            "manifest_ref": store.receipt(descriptor["snapshot_id"])["manifest_ref"],
            "restore_and_readback_s": time.perf_counter() - started,
            "recovery_scope": "PORTABLE_FRESH_ENGINE_NATIVE_DECODE",
        }
        return self.portable_restore_by_rid[req.rid]

    def capture_native_input_intervals(self, batch):
        """Copy physical-import input positions before this actual native forward."""
        rows = []
        for index, req in enumerate(batch.reqs):
            metrics = self.resume_metrics.get(req.rid, {})
            prefix = metrics.get("committed_prefix_tokens")
            if type(prefix) is not int:
                continue
            try:
                if batch.forward_mode.is_extend():
                    start, end = req.prefix_len, req.extend_end
                    kind = "extend"
                elif batch.forward_mode.is_decode():
                    lengths = batch.seq_lens_cpu
                    if getattr(lengths, "is_cuda", False):
                        raise ValueError("counter cannot copy CUDA lengths")
                    value = lengths[index]
                    start = int(value.item() if hasattr(value, "item") else value) - 1
                    end, kind = start + 1, "decode"
                else:
                    raise ValueError("unsupported physical input mode")
                if (
                    type(start) is not int
                    or type(end) is not int
                    or start < 0
                    or end <= start
                    or type(batch.forward_iter) is not int
                ):
                    raise ValueError("missing native input interval")
                rows.append(
                    (
                        req.rid,
                        batch.forward_iter,
                        kind,
                        start,
                        end,
                        prefix,
                        str(get_serving().weight_version),
                    )
                )
            except Exception:
                rows.append(
                    (
                        req.rid,
                        getattr(batch, "forward_iter", None),
                        "unavailable",
                        None,
                        None,
                        prefix,
                        None,
                    )
                )
        return tuple(rows)

    def observe_native_input_intervals(self, intervals):
        """Count only intervals carried by this successfully processed result."""
        for rid, forward, kind, start, end, prefix, version in intervals or ():
            metrics = self.resume_metrics.get(rid)
            if metrics is None or "committed_prefix_tokens" not in metrics:
                continue
            if metrics["actual_input_counter_status"] == "UNAVAILABLE":
                continue
            if (
                kind == "unavailable"
                or prefix != metrics["committed_prefix_tokens"]
                or version != str(get_serving().weight_version)
            ):
                metrics["actual_input_counter_status"] = "UNAVAILABLE"
                metrics["actual_recomputed_tokens"] = None
                continue
            if forward in metrics["actual_completed_forward_iters"]:
                continue
            metrics["actual_completed_forward_iters"].append(forward)
            metrics["actual_processed_input_tokens"] += end - start
            metrics["actual_" + kind + "_input_tokens"] += end - start
            metrics["actual_recomputed_tokens"] += max(
                0, min(end, prefix) - max(start, 0)
            )
            metrics["actual_input_counter_status"] = "OBSERVED_COMPLETED_NATIVE_FORWARD"

    def observe_batch_result(self, batch):
        for index, req in enumerate(batch.reqs):
            # Observe the actual version after the real batch result was processed.
            spans = getattr(req, "_token_kv_output_version_spans", [])
            previous_end = spans[-1]["end"] if spans else 0
            end = len(req.output_ids)
            version = get_serving().weight_version
            if end > previous_end:
                if spans and spans[-1]["version"] == version:
                    spans[-1]["end"] = end
                else:
                    spans.append(
                        {"version": version, "start": previous_end, "end": end}
                    )
                req._token_kv_output_version_spans = spans
            metrics = self.resume_metrics.get(req.rid)
            if metrics is None:
                continue
            if batch.forward_mode.is_decode():
                metrics["resumed_decode_input_tokens"] += 1
                metrics["resumed_forward_batches"] += 1
            elif batch.forward_mode.is_extend():
                lengths = getattr(batch, "extend_lens", None)
                if lengths is None:
                    metrics["resumed_prefill_tokens"] = None
                elif metrics["resumed_prefill_tokens"] is not None:
                    metrics["resumed_prefill_tokens"] += int(lengths[index])
                metrics["resumed_forward_batches"] += 1

    def _scheduler_process_identity(self):
        """Observe this actual scheduler CUDA context; no allocation or inference."""
        observed = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "tp_rank": get_parallel().tp_rank,
            "cuda_context_initialized": torch.cuda.is_initialized(),
        }
        try:
            if not observed["cuda_context_initialized"]:
                raise ValueError("Scheduler CUDA context is not initialized")
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            uuid = getattr(properties, "uuid", None)
            if uuid is None:
                raise ValueError("Scheduler CUDA device UUID is unavailable")
            if isinstance(uuid, bytes):
                try:
                    uuid = uuid.decode("ascii")
                except UnicodeDecodeError:
                    uuid = uuid.hex()
            observed.update(status="OBSERVED", device_index=device, gpu_uuid=str(uuid))
        except Exception as error:
            observed.update(
                status="UNAVAILABLE",
                gpu_uuid=None,
                error_type=type(error).__name__,
                error_message=str(error)[:240],
            )
        return observed

    def handle(self, request):
        started = time.perf_counter()
        local = {"tp_rank": get_parallel().tp_rank}
        try:
            failures = self._check_runtime()
            if request.operation == "inspect":
                local.update(
                    status="PASS" if not failures else "UNSUPPORTED",
                    identity=self._identity() if not failures else None,
                    local_cache_layout=self._cache_layout() if not failures else None,
                    supported=not failures,
                    runtime_constraints=failures,
                    resume_metrics_by_rid=self.resume_metrics,
                    portable_restore_by_rid=self.portable_restore_by_rid,
                    engine_instance_id=self.engine_instance_id,
                    scheduler_process_identity=self._scheduler_process_identity(),
                )
            elif request.operation == "preempt":
                local.update(self._preempt(request, failures), status="PASS")
            elif request.operation == "publish_weight_version":
                if failures:
                    raise ValueError("; ".join(failures))
                # Native weight publication runs on an idle scheduler without
                # requiring the separate snapshot pause/resume protocol.
                if not self.scheduler.is_fully_idle():
                    raise ValueError(
                        "publish_weight_version requires no active, queued, or overlap requests"
                    )
                if not request.expected_weight_version:
                    raise ValueError("expected_weight_version is required")
                if not self.scheduler.flush_cache():
                    raise ValueError("native flush_cache failed")
                self.scheduler.record_weight_version_change(
                    request.expected_weight_version
                )
                local.update(
                    status="PASS",
                    identity=self._identity(),
                    flushed_cache=True,
                )
            else:
                if failures:
                    raise ValueError("; ".join(failures))
                if not self.scheduler._engine_paused:
                    raise ValueError(
                        "pause_generation(in_place) must precede snapshot operations"
                    )
                drained_overlap_results = self._drain_overlap_results()
                if not request.snapshot_id or not request.storage_dir:
                    raise ValueError("snapshot_id and storage_dir are required")
                identity = self._validate_identity(request)
                store = TokenKVSnapshotStore(request.storage_dir)
                if request.operation == "export":
                    receipt = self._export(request, identity, store)
                elif request.operation in ("release", "restore"):
                    record = self.records.get(request.snapshot_id)
                    if record is None:
                        raise ValueError("No retained CPU request for snapshot_id")
                    if request.operation == "release":
                        receipt = self._release(request, identity, store, record)
                    else:
                        receipt = self._restore(request, identity, store, record)
                else:
                    raise ValueError("Unknown snapshot operation")
                local.update(
                    receipt,
                    status="PASS",
                    identity=identity,
                    overlap_results_drained=drained_overlap_results,
                )
        except Exception as error:
            local.update(status="FAILED", error=str(error))
        local["operation_s"] = time.perf_counter() - started
        results = [local]
        if get_parallel().tp_size > 1:
            results = [None] * get_parallel().tp_size
            torch.distributed.all_gather_object(
                results, local, group=self.scheduler.tp_cpu_group
            )
        return TokenKVSnapshotReqOutput(
            success=all(result["status"] == "PASS" for result in results),
            results=results,
        )
