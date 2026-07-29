"""Thin Scheduler adapter for the Rust-owned PD transport.

This module deliberately contains no peer protocol, Room FSM, deadline, or
native-transfer state machine. It snapshots stable Scheduler buffers and maps
request ids to opaque Rust handles; every state transition is a coarse-grained
batch call into ``PdTransport``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from sglang.srt.disaggregation.rust_pd_buffers import (
    PD_AUX_BYTES,
    PD_AUX_SLOTS,
    PD_COMPLETION_BYTES,
    PD_COMPLETION_SLOTS,
    StablePdRegionTable,
)
from sglang.srt.disaggregation.rust_pd_contract import contract_digests

PD_BATCH_MAX = 8


@dataclass(frozen=True)
class PdRequestSidecar:
    version: int
    bootstrap_host: str
    bootstrap_port: int
    bootstrap_room: int
    attempt_id: str
    batch_index: int
    request_digest: str
    decode_process_epoch: str

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("PD_REQUEST_INVALID")
        _validate_uuid4(self.attempt_id)
        _validate_uuid4(self.decode_process_epoch)
        if not 0 <= self.batch_index < PD_BATCH_MAX:
            raise ValueError("PD_REQUEST_INVALID")
        if not 0 <= self.bootstrap_room <= 2**63 - 1:
            raise ValueError("PD_REQUEST_INVALID")
        if (
            not isinstance(self.bootstrap_host, str)
            or not self.bootstrap_host
            or not isinstance(self.bootstrap_port, int)
            or not 1 <= self.bootstrap_port <= 65535
        ):
            raise ValueError("PD_REQUEST_INVALID")
        _validate_digest(self.request_digest)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PdRequestSidecar:
        fields = {
            "version",
            "bootstrap_host",
            "bootstrap_port",
            "bootstrap_room",
            "attempt_id",
            "batch_index",
            "request_digest",
            "decode_process_epoch",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("PD_REQUEST_INVALID")
        try:
            return cls(**{field: value[field] for field in fields})
        except (TypeError, KeyError) as error:
            raise ValueError("PD_REQUEST_INVALID") from error


@dataclass
class _RequestBinding:
    request: Any
    handle: int
    terminal_generation: int
    first_token_committed: bool = False


class RustPdSchedulerAdapter:
    """Opaque request/handle mapping and batch-only PyO3 call site."""

    def __init__(
        self,
        transport: Any,
        role: str,
        region_table: StablePdRegionTable,
        pool: Any,
        scheduler: Any = None,
    ) -> None:
        if role not in ("prefill", "decode"):
            raise ValueError("PD_UNSUPPORTED")
        self.transport = transport
        self.role = role
        self.region_table = region_table
        self.pool = pool
        self.scheduler = scheduler
        self._bindings: dict[str, _RequestBinding] = {}
        self._pending: list[Any] = []
        self._inflight: list[Any] = []

    @classmethod
    def from_scheduler(cls, scheduler: Any) -> RustPdSchedulerAdapter:
        """Build the one Rust-PD owner after the Scheduler pools are stable."""
        _validate_scheduler_support(scheduler)
        pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        generation = int(getattr(pool, "pd_generation", 0)) + 1
        pool.pd_generation = generation

        tensor_factory = getattr(scheduler, "_rust_pd_pinned_tensor_factory", None)
        if tensor_factory is None:
            import torch

            def tensor_factory(shape: tuple[int, int]) -> Any:
                return torch.empty(
                    shape,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )

        aux_tensor = tensor_factory((PD_AUX_SLOTS, PD_AUX_BYTES))
        completion_tensor = tensor_factory((PD_COMPLETION_SLOTS, PD_COMPLETION_BYTES))
        region_table = StablePdRegionTable.capture(
            pool,
            aux_tensor,
            completion_tensor,
            generation=generation,
        )

        config = _transport_config(scheduler, region_table)
        transport_factory = getattr(scheduler, "_rust_pd_transport_factory", None)
        if transport_factory is None:
            from sglang.srt.server._core import PdTransport

            transport_factory = PdTransport
        transport = transport_factory(json.dumps(config, separators=(",", ":")))
        transport.start()
        adapter = cls(
            transport,
            str(scheduler.server_args.disaggregation_mode),
            region_table,
            pool,
            scheduler,
        )
        # Keep the pinned owners and readiness handle alongside the adapter.
        adapter.aux_tensor = aux_tensor
        adapter.completion_tensor = completion_tensor
        adapter.readiness_handle = transport.readiness()
        return adapter

    @property
    def active_count(self) -> int:
        return len(self._bindings)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    def enqueue(self, request: Any, *, is_retracted: bool = False) -> None:
        if is_retracted:
            raise RuntimeError("PD_UNSUPPORTED")
        self._sidecar(request)
        if (
            request.rid in self._bindings
            or any(item.rid == request.rid for item in self._pending)
            or any(item.rid == request.rid for item in self._inflight)
        ):
            raise RuntimeError("PD_STALE_EPOCH")
        self._pending.append(request)

    def flush_pending(self) -> list[Any]:
        """Create queued Rooms in Scheduler-sized batches.

        Prefill requests become compute-ready after sender initialization.
        Decode requests stay in the adapter until destination transfer reaches
        a validated terminal result.
        """
        compute_ready: list[Any] = []
        while self._pending:
            requests = self._pending[:PD_BATCH_MAX]
            self.create_many(requests)
            try:
                if self.role == "prefill":
                    self.sender_init_many(requests)
                    compute_ready.extend(requests)
                else:
                    self._prepare_decode_batch(requests)
                    self.receiver_prepare_many(requests)
                    self._inflight.extend(requests)
            except Exception:
                self.abort_many(requests)
                raise
            finally:
                del self._pending[: len(requests)]
        return compute_ready

    def add_prefill_inflight(self, requests: Sequence[Any]) -> None:
        self._require_role("prefill")
        self._bindings_for(requests)
        known = {request.rid for request in self._inflight}
        if any(request.rid in known for request in requests):
            raise RuntimeError("PD_STALE_EPOCH")
        self._inflight.extend(requests)

    def poll_inflight(self) -> tuple[list[Any], list[tuple[Any, Any]]]:
        """Return successful requests and typed failures, preserving order."""
        if not self._inflight:
            return [], []
        requests = list(self._inflight)
        results = self.poll_many(requests)
        completed: list[Any] = []
        failed: list[tuple[Any, Any]] = []
        remaining: list[Any] = []
        for request, result in zip(requests, results, strict=True):
            if result.status == 4:
                completed.append(request)
            elif result.status == 0:
                failed.append((request, result))
            else:
                remaining.append(request)
        self._inflight = remaining
        return completed, failed

    def clear_terminal(self, requests: Sequence[Any]) -> None:
        if requests:
            self.clear_many(requests)

    def abort_matching(
        self, rid: str, *, abort_all: bool = False
    ) -> tuple[list[Any], list[Any]]:
        """Abort queued and Rust-owned requests without touching legacy owners."""

        def matches(request: Any) -> bool:
            return abort_all or request.rid.startswith(rid)

        queued = [request for request in self._pending if matches(request)]
        self._pending = [request for request in self._pending if not matches(request)]

        active_by_rid = {
            binding.request.rid: binding.request
            for binding in self._bindings.values()
            if matches(binding.request)
        }
        active = list(active_by_rid.values())
        for offset in range(0, len(active), PD_BATCH_MAX):
            self.abort_many(active[offset : offset + PD_BATCH_MAX])
        return queued, active

    def _prepare_decode_batch(self, requests: Sequence[Any]) -> None:
        if self.scheduler is None:
            return
        scheduler = self.scheduler
        allocator = scheduler.token_to_kv_pool_allocator
        page_size = allocator.page_size
        if page_size != 64:
            raise RuntimeError("PD_UNSUPPORTED")

        # Frozen Rust-PD mode has no radix, SWA, HiCache, chunking, or
        # rebootstrap. Allocate the prompt pages directly with the existing
        # Scheduler allocators, then expose only logical page ids to Rust.
        import torch

        from sglang.srt.disaggregation.decode import alloc_for_decode_prealloc

        allocated: list[Any] = []
        try:
            for request in requests:
                fill_len = len(request.origin_input_ids)
                if not 1 <= fill_len <= 4096:
                    raise ValueError("PD_REQUEST_INVALID")
                if scheduler.req_to_token_pool.alloc([request]) is None:
                    raise RuntimeError("PD_CAPACITY_EXHAUSTED")
                kv_loc = alloc_for_decode_prealloc(
                    allocator,
                    req=request,
                    fill_len=fill_len,
                    delta_len=fill_len,
                    prefix_len=0,
                    total_prefix_len=0,
                    prefix_indices=None,
                    uses_swa_tail=False,
                    swa_tail_len=fill_len,
                )
                if kv_loc is None:
                    raise RuntimeError("PD_CAPACITY_EXHAUSTED")
                scheduler.req_to_token_pool.write(
                    (request.req_pool_idx, slice(0, len(kv_loc))), kv_loc
                )
                request.kv_committed_len = fill_len
                request.full_untruncated_fill_ids = (
                    request.origin_input_ids + request.output_ids
                )
                request.prefix_indices = torch.empty((0,), dtype=torch.int64)
                request.set_extend_range(0, fill_len)
                page_ids = (
                    kv_loc[::page_size]
                    .to(device="cpu", dtype=torch.int64)
                    .div(page_size, rounding_mode="floor")
                    .tolist()
                )
                if not page_ids:
                    raise RuntimeError("PD_PROTOCOL_MISMATCH")
                request.pd_destination_pages = tuple(int(page) for page in page_ids)
                allocated.append(request)
        except Exception:
            # Allocation rollback stays with the Scheduler allocator. Rust has
            # not prepared or submitted the Room yet at this point.
            from sglang.srt.mem_cache.common import release_kv_cache

            for request in allocated:
                release_kv_cache(request, scheduler.tree_cache, is_insert=False)
            raise

    def create_many(self, requests: Sequence[Any]) -> None:
        self._validate_requests(requests, allow_existing=False)
        sidecars = [self._sidecar(request) for request in requests]
        if self.role == "prefill":
            results = self.transport.sender_create_many(
                [sidecar.decode_process_epoch for sidecar in sidecars],
                [sidecar.bootstrap_room for sidecar in sidecars],
                [sidecar.attempt_id for sidecar in sidecars],
                [sidecar.request_digest for sidecar in sidecars],
            )
        else:
            results = self.transport.receiver_create_many(
                [sidecar.bootstrap_room for sidecar in sidecars],
                [sidecar.attempt_id for sidecar in sidecars],
                [sidecar.request_digest for sidecar in sidecars],
            )
        self._store_created(requests, results)

    def sender_init_many(self, requests: Sequence[Any]) -> None:
        self._require_role("prefill")
        self._batch_transition(requests, self.transport.sender_init_many)

    def sender_send_chunks(
        self,
        requests: Sequence[Any],
        transfer_bytes: Sequence[int],
    ) -> None:
        self._require_role("prefill")
        bindings = self._bindings_for(requests)
        if len(transfer_bytes) != len(bindings) or any(
            not isinstance(value, int) or value <= 0 for value in transfer_bytes
        ):
            raise ValueError("PD_REQUEST_INVALID")
        source_pages = [self._source_pages(request) for request in requests]
        first_token_ids = [
            (
                int(request.output_ids[-1])
                if request.output_ids and request.sampling_params.max_new_tokens != 0
                else None
            )
            for request in requests
        ]
        valid_token_counts = [len(request.origin_input_ids) for request in requests]
        cuda_stream = 0
        if self.scheduler is not None:
            stream_hook = getattr(
                self.scheduler, "_rust_pd_compute_stream_handle", None
            )
            if stream_hook is not None:
                cuda_stream = int(stream_hook())
            else:
                import torch

                cuda_stream = int(
                    torch.cuda.current_stream(
                        device=f"cuda:{self.scheduler.ps.gpu_id}"
                    ).cuda_stream
                )
        results = self.transport.sender_send_chunks(
            [binding.handle for binding in bindings],
            list(transfer_bytes),
            source_pages,
            first_token_ids,
            valid_token_counts,
            cuda_stream,
        )
        self._require_success(results, [binding.handle for binding in bindings])

    def receiver_prepare_many(self, requests: Sequence[Any]) -> None:
        self._require_role("decode")
        bindings = self._bindings_for(requests)
        handles = [binding.handle for binding in bindings]
        results = self.transport.receiver_prepare_many(
            handles,
            [list(request.pd_destination_pages) for request in requests],
            [len(request.origin_input_ids) for request in requests],
        )
        self._require_success(results, handles)

    def poll_many(self, requests: Sequence[Any]) -> list[Any]:
        bindings = self._bindings_for(requests)
        results = self.transport.poll_many([binding.handle for binding in bindings])
        if len(results) != len(bindings):
            raise RuntimeError("PD_PROTOCOL_MISMATCH")
        for binding, result in zip(bindings, results, strict=True):
            if result.handle != binding.handle:
                raise RuntimeError("PD_STALE_EPOCH")
            if not result.ok:
                continue
            if result.terminal_generation != binding.terminal_generation:
                raise RuntimeError("PD_STALE_EPOCH")
            if result.first_token_id is not None:
                if result.first_token_consumed or binding.first_token_committed:
                    raise RuntimeError("PD_PROTOCOL_MISMATCH")
                binding.request.output_ids.append(result.first_token_id)
                binding.first_token_committed = True
        return list(results)

    def abort_many(
        self, requests: Sequence[Any], reason: str = "PD_ABORTED"
    ) -> list[Any]:
        bindings = self._bindings_for(requests)
        results = self.transport.abort_many(
            [binding.handle for binding in bindings], reason
        )
        self._require_success(results, [binding.handle for binding in bindings])
        return list(results)

    def clear_many(self, requests: Sequence[Any]) -> None:
        bindings = self._bindings_for(requests)
        results = self.transport.clear_many([binding.handle for binding in bindings])
        self._require_success(results, [binding.handle for binding in bindings])
        for request in requests:
            del self._bindings[request.rid]

    def _batch_transition(self, requests: Sequence[Any], method: Any) -> None:
        bindings = self._bindings_for(requests)
        handles = [binding.handle for binding in bindings]
        self._require_success(method(handles), handles)

    def _store_created(self, requests: Sequence[Any], results: Sequence[Any]) -> None:
        if len(results) != len(requests):
            raise RuntimeError("PD_PROTOCOL_MISMATCH")
        created: list[int] = []
        try:
            for request, result in zip(requests, results, strict=True):
                if not result.ok or result.handle == 0:
                    raise RuntimeError(result.pd_reason)
                terminal_generation = getattr(result, "terminal_generation", 0)
                if not 1 <= terminal_generation <= 2**64 - 1:
                    raise RuntimeError("PD_PROTOCOL_MISMATCH")
                self._bindings[request.rid] = _RequestBinding(
                    request=request,
                    handle=result.handle,
                    terminal_generation=terminal_generation,
                )
                created.append(result.handle)
        except Exception:
            if created:
                self.transport.abort_many(created, "PD_ABORTED")
            for request in requests:
                self._bindings.pop(request.rid, None)
            raise

    def _bindings_for(self, requests: Sequence[Any]) -> list[_RequestBinding]:
        self._validate_requests(requests, allow_existing=True)
        try:
            return [self._bindings[request.rid] for request in requests]
        except KeyError as error:
            raise RuntimeError("PD_STALE_EPOCH") from error

    def _validate_requests(
        self, requests: Sequence[Any], *, allow_existing: bool
    ) -> None:
        self.region_table.verify(self.pool)
        if not 1 <= len(requests) <= PD_BATCH_MAX:
            raise ValueError("PD_REQUEST_INVALID")
        rids = [request.rid for request in requests]
        if len(set(rids)) != len(rids):
            raise ValueError("PD_REQUEST_INVALID")
        for request in requests:
            self._sidecar(request)
            if not allow_existing and request.rid in self._bindings:
                raise RuntimeError("PD_STALE_EPOCH")

    @staticmethod
    def _sidecar(request: Any) -> PdRequestSidecar:
        sidecar = getattr(request, "pd_sidecar", None)
        if isinstance(sidecar, Mapping):
            sidecar = PdRequestSidecar.from_mapping(sidecar)
            request.pd_sidecar = sidecar
        if not isinstance(sidecar, PdRequestSidecar):
            raise ValueError("PD_REQUEST_INVALID")
        return sidecar

    @staticmethod
    def _require_success(results: Sequence[Any], handles: Sequence[int]) -> None:
        if len(results) != len(handles):
            raise RuntimeError("PD_PROTOCOL_MISMATCH")
        for handle, result in zip(handles, results, strict=True):
            if result.handle != handle:
                raise RuntimeError("PD_STALE_EPOCH")
            if not result.ok:
                raise RuntimeError(result.pd_reason)

    def _require_role(self, role: str) -> None:
        if self.role != role:
            raise RuntimeError("PD_UNSUPPORTED")

    def _source_pages(self, request: Any) -> list[int]:
        if self.scheduler is None:
            pages = getattr(request, "pd_source_pages", None)
            if pages is None:
                raise RuntimeError("PD_PROTOCOL_MISMATCH")
            return [int(page) for page in pages]
        page_size = self.scheduler.token_to_kv_pool_allocator.page_size
        locations = self.scheduler.req_to_token_pool.req_to_token[
            request.req_pool_idx, : len(request.origin_input_ids)
        ]
        return [
            int(page)
            for page in locations[::page_size]
            .to(device="cpu")
            .div(page_size, rounding_mode="floor")
            .tolist()
        ]


def _validate_uuid4(value: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError("PD_REQUEST_INVALID") from error
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("PD_REQUEST_INVALID")


def _validate_digest(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value == "0" * 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("PD_REQUEST_INVALID")


def _validate_scheduler_support(scheduler: Any) -> None:
    args = scheduler.server_args
    model = scheduler.model_config
    role = str(args.disaggregation_mode)
    expected_gpu = 4 if role == "prefill" else 5 if role == "decode" else None
    unsupported = (
        expected_gpu is None
        or scheduler.ps.tp_size != 1
        or scheduler.ps.pp_size != 1
        or scheduler.ps.dp_size != 1
        or scheduler.ps.gpu_id != expected_gpu
        or scheduler.page_size != 64
        or str(model.dtype) not in ("torch.bfloat16", "bfloat16")
        or model.num_hidden_layers != 28
        or model.num_attention_heads != 16
        or model.num_key_value_heads != 8
        or model.head_dim != 128
        or bool(model.is_multimodal)
        or args.attention_backend != "flashinfer"
        or not args.disable_radix_cache
        or not args.disable_overlap_schedule
        or not args.disable_cuda_graph
        or args.chunked_prefill_size not in (None, -1)
        or bool(args.enable_hierarchical_cache)
        or bool(args.enable_lora)
        or args.speculative_algorithm is not None
        or getattr(args, "cpu_offload_gb", 0) != 0
        or getattr(args, "quantization", None) is not None
    )
    architectures = getattr(getattr(model, "hf_config", None), "architectures", [])
    if unsupported or architectures != ["Qwen3ForCausalLM"]:
        raise ValueError("PD_UNSUPPORTED")
    if not args.pd_control_psk_file:
        raise ValueError("PD_REQUEST_INVALID")


def _transport_config(
    scheduler: Any,
    region_table: StablePdRegionTable,
) -> dict[str, Any]:
    args = scheduler.server_args
    role = str(args.disaggregation_mode)
    mock_data_plane = bool(getattr(scheduler, "_rust_pd_mock_data_plane", False))
    model_digest, tokenizer_digest, layout_digest, native_digest = contract_digests(
        str(args.model_path),
        str(args.tokenizer_path or args.model_path),
    )
    return {
        "role": role,
        "model_manifest_digest": model_digest,
        "tokenizer_manifest_digest": tokenizer_digest,
        "layout_fingerprint": layout_digest,
        "native_abi_digest": native_digest,
        "mooncake_host": "127.0.0.1" if mock_data_plane else str(args.host),
        "mooncake_ports": [19000 if mock_data_plane or role == "prefill" else 19001],
        "control_host": str(args.host),
        "control_port": int(args.disaggregation_bootstrap_port),
        "pd_control_psk_file": str(args.pd_control_psk_file),
        "mock_data_plane": mock_data_plane,
        "regions": region_table.transport_values(),
    }
