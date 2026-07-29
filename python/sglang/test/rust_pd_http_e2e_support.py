from __future__ import annotations

import json
from array import array
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sglang.srt.disaggregation.rust_pd import RustPdSchedulerAdapter
from sglang.srt.disaggregation.rust_pd_buffers import StablePdRegionTable
from sglang.srt.managers.rust_server import RustServer

API_KEY = "pd-http-e2e-secret"
MODEL_PATH = Path("/mnt/models/Qwen/Qwen3-0.6B")
PROCESS_EPOCH = {
    "prefill": "11111111-1111-4111-8111-111111111111",
    "decode": "22222222-2222-4222-8222-222222222222",
}
REGISTRATION_EPOCH = {
    "prefill": "33333333-3333-4333-8333-333333333333",
    "decode": "44444444-4444-4444-8444-444444444444",
}
RESOURCE_FIELDS = (
    "active_rooms",
    "active_handles",
    "result_slots",
    "pending_prepares",
    "wire_plans",
    "native_leases",
    "source_kv_pages",
    "destination_kv_pages",
    "aux_slots",
    "completion_slots",
    "request_slots",
    "in_flight_transfers",
    "native_batches",
    "pending_bytes",
    "quarantined_rooms",
)
CLEAN_RESOURCES = (0,) * (3 + len(RESOURCE_FIELDS))


class _FakeTensor:
    def __init__(
        self,
        address: int,
        *,
        shape: tuple[int, ...] = (128, 8, 128),
        stride: tuple[int, ...] = (1024, 128, 1),
        device: str,
        dtype: str = "torch.bfloat16",
        pinned: bool = False,
    ) -> None:
        self._address = address
        self.shape = shape
        self._stride = stride
        device_type, device_index = device.split(":")
        self.device = SimpleNamespace(type=device_type, index=int(device_index))
        self.dtype = dtype
        self._pinned = pinned

    def data_ptr(self) -> int:
        return self._address

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def element_size(self) -> int:
        return 2 if self.dtype == "torch.bfloat16" else 1

    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result

    def is_contiguous(self) -> bool:
        return True

    def is_pinned(self) -> bool:
        return self._pinned


class _FakePool:
    page_size = 64
    pd_generation = 1

    def __init__(self, gpu: int) -> None:
        device = f"cuda:{gpu}"
        self.k_buffer = [
            _FakeTensor(0x100000 + index * 0x20000, device=device)
            for index in range(28)
        ]
        self.v_buffer = [
            _FakeTensor(0x500000 + index * 0x20000, device=device)
            for index in range(28)
        ]

    def _pd_registerable_tensors(self) -> list[_FakeTensor]:
        return [*self.k_buffer, *self.v_buffer]


def region_table(role: str) -> tuple[_FakePool, StablePdRegionTable]:
    pool = _FakePool(4 if role == "prefill" else 5)
    aux = _FakeTensor(
        0x900000,
        shape=(32, 64),
        stride=(64, 1),
        device="cpu:0",
        dtype="torch.uint8",
        pinned=True,
    )
    completion = _FakeTensor(
        0xA00000,
        shape=(32, 192),
        stride=(192, 1),
        device="cpu:0",
        dtype="torch.uint8",
        pinned=True,
    )
    return pool, StablePdRegionTable.capture(pool, aux, completion, generation=1)


def transport_config(
    role: str, control_port: int, data_port: int, psk_file: str
) -> str:
    return json.dumps(
        {
            "role": role,
            "process_epoch": PROCESS_EPOCH[role],
            "registration_epoch": REGISTRATION_EPOCH[role],
            "model_manifest_digest": "11" * 32,
            "tokenizer_manifest_digest": "22" * 32,
            "layout_fingerprint": "33" * 32,
            "native_abi_digest": "44" * 32,
            "mooncake_host": "127.0.0.1",
            "mooncake_ports": [data_port],
            "control_host": "127.0.0.1",
            "control_port": control_port,
            "pd_control_psk_file": psk_file,
            "mock_data_plane": True,
        },
        separators=(",", ":"),
    )


def server_args(role: str, http_port: int, control_port: int, psk_file: str) -> str:
    return json.dumps(
        {
            "model_path": str(MODEL_PATH),
            "tokenizer_path": str(MODEL_PATH),
            "served_model_name": "qwen3-0.6b",
            "host": "127.0.0.1",
            "port": http_port,
            "tokenizer_worker_num": 1,
            "detokenizer_worker_num": 1,
            "model_config": {"context_len": 4096, "vocab_size": 151936},
            "api_key": API_KEY,
            "disaggregation_mode": role,
            "disaggregation_bootstrap_port": control_port,
            "pd_control_psk_file": psk_file,
        },
        separators=(",", ":"),
    )


def prepare_request(request: Any) -> None:
    request.origin_input_ids = list(request.input_ids)
    request.output_ids = []
    page_count = (len(request.origin_input_ids) + 63) // 64
    offset = int(request.pd_sidecar["batch_index"]) * 64
    pages = list(range(offset, offset + page_count))
    request.pd_source_pages = pages
    request.pd_destination_pages = pages


def scheduler_request(request: Any) -> SimpleNamespace:
    return SimpleNamespace(
        rid=request.rid,
        input_ids=list(request.input_ids),
        sampling_params=request.sampling_params,
        pd_sidecar=request.pd_sidecar,
        origin_input_ids=list(request.input_ids),
        output_ids=[],
    )


def output_payload(
    requests: list[Any], finish_reasons: list[dict[str, Any]]
) -> SimpleNamespace:
    return SimpleNamespace(
        rids=[request.rid for request in requests],
        finished_reasons=finish_reasons,
        output_ids=[array("i", request.output_ids) for request in requests],
        prompt_tokens=[len(request.origin_input_ids) for request in requests],
        output_token_logprobs_val=None,
        input_token_logprobs_val=None,
        output_top_logprobs_val=None,
        input_top_logprobs_val=None,
        output_token_ids_logprobs_val=None,
        input_token_ids_logprobs_val=None,
        output_hidden_states=None,
    )


def typed_failure(
    rust_server: RustServer, request: Any, reason: str, status: int
) -> None:
    rust_server.push_generation(
        output_payload(
            [request],
            [
                {
                    "type": "abort",
                    "message": "classification must use pd_reason",
                    "status_code": status,
                    "err_type": None,
                    "pd_reason": reason,
                }
            ],
        )
    )


def normal_batch(
    role: str,
    adapter: RustPdSchedulerAdapter,
    rust_server: RustServer,
    requests: list[Any],
    *,
    publish: bool = True,
) -> list[int]:
    for request in requests:
        prepare_request(request)
        adapter.enqueue(request)

    ready = adapter.flush_pending()
    if role == "prefill":
        if ready != requests:
            raise AssertionError("prefill batch did not become compute-ready")
        for request in requests:
            sentinel = request.origin_input_ids[0]
            first_token = (
                None
                if request.sampling_params.max_new_tokens == 0
                else 151645 if sentinel == 9010 else 77 if sentinel == 9011 else 42
            )
            if first_token is not None:
                request.output_ids.append(first_token)
        transfer_bytes = [
            len(request.origin_input_ids) * 56 * 2_048 for request in requests
        ]
        adapter.sender_send_chunks(requests, transfer_bytes)
        adapter.add_prefill_inflight(requests)
        completed, failures = adapter.poll_inflight()
    else:
        if ready:
            raise AssertionError("decode must wait for destination visibility")
        completed, failures = adapter.poll_inflight()
        for request in completed:
            terminal_on_first = request.origin_input_ids[0] in (9010, 9011)
            if (
                not terminal_on_first
                and len(request.output_ids) < request.sampling_params.max_new_tokens
            ):
                request.output_ids.append(43)

    if failures or completed != requests:
        raise AssertionError(
            f"{role} terminal mismatch: completed={len(completed)} failures={failures}"
        )
    finish_reasons = []
    for request in requests:
        sentinel = request.origin_input_ids[0]
        if sentinel in (9010, 9011):
            finish_reasons.append({"type": "stop", "matched": request.output_ids[-1]})
        else:
            finish_reasons.append({"type": "length", "length": len(request.output_ids)})
    if publish:
        rust_server.push_generation(output_payload(requests, finish_reasons))
        adapter.clear_terminal(requests)
    return [int(request.pd_sidecar.bootstrap_room) for request in requests]


def resource_counts(adapter: RustPdSchedulerAdapter, transport: Any) -> tuple[int, ...]:
    snapshot = transport.resource_snapshot()
    return (
        adapter.active_count,
        adapter.pending_count,
        adapter.inflight_count,
        *(int(getattr(snapshot, field)) for field in RESOURCE_FIELDS),
    )
