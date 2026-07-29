from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.rust_pd import (
    PdRequestSidecar,
    RustPdSchedulerAdapter,
    StablePdRegionTable,
)


class FakeTensor:
    def __init__(
        self,
        address,
        shape=(128, 8, 128),
        stride=(1024, 128, 1),
        device="cuda:5",
        dtype="torch.bfloat16",
        pinned=False,
    ):
        self._address = address
        self.shape = shape
        self._stride = stride
        self.device = SimpleNamespace(
            type=device.split(":")[0], index=int(device.split(":")[1])
        )
        self.dtype = dtype
        self._pinned = pinned

    def data_ptr(self):
        return self._address

    def stride(self):
        return self._stride

    def element_size(self):
        return 2 if self.dtype == "torch.bfloat16" else 1

    def numel(self):
        count = 1
        for value in self.shape:
            count *= value
        return count

    def is_contiguous(self):
        return True

    def is_pinned(self):
        return self._pinned


class FakePool:
    page_size = 64

    def __init__(self, device="cuda:5"):
        self.pd_generation = 3
        self.k_buffer = [
            FakeTensor(0x100000 + index * 0x10000, device=device) for index in range(28)
        ]
        self.v_buffer = [
            FakeTensor(0x300000 + index * 0x10000, device=device) for index in range(28)
        ]

    def _pd_registerable_tensors(self):
        return self.k_buffer + self.v_buffer


class Item:
    def __init__(
        self,
        handle,
        ok=True,
        pd_reason="PD_SUCCESS",
        retryable=False,
        terminal_generation=7,
    ):
        self.handle = handle
        self.terminal_generation = terminal_generation
        self.ok = ok
        self.pd_reason = pd_reason
        self.retryable = retryable


class Poll:
    def __init__(self, handle, first_token_id, first_token_consumed):
        self.handle = handle
        self.ok = True
        self.status = 4
        self.pd_reason = "PD_SUCCESS"
        self.retryable = False
        self.transfer_bytes = 64
        self.transfer_latency_ms = 2
        self.terminal_generation = 7
        self.first_token_id = first_token_id
        self.first_token_consumed = first_token_consumed


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.poll_count = 0

    def sender_create_many(self, epochs, rooms, attempts, digests):
        self.calls.append(("sender_create_many", list(rooms)))
        return [Item(100 + index) for index in range(len(rooms))]

    def sender_init_many(self, handles):
        self.calls.append(("sender_init_many", list(handles)))
        return [Item(handle) for handle in handles]

    def sender_send_chunks(
        self,
        handles,
        transfer_bytes,
        source_pages,
        first_token_ids,
        valid_token_counts,
        cuda_stream,
    ):
        self.calls.append(
            ("sender_send_chunks", list(handles), list(transfer_bytes), cuda_stream)
        )
        return [Item(handle) for handle in handles]

    def receiver_create_many(self, rooms, attempts, digests):
        self.calls.append(("receiver_create_many", list(rooms)))
        return [Item(200 + index) for index in range(len(rooms))]

    def receiver_prepare_many(self, handles, destination_pages, valid_token_counts):
        self.calls.append(("receiver_prepare_many", list(handles)))
        return [Item(handle) for handle in handles]

    def poll_many(self, handles):
        self.calls.append(("poll_many", list(handles)))
        self.poll_count += 1
        return [
            Poll(handle, 42 if self.poll_count == 1 else None, self.poll_count != 1)
            for handle in handles
        ]

    def abort_many(self, handles, reason):
        self.calls.append(("abort_many", list(handles), reason))
        return [Item(handle) for handle in handles]

    def clear_many(self, handles):
        self.calls.append(("clear_many", list(handles)))
        return [Item(handle) for handle in handles]


def sidecar(index, room):
    return PdRequestSidecar(
        version=1,
        bootstrap_host="prefill.internal",
        bootstrap_port=8998,
        bootstrap_room=room,
        attempt_id=f"44444444-4444-4{index:03x}-8444-44444444444{index}",
        batch_index=index,
        request_digest=f"{index + 1:02x}" * 32,
        decode_process_epoch="22222222-2222-4222-8222-222222222222",
    )


def request(index, room):
    return SimpleNamespace(
        rid=f"rid-{index}",
        pd_sidecar=sidecar(index, room),
        output_ids=[],
        origin_input_ids=[1],
        sampling_params=SimpleNamespace(max_new_tokens=1),
        pd_source_pages=[index],
        pd_destination_pages=[index],
    )


def region_table(device="cuda:5"):
    pool = FakePool(device)
    aux = FakeTensor(
        0x500000,
        shape=(32, 64),
        stride=(64, 1),
        device="cpu:0",
        dtype="torch.uint8",
        pinned=True,
    )
    completion = FakeTensor(
        0x600000,
        shape=(32, 192),
        stride=(192, 1),
        device="cpu:0",
        dtype="torch.uint8",
        pinned=True,
    )
    return pool, StablePdRegionTable.capture(pool, aux, completion, generation=3)


def test_region_table_is_exact_and_detects_pool_address_or_generation_change():
    pool, table = region_table()
    assert [descriptor.region_id for descriptor in table.descriptors] == list(range(58))
    assert len(table.tensor_owners) == 58
    table.verify(pool)

    pool.k_buffer[0]._address += 64
    with pytest.raises(RuntimeError, match="PD_STALE_EPOCH"):
        table.verify(pool)

    pool, table = region_table()
    pool.pd_generation += 1
    with pytest.raises(RuntimeError, match="PD_STALE_EPOCH"):
        table.verify(pool)


def test_prefill_hot_path_uses_one_batch_call_per_scheduler_phase():
    pool, table = region_table("cuda:4")
    transport = FakeTransport()
    adapter = RustPdSchedulerAdapter(transport, "prefill", table, pool)
    requests = [request(0, 0), request(1, 2**63 - 1)]

    adapter.create_many(requests)
    adapter.sender_init_many(requests)
    adapter.sender_send_chunks(requests, [64, 128])

    assert transport.calls == [
        ("sender_create_many", [0, 2**63 - 1]),
        ("sender_init_many", [100, 101]),
        ("sender_send_chunks", [100, 101], [64, 128], 0),
    ]


def test_decode_first_token_and_terminal_generation_are_committed_once():
    pool, table = region_table()
    transport = FakeTransport()
    adapter = RustPdSchedulerAdapter(transport, "decode", table, pool)
    requests = [request(0, 0)]

    adapter.create_many(requests)
    adapter.receiver_prepare_many(requests)
    first = adapter.poll_many(requests)
    second = adapter.poll_many(requests)

    assert first[0].first_token_id == 42
    assert second[0].first_token_id is None
    assert requests[0].output_ids == [42]

    adapter.clear_many(requests)
    assert adapter.active_count == 0
