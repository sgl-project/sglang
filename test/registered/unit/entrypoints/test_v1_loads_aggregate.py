"""Unit tests for /v1/loads load snapshot response behavior."""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace

import msgspec.msgpack

from sglang.srt.entrypoints.v1_loads import get_loads
from sglang.srt.managers.load_snapshot import (
    HEADER_STRUCT,
    MAGIC,
    SLOT_LEN_STRUCT,
    SLOT_SIZE,
    VERSION,
    DisaggregationMetrics,
    LoadSnapshot,
    QueueMetrics,
    ShmLoadSnapshotReader,
    ShmLoadSnapshotWriter,
    slot_offset,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()


register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _temp_path() -> str:
    fd, path = tempfile.mkstemp()
    os.close(fd)
    os.unlink(path)
    return path


class _FakeTokenizerManager(TokenizerControlMixin):
    def __init__(self, reader, dp_size: int):
        self.load_snapshot_reader = reader
        self.elastic_worker_count = dp_size
        self.server_args = SimpleNamespace(
            dp_size=dp_size,
            enable_dp_attention=False,
            nnodes=1,
        )

    def auto_create_handle_loop(self):
        pass


class _FakeHttpTokenizerManager:
    metrics_collector = None

    def __init__(self, loads):
        self.loads = loads

    async def get_loads(self, include=None, dp_rank=None):
        results = []
        for load in self.loads:
            if dp_rank is not None and load.dp_rank != dp_rank:
                continue
            results.append(load)
        return results


class TestLoadsResponse(CustomTestCase):
    def test_response_omits_server_side_aggregate_and_redundant_fields(self):
        manager = _FakeHttpTokenizerManager(
            [
                LoadSnapshot(
                    dp_rank=0,
                    num_running_reqs=3,
                    num_waiting_reqs=2,
                    num_total_tokens=256,
                )
            ]
        )

        response = asyncio.run(get_loads(tokenizer_manager=manager))

        self.assertNotIn("dp_rank_count", response)
        self.assertNotIn("aggregate", response)
        self.assertEqual(len(response["loads"]), 1)
        self.assertNotIn("num_total_reqs", response["loads"][0])
        self.assertEqual(response["loads"][0]["num_running_reqs"], 3)
        self.assertEqual(response["loads"][0]["num_waiting_reqs"], 2)


class TestGetLoads(CustomTestCase):
    def test_load_snapshot_wire_format_is_msgpack_slots(self):
        path = _temp_path()
        writer = ShmLoadSnapshotWriter(path, dp_size=2, dp_rank=1)
        try:
            writer.write(
                LoadSnapshot(
                    dp_rank=1,
                    num_running_reqs=3,
                    num_waiting_reqs=2,
                    token_usage=0.25,
                )
            )

            with open(path, "rb") as f:
                data = f.read()

            self.assertEqual(len(data), HEADER_STRUCT.size + 2 * SLOT_SIZE)
            magic, version, dp_size, slot_size = HEADER_STRUCT.unpack_from(data, 0)
            self.assertEqual(magic, MAGIC)
            self.assertEqual(version, VERSION)
            self.assertEqual(dp_size, 2)
            self.assertEqual(slot_size, SLOT_SIZE)

            offset = slot_offset(1, slot_size)
            (payload_len,) = SLOT_LEN_STRUCT.unpack_from(data, offset)
            payload_start = offset + SLOT_LEN_STRUCT.size
            payload = data[payload_start : payload_start + payload_len]
            decoded = msgspec.msgpack.decode(payload)

            self.assertEqual(decoded["dp_rank"], 1)
            self.assertEqual(decoded["num_running_reqs"], 3)
            self.assertEqual(decoded["num_waiting_reqs"], 2)
            self.assertEqual(decoded["token_usage"], 0.25)
        finally:
            writer.close()
            if os.path.exists(path):
                os.unlink(path)

    def test_reads_snapshot_and_filters_sections(self):
        path = _temp_path()
        writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
        reader = ShmLoadSnapshotReader(path, dp_size=1)
        try:
            initial_load = reader.read(0)
            self.assertIsNotNone(initial_load)
            self.assertEqual(initial_load.num_total_tokens, 0)

            writer.write(
                LoadSnapshot(
                    dp_rank=0,
                    timestamp=1.25,
                    num_running_reqs=3,
                    num_waiting_reqs=2,
                    num_used_tokens=128,
                    num_total_tokens=256,
                    max_total_num_tokens=4096,
                    token_usage=0.125,
                    gen_throughput=99.5,
                    cache_hit_rate=0.75,
                    utilization=0.5,
                    max_running_requests=128,
                    disaggregation=DisaggregationMetrics(
                        mode="decode", decode_transfer_queue_reqs=4
                    ),
                    queues=QueueMetrics(waiting=2, grammar=1, paused=0, retracted=3),
                )
            )

            manager = _FakeTokenizerManager(reader, dp_size=1)
            loads = asyncio.run(manager.get_loads(include=["core"], dp_rank=0))

            self.assertEqual(len(loads), 1)
            self.assertEqual(loads[0].num_total_tokens, 256)

            d = loads[0].to_dict({"core"})
            self.assertNotIn("disaggregation", d)
            self.assertNotIn("queues", d)

            loads_all = asyncio.run(manager.get_loads(include=["all"], dp_rank=0))
            d_all = loads_all[0].to_dict()
            self.assertIn("disaggregation", d_all)
            self.assertIn("queues", d_all)
        finally:
            reader.close()
            writer.close()
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
