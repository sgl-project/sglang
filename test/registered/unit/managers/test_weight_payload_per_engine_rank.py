"""Tensor weight payloads are one per engine rank, in pipeline-major order."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _payloads_tagged_by_rank(world_size):
    return [
        MultiprocessingSerializer.serialize(torch.full((2,), float(rank)))
        for rank in range(world_size)
    ]


class TestDeserializeOwnRank(unittest.TestCase):
    def setUp(self):
        self.worker = TpModelWorker.__new__(TpModelWorker)
        # The CUDA-IPC reduction patch is process-wide and breaks serializing
        # CPU tensors afterwards.
        patcher = patch("sglang.srt.managers.tp_worker.monkey_patch_torch_reductions")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_later_pipeline_stage_reads_its_world_rank_entry(self):
        with get_parallel().override(tp_size=2, pp_size=2, pp_rank=1, tp_rank=0):
            tensor = self.worker.deserialize_own_rank(_payloads_tagged_by_rank(4))
        self.assertTrue(torch.equal(tensor, torch.full((2,), 2.0)))

    def test_a_single_stage_reads_its_tp_rank_entry(self):
        with get_parallel().override(tp_size=2, pp_size=1, pp_rank=0, tp_rank=1):
            tensor = self.worker.deserialize_own_rank(_payloads_tagged_by_rank(2))
        self.assertTrue(torch.equal(tensor, torch.full((2,), 1.0)))


class TestSerializePerEngineRank(unittest.TestCase):
    def test_engine_serializes_one_payload_per_engine_rank(self):
        engine = Engine.__new__(Engine)
        with get_parallel().override(tp_size=2, pp_size=2):
            payloads = engine._serialize_tensors_per_rank(
                [("w", torch.zeros(2))], load_format=None
            )
        self.assertEqual(len(payloads), 4)

    def test_http_adapter_posts_one_payload_per_engine_rank(self):
        adapter = HttpServerEngineAdapter.__new__(HttpServerEngineAdapter)
        adapter.server_args = ServerArgs(model_path="dummy", tp_size=2, pp_size=2)
        with patch.object(adapter, "_make_request") as make_request:
            adapter.update_weights_from_tensor([("w", torch.zeros(2))])
        body = make_request.call_args.args[1]
        self.assertEqual(len(body["serialized_named_tensors"]), 4)


if __name__ == "__main__":
    unittest.main()
