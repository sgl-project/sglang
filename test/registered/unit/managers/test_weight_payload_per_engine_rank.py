"""Tensor weight payloads are one per engine rank, in pipeline-major order."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter
from sglang.srt.managers.tokenizer_control_mixin import (
    check_one_payload_per_engine_rank,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.utils import update_weights
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
            payloads = engine.serialize_tensors_per_rank(
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

    def test_weight_sync_producer_sends_one_payload_per_engine_rank(self):
        engine = Engine.__new__(Engine)
        engine.update_weights_from_tensor = AsyncMock(return_value=(True, "Success"))
        trainer_mesh = MagicMock()
        trainer_mesh["tp"].mesh.size.return_value = (2,)
        trainer_mesh["tp"].mesh.tolist.return_value = [0, 1]
        trainer_mesh["tp"].get_local_rank.return_value = 0

        def gather_both_trainer_ranks(obj, object_gather_list, dst, group):
            [(name, copies)] = obj
            rank_one = [
                MultiprocessingSerializer.serialize(torch.ones(2)) for _ in copies
            ]
            object_gather_list[:] = [obj, [(name, rank_one)]]

        with (
            patch("sglang.srt.utils.patch_torch.monkey_patch_torch_reductions"),
            patch("sglang.srt.weight_sync.utils.dist.broadcast_object_list"),
            patch(
                "sglang.srt.weight_sync.utils.dist.gather_object",
                side_effect=gather_both_trainer_ranks,
            ),
            get_parallel().override(tp_size=2, pp_size=2),
        ):
            asyncio.run(
                update_weights(engine, [("w", torch.zeros(2))], "tp", trainer_mesh)
            )
        request = engine.update_weights_from_tensor.call_args.args[0]
        self.assertEqual(len(request.serialized_named_tensors), 4)
        [(name, handles)] = MultiprocessingSerializer.deserialize(
            request.serialized_named_tensors[3]
        )
        self.assertEqual(name, "w")
        trainer_rank_per_engine_rank = [
            MultiprocessingSerializer.deserialize(v)[0].item() for v in handles.values
        ]
        self.assertEqual(trainer_rank_per_engine_rank, [0, 1, 0, 1])


class TestPayloadCountCheck(unittest.TestCase):
    def test_a_per_tp_rank_count_is_rejected_under_pipeline_parallelism(self):
        with get_parallel().override(tp_size=2, pp_size=2):
            with self.assertRaisesRegex(ValueError, r"4\b.*got 2"):
                check_one_payload_per_engine_rank([b""] * 2)


if __name__ == "__main__":
    unittest.main()
