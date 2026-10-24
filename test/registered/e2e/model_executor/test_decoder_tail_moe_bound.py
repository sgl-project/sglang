"""Decoder-tail counter ownership through real CUDA graph capture and replay."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
)
from sglang.srt.model_executor.runner.row_bucket_graph_runner import (
    RowBucketGraphRunner,
    _Graph,
)
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.srt.models.deepseek_v4_replay_tail import DecoderReplayTail
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def _batch(count_shape=()):
    empty = torch.empty(0, dtype=torch.int32, device="cuda")
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=empty,
        req_pool_indices=empty,
        seq_lens=empty,
        out_cache_loc=empty,
        seq_lens_sum=0,
    )
    batch.num_token_non_padded = torch.tensor(
        27, dtype=torch.int32, device="cuda"
    ).reshape(count_shape)
    batch.global_num_token_non_padded = torch.tensor(
        97, dtype=torch.int32, device="cuda"
    ).reshape(count_shape)
    batch.global_num_token_non_padded_cpu = 97
    batch.attn_tp_sequence_sharded = True
    return batch


def _counter_rows(rows, batch):
    bound = get_forward_context().moe_num_token_non_padded()
    return torch.stack(
        [
            bound.expand(rows),
            batch.num_token_non_padded.expand(rows),
            (torch.arange(rows, device="cuda") < bound).to(torch.int32),
        ],
        dim=1,
    )


class TestDecoderTailMoeBound(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def _record(self, captures):
        # Replace only distributed breakable recording; the runner still owns
        # batch copies, count allocations, replay fills and graph cache selection.
        def record(body, rows, local, batch, start, free_before):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                body()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = body()
            torch.cuda.current_stream().wait_stream(stream)
            captures.append((rows, local, batch.global_num_token_non_padded, batch))
            return _Graph(graph, local, outputs, (batch,))

        return record

    def test_tail_run_refreshes_global_and_local_counts_across_buckets(self):
        """A gathered MoE reads the tail's live GLOBAL prefix after shape changes,
        without capturing or modifying its parent batch's device scalar."""
        for rank in (0, 3):
            with (
                self.subTest(rank=rank),
                get_parallel().override(
                    attn_dp_size=1, attn_cp_size=1, attn_tp_size=4, attn_tp_rank=rank
                ),
                patch_communicator(
                    "is_moe_input_scattered_across_dp_ranks", return_value=False
                ),
                patch.object(
                    moe_utils,
                    "is_moe_input_scattered_across_dp_ranks",
                    return_value=False,
                ),
                patch_communicator("is_enable_moe_cp_allgather", return_value=False),
            ):
                parent = _batch((1,) if rank == 3 else ())
                local = parent.num_token_non_padded
                global_count = parent.global_num_token_non_padded
                tail = DecoderReplayTail(
                    model=SimpleNamespace(late_layer_vl_routing=False),
                    layer_ids=range(1),
                    row_step=8,
                    max_rows=16,
                )
                captures = []

                def layer_stack(state, *, forward_batch, **kwargs):
                    return (
                        _counter_rows(state.streams.shape[0], forward_batch),
                        None,
                        [],
                    )

                with (
                    forward_context(
                        ForwardContext(attn_backend=None), forward_batch=parent
                    ),
                    patch.object(tail, "_layer_stack", side_effect=layer_stack),
                    patch.object(
                        tail, "_break_context", side_effect=lambda batch: nullcontext()
                    ),
                    patch.object(
                        tail.runner, "_record", side_effect=self._record(captures)
                    ),
                ):
                    for live, local_by_rank in (
                        (10, [4, 4, 2, 0]),
                        (3, [2, 1, 0, 0]),
                        (7, [2, 2, 2, 1]),
                        (10, [4, 4, 2, 0]),
                    ):
                        value, pre, aux = tail.run(
                            state=HcState(torch.zeros(live, 1, 1, device="cuda")),
                            positions=torch.arange(live, device="cuda"),
                            input_ids=torch.arange(live, device="cuda"),
                            input_ids_global=None,
                            forward_batch=parent,
                            capture_dspark=False,
                        )
                        expected = torch.tensor(
                            [live, local_by_rank[rank], 1],
                            device="cuda",
                            dtype=torch.int32,
                        ).expand(live, 3)
                        torch.testing.assert_close(value, expected)
                        self.assertIsNone(pre)
                        self.assertEqual(aux, [])
                        self.assertIs(parent.num_token_non_padded, local)
                        self.assertIs(parent.global_num_token_non_padded, global_count)
                        self.assertEqual((local.item(), global_count.item()), (27, 97))
                        self.assertIs(
                            get_forward_context().moe_num_token_non_padded(),
                            global_count,
                        )
                self.assertEqual([entry[0] for entry in captures], [16, 8])
                for _, captured_local, captured_global, _ in captures:
                    self.assertNotEqual(captured_local.data_ptr(), local.data_ptr())
                    self.assertNotEqual(
                        captured_global.data_ptr(), global_count.data_ptr()
                    )
                    self.assertNotEqual(
                        captured_local.data_ptr(), captured_global.data_ptr()
                    )
                    self.assertEqual(captured_local.shape, local.shape)
                    self.assertEqual(captured_global.shape, global_count.shape)
                    self.assertEqual(
                        captured_local.untyped_storage().data_ptr(),
                        captured_global.untyped_storage().data_ptr(),
                    )
                    self.assertEqual(captured_local.untyped_storage().nbytes(), 8)

    def test_row_graph_refreshes_an_idle_global_prefix(self):
        """An idle replay must overwrite a previous live GLOBAL count even when
        it reuses a nonempty peer-selected bucket."""
        with (
            get_parallel().override(
                attn_dp_size=1, attn_cp_size=1, attn_tp_size=4, attn_tp_rank=3
            ),
            patch_communicator(
                "is_moe_input_scattered_across_dp_ranks", return_value=False
            ),
            patch.object(
                moe_utils, "is_moe_input_scattered_across_dp_ranks", return_value=False
            ),
            patch_communicator("is_enable_moe_cp_allgather", return_value=False),
        ):
            parent = _batch()
            runner = RowBucketGraphRunner(row_step=8, max_rows=8, label="test")
            captures = []
            with (
                forward_context(
                    ForwardContext(attn_backend=None), forward_batch=parent
                ),
                patch.object(runner, "_record", side_effect=self._record(captures)),
            ):
                for live, local in ((7, 1), (0, 0), (3, 0)):
                    (value,) = runner.run(
                        key=(),
                        live=[torch.zeros(live, 1, device="cuda")],
                        forward_batch=parent,
                        body=lambda inputs, batch: [
                            _counter_rows(inputs[0].shape[0], batch)
                        ],
                        context=lambda batch: nullcontext(),
                        bucket_from=8,
                        out_rows=8,
                        global_num_rows=live,
                    )
                    expected = torch.stack(
                        [
                            torch.full((8,), live, dtype=torch.int32, device="cuda"),
                            torch.full((8,), local, dtype=torch.int32, device="cuda"),
                            (torch.arange(8, device="cuda") < live).to(torch.int32),
                        ],
                        dim=1,
                    )
                    torch.testing.assert_close(value, expected)
                    self.assertEqual(parent.global_num_token_non_padded.item(), 97)
                    self.assertEqual(parent.num_token_non_padded.item(), 27)
            self.assertEqual(len(captures), 1)


if __name__ == "__main__":
    unittest.main()
