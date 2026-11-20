"""Decoder-tail counter ownership through real CUDA graph capture and replay."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.deepseek_v4_replay_tail as replay_tail
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
)
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner.row_bucket_graph_runner import (
    RowBucketGraphRunner,
    _Graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.srt.models.deepseek_v4_replay_tail import DecoderReplayTail
from sglang.srt.runtime_context import get_flags, get_parallel
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

    def test_cp_tail_replay_owns_logical_counts_and_keeps_physical_rows(self):
        """CP padding must not dispatch stale rows after a bucket shrinks."""
        for rank in (0, 3):
            with (
                self.subTest(rank=rank),
                get_parallel().override(
                    attn_dp_size=1, attn_cp_size=4, attn_cp_rank=rank, attn_tp_size=1
                ),
                get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE),
                patch.object(cp_base, "_STRATEGY", InterleaveCPStrategy(cp_size=4)),
            ):
                parent = _batch()
                parent.attn_tp_sequence_sharded = False
                tail = DecoderReplayTail(
                    model=SimpleNamespace(late_layer_vl_routing=False),
                    layer_ids=range(1),
                    row_step=4,
                    max_rows=8,
                )
                captures = []

                def layer_stack(state, *, positions, forward_batch, **kwargs):
                    bound = get_forward_context().moe_num_token_non_padded()
                    rows = state.streams.shape[0]
                    mask = torch.arange(rows, device="cuda") < bound
                    value = torch.where(mask, state.streams[:, 0, 0] + positions, 0)[
                        :, None
                    ]
                    return value, None, [value + 1]

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
                    for step, (physical, logical) in enumerate(
                        ((6, 5), (3, 2), (3, 0), (6, 1), (3, 3), (6, 6))
                    ):
                        counts = [physical] * 4
                        counts[rank] = logical
                        cp_count = torch.tensor(99, device="cuda", dtype=torch.int32)
                        cp = InterleaveContextParallelMetadata(
                            per_rank_actual_token=[physical] * 4,
                            per_rank_logical_token=counts,
                            local_index=torch.arange(physical, device="cuda"),
                            gather_index=torch.arange(sum(counts), device="cuda"),
                            moe_local_token_count=cp_count,
                        )
                        parent.attn_cp_metadata = cp
                        x = torch.arange(physical, device="cuda").float() + step * 100
                        positions = torch.arange(physical, device="cuda") * 4 + rank
                        value, pre, aux = tail.run(
                            state=HcState(x[:, None, None]),
                            positions=positions,
                            input_ids=positions,
                            input_ids_global=None,
                            forward_batch=parent,
                            capture_dspark=False,
                        )
                        expected = (x + positions)[:, None]
                        expected[logical:] = 0
                        torch.testing.assert_close(value, expected)
                        torch.testing.assert_close(aux[0], expected + 1)
                        self.assertIsNone(pre)
                        self.assertEqual(value.shape[0], physical)
                        self.assertIs(parent.attn_cp_metadata, cp)
                        self.assertIs(cp.moe_local_token_count, cp_count)
                        self.assertEqual(cp_count.item(), 99)
                        self.assertEqual(parent.num_token_non_padded.item(), 27)
                        self.assertEqual(parent.global_num_token_non_padded.item(), 97)
                        self.assertIs(
                            get_forward_context().moe_num_token_non_padded(), cp_count
                        )
                self.assertEqual([row for row, *_ in captures], [8, 4])
                for _, graph_count, _, _ in captures:
                    self.assertNotEqual(
                        graph_count.data_ptr(), parent.num_token_non_padded.data_ptr()
                    )
                    self.assertNotEqual(graph_count.data_ptr(), cp_count.data_ptr())

    def test_cp_tail_break_refreshes_layout_and_crops_collective_padding(self):
        """A replayed KV break must use the step's CP layout and physical size."""
        with (
            get_parallel().override(
                attn_dp_size=1, attn_cp_size=4, attn_cp_rank=0, attn_tp_size=1
            ),
            get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE),
            patch.object(cp_base, "_STRATEGY", InterleaveCPStrategy(cp_size=4)),
        ):
            parent = _batch()
            parent.attn_tp_sequence_sharded = False
            tail = DecoderReplayTail(
                model=SimpleNamespace(late_layer_vl_routing=False),
                layer_ids=range(1),
                row_step=4,
                max_rows=8,
            )
            tail._layers = SimpleNamespace(
                attention_layers=[],
                moe_layers=[],
                moe_fusions=[],
                dsa_indexers=[],
                mha_companion_layers=[],
            )
            cache = torch.full((64,), -1.0, device="cuda")
            seen = []

            def store(x, positions, batch, backend, **kwargs):
                seen.append(
                    (
                        batch.attn_cp_metadata,
                        x.shape[0],
                        torch.cuda.is_current_stream_capturing(),
                    )
                )
                cache[batch.attn_cp_metadata.local_index] = x[:, 0] + positions

            attention = SimpleNamespace(_compute_kv_to_cache=store)

            def layer_stack(state, *, positions, **kwargs):
                x = state.streams[:, 0] * 2
                replay_tail.bcg_tail_kv_store(attention, x, positions, None)
                return x + positions[:, None], None, []

            def record(body, rows, local, batch, start, free_before):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    body()
                torch.cuda.current_stream().wait_stream(stream)
                graph = BreakableCUDAGraph()
                with BreakableCUDAGraphCapture(
                    cuda_graph=graph, pool=torch.cuda.graph_pool_handle(), stream=stream
                ):
                    outputs = body()
                torch.cuda.current_stream().wait_stream(stream)
                return _Graph(graph, local, outputs, (batch,))

            with (
                forward_context(
                    ForwardContext(attn_backend=SimpleNamespace(forward_metadata=None)),
                    forward_batch=parent,
                ),
                patch.object(tail, "_layer_stack", side_effect=layer_stack),
                patch.object(tail.runner, "_record", side_effect=record),
            ):
                for step, physical in enumerate((6, 3, 6, 3)):
                    cp = InterleaveContextParallelMetadata(
                        per_rank_actual_token=[physical] * 4,
                        per_rank_logical_token=[physical] * 4,
                        local_index=torch.arange(physical, device="cuda") + step * 8,
                        gather_index=torch.arange(physical * 4, device="cuda"),
                    )
                    parent.attn_cp_metadata = cp
                    x = torch.arange(physical, device="cuda").float() + step * 100
                    positions = torch.arange(physical, device="cuda") * 4
                    cache.fill_(-1)
                    value, _, _ = tail.run(
                        state=HcState(x[:, None, None]),
                        positions=positions,
                        input_ids=positions,
                        input_ids_global=None,
                        forward_batch=parent,
                        capture_dspark=False,
                    )
                    expected = x * 2 + positions
                    torch.testing.assert_close(value[:, 0], expected)
                    expected_cache = torch.full_like(cache, -1)
                    expected_cache[cp.local_index] = expected
                    torch.testing.assert_close(cache, expected_cache)
                    self.assertIs(seen[-1][0].local_index, cp.local_index)
                    self.assertEqual(seen[-1][1:], (physical, False))
            self.assertTrue(all(not capturing for _, _, capturing in seen))
            self.assertEqual(len(tail.runner._graphs), 2)


if __name__ == "__main__":
    unittest.main()
