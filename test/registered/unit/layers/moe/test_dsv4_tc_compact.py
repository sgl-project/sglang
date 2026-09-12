"""DP graph padding must not become MoE work or displace hash-routing IDs.

The collective fixture represents an external variable-size TP transport.
Expected outputs come from rank-labelled token records, independently of the
bridge's packing/scattering implementation. No server or GPU is used.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers import dp_attention as dp_module  # noqa: E402
from sglang.srt.layers.dp_attention import DpPaddingMode  # noqa: E402
from sglang.srt.layers.moe import dsv4_tc_compact as compact  # noqa: E402
from sglang.srt.model_executor import forward_batch_info as batch_module  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_flags, get_forward  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _rank_records(counts, hidden_size=4):
    ids = [1000 * rank + torch.arange(count) for rank, count in enumerate(counts)]
    hidden = [
        torch.stack([tokens.float() + col / 4 for col in range(hidden_size)], dim=1)
        for tokens in ids
    ]
    return hidden, ids


class _ReferenceTransport:
    """CPU transport oracle; checks that callers send only their live prefix."""

    def __init__(self, case, rank, counts, hidden, ids):
        self.case = case
        self.rank_in_group = rank
        self.world_size = len(counts)
        self.counts = list(counts)
        self.hidden = hidden
        self.ids = ids

    def all_gatherv(self, input_, *, sizes, output):
        self.case.assertEqual(list(sizes), self.counts)
        source = self.hidden if input_.is_floating_point() else self.ids
        expected_local = source[self.rank_in_group].reshape_as(input_)
        torch.testing.assert_close(input_, expected_local, rtol=0, atol=0)
        output.copy_(torch.cat(source).reshape_as(output))
        return output

    def reduce_scatterv(self, input_, *, output, sizes):
        if sizes is None:
            self.case.assertEqual(len(set(self.counts)), 1)
        else:
            self.case.assertEqual(list(sizes), self.counts)
        self.case.assertEqual(input_.shape[0], sum(self.counts))
        self.case.assertEqual(output.shape[0], self.counts[self.rank_in_group])
        # Every rank's fake expert contributes (rank+1) times the same token
        # function. This emulates SUM reduction and makes a missing/duplicate
        # reduction observable instead of accidentally accepting a local copy.
        reduced = input_ / (self.rank_in_group + 1) * sum(range(1, self.world_size + 1))
        first = sum(self.counts[: self.rank_in_group])
        output.copy_(reduced[first : first + output.shape[0]])
        return output


class _TokenExpert:
    def __init__(
        self,
        case,
        rank,
        counts,
        ids,
        *,
        has_ids=True,
        separate_shared=False,
    ):
        self.case, self.rank, self.counts = case, rank, counts
        self.ids = ids
        self.is_hash = has_ids
        self._shared_expert_tp1 = separate_shared
        self.shared_experts = object() if separate_shared else None

    def _forward_shared_experts(self, hidden):
        return hidden * 3 + 7

    def __call__(
        self, hidden, batch, *, input_ids, input_ids_global, skip_shared_experts
    ):
        self.case.assertEqual(hidden.shape[0], sum(self.counts))
        self.case.assertEqual(batch.global_num_tokens_cpu, list(self.counts))
        self.case.assertEqual(batch.global_num_tokens_gpu.tolist(), list(self.counts))
        self.case.assertEqual(batch.global_dp_buffer_len, sum(self.counts))
        self.case.assertTrue(batch.dp_padding_mode.is_sum_len())
        self.case.assertIsNone(batch.dp_local_start_pos)
        self.case.assertIsNone(batch.dp_local_num_tokens)
        self.case.assertTrue(get_forward().mlp_reduce_scatter)
        self.case.assertFalse(get_forward().fuse_mlp_allreduce)
        self.case.assertEqual(skip_shared_experts, self._shared_expert_tp1)
        if self.is_hash:
            expected_ids = torch.cat(self.ids)
            torch.testing.assert_close(input_ids, expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(input_ids_global, expected_ids, rtol=0, atol=0)
            token_term = input_ids.to(hidden.dtype).reshape(-1, 1)
        else:
            self.case.assertIsNone(input_ids)
            self.case.assertIsNone(input_ids_global)
            token_term = 0
        return (hidden * 2 + token_term) * (self.rank + 1)


class TestDsv4TcCompactMoe(CustomTestCase):
    def _case(self, counts, rank, *, has_ids=True, separate_shared=False):
        bucket, width = max(max(counts), 4), 4
        hidden, ids = _rank_records(counts, width)
        local = torch.full((bucket, width), -9876.0)
        local[: counts[rank]].copy_(hidden[rank])
        local_ids = torch.full((bucket,), -9876, dtype=torch.int64)
        local_ids[: counts[rank]].copy_(ids[rank])
        output = torch.full_like(local, float("nan"))
        metadata = SimpleNamespace(
            global_num_tokens_cpu=[bucket] * len(counts),
            global_num_tokens_gpu=torch.full((len(counts),), bucket),
            global_dp_buffer_len=bucket * len(counts),
            dp_padding_mode=DpPaddingMode.MAX_LEN,
            dp_local_start_pos=torch.tensor(999),
            dp_local_num_tokens=torch.tensor(bucket),
        )
        expert = _TokenExpert(
            self,
            rank,
            counts,
            ids,
            has_ids=has_ids,
            separate_shared=separate_shared,
        )
        group = _ReferenceTransport(self, rank, counts, hidden, ids)
        with get_forward().scoped(mlp_reduce_scatter=False, fuse_mlp_allreduce=True):
            compact.run_compact_dp_moe(
                local,
                local_ids if has_ids else None,
                output,
                counts=counts,
                local_rank=rank,
                tp_group=group,
                moe_layer=expert,
                forward_batch=metadata,
                counts_gpu=torch.tensor(counts),
            )
            self.assertFalse(get_forward().mlp_reduce_scatter)
            self.assertTrue(get_forward().fuse_mlp_allreduce)
        self.assertEqual(metadata.global_num_tokens_cpu, [bucket] * len(counts))
        torch.testing.assert_close(
            metadata.global_num_tokens_gpu, torch.full((len(counts),), bucket)
        )
        self.assertTrue(metadata.dp_padding_mode.is_max_len())
        token_term = ids[rank].reshape(-1, 1) if has_ids else 0
        expected = (hidden[rank] * 2 + token_term) * sum(range(1, len(counts) + 1))
        if separate_shared:
            expected = expected + hidden[rank] * 3 + 7
        torch.testing.assert_close(output[: counts[rank]], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            output[counts[rank] :],
            torch.zeros_like(output[counts[rank] :]),
            rtol=0,
            atol=0,
        )

    def test_ragged_hash_routing_keeps_rank_order_and_clears_idle_tails(self):
        for counts, ranks in (
            ([3, 0, 1, 4, 0, 2, 0, 1], (0, 1, 3, 7)),
            ([2] * 8, (3,)),
            ([0] * 8, (0,)),
        ):
            for rank in ranks:
                with self.subTest(counts=counts, rank=rank):
                    self._case(counts, rank)

    def test_non_hash_experts_do_not_fabricate_ids(self):
        self._case([1, 0, 2, 0, 0, 1, 0, 3], 2, has_ids=False)

    def test_replicated_shared_expert_is_added_once_after_tp_reduction(self):
        self._case([3, 0, 1, 4, 0, 2, 0, 1], 7, separate_shared=True)

    def test_dp_gather_tracking_keeps_one_graph_across_tc_capture_phases(self):
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            enable_tc_piecewise_cuda_graph,
        )

        graphs = []

        def backend(graph, _inputs):
            graphs.append(graph)
            return graph.forward

        def body(hidden):
            dp_module._note_dp_gather_in_prefill_graph()
            return hidden + 1

        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        hidden = torch.zeros(4)
        dp = get_flags().dp
        with dp.override(
            capturing_prefill_graph=False, prefill_graph_has_dp_gather=False
        ):
            compiled = torch.compile(body, backend=backend, fullgraph=True)
            with enable_tc_piecewise_cuda_graph():
                for capturing in (False, True, True, False):
                    dp.capturing_prefill_graph = capturing
                    torch.testing.assert_close(compiled(hidden), hidden + 1)
            self.assertEqual(len(graphs), 1)
            self.assertTrue(dp.prefill_graph_has_dp_gather)

        # The same capture flag must stay inert in ordinary torch.compile,
        # while an eager CUDA-graph capture still records the gather.
        torch._dynamo.reset()
        with dp.override(
            capturing_prefill_graph=True, prefill_graph_has_dp_gather=False
        ):
            compiled = torch.compile(body, backend=backend, fullgraph=True)
            torch.testing.assert_close(compiled(hidden), hidden + 1)
            self.assertFalse(dp.prefill_graph_has_dp_gather)
            dp_module._note_dp_gather_in_prefill_graph()
            self.assertTrue(dp.prefill_graph_has_dp_gather)

    def test_true_counts_survive_graph_padding_and_idle_conversion(self):
        counts = [0, 3, 1, 0, 0, 2, 0, 0]
        for rank, mode in [(0, ForwardMode.IDLE), (1, ForwardMode.MIXED)]:
            with self.subTest(rank=rank, mode=mode):
                n = counts[rank]
                bs = int(n > 0)
                lengths = torch.tensor([n] if n else [], dtype=torch.int32)
                batch = ForwardBatch(
                    forward_mode=mode,
                    batch_size=bs,
                    input_ids=torch.arange(n, dtype=torch.int64),
                    req_pool_indices=torch.zeros(bs, dtype=torch.int64),
                    seq_lens=lengths.clone(),
                    orig_seq_lens=lengths.clone(),
                    seq_lens_cpu=lengths.clone(),
                    out_cache_loc=torch.arange(n),
                    seq_lens_sum=n,
                    positions=torch.arange(n),
                    is_extend_in_batch=True,
                    can_run_dp_prefill_cuda_graph=True,
                    global_num_tokens_cpu=list(counts),
                    global_num_tokens_gpu=torch.tensor(counts),
                    global_num_tokens_for_logprob_cpu=list(counts),
                    global_num_token_non_padded_cpu=n,
                )
                runner = SimpleNamespace(
                    model_config=SimpleNamespace(hf_config=SimpleNamespace()),
                    is_draft_worker=False,
                    attn_tp_sequence_sharded=lambda _: False,
                    attn_backend=SimpleNamespace(
                        get_cpu_graph_seq_len_fill_value=lambda: 1
                    ),
                )
                execution = SimpleNamespace(
                    graph=SimpleNamespace(
                        cuda_graph_config=SimpleNamespace(
                            prefill=SimpleNamespace(
                                bs=[4, 8], backend=compact.Backend.TC_PIECEWISE
                            )
                        )
                    )
                )
                with (
                    patch.object(compact, "compact_moe_enabled", return_value=True),
                    patch.object(batch_module, "_is_cpu", True),
                    patch.object(
                        batch_module,
                        "get_parallel",
                        return_value=SimpleNamespace(attn_tp_size=1, attn_dp_rank=rank),
                    ),
                    patch.object(batch_module, "get_exec", return_value=execution),
                    patch.object(batch_module, "mambaish_config", return_value=None),
                    patch.object(batch_module, "set_dp_buffer_len"),
                    patch.object(batch_module, "set_is_extend_in_batch"),
                    patch.object(dp_module, "get_attention_dp_size", return_value=8),
                    patch.object(
                        batch_module,
                        "prefill_graph_tolerates_sum_len",
                        return_value=False,
                    ),
                ):
                    # The actual preparation method pads buffers and fabricates
                    # idle EXTEND rows. Only topology/external allocation hooks
                    # are replaced; no preparation behavior is mocked.
                    batch.prepare_mlp_sync_batch(runner)
                self.assertEqual(batch.global_num_tokens_cpu, [4] * 8)
                batch.global_num_tokens_gpu.fill_(99)
                batch.global_num_tokens_cpu[0] = 99
                self.assertEqual(batch.moe_real_num_tokens_cpu, counts)
                torch.testing.assert_close(
                    batch.moe_real_num_tokens_gpu, torch.tensor(counts)
                )
                self.assertEqual(batch.input_ids.shape[0], 4)
                if mode.is_idle():
                    self.assertEqual(batch.forward_mode, ForwardMode.EXTEND)
                    self.assertEqual(batch.global_num_token_non_padded_cpu, 0)


if __name__ == "__main__":
    unittest.main()
