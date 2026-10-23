"""Chunk layouts preserve rank-local order without growing embedding storage."""

import unittest

from sglang.srt.layers.engram_prefetch import EngramPrefetchLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestEngramPrefetchLayout(CustomTestCase):
    def test_uneven_segments_and_idle_rank_reconstruct_global_order(self):
        counts = (259, 0, 131)
        for rank, count in enumerate(counts):
            plan = EngramPrefetchLayout(counts, rank, count, 128)
            global_rows = list(range(sum(counts)))
            consumed = []
            for block in range(plan.num_chunks):
                slices = plan.global_slices(block)
                packed = [row for part in slices for row in global_rows[part]]
                self.assertLessEqual(len(packed), len(counts) * 128)
                start = sum(plan.block_sizes(block)[:rank])
                length = plan.block_sizes(block)[rank]
                consumed.extend(packed[start : start + length])
            offset = sum(counts[:rank])
            self.assertEqual(consumed, global_rows[offset : offset + count])

    def test_graph_padding_and_tail_never_expose_another_segment(self):
        plan = EngramPrefetchLayout((129, 129), 1, 144, 128)
        self.assertEqual(plan.block_sizes(1), (1, 1))
        self.assertEqual(plan.local_slice(1), slice(128, 129))
        self.assertEqual(plan.global_slices(1), (slice(128, 129), slice(257, 258)))
        self.assertEqual(EngramPrefetchLayout((0, 0), 0, 0, 128).num_chunks, 0)

    def test_reserved_rows_do_not_extend_past_local_activations(self):
        plan = EngramPrefetchLayout((260, 260), 1, 131, 128)
        # Collectives retain every reserved row; only real local activations
        # may reach the projection and gate.
        self.assertEqual(plan.block_sizes(1), (128, 128))
        self.assertEqual(plan.local_slice(1), slice(128, 131))
        self.assertEqual(plan.local_slice(2), slice(131, 131))

    def test_per_rank_elastic_ep_rejected_before_allocating_staging(self):
        """Elastic WORLD gathers can exceed the fixed TP/DP segment plan."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.layers.engram_prefetch import EngramPrefetcher

        layer = SimpleNamespace(
            layer_hash_index=0,
            embed=SimpleNamespace(host_table=object(), _shared=False),
            q_weight=SimpleNamespace(device=SimpleNamespace(type="cuda")),
        )
        with (
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(elastic_ep_backend="mooncake")
                ),
            ),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1, attn_cp_size=1),
            ),
            patch(
                "sglang.srt.runtime_context.get_stream",
                side_effect=AssertionError("must reject before allocating resources"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "elastic EP"):
                EngramPrefetcher([layer], 128, 0)

    def test_late_cut_uses_same_prefetch_phases_for_extend_and_idle(self):
        """DP peers must exchange tail hashes in the same collective phase."""
        from types import SimpleNamespace
        from unittest.mock import patch

        import torch

        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4 as dsv4
        from sglang.srt.runtime_context import get_parallel

        class Prefetch:
            def __init__(self):
                self.phases = []

            def begin(self, hashes, batch, *, layer_ids, cp_all_tokens):
                self.phases.append(layer_ids)

            def finish(self):
                pass

        class Engram:
            def __init__(self, index):
                self.layer_hash_index = index

            def __call__(self, x, *args, **kwargs):
                return x + 1

        class Block(torch.nn.Module):
            hc_cfg = None

            def __init__(self, engram=None):
                super().__init__()
                self.engram = engram

            def forward_hc_pre_from_prev(self, *, state, **kwargs):
                return state

        model = dsv4.DeepseekV4Model.__new__(dsv4.DeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.layers = torch.nn.ModuleList(
            [Block(Engram(0)), Block(), Block(Engram(1)), Block(), Block(Engram(2))]
        )
        model.start_layer, model.end_layer, model.late_layer_start = 0, 5, 2
        model.tail_writes_source_kv = False
        model.decoder_replay_tail = None
        model.pp_group = SimpleNamespace(world_size=1)
        model.config = SimpleNamespace(model_type="deepseek_v41", vision_n_layers=0)
        model.engram_prefetch = Prefetch()
        model.engram_hasher = lambda ids, batch: torch.zeros(
            len(ids), 3, 4, dtype=torch.int64
        )
        tail = SimpleNamespace(
            scatter_indices=None,
            extend_seq_lens_cpu=[1],
            rows=lambda value: value[-1:],
            positions=torch.tensor([2]),
        )
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
            enter_late_layer_tail=lambda *args: object(),
            exit_late_layer_tail=lambda *args: None,
        )
        for mode in (ForwardMode.EXTEND, ForwardMode.IDLE, ForwardMode.DECODE):
            count = 0 if mode.is_idle() else 3
            positions = torch.arange(count)
            batch = SimpleNamespace(
                forward_mode=mode,
                extend_seq_lens_cpu=[count],
                capture_hidden_mode=None,
                return_logprob=False,
            )
            model.engram_prefetch.phases.clear()
            with (
                get_parallel().override(
                    tp_size=1, tp_rank=0, attn_dp_size=1, attn_cp_size=1
                ),
                patch.object(dsv4, "_is_cuda", False),
                patch.object(dsv4, "_is_hip", False),
                patch.object(dsv4, "get_attn_backend", return_value=backend),
                patch.object(dsv4, "check_cuda_graph_backend", return_value=True),
            ):
                model._forward_layers_hc_pre_from_prev(
                    positions,
                    torch.zeros(count, 2, 4),
                    batch,
                    positions,
                    positions,
                    False,
                    [],
                )
            self.assertEqual(model.engram_prefetch.phases, [(0,), (1, 2)], mode)


if __name__ == "__main__":
    unittest.main()
