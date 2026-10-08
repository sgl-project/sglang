"""Encoder-SWA replay folded into the extend forward: row bookkeeping of the
folded batch and the paths that consume it. CPU only."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    late_layer_tail_layout,
)
from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    DSV4AttnMetadata,
)
from sglang.srt.mem_cache.dsv41_request_window import window_layout
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.encoder_swa_replay import _fold_batch, drop_folded_rows
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MAX_CTX = 1024


def _loc(slot, pos):
    return slot * MAX_CTX + pos


def _tok(slot, pos):
    return 1000 * (slot + 1) + pos


# (req slot, cached prefix, new tokens). Slot 1 misses; slot 2 hits with a full
# 128-token replay and a folded extend longer than the 128-row tail; slot 3 hits
# a 64-token prefix (replay from 0) and its whole folded extend is its tail.
REQS = [(1, 0, 40), (2, 512, 30), (3, 64, 20)]
HITS = [(1, 384, 512), (2, 0, 64)]  # (batch index, replay start, prefix)


class _Fixture:
    def __init__(self):
        self.req_to_token = torch.tensor(
            [[_loc(s, p) for p in range(MAX_CTX)] for s in range(4)], dtype=torch.int32
        )
        self.runner = SimpleNamespace(
            device="cpu",
            req_to_token_pool=SimpleNamespace(req_to_token=self.req_to_token),
        )
        reqs = [
            SimpleNamespace(
                full_untruncated_fill_ids=[_tok(s, p) for p in range(p + n)]
            )
            for s, p, n in REQS
        ]
        rows = [(s, p, p + n) for s, p, n in REQS]
        self.extend_num_tokens = sum(n for _, _, n in REQS)
        self.batch = SimpleNamespace(
            reqs=reqs,
            req_pool_indices=torch.tensor([s for s, _, _ in REQS]),
            prefix_lens=[p for _, p, _ in REQS],
            extend_lens=[n for _, _, n in REQS],
            extend_num_tokens=self.extend_num_tokens,
            input_ids=torch.tensor(
                [_tok(s, q) for s, a, b in rows for q in range(a, b)]
            ),
            out_cache_loc=torch.tensor(
                [_loc(s, q) for s, a, b in rows for q in range(a, b)]
            ),
            prefill_input_ids_cpu=None,
            # Slot 3 asks for input logprobs from its 6th new token on.
            extend_logprob_start_lens=[40, 30, 5],
            engram_history=None,
            global_num_tokens=[self.extend_num_tokens],
            global_num_tokens_for_logprob=[1 + 1 + 15],
            can_run_decode_cuda_graph=False,
            can_run_dp_draft_cuda_graph=False,
            dp_spec_prefill_coordination_applied=False,
        )
        self.folded = _fold_batch(
            batch=self.batch,
            runner=self.runner,
            rows=[(i, start, end) for i, start, end in HITS],
        )

    def folded_positions(self):
        fb = self.folded.batch
        return torch.cat(
            [torch.arange(p, p + n) for p, n in zip(fb.prefix_lens, fb.extend_lens)]
        )

    def folded_seq_lens(self):
        fb = self.folded.batch
        return [p + n for p, n in zip(fb.prefix_lens, fb.extend_lens)]


class TestFoldedExtendRows(CustomTestCase):
    def test_mixed_hit_and_miss_fold_maps_every_row(self):
        """Hits extend from their replay start into their cached KV slots; every
        per-row view (tokens, KV slots, window floor, logprob range, hidden rows)
        stays aligned with the scheduled batch."""
        fx = _Fixture()
        fb, folded = fx.folded.batch, fx.folded
        starts = [0, 384, 0]
        self.assertEqual(fb.prefix_lens, starts)
        self.assertEqual(fb.extend_lens, [40, 158, 84])
        self.assertEqual(fb.extend_num_tokens, 282)
        spans = [(s, a, p + n) for (s, p, n), a in zip(REQS, starts)]
        torch.testing.assert_close(
            fb.input_ids,
            torch.tensor([_tok(s, q) for s, a, b in spans for q in range(a, b)]),
        )
        torch.testing.assert_close(
            fb.out_cache_loc,
            torch.tensor([_loc(s, q) for s, a, b in spans for q in range(a, b)]),
        )
        # A miss is never floored; a hit's rows all floor at its replay start.
        torch.testing.assert_close(
            folded.row_floor,
            torch.tensor([0] * 40 + [384] * 158 + [0] * 84),
        )
        torch.testing.assert_close(
            folded.compress_skip, torch.tensor([0, 128, 64], dtype=torch.int32)
        )
        # Input logprobs cover the same tokens, so their row count is unchanged.
        self.assertEqual(fb.extend_logprob_start_lens, [40, 158, 69])
        logprob_rows = [
            max(n - s, 1) for n, s in zip(fb.extend_lens, fb.extend_logprob_start_lens)
        ]
        self.assertEqual(sum(logprob_rows), fx.batch.global_num_tokens_for_logprob[0])
        # Full hidden states of the folded forward reduce to the scheduled rows.
        out = SimpleNamespace(
            hidden_states=fb.input_ids[:, None].float(),
            hidden_states_token_indices=None,
        )
        drop_folded_rows(logits_output=out, folded=folded)
        torch.testing.assert_close(
            out.hidden_states, fx.batch.input_ids[:, None].float()
        )


class TestFoldWithDecoderSwaTail(CustomTestCase):
    """--enable-encoder-swa-bounded-replay with --enable-decoder-swa-bounded-replay:
    the late layers run only each request's last SWA_WINDOW folded rows."""

    def _backend(self, fx):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.encoder_replay = False
        backend.encoder_row_floor = fx.folded.row_floor
        backend.req_to_token = fx.req_to_token
        backend.page_size = 256
        backend.index_topk = 512
        backend.present_ratios = (0,)
        backend.low_ratios = ()
        backend.has_c4 = backend.has_c128 = False
        backend.trtllm_attn = False
        backend.cuda_int32_kwargs = {"dtype": torch.int32, "device": "cpu"}
        backend.token_to_kv_pool = SimpleNamespace(
            request_window=SimpleNamespace(capacity=2 * SWA_WINDOW)
        )
        return backend

    def test_tail_keeps_its_own_floor_under_fold(self):
        """A prefix hit with both flags on builds the tail metadata; each tail row
        is floored at its tail start or its request's replay start, whichever is
        later. It used to take the fold's per-row floor of all folded rows and
        fail a row-count assert."""
        fx = _Fixture()
        fb = fx.folded.batch
        seq_lens = fx.folded_seq_lens()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=list(fb.extend_lens),
            seq_lens_cpu=torch.tensor(seq_lens),
            seq_lens=torch.tensor(seq_lens),
            req_pool_indices=fb.req_pool_indices,
            out_cache_loc=fb.out_cache_loc,
            positions=fx.folded_positions(),
        )
        with (
            patch.object(DSV4AttnMetadata, "init_compression_metadata"),
            patch.object(DSV4AttnMetadata, "init_flashmla_related"),
        ):
            metadata = self._backend(fx)._build_late_layer_tail_metadata(forward_batch)

        # Tails: the miss's 40 rows, the last 128 of the 158-row hit (from 414,
        # above its replay start 384), and the whole 84-row hit (from 0).
        tail = metadata.late_layer_tail
        self.assertEqual(tail.extend_seq_lens_cpu, [40, 128, 84])
        floor = torch.tensor([0] * 40 + [414] * 128 + [0] * 84)
        req = torch.tensor([1] * 40 + [2] * 128 + [3] * 84)
        expected = window_layout(
            req,
            tail.positions,
            capacity=2 * SWA_WINDOW,
            floor=floor,
            num_groups=3,
        )
        layout = metadata.core_attn_metadata.request_window_layout
        torch.testing.assert_close(layout.lengths, expected.lengths)
        torch.testing.assert_close(layout.indices, expected.indices)

    def test_tail_hidden_rows_map_to_scheduled_rows(self):
        """With DSpark the model returns the tail's hidden rows and their indices
        into the folded extend. The draft indexes scheduled-row cache locations
        with them, so they must come back in scheduled-row space without the
        replayed rows."""
        fx = _Fixture()
        fb = fx.folded.batch
        token_indices, _, _ = late_layer_tail_layout(
            extend_lens_cpu=list(fb.extend_lens),
            seq_lens_cpu=fx.folded_seq_lens(),
            tail_len=SWA_WINDOW,
            device=torch.device("cpu"),
        )
        out = SimpleNamespace(
            hidden_states=fb.input_ids[token_indices][:, None].float(),
            hidden_states_token_indices=token_indices,
        )
        drop_folded_rows(logits_output=out, folded=fx.folded)

        # Kept: all 40 miss rows, the hit's 30 new rows, the small hit's 20.
        expected = torch.cat(
            [torch.arange(0, 40), torch.arange(40, 70), torch.arange(70, 90)]
        )
        torch.testing.assert_close(out.hidden_states_token_indices, expected)
        torch.testing.assert_close(
            out.hidden_states, fx.batch.input_ids[expected][:, None].float()
        )


class TestFoldMlpSync(CustomTestCase):
    """Pure TP that still syncs the MLP (e.g. TP4 with --moe-dense-tp-size 1)."""

    ATTN_TP = 4

    def setUp(self):
        override = get_context().override_server_args(
            tp_size=self.ATTN_TP,
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(bs=[])),
        )
        override.install()
        self.addCleanup(override.restore)

    def _forward_batch(self, batch):
        num_rows = batch.input_ids.shape[0]
        seq_lens = [p + n for p, n in zip(batch.prefix_lens, batch.extend_lens)]
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(seq_lens),
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=torch.tensor(seq_lens),
            out_cache_loc=batch.out_cache_loc,
            seq_lens_sum=sum(seq_lens),
            positions=torch.zeros(num_rows, dtype=torch.int64),
            seq_lens_cpu=torch.tensor(seq_lens),
            extend_num_tokens=batch.extend_num_tokens,
            extend_seq_lens=torch.tensor(batch.extend_lens),
            extend_seq_lens_cpu=list(batch.extend_lens),
            extend_prefix_lens_cpu=list(batch.prefix_lens),
            is_extend_in_batch=True,
        )
        fb.init_mlp_sync_metadata(batch, torch.device("cpu"))
        return fb

    def _mlp_sync(self, fb):
        runner = MagicMock()
        runner.attn_backend.get_cuda_graph_seq_len_fill_value.return_value = 1
        runner.attn_backend.get_cpu_graph_seq_len_fill_value.return_value = 1
        runner.is_draft_worker = False
        runner.attn_tp_sequence_sharded.return_value = False
        tp_group = GroupCoordinator.__new__(GroupCoordinator)
        tp_group.world_size = self.ATTN_TP
        tp_group.rank_in_group = 0
        with (
            get_parallel().override(tp_rank=0, attn_tp_rank=0),
            patch.object(parallel_state, "_TP", tp_group),
            # CPU runners have no driver to pin the synced token counts with.
            patch("sglang.srt.model_executor.forward_batch_info._is_cpu", True),
        ):
            fb.prepare_mlp_sync_batch(runner)

    def test_folded_rows_count_in_mlp_sync_totals(self):
        """The scheduler's gathered count covers the scheduled rows only. A folded
        batch must sync its folded row count, or the padding to that count
        shrinks input_ids (a negative pad); logprob rows keep their count."""
        fx = _Fixture()
        fb = self._forward_batch(fx.folded.batch)
        self._mlp_sync(fb)

        padded = 284  # 282 folded rows aligned to the attention TP size
        self.assertEqual(fb.global_num_tokens_cpu, [padded])
        self.assertEqual(fb.extend_num_tokens, padded)
        self.assertEqual(fb.input_ids.shape[0], padded)
        self.assertEqual(fb.out_cache_loc.shape[0], padded)
        torch.testing.assert_close(fb.input_ids[:282], fx.folded.batch.input_ids)
        self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [17])
        # Hidden states of the padded forward still reduce to the scheduled rows.
        out = SimpleNamespace(
            hidden_states=fb.input_ids[:, None].float(),
            hidden_states_token_indices=None,
        )
        drop_folded_rows(logits_output=out, folded=fx.folded)
        torch.testing.assert_close(
            out.hidden_states, fx.batch.input_ids[:, None].float()
        )


if __name__ == "__main__":
    unittest.main()
