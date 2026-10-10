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
from sglang.srt.layers import dp_attention
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    DSV4AttnMetadata,
)
from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.mem_cache.dsv41_request_window import window_layout
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.encoder_swa_replay import (
    _build_replay_batch,
    _check_folded_counts,
    _fold_batch,
    _replay_spans,
    decoder_swa_trim_rows,
    drop_folded_rows,
    encoder_swa_fold_rows,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models import deepseek_v4
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
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
            forward_mode=ForwardMode.EXTEND,
            # Every request is new to the batch, so each window resets.
            encoder_swa_reset=[True] * len(REQS),
            global_num_tokens_for_logprob=[1 + 1 + 15],
            can_run_decode_cuda_graph=False,
            can_run_dp_draft_cuda_graph=False,
            dp_spec_prefill_coordination_applied=False,
        )
        # The scheduler's MLP-sync count already includes the rows the fold adds.
        self.batch.global_num_tokens = [
            self.extend_num_tokens + encoder_swa_fold_rows(self.batch)
        ]
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
        """The scheduler's gathered count covers the folded rows, so the padding to
        that count never shrinks input_ids (a negative pad); logprob rows keep
        their count."""
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

    def test_batched_replay_rows_count_in_mlp_sync_totals(self):
        """Backends that do not fold run one replay forward for all hits of a step.
        Its rows (128 per hit) are not the scheduled rows the gather counted, so it
        must sync its own count, or the padding shrinks input_ids."""
        fx = _Fixture()
        hits = [(1, 512, 64), (2, 640, 64)]  # (req slot, cached prefix, new tokens)
        rows = [(s, p, p + n) for s, p, n in hits]
        batch = SimpleNamespace(
            **{
                **vars(fx.batch),
                "reqs": [
                    SimpleNamespace(
                        full_untruncated_fill_ids=[_tok(s, q) for q in range(p + n)]
                    )
                    for s, p, n in hits
                ],
                "req_pool_indices": torch.tensor([s for s, _, _ in hits]),
                "req_pool_indices_cpu": torch.tensor([s for s, _, _ in hits]),
                "prefix_lens": [p for _, p, _ in hits],
                "extend_lens": [n for _, _, n in hits],
                "extend_num_tokens": 128,
                "input_ids": torch.tensor(
                    [_tok(s, q) for s, a, b in rows for q in range(a, b)]
                ),
                "out_cache_loc": torch.tensor(
                    [_loc(s, q) for s, a, b in rows for q in range(a, b)]
                ),
                "extend_logprob_start_lens": [64, 64],
                "global_num_tokens": [128],
                "global_num_tokens_for_logprob": [2],
            }
        )
        runner = SimpleNamespace(
            **vars(fx.runner),
            model=SimpleNamespace(model=SimpleNamespace(engram_hasher=None)),
        )
        replay = _build_replay_batch(
            batch=batch, runner=runner, rows=[(0, 384, 512), (1, 512, 640)]
        )
        fb = self._forward_batch(replay)
        self._mlp_sync(fb)

        self.assertEqual(fb.global_num_tokens_cpu, [256])
        self.assertEqual(fb.extend_num_tokens, 256)
        self.assertEqual(fb.input_ids.shape[0], 256)
        torch.testing.assert_close(
            fb.input_ids,
            torch.tensor(
                [_tok(1, q) for q in range(384, 512)]
                + [_tok(2, q) for q in range(512, 640)]
            ),
        )
        # One sampled row per replayed request, as the scheduler would count it.
        self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [2])


class TestFoldUnderDpAttention(CustomTestCase):
    """Attention DP gathers every rank's row count before the forward, so the
    scheduler must count the replay rows the worker folds in afterwards."""

    DP = 4
    RANK = 1
    PEERS = {0: 4096, 2: 0, 3: 130}  # an 4K extend, an idle rank, a 130-row extend

    def test_fold_rows_are_the_rows_the_fold_adds(self):
        fx = _Fixture()
        self.assertEqual(_replay_spans(fx.batch), HITS)
        self.assertEqual(
            encoder_swa_fold_rows(fx.batch), fx.folded.num_rows - fx.extend_num_tokens
        )
        self.assertEqual(encoder_swa_fold_rows(fx.batch), 128 + 64)

    def test_no_fold_rows_without_window_resets(self):
        fx = _Fixture()
        for reset, mode in (
            (None, ForwardMode.EXTEND),  # flag off
            (
                [False] * len(REQS),
                ForwardMode.EXTEND,
            ),  # later chunks of running requests
            ([True] * len(REQS), ForwardMode.DECODE),
            ([True] * len(REQS), ForwardMode.IDLE),
        ):
            batch = SimpleNamespace(
                **{**vars(fx.batch), "encoder_swa_reset": reset, "forward_mode": mode}
            )
            self.assertEqual(encoder_swa_fold_rows(batch), 0, (reset, mode))

    def _gather(self, batch, *, folds, decoder=False, peer_trims=None):
        """The real scheduler gather on DP rank RANK; the peers report PEERS
        tokens and peer_trims decoder tail trims."""
        peer_trims = peer_trims or {}

        def fake_all_gather_single(output, local, group, **_):
            rows = []
            for r in range(self.DP):
                row = local.clone()
                if r != self.RANK:
                    row[0] = row[1] = self.PEERS[r]
                    row[11] = peer_trims.get(r, 0)  # decoder_trim_rows
                rows.append(row)
            output.copy_(torch.stack(rows).flatten())

        exec_cfg = SimpleNamespace(
            features=SimpleNamespace(enable_decoder_swa_bounded_replay=decoder),
            overlap=SimpleNamespace(enable_two_batch_overlap=False),
        )

        tbo = MagicMock()
        tbo.prepare_all_gather.return_value = (False, ForwardMode.EXTEND.value)
        tbo.compute_output.return_value = (None, ForwardMode.EXTEND)
        parallel = SimpleNamespace(
            num_dp_ranks=self.DP,
            attn_tp_size=1,
            attn_cp_size=1,
            tp_group=SimpleNamespace(
                device_group=object(),
                device="cpu",
                cpu_group=object(),
                active_ranks_cpu=torch.ones(self.DP, dtype=torch.int64),
            ),
        )
        with (
            patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(
                dp_attn, "should_skip_scheduler_all_gather", return_value=False
            ),
            patch.object(dp_attn, "get_parallel", return_value=parallel),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(dp_attn, "get_exec", return_value=exec_cfg),
            patch.object(
                dp_attn, "all_gather_single", side_effect=fake_all_gather_single
            ),
        ):
            return dp_attn.prepare_mlp_sync_batch_raw(
                batch,
                model_runner=SimpleNamespace(
                    prefill_cuda_graph_runner=None,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    model_config=object(),
                    attn_backend=SimpleNamespace(folds_encoder_swa_replay=folds),
                ),
                get_idle_batch=MagicMock(side_effect=AssertionError("has a batch")),
                disable_cuda_graph=True,
                require_mlp_tp_gather=True,
                disable_overlap_schedule=True,
                offload_tags=set(),
            )

    def _scheduled_batch(self):
        fx = _Fixture()
        batch = SimpleNamespace(
            **vars(fx.batch),
            return_logprob=True,
            batch_size=lambda: len(REQS),
            spec_info=None,
            seq_lens_cpu=None,
            dp_balance_stats=None,
        )
        batch.reqs = [
            SimpleNamespace(rid=f"r{i}", **vars(r)) for i, r in enumerate(batch.reqs)
        ]
        return fx, batch

    def test_dp_gather_carries_this_ranks_folded_rows(self):
        """Every rank sizes its DP gather and scatter buffers from this count; it
        must be the folded row count the worker will run, not the scheduled one."""
        fx, batch = self._scheduled_batch()
        out = self._gather(batch, folds=True)
        self.assertEqual(out.global_num_tokens, [4096, fx.folded.num_rows, 0, 130])
        self.assertEqual(fx.folded.num_rows, 282)
        # The fold leaves each request's sampled-row count unchanged.
        self.assertEqual(out.global_num_tokens_for_logprob, [4096, 17, 0, 130])

    def test_dp_gather_keeps_scheduled_rows_without_the_fold(self):
        # Backends that replay in a separate forward keep the scheduled count.
        _, batch = self._scheduled_batch()
        out = self._gather(batch, folds=False)
        self.assertEqual(out.global_num_tokens, [4096, 90, 0, 130])

    def test_folded_batch_checks_its_own_dp_slot(self):
        """The worker's folded batch must match its slot of the gathered counts;
        a scheduled-only count there would desync the DP gather, so it raises."""
        fx = _Fixture()
        good = [4096, 282, 0, 130]
        stale = [4096, 90, 0, 130]
        with (
            patch.object(dp_attention, "dp_gather_width", return_value=self.DP),
            patch.object(dp_attention, "dp_gather_slot", return_value=self.RANK),
        ):
            _check_folded_counts(
                folded=SimpleNamespace(
                    **{**vars(fx.folded.batch), "global_num_tokens": good}
                )
            )
            with self.assertRaisesRegex(ValueError, "does not cover the 282"):
                _check_folded_counts(
                    folded=SimpleNamespace(
                        **{**vars(fx.folded.batch), "global_num_tokens": stale}
                    )
                )


class TestDecoderTailUnderDpAttention(CustomTestCase):
    """Decoder SWA replay drops all but each request's last SWA_WINDOW rows before
    the late layers. Under attention DP the late layers' MoE gather spans every
    rank, so the scheduler gathers each rank's trimmed rows and every rank, trimming
    or not, resizes its gather at the same layer."""

    def test_trim_rows_match_the_backend_tail(self):
        fx = _Fixture()
        folded_lens = fx.folded.batch.extend_lens  # [40, 158, 84]
        _, tail_lens, _ = late_layer_tail_layout(
            extend_lens_cpu=folded_lens,
            seq_lens_cpu=fx.folded_seq_lens(),
            tail_len=SWA_WINDOW,
            device="cpu",
        )
        trim = decoder_swa_trim_rows(fx.batch, folds_encoder=True)
        self.assertEqual(trim, fx.folded.num_rows - sum(tail_lens))
        self.assertEqual(trim, 30)
        # Without the fold the 30-token extend of the hit stays under one window.
        self.assertEqual(decoder_swa_trim_rows(fx.batch, folds_encoder=False), 0)

    def test_decoder_only_trim_matches_backend_tail(self):
        # A cold extend and two prefix-hit extends, without encoder replay.
        lens, prefixes = [4096, 282, 40], [0, 512, 64]
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_lens=lens,
            prefix_lens=prefixes,
            encoder_swa_reset=None,
        )
        _, tail_lens, _ = late_layer_tail_layout(
            extend_lens_cpu=lens,
            seq_lens_cpu=torch.tensor([p + n for p, n in zip(prefixes, lens)]),
            tail_len=SWA_WINDOW,
            device="cpu",
        )
        self.assertEqual(
            decoder_swa_trim_rows(batch, folds_encoder=False),
            sum(lens) - sum(tail_lens),
        )
        self.assertEqual(tail_lens, [128, 128, 40])

    def test_no_trim_outside_plain_extends(self):
        fx = _Fixture()
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.TARGET_VERIFY):
            batch = SimpleNamespace(**{**vars(fx.batch), "forward_mode": mode})
            self.assertEqual(decoder_swa_trim_rows(batch, folds_encoder=True), 0, mode)

    def test_dp_gather_carries_every_ranks_trim(self):
        dp = TestFoldUnderDpAttention()
        _, batch = dp._scheduled_batch()
        out = dp._gather(batch, folds=True, decoder=True, peer_trims={0: 3968, 3: 2})
        self.assertEqual(out.global_num_tokens, [4096, 282, 0, 130])
        self.assertEqual(out.global_decoder_trim_rows, [3968, 30, 0, 2])

    def test_no_resize_when_no_rank_trims(self):
        dp = TestFoldUnderDpAttention()
        _, batch = dp._scheduled_batch()
        batch.encoder_swa_reset = [False] * len(REQS)  # no fold: extends of 40/30/20
        out = dp._gather(batch, folds=True, decoder=True)
        self.assertIsNone(out.global_decoder_trim_rows)
        # One trimming peer is enough for every rank to resize.
        out = dp._gather(batch, folds=True, decoder=True, peer_trims={0: 3968})
        self.assertEqual(out.global_decoder_trim_rows, [3968, 0, 0, 0])

    def _forward_batch(self, rows, trims, rank):
        return SimpleNamespace(
            dp_padding_mode=DpPaddingMode.SUM_LEN,
            global_num_tokens_cpu=list(rows),
            global_num_tokens_padded_cpu=list(rows),
            global_num_tokens_gpu=torch.tensor(rows),
            global_dp_buffer_len=sum(rows),
            global_decoder_trim_rows_cpu=trims,
            dp_local_start_pos=torch.tensor(sum(rows[:rank])),
            dp_local_num_tokens=torch.tensor(rows[rank]),
        )

    def test_late_layers_resize_and_restore_the_dp_gather(self):
        """Ranks 0 and 1 trim; rank 2 is idle and rank 3 decodes 130 rows. All four
        publish the same late sizes, and the exit restores the full ones."""
        rows, trims = [4096, 282, 0, 130], [3968, 30, 0, 0]
        for rank in range(4):
            fb = self._forward_batch(rows, trims, rank)
            published = []
            with patch.object(
                deepseek_v4,
                "set_dp_buffer_len_from_batch",
                side_effect=lambda b: published.append(
                    (list(b.global_num_tokens_padded_cpu), b.global_dp_buffer_len)
                ),
            ):
                saved = deepseek_v4._enter_dp_late_layers(fb)
                self.assertEqual(fb.global_num_tokens_cpu, [128, 252, 0, 130])
                self.assertEqual(fb.global_num_tokens_gpu.tolist(), [128, 252, 0, 130])
                self.assertIsNone(
                    fb.dp_local_start_pos
                )  # recomputed from the late sizes
                deepseek_v4._exit_dp_late_layers(fb, saved)
            self.assertEqual(published, [([128, 252, 0, 130], 510), (rows, 4508)])
            self.assertEqual(fb.global_num_tokens_cpu, rows)
            self.assertEqual(fb.global_num_tokens_gpu.tolist(), rows)
            self.assertEqual(fb.dp_local_num_tokens.item(), rows[rank])

    def test_trimming_rank_late_rows_equal_its_tail(self):
        # Rank 1 runs the fixture's folded extend; its late slot is its tail rows.
        fx = _Fixture()
        _, tail_lens, _ = late_layer_tail_layout(
            extend_lens_cpu=fx.folded.batch.extend_lens,
            seq_lens_cpu=fx.folded_seq_lens(),
            tail_len=SWA_WINDOW,
            device="cpu",
        )
        trim = decoder_swa_trim_rows(fx.batch, folds_encoder=True)
        fb = self._forward_batch([0, fx.folded.num_rows], [0, trim], 1)
        with patch.object(deepseek_v4, "set_dp_buffer_len_from_batch"):
            deepseek_v4._enter_dp_late_layers(fb)
        self.assertEqual(fb.global_num_tokens_cpu[1], sum(tail_lens))


class TestDpReplayLaunchPolicy(CustomTestCase):
    def _validate(self, encoder, decoder, *, cuda=True, tp=4, dp=4, a2a="none"):
        from sglang.srt.arg_groups import deepseek_v4_hook
        from sglang.srt.runtime_context import override_platform

        cfg = SimpleNamespace(
            enable_encoder_swa_bounded_replay=encoder,
            enable_decoder_swa_bounded_replay=decoder,
            tp_size=tp,
            attn_dp_size=dp,
            attn_cp_size=1,
            moe_a2a_backend=a2a,
            dsv4_attn_backend="dsv4",
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(backend="disabled")),
            max_running_requests=64,
            chunked_prefill_size=32768,
            enable_unified_cache_external_linker=False,
            enable_unified_memory=False,
            disaggregation_mode="null",
            enable_mixed_chunk=False,
            enable_lora=False,
            enable_session_radix_cache=False,
            speculative_algorithm=None,
            enable_hisparse=False,
            enable_two_batch_overlap=False,
            pp_size=1,
        )
        model = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"))
        with (
            override_platform(is_cuda=cuda, is_hip=not cuda),
            patch.object(deepseek_v4_hook, "resolving_view", return_value=cfg),
            patch.object(deepseek_v4_hook, "model_config_of", return_value=model),
            patch.object(deepseek_v4_hook, "is_gfx95_supported", return_value=True),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
        ):
            deepseek_v4_hook.validate_deepseek_v41_features(object())

    def test_cuda_flags_together_or_separately(self):
        for encoder, decoder in ((True, False), (False, True), (True, True)):
            with self.subTest(encoder=encoder, decoder=decoder):
                self._validate(encoder, decoder)

    def test_unvalidated_dp_layouts_stay_rejected(self):
        for encoder, decoder in ((True, False), (False, True), (True, True)):
            for layout in (dict(cuda=False), dict(tp=8), dict(a2a="deepep")):
                with self.subTest(encoder=encoder, decoder=decoder, layout=layout):
                    with self.assertRaisesRegex(ValueError, "DP attention"):
                        self._validate(encoder, decoder, **layout)


if __name__ == "__main__":
    unittest.main()
