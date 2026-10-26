"""Encoder-SWA replay folded into the extend forward: row bookkeeping of the
folded batch and the paths that consume it. CPU only."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
    late_layer_tail_layout,
)
from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    DSV4AttnMetadata,
    late_layer_whole_extend,
)
from sglang.srt.managers.mm_utils import pad_mm_tokens
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.dsv41_request_window import window_layout
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.encoder_swa_replay import (
    _build_replay_batch,
    _fold_batch,
    drop_folded_rows,
    folded_replay_rows,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_context, get_parallel, override_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The fake request window's zero row: past every slot the fixtures use.
_ZERO_ROW = 64 * 2 * SWA_WINDOW

_ENV_GATE = "sglang.srt.layers.attention.dsv4.unified_kv_gate"
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
REPLAY_ROWS = 128 + 64


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
            forward_mode=ForwardMode.EXTEND,
            reqs=reqs,
            encoder_swa_reset=[True] * len(REQS),
            req_pool_indices=torch.tensor([s for s, _, _ in REQS]),
            req_pool_indices_cpu=torch.tensor([s for s, _, _ in REQS]),
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
            global_num_tokens=[self.extend_num_tokens + REPLAY_ROWS],
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


def _backend(fx):
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
    backend.model_runner = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.NONE)
    backend.cuda_int32_kwargs = {"dtype": torch.int32, "device": "cpu"}
    backend.token_to_kv_pool = SimpleNamespace(
        request_window=SimpleNamespace(capacity=2 * SWA_WINDOW, zero_row=_ZERO_ROW)
    )
    backend.req_to_token_pool = SimpleNamespace(req_to_token=fx.req_to_token)
    backend.online_c128_mtp = SimpleNamespace(prepare_forward=lambda *_, **__: 0)
    backend.is_dspark_draft = False
    backend.topk = backend.speculative_num_steps = 0
    return backend


def _build_metadata(fx, forward_batch, *, use_prefill_cuda_graph=False):
    with (
        patch.object(DSV4AttnMetadata, "init_compression_metadata"),
        patch.object(DSV4AttnMetadata, "init_flashmla_related"),
    ):
        return _backend(fx)._build_forward_metadata(
            forward_batch, use_prefill_cuda_graph=use_prefill_cuda_graph
        )


class TestFoldedExtendRows(CustomTestCase):
    def test_image_prefix_hit_reembeds_replay_rows_at_absolute_offsets(self):
        fx = _Fixture()
        embedding = torch.nn.Embedding(5000, 3)
        image_id = 42
        mm_inputs = [None]
        expected = []
        scheduled_ids = []
        for i, ((slot, prefix, length), req) in enumerate(zip(REQS, fx.batch.reqs)):
            ids = array("q", req.full_untruncated_fill_ids)
            full_embeddings = embedding(torch.tensor(ids)).detach()
            if i:
                # The long hit begins replay inside an image; the short hit
                # replays both text and image rows from the start of the prompt.
                start, end = (380, 530) if i == 1 else (8, 75)
                features = torch.arange((end - start + 1) * 3).reshape(-1, 3).float()
                item = MultimodalDataItem(
                    modality=Modality.IMAGE,
                    hash=i,
                    offsets=[(start, end)],
                    precomputed_embeddings=features,
                )
                item.set_pad_value()
                mm = MultimodalInputs(mm_items=[item], im_token_id=image_id)
                mm_inputs.append(mm)
                ids[start : end + 1] = array("q", [image_id] * len(features))
                ids = pad_mm_tokens(ids, mm)
                full_embeddings[start : end + 1] = features
            req.full_untruncated_fill_ids = ids
            scheduled_ids.extend(ids[prefix : prefix + length])
            expected.append(full_embeddings[max(0, prefix - 128) : prefix + length])
        fx.batch.input_ids = torch.tensor(scheduled_ids)
        fx.batch.multimodal_inputs = mm_inputs
        original_ids = fx.batch.input_ids.clone()
        folded = _fold_batch(batch=fx.batch, runner=fx.runner, rows=HITS)
        fb = SimpleNamespace(
            mm_inputs=folded.batch.multimodal_inputs,
            extend_prefix_lens_cpu=folded.batch.prefix_lens,
            extend_seq_lens_cpu=folded.batch.extend_lens,
            mm_input_embeds=None,
        )
        model = SimpleNamespace(
            get_input_embeddings=lambda: embedding,
            get_image_feature=MagicMock(side_effect=AssertionError("cached features")),
        )
        actual = DeepseekV4ForCausalLM._prepare_mm_embeddings(
            model, folded.batch.input_ids, fb
        )
        torch.testing.assert_close(actual, torch.cat(expected))
        torch.testing.assert_close(fx.batch.input_ids, original_ids)
        self.assertIs(fb.mm_inputs, mm_inputs)
        model.get_image_feature.assert_not_called()

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
        self.assertEqual(folded_replay_rows(fx.batch), REPLAY_ROWS)
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

    def test_replay_rows_follow_window_resets_only(self):
        """A request that keeps its window (a later chunk) adds no replay rows."""
        fx = _Fixture()
        fx.batch.encoder_swa_reset = [False, False, True]
        self.assertEqual(folded_replay_rows(fx.batch), 64)
        fx.batch.forward_mode = ForwardMode.DECODE
        self.assertEqual(folded_replay_rows(fx.batch), 0)


class TestFoldWithDecoderSwaTail(CustomTestCase):
    """Encoder SWA bounded replay combined with decoder SWA bounded replay:
    the late layers run only each request's last SWA_WINDOW folded rows."""

    def test_tail_keeps_its_own_floor_under_fold(self):
        """A prefix hit with both flags on builds the tail metadata; each tail row
        is floored one window back or at its request's replay start, whichever is
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
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_logprob=False,
        )
        with (
            patch.object(DSV4AttnMetadata, "init_compression_metadata"),
            patch.object(DSV4AttnMetadata, "init_flashmla_related"),
        ):
            metadata = _backend(fx)._build_late_layer_tail_metadata(forward_batch)

        # Tails: the miss's 40 rows, the last 128 of the 158-row hit (from 414,
        # above its replay start 384), and the whole 84-row hit (from 0).
        tail = metadata.late_layer_tail
        self.assertEqual(tail.extend_seq_lens_cpu, [40, 128, 84])
        floor = torch.tensor([0] * 40 + [414] * 128 + [0] * 84)
        req = torch.tensor([1] * 40 + [2] * 128 + [3] * 84)
        expected = window_layout(
            req,
            tail.positions,
            zero_row=_ZERO_ROW,
            capacity=2 * SWA_WINDOW,
            floor=floor,
            num_groups=3,
        )
        layout = metadata.core_attn_metadata.request_window_layout
        torch.testing.assert_close(layout.lengths, expected.lengths)
        torch.testing.assert_close(layout.indices, expected.indices)

    def test_short_tail_floors_one_window_back(self):
        """A final chunk shorter than a window reads the rows the previous chunk's
        tail wrote, so its tail is floored at seq_len - SWA_WINDOW."""
        cpu = torch.device("cpu")
        _, _, floor = late_layer_tail_layout(
            extend_lens_cpu=[40], seq_lens_cpu=[300], tail_len=SWA_WINDOW, device=cpu
        )
        torch.testing.assert_close(floor, torch.full((40,), 172, dtype=torch.int32))

        # A short chunk, a full-window tail, and a short fresh request.
        _, _, floor = late_layer_tail_layout(
            extend_lens_cpu=[40, 200, 50],
            seq_lens_cpu=[300, 200, 50],
            tail_len=SWA_WINDOW,
            device=cpu,
        )
        expected = torch.tensor([172] * 40 + [72] * 128 + [0] * 50, dtype=torch.int32)
        torch.testing.assert_close(floor, expected)

    def test_tail_hidden_rows_map_to_scheduled_rows(self):
        """With DSpark the model returns the tail's hidden rows and their indices
        into the folded extend. The draft indexes scheduled-row cache locations
        with them, so they must come back in scheduled-row space without the
        replayed rows."""
        fx = _Fixture()
        fb = fx.folded.batch
        token_indices, tail_lens, _ = late_layer_tail_layout(
            extend_lens_cpu=list(fb.extend_lens),
            seq_lens_cpu=fx.folded_seq_lens(),
            tail_len=SWA_WINDOW,
            device=torch.device("cpu"),
        )
        self.assertEqual(tail_lens, [40, 128, 84])
        out = SimpleNamespace(
            hidden_states=fb.input_ids[token_indices][:, None].float(),
            hidden_states_token_indices=token_indices,
        )
        drop_folded_rows(logits_output=out, folded=fx.folded, tail_lens=tail_lens)

        # Kept: all 40 miss rows, the hit's 30 new rows, the small hit's 20.
        expected = torch.cat(
            [torch.arange(0, 40), torch.arange(40, 70), torch.arange(70, 90)]
        )
        torch.testing.assert_close(out.hidden_states_token_indices, expected)
        torch.testing.assert_close(
            out.hidden_states, fx.batch.input_ids[expected][:, None].float()
        )

    def test_prompt_logprob_reader_keeps_its_whole_tail(self):
        """A prompt-logprob reader's tail is its whole folded extend, replay rows included."""
        fx = _Fixture()
        fx.batch.extend_logprob_start_lens = [40, 10, 5]
        folded = _fold_batch(batch=fx.batch, runner=fx.runner, rows=HITS)
        fb = folded.batch
        whole = late_layer_whole_extend(
            SimpleNamespace(
                extend_seq_lens_cpu=list(fb.extend_lens),
                extend_logprob_start_lens_cpu=fb.extend_logprob_start_lens,
                return_logprob=True,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            ),
            dspark=True,
        )
        self.assertEqual(whole, [False, True, True])
        token_indices, tail_lens, _ = late_layer_tail_layout(
            extend_lens_cpu=list(fb.extend_lens),
            seq_lens_cpu=fx.folded_seq_lens(),
            tail_len=SWA_WINDOW,
            device=torch.device("cpu"),
            whole_extend=whole,
        )
        self.assertEqual(tail_lens, [40, 158, 84])
        out = SimpleNamespace(
            hidden_states=fb.input_ids[token_indices][:, None].float(),
            hidden_states_token_indices=token_indices,
        )
        drop_folded_rows(logits_output=out, folded=folded, tail_lens=tail_lens)

        torch.testing.assert_close(out.hidden_states_token_indices, torch.arange(90))
        torch.testing.assert_close(
            out.hidden_states, fx.batch.input_ids[:, None].float()
        )


def _extend_batch(fx):
    """The folded extend's forward batch, as the backend's metadata reads it."""
    fb = fx.folded.batch
    seq_lens = fx.folded_seq_lens()
    lens = torch.tensor(fb.extend_lens)
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(fb.extend_lens),
        spec_info=None,
        max_seq_len_override=None,
        extend_seq_lens=lens,
        extend_seq_lens_cpu=list(fb.extend_lens),
        extend_start_loc=torch.cumsum(lens, 0) - lens,
        seq_lens_cpu=torch.tensor(seq_lens),
        seq_lens=torch.tensor(seq_lens),
        req_pool_indices=fb.req_pool_indices,
        out_cache_loc=fb.out_cache_loc,
        positions=fx.folded_positions(),
        encoder_swa_replay=False,
        encoder_swa_row_floor=fx.folded.row_floor,
        encoder_swa_compress_rows=fx.folded.keep_rows,
    )


class TestFoldedReplayCompressorWrites(CustomTestCase):
    """Folded replay rows rebuild only their SWA window: their main KV, compressed
    groups and index keys are cached, so the compressors write only the new rows."""

    def _expected_out_loc(self, fx):
        """Replay rows on the padding slot, new rows on their own KV slots."""
        fb = fx.folded.batch
        replayed = torch.cat(
            [
                torch.arange(start, start + n) < prefix
                for start, n, (_, prefix, _) in zip(
                    fb.prefix_lens, fb.extend_lens, REQS
                )
            ]
        )
        return fb.out_cache_loc.masked_fill(replayed, 0)

    def test_extend_metadata_puts_replay_rows_on_the_padding_slot(self):
        fx = _Fixture()
        metadata = _build_metadata(fx, _extend_batch(fx))

        raw_out_loc = metadata.core_attn_metadata.raw_out_loc
        self.assertEqual(int((raw_out_loc == 0).sum()), REPLAY_ROWS)
        torch.testing.assert_close(raw_out_loc, self._expected_out_loc(fx))

    def test_unfused_compressor_writes_each_row_at_its_own_slot(self):
        fx = _Fixture()
        backend = _backend(fx)
        out_loc = self._expected_out_loc(fx)
        backend.forward_metadata = SimpleNamespace(
            core_metadata=SimpleNamespace(c1_out_loc=out_loc, raw_out_loc=out_loc)
        )
        written = []
        backend._low_ratio_write_group = lambda layer, kv, slots, pos, **_: (
            written.append((kv, slots))
        )
        layer = SimpleNamespace(
            compress_ratio=1,
            layer_id=0,
            compressor=SimpleNamespace(
                use_fused_compress=False, project=lambda x: (x, None)
            ),
        )
        fb = fx.folded.batch
        req = fb.req_pool_indices.repeat_interleave(torch.tensor(fb.extend_lens))
        rows = torch.arange(out_loc.shape[0], dtype=torch.float32)[:, None]
        forward_batch = _extend_batch(fx)
        backend._low_ratio_compress(
            layer, rows, req, forward_batch.positions, forward_batch
        )

        ((kv, slots),) = written
        torch.testing.assert_close(kv, rows)
        torch.testing.assert_close(slots, out_loc)


class TestFoldedReplayUnderPrefillGraph(CustomTestCase):
    """The breakable prefill graph replays a folded batch padded to its captured
    bucket and refreshes the captured metadata in place before each replay."""

    PAD = 32

    def _graph_batch(self, fx, *, compress_rows):
        fb = _extend_batch(fx)
        pad = torch.zeros(self.PAD, dtype=fb.out_cache_loc.dtype)
        fb.out_cache_loc = torch.cat([fb.out_cache_loc, pad])
        fb.positions = torch.cat([fb.positions, pad])
        fb.encoder_swa_compress_rows = compress_rows
        return fb

    def test_padding_rows_get_no_window_and_commit_nowhere(self):
        fx = _Fixture()
        live = sum(fx.folded.batch.extend_lens)
        graph = _build_metadata(
            fx,
            self._graph_batch(fx, compress_rows=fx.folded.keep_rows),
            use_prefill_cuda_graph=True,
        ).core_attn_metadata.request_window_layout
        eager = _build_metadata(
            fx, _extend_batch(fx)
        ).core_attn_metadata.request_window_layout

        self.assertEqual(graph.pos.numel(), live + self.PAD)
        # The graph fixes its window groups, so workspace offsets differ from
        # eager; each live row still sees the same window and commits the same.
        live_rows = graph.leading_rows(live)
        torch.testing.assert_close(live_rows.lengths, eager.lengths)
        torch.testing.assert_close(live_rows.commit_mask, eager.commit_mask)
        torch.testing.assert_close(
            (live_rows.indices >= 0).sum(-1), (eager.indices >= 0).sum(-1)
        )
        self.assertEqual(live_rows.write_loc.unique().numel(), live)
        self.assertFalse(bool(graph.commit_mask[live:].any()))
        self.assertTrue(bool((graph.lengths[live:] == 0).all()))
        self.assertTrue(bool((graph.indices[live:] == -1).all()))
        self.assertTrue(
            bool((graph.write_loc[live:] > graph.write_loc[:live].max()).all())
        )

    def test_graph_slots_do_not_alias_the_static_out_cache_loc(self):
        """A capture has no replay rows; the replay that has them refreshes the
        captured slots in place, which must leave the batch's input untouched."""
        fx = _Fixture()
        fb = self._graph_batch(fx, compress_rows=None)
        raw = _build_metadata(
            fx, fb, use_prefill_cuda_graph=True
        ).core_attn_metadata.raw_out_loc
        self.assertNotEqual(
            raw.untyped_storage().data_ptr(),
            fb.out_cache_loc.untyped_storage().data_ptr(),
        )
        torch.testing.assert_close(raw, fb.out_cache_loc)


class TestEncoderReplayPrefillGraphGate(CustomTestCase):
    def _validate(self, prefill_backend):
        args = ServerArgs(
            model_path="local-deepseek-v41",
            attention_backend="dsv4",
            enable_encoder_swa_bounded_replay=True,
            enable_decoder_swa_bounded_replay=False,
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=prefill_backend)
            ),
        )
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        )
        with patch(f"{_ENV_GATE}.is_unified_kv_triton", return_value=False):
            deepseek_v4_hook.validate_deepseek_v41_features(args)

    @override_platform(is_cuda=True, is_hip=False)
    def test_cuda_accepts_only_the_breakable_prefill_graph(self):
        self._validate(Backend.BREAKABLE)
        for backend in (Backend.FULL, Backend.TC_PIECEWISE):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValueError, "other than breakable"),
            ):
                self._validate(backend)

    @override_platform(is_cuda=False, is_hip=True)
    def test_rocm_rejects_the_prefill_graph(self):
        with (
            patch.object(deepseek_v4_hook, "is_gfx95_supported", return_value=True),
            self.assertRaisesRegex(ValueError, "prefill CUDA graph on ROCm"),
        ):
            self._validate(Backend.BREAKABLE)


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
        """The scheduler's gathered count includes the replay rows, so the folded
        batch pads to exactly its own rows; logprob rows keep their count."""
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


class _PastEncoderReplayChecks(Exception):
    """Validation read a field the encoder replay checks do not need."""


class _View(SimpleNamespace):
    def __getattr__(self, name):
        raise _PastEncoderReplayChecks(name)


class TestEncoderReplayChunkFloor(CustomTestCase):
    def _validate(self, chunked_prefill_size):
        view = _View(
            enable_encoder_swa_bounded_replay=True,
            max_running_requests=8,
            chunked_prefill_size=chunked_prefill_size,
            page_size=256,
        )
        with patch.object(deepseek_v4_hook, "resolving_view", return_value=view):
            deepseek_v4_hook.validate_deepseek_v41_encoder_replay_budget(object())

    def test_chunk_holds_the_replay_rows_and_a_page(self):
        self._validate(384)
        with self.assertRaisesRegex(ValueError, "at least 384 chunked prefill"):
            self._validate(383)


class TestEncoderReplayDpStartupCheck(CustomTestCase):
    """ROCm replays hits in a forward of their own, so it must refuse attention DP."""

    def _validate(self, *, is_hip, attn_dp_size):
        from sglang.srt.arg_groups import deepseek_v4_hook
        from sglang.srt.model_executor.cuda_graph_config import Backend
        from sglang.srt.runtime_context import override_platform

        view = _View(
            dsv4_attn_backend="dsv4",
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED)
            ),
            enable_encoder_swa_bounded_replay=True,
            attn_cp_size=1,
            attn_dp_size=attn_dp_size,
            ep_join_mode=None,
            enable_unified_cache_external_linker=False,
            enable_unified_memory=False,
            disaggregation_mode="null",
            enable_mixed_chunk=False,
            enable_lora=False,
            enable_session_radix_cache=False,
        )
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        )
        with (
            override_platform(
                is_cuda=not is_hip, is_hip=is_hip, is_npu=False, is_musa=False
            ),
            patch.object(deepseek_v4_hook, "is_gfx95_supported", return_value=True),
            patch.object(deepseek_v4_hook, "resolving_view", return_value=view),
            patch.object(
                deepseek_v4_hook, "model_config_of", return_value=model_config
            ),
        ):
            deepseek_v4_hook.validate_deepseek_v41_features(object())

    def test_rocm_rejects_attention_dp(self):
        with self.assertRaisesRegex(ValueError, "DP attention on ROCm"):
            self._validate(is_hip=True, attn_dp_size=4)

    def test_rocm_without_dp_and_cuda_with_dp_pass(self):
        for is_hip, attn_dp_size in ((True, 1), (False, 4)):
            with self.subTest(is_hip=is_hip, attn_dp_size=attn_dp_size):
                with self.assertRaises(_PastEncoderReplayChecks):
                    self._validate(is_hip=is_hip, attn_dp_size=attn_dp_size)


if __name__ == "__main__":
    unittest.main()
