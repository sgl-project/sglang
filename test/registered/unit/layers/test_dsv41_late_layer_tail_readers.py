"""Decoder SWA bounded replay: the rows the late layers run (whole extends for
requests that read prompt rows) and how DP ranks learn each other's. CPU only."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    late_layer_tail_layout,
)
from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    late_layer_whole_extend,
)
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.models.deepseek_common import decoder_swa_bounded_replay
from sglang.srt.models.deepseek_common.decoder_swa_bounded_replay import (
    late_layer_dp_counts,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4Model
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CPU = torch.device("cpu")


def _batch(extend_lens, starts=None, capture=CaptureHiddenMode.NULL):
    return SimpleNamespace(
        extend_seq_lens_cpu=extend_lens,
        extend_logprob_start_lens_cpu=starts,
        return_logprob=starts is not None,
        capture_hidden_mode=capture,
    )


class TestWholeExtendRequests(CustomTestCase):
    def test_no_prompt_reader_keeps_the_window_tail(self):
        # No logprobs and output-only logprobs (start == extend) read no
        # prompt row.
        for batch in (
            _batch([300, 50]),
            _batch([300, 50], starts=[300, 50]),
            _batch([300], capture=CaptureHiddenMode.LAST),
        ):
            self.assertIsNone(late_layer_whole_extend(batch, dspark=False))

    def test_dspark_capture_reads_the_tail_only(self):
        # DSpark's target prefill, and its graph capture batches, capture FULL
        # hidden states, but its draft reads tail rows only.
        batch = _batch([300], capture=CaptureHiddenMode.FULL)
        self.assertIsNone(late_layer_whole_extend(batch, dspark=True))

    def test_any_prompt_logprob_reads_the_whole_extend(self):
        # Inside the last window too, and in a prompt shorter than a window:
        # those rows would otherwise get the window floor.
        batch = _batch([300, 300, 300, 50], starts=[0, 300, 300 - SWA_WINDOW, 0])
        whole = late_layer_whole_extend(batch, dspark=False)
        self.assertEqual(whole, [True, False, True, True])

    def test_full_hidden_capture_reads_every_row(self):
        batch = _batch([300, 50], capture=CaptureHiddenMode.FULL)
        self.assertEqual(late_layer_whole_extend(batch, dspark=False), [True, True])


class TestWholeExtendLayout(CustomTestCase):
    def test_other_requests_match_the_window_layout(self):
        extend_lens, seq_lens = [200, 300, 50], [300, 600, 50]
        token_indices, tail_lens, floor = late_layer_tail_layout(
            extend_lens_cpu=extend_lens,
            seq_lens_cpu=seq_lens,
            tail_len=SWA_WINDOW,
            device=CPU,
            whole_extend=[False, True, False],
        )
        base_indices, base_lens, base_floor = late_layer_tail_layout(
            extend_lens_cpu=extend_lens,
            seq_lens_cpu=seq_lens,
            tail_len=SWA_WINDOW,
            device=CPU,
        )
        self.assertEqual(tail_lens, [128, 300, 50])
        self.assertEqual(base_lens, [128, 128, 50])
        # The first and last requests keep the window layout row for row.
        torch.testing.assert_close(token_indices[:128], base_indices[:128])
        torch.testing.assert_close(floor[:128], base_floor[:128])
        torch.testing.assert_close(token_indices[428:], base_indices[256:])
        torch.testing.assert_close(floor[428:], base_floor[256:])
        # The whole request: every row of its extend, each with its plain window
        # (a floor a window below the extend start).
        torch.testing.assert_close(token_indices[128:428], torch.arange(200, 500))
        expected = torch.full((300,), 600 - 300 - SWA_WINDOW, dtype=torch.int32)
        torch.testing.assert_close(floor[128:428], expected)

    def test_short_whole_extend_keeps_its_plain_window(self):
        # A chunk shorter than a window is floored one window back otherwise.
        _, tail_lens, floor = late_layer_tail_layout(
            extend_lens_cpu=[40],
            seq_lens_cpu=[300],
            tail_len=SWA_WINDOW,
            device=CPU,
            whole_extend=[True],
        )
        self.assertEqual(tail_lens, [40])
        expected = torch.full((40,), 300 - 40 - SWA_WINDOW, dtype=torch.int32)
        torch.testing.assert_close(floor, expected)


class TestPrefillGraphTailCapacity(CustomTestCase):
    """The backend's graph check and the DP vote's view of it agree: no DP rank
    replays the prefill graph while another runs the step eagerly."""

    def _backend(self, capacity):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.enable_decoder_swa_bounded_replay = True
        backend._prefill_graph_tail_capacity = lambda: capacity
        backend.model_runner = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.NONE)
        return backend

    def _both(self, extend_lens, capacity, starts=None, full=False):
        hidden = CaptureHiddenMode.FULL if full else CaptureHiddenMode.NULL
        forward_batch = _batch(extend_lens, starts, capture=hidden)
        forward_batch.seq_lens_cpu = torch.tensor([400] * len(extend_lens))
        schedule_batch = SimpleNamespace(
            extend_lens=extend_lens,
            extend_logprob_start_lens=starts,
            return_logprob=starts is not None,
            reqs=[SimpleNamespace(return_hidden_states_mode=hidden)] * len(extend_lens),
        )
        backend = self._backend(capacity)
        with (
            patch.object(
                deepseek_v4_backend, "_prefill_graph_max_seq_len", return_value=16384
            ),
            patch.object(
                deepseek_v4_backend,
                "get_server_return_hidden_states_mode",
                return_value=CaptureHiddenMode.NULL,
            ),
        ):
            can_run = backend.can_run_prefill_cuda_graph(forward_batch)
            vote = backend.vote_prefill_cuda_graph(schedule_batch)
        self.assertEqual(can_run, vote)
        return vote

    def test_whole_extend_counts_against_the_tail_block(self):
        self.assertTrue(self._both([300, 300], capacity=256))
        # A prompt-logprob request's tail no longer fits: run the step eagerly.
        self.assertFalse(self._both([300, 300], 256, starts=[0, 300]))
        self.assertTrue(self._both([300, 300], 428, starts=[0, 300]))
        self.assertFalse(self._both([300], 256, full=True))


class TestTailReadersCheck(CustomTestCase):
    def _check(self, batch, tail_lens, dspark=False):
        model = SimpleNamespace(dspark_layers_to_capture=[1] if dspark else None)
        tail = SimpleNamespace(extend_seq_lens_cpu=tail_lens)
        DeepseekV4Model._check_late_layer_tail_readers(model, batch, tail)

    def test_tail_covering_the_readers_passes(self):
        self._check(_batch([300, 50], starts=[300, 50]), [128, 50])
        self._check(_batch([300, 50], starts=[200, 0]), [128, 50])
        self._check(_batch([300, 50], starts=[0, 0]), [300, 50])
        self._check(_batch([300], capture=CaptureHiddenMode.FULL), [300])
        self._check(_batch([300], capture=CaptureHiddenMode.FULL), [128], dspark=True)

    def test_tail_missing_rows_raises(self):
        with self.assertRaises(ValueError):
            self._check(_batch([300], starts=[0]), [128])
        with self.assertRaises(ValueError):
            self._check(_batch([300], capture=CaptureHiddenMode.FULL), [128])


class TestLateLayerRowsUnderDpAttention(CustomTestCase):
    """A rank replaying the prefill graph never runs the model's late-layer code,
    so its eager peers learn its late-layer rows before the forward."""

    def setUp(self):
        override = get_context().override_server_args(
            tp_size=2,
            dp_size=2,
            enable_dp_attention=True,
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(bs=[])),
        )
        override.install()
        self.addCleanup(override.restore)

    def _dp_rank(self, rank):
        return get_parallel().override(
            attn_tp_size=1,
            attn_dp_size=2,
            attn_dp_rank=rank,
            tp_group=SimpleNamespace(cpu_group=None),
        )

    def test_rows_are_exchanged_before_the_forward(self):
        # Rank 0 prefills 300 rows; rank 1 is idle.
        n = 300
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.arange(n),
            req_pool_indices=torch.tensor([1]),
            seq_lens=torch.tensor([n]),
            out_cache_loc=torch.arange(n),
            seq_lens_sum=n,
            positions=torch.arange(n),
            seq_lens_cpu=torch.tensor([n]),
            extend_num_tokens=n,
            extend_seq_lens=torch.tensor([n]),
            extend_seq_lens_cpu=[n],
            extend_prefix_lens_cpu=[0],
            is_extend_in_batch=True,
        )
        fb.init_mlp_sync_metadata(
            SimpleNamespace(
                global_num_tokens=[n, 0],
                global_num_tokens_for_logprob=[1, 0],
                dp_spec_prefill_coordination_applied=False,
                draft_global_num_tokens=None,
                can_run_decode_cuda_graph=False,
                can_run_dp_draft_cuda_graph=False,
            ),
            CPU,
        )
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.enable_decoder_swa_bounded_replay = True
        backend.model_runner = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.NONE)
        backend.get_cpu_graph_seq_len_fill_value = lambda: 1
        runner = MagicMock(attn_backend=backend, is_draft_worker=False)
        runner.attn_tp_sequence_sharded.return_value = False

        def idle_peer(out, local, group):
            out.copy_(torch.cat([local, local.new_zeros(1)]))

        with (
            self._dp_rank(0),
            # CPU runners have no driver to pin the synced token counts with.
            patch.object(forward_batch_info, "_is_cpu", True),
            patch.object(
                forward_batch_info, "all_gather_single", side_effect=idle_peer
            ),
        ):
            fb.prepare_mlp_sync_batch(runner)
        self.assertEqual(fb.global_late_layer_num_tokens_cpu, [SWA_WINDOW, 0])

        for rank, local_rows in ((0, SWA_WINDOW), (1, 0)):
            with (
                self.subTest(rank=rank),
                self._dp_rank(rank),
                patch.object(
                    decoder_swa_bounded_replay,
                    "all_gather_single",
                    side_effect=AssertionError("a collective inside the forward"),
                ),
            ):
                counts = late_layer_dp_counts(fb, local_rows, in_graph=False)
            self.assertEqual(counts, [SWA_WINDOW, 0])


if __name__ == "__main__":
    unittest.main()
