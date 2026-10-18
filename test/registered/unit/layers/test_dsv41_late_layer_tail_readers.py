"""Decoder SWA bounded replay: requests that read prompt rows (prompt logprobs,
every hidden state) run their whole extend through the late layers. CPU only."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.models.deepseek_v4 import DeepseekV4Model
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


if __name__ == "__main__":
    unittest.main()
