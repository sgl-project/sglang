"""Decoder SWA bounded replay with SGLANG_OPT_DSV41_PROMPT_WINDOW_TAIL: a prefill
chunk runs the late layers over its rows in the prompt's last window only, so any
chunking of a prompt computes exactly the rows and floors of the unchunked prompt
there, plus one placeholder row per chunk outside it. CPU only."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
    late_layer_tail_layout,
    late_layer_tail_lens,
)
from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    late_layer_num_tokens,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    prompt_rest_lens_of,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CPU = torch.device("cpu")


def _layout(extend_lens, seq_lens, rest=None, whole_extend=None):
    return late_layer_tail_layout(
        extend_lens_cpu=extend_lens,
        seq_lens_cpu=seq_lens,
        tail_len=SWA_WINDOW,
        device=CPU,
        whole_extend=whole_extend,
        prompt_rest_lens=rest,
    )


def _chunked_rows(prompt_len, chunks):
    """(absolute position, floor) of every tail row over a prompt's chunks, each
    chunk one single-request step."""
    rows, start = [], 0
    for n in chunks:
        end = start + n
        token_indices, _, floor = _layout([n], [end], rest=[prompt_len - end])
        rows += list(zip((start + token_indices).tolist(), floor.tolist()))
        start = end
    assert start == prompt_len
    return rows


class TestTailLens(CustomTestCase):
    def test_final_chunks_keep_the_window_tail(self):
        extend_lens = [300, 50, 128]
        self.assertEqual(
            late_layer_tail_lens(extend_lens, SWA_WINDOW, prompt_rest_lens=[0, 0, 0]),
            late_layer_tail_lens(extend_lens, SWA_WINDOW),
        )

    def test_earlier_chunks_keep_their_rows_in_the_prompt_window(self):
        # Far from the end: one row. Straddling the window: the rows inside it.
        # Inside the window: the whole chunk.
        self.assertEqual(
            late_layer_tail_lens(
                [32768, 300, 40, 300],
                SWA_WINDOW,
                prompt_rest_lens=[5000, 50, 20, SWA_WINDOW],
            ),
            [1, SWA_WINDOW - 50, 40, 1],
        )

    def test_whole_extend_wins(self):
        self.assertEqual(
            late_layer_tail_lens(
                [300, 300], SWA_WINDOW, [True, False], prompt_rest_lens=[5000, 5000]
            ),
            [300, SWA_WINDOW],
        )
        # No whole request: the prompt window applies, as in the layout.
        self.assertEqual(
            late_layer_tail_lens([300], SWA_WINDOW, [False], prompt_rest_lens=[5000]),
            [1],
        )


class TestTailLayout(CustomTestCase):
    def test_final_chunks_match_the_window_layout(self):
        # Negative control: final chunks (nothing left) change no row or floor.
        for extend_lens, seq_lens in (([300], [900]), ([300, 50, 40], [900, 50, 400])):
            with self.subTest(extend_lens=extend_lens):
                got = _layout(extend_lens, seq_lens, rest=[0] * len(extend_lens))
                want = _layout(extend_lens, seq_lens)
                torch.testing.assert_close(got[0], want[0])
                self.assertEqual(got[1], want[1])
                torch.testing.assert_close(got[2], want[2])

    def test_placeholder_row_reads_only_itself(self):
        token_indices, tail_lens, floor = _layout([32768], [65536], rest=[70000])
        self.assertEqual(tail_lens, [1])
        self.assertEqual(token_indices.tolist(), [32767])
        self.assertEqual(floor.tolist(), [65535])

    def test_batched_layout_matches_single_requests(self):
        extend_lens = [32768, 300, 40, 200]
        seq_lens = [40000, 900, 60, 200]
        rest = [9000, 50, 20, 0]
        token_indices, tail_lens, floor = _layout(extend_lens, seq_lens, rest)
        want_indices, want_floor, start = [], [], 0
        for n, s, r in zip(extend_lens, seq_lens, rest):
            idx, _, fl = _layout([n], [s], rest=[r])
            want_indices.append(start + idx)
            want_floor.append(fl)
            start += n
        self.assertEqual(tail_lens, [1, SWA_WINDOW - 50, 40, SWA_WINDOW])
        torch.testing.assert_close(token_indices, torch.cat(want_indices))
        torch.testing.assert_close(floor, torch.cat(want_floor))


class TestChunkingInvariance(CustomTestCase):
    """The rows in the prompt's last window, and the floor each reads from, do
    not depend on how the prompt was chunked."""

    def test_any_chunking_computes_the_unchunked_window(self):
        prompt_len = 1000
        whole = _chunked_rows(prompt_len, [prompt_len])
        floor = prompt_len - SWA_WINDOW
        self.assertEqual(whole, [(p, floor) for p in range(floor, prompt_len)])
        for chunks in (
            [400, 400, 200],
            [900, 60, 40],
            [800, 100, 60, 40],
            [872, 1, 127],
            [700, 200, 100],
        ):
            with self.subTest(chunks=chunks):
                rows = _chunked_rows(prompt_len, chunks)
                window = [(p, f) for p, f in rows if p >= floor]
                self.assertEqual(window, whole)
                # Placeholders sit below the window and read only their own row.
                for p, f in rows:
                    if p < floor:
                        self.assertEqual(f, p)

    def test_window_rows_never_read_below_the_window(self):
        rows = _chunked_rows(1000, [900, 60, 40])
        self.assertTrue(all(f == 1000 - SWA_WINDOW for p, f in rows if p >= 872))


class TestPromptRestLens(CustomTestCase):
    def test_rest_from_the_batch_ends(self):
        prompt = SimpleNamespace(full_untruncated_fill_ids=list(range(1000)))
        decode = SimpleNamespace(full_untruncated_fill_ids=list(range(70)))
        # A chunk ending at 400, a replay batch ending at its prefix 128, and a
        # decode row carried in the extend.
        self.assertEqual(
            prompt_rest_lens_of([prompt, prompt, decode], [400, 128, 70], {id(decode)}),
            [600, 872, 0],
        )

    def test_rows_counted_for_dp_match_the_layout(self):
        extend_lens, rest = [32768, 300, 40], [9000, 50, 20]
        fb = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=extend_lens,
            extend_logprob_start_lens_cpu=None,
            return_logprob=False,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            extend_prompt_rest_lens_cpu=rest,
        )
        _, tail_lens, _ = _layout(extend_lens, [40000, 900, 60], rest)
        self.assertEqual(late_layer_num_tokens(fb, dspark=True), sum(tail_lens))
        fb.extend_prompt_rest_lens_cpu = None
        self.assertEqual(late_layer_num_tokens(fb, dspark=True), 2 * SWA_WINDOW + 40)


if __name__ == "__main__":
    unittest.main()
