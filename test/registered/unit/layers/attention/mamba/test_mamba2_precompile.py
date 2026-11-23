"""Unit tests for the Mamba2 prefill precompile sweep layouts (no GPU).

The sweep in ``mamba2_precompile`` only removes live-traffic compiles
(issue #51628) if its batch layouts reach every Triton specialization the
prefill kernels can see. These tests check that coverage on the layouts
alone. The GPU check that nothing compiles after the sweep lives in
``test/registered/kernels/ops/mamba/test_mamba2_prefill_precompile.py``.
The last test checks that Nano VL encoder-only workers skip the sweep.
"""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.mamba.mamba2_precompile import (
    _tracked_requests,
    prefill_layouts,
)
from sglang.srt.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CHUNK_SIZE = 256


def _int_class(value: int) -> str:
    # The integer classes Triton specializes kernel arguments on.
    if value == 1:
        return "one"
    return "div16" if value % 16 == 0 else "other"


def _covered_triples(layouts) -> set:
    # (logical chunks, chunks, tokens) integer classes the layouts produce.
    covered = set()
    for seq_lens in layouts:
        num_tokens = sum(seq_lens)
        num_chunks = -(-num_tokens // CHUNK_SIZE)
        ends = itertools.accumulate(seq_lens[:-1])
        num_logical = num_chunks + sum(end % CHUNK_SIZE != 0 for end in ends)
        covered.add(
            (
                _int_class(num_logical),
                _int_class(num_chunks),
                _int_class(num_tokens),
            )
        )
    return covered


def _reachable_triples(max_tokens: int) -> set:
    # Every (logical chunks, chunks, tokens) class triple a batch of at most
    # max_tokens tokens can produce. Each request boundary off the chunk grid
    # adds one logical chunk, so any count between the two ends is reachable.
    # Past 33 extra logical chunks no new class shows up, so the range stops.
    reachable = set()
    for num_tokens in range(1, max_tokens + 1):
        num_chunks = -(-num_tokens // CHUNK_SIZE)
        off_grid = num_tokens - 1 - (num_tokens - 1) // CHUNK_SIZE
        for num_logical in range(num_chunks, num_chunks + min(off_grid, 33) + 1):
            reachable.add(
                (
                    _int_class(num_logical),
                    _int_class(num_chunks),
                    _int_class(num_tokens),
                )
            )
    return reachable


class TestPrefillLayouts(CustomTestCase):
    def test_cover_integer_classes(self):
        token_classes, chunk_classes, request_classes = set(), set(), set()
        for seq_lens in prefill_layouts(CHUNK_SIZE):
            num_tokens = sum(seq_lens)
            token_classes.add(_int_class(num_tokens))
            chunk_classes.add(_int_class(-(-num_tokens // CHUNK_SIZE)))
            request_classes.add(_int_class(len(seq_lens)))
        expected = {"one", "div16", "other"}
        self.assertEqual(token_classes, expected)
        self.assertEqual(chunk_classes, expected)
        self.assertEqual(request_classes, expected)

    def test_cover_logical_chunk_counts(self):
        # _state_passing_fwd_kernel takes the token count, the chunk count and,
        # with initial states, the logical chunk count as separate integer
        # arguments, so the sweep must cover combinations of their classes, not
        # just each class on its own.
        covered = _covered_triples(prefill_layouts(CHUNK_SIZE))
        # (logical chunks, chunks, tokens): every combination a batch of up to
        # 16 chunks can produce. The last entry needs 16 requests in a single
        # chunk with a token count that is neither 1 nor a multiple of 16,
        # such as 15 1-token requests and one 2-token request.
        required = {
            ("div16", "div16", "div16"),
            ("div16", "div16", "other"),
            ("div16", "one", "div16"),
            ("div16", "other", "div16"),
            ("div16", "other", "other"),
            ("one", "one", "div16"),
            ("one", "one", "one"),
            ("one", "one", "other"),
            ("other", "div16", "div16"),
            ("other", "div16", "other"),
            ("other", "one", "div16"),
            ("other", "one", "other"),
            ("other", "other", "div16"),
            ("other", "other", "other"),
            ("div16", "one", "other"),
        }
        self.assertLessEqual(required, covered)

    def test_cap_at_max_tokens(self):
        # Under a cap (the chunked prefill size), the sweep stays within it
        # and still reaches every class triple a batch under the cap can have.
        for max_tokens in (1, 17, 300, 2048, 4096):
            with self.subTest(max_tokens=max_tokens):
                layouts = prefill_layouts(CHUNK_SIZE, max_tokens=max_tokens)
                self.assertLessEqual(max(map(sum, layouts)), max_tokens)
                self.assertEqual(
                    _covered_triples(layouts), _reachable_triples(max_tokens)
                )

    def test_exercise_state_tracking(self):
        tracked_counts = {
            len(_tracked_requests(seq_lens=seq_lens, chunk_size=CHUNK_SIZE)[0])
            for seq_lens in prefill_layouts(CHUNK_SIZE)
        }
        self.assertLessEqual({1, 2}, tracked_counts)


class TestTrackedRequests(CustomTestCase):
    def test_start_off_grid(self):
        seq_idx, end_locs = _tracked_requests(
            seq_lens=(100, 300, 50, 512), chunk_size=CHUNK_SIZE
        )
        # Request 1 starts at 100 and covers one full chunk; request 2 is
        # shorter than a chunk; request 3 starts at 450 and covers two chunks.
        self.assertEqual(seq_idx, [1, 3])
        self.assertEqual(end_locs, [100 + 256, 450 + 512])


class TestNanoVLPrecompileHook(CustomTestCase):
    # Nano VL only forwards the hook; an encoder-only worker has no
    # language model to warm up, so it must not forward it.
    def _forwarded(self, encoder_only: bool) -> Mock:
        model = SimpleNamespace(
            config=SimpleNamespace(encoder_only=encoder_only),
            language_model=Mock(),
        )
        NemotronH_Nano_VL_V2.precompile_kernels_after_loading(model)
        return model.language_model.precompile_kernels_after_loading

    def test_encoder_only_worker_skips(self):
        self._forwarded(encoder_only=True).assert_not_called()

    def test_language_model_worker_precompiles(self):
        self._forwarded(encoder_only=False).assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
