import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.overlap_utils import decide_needs_cpu_seq_lens

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Backend:
    def __init__(
        self,
        *,
        needs_cpu_seq_lens: bool = True,
        supports_ngram_gpu_only_seq_lens: bool = False,
    ):
        self.needs_cpu_seq_lens = needs_cpu_seq_lens
        self.supports_ngram_gpu_only_seq_lens = supports_ngram_gpu_only_seq_lens


def _server_args(*, disable_overlap=False, enable_tbo=False):
    return SimpleNamespace(
        speculative_algorithm="NGRAM",
        disable_overlap_schedule=disable_overlap,
        enable_two_batch_overlap=enable_tbo,
    )


class TestOverlapSeqLensPolicy(unittest.TestCase):
    def test_ngram_without_precompute_keeps_cpu_lengths(self):
        backend = _Backend(needs_cpu_seq_lens=False)
        with envs.SGLANG_ENABLE_NGRAM_PRECOMPUTE.override(False):
            self.assertTrue(decide_needs_cpu_seq_lens(_server_args(), [backend]))

    def test_ngram_precompute_uses_gpu_only_lengths(self):
        backend = _Backend(supports_ngram_gpu_only_seq_lens=True)
        with envs.SGLANG_ENABLE_NGRAM_PRECOMPUTE.override(True):
            self.assertFalse(decide_needs_cpu_seq_lens(_server_args(), [backend]))

    def test_ngram_precompute_rejects_unsupported_backend(self):
        with envs.SGLANG_ENABLE_NGRAM_PRECOMPUTE.override(True):
            with self.assertRaisesRegex(ValueError, "_Backend"):
                decide_needs_cpu_seq_lens(_server_args(), [_Backend()])

    def test_ngram_precompute_rejects_disabled_overlap(self):
        backend = _Backend(supports_ngram_gpu_only_seq_lens=True)
        with envs.SGLANG_ENABLE_NGRAM_PRECOMPUTE.override(True):
            with self.assertRaisesRegex(ValueError, "overlap scheduling"):
                decide_needs_cpu_seq_lens(_server_args(disable_overlap=True), [backend])

    def test_ngram_precompute_rejects_tbo(self):
        backend = _Backend(supports_ngram_gpu_only_seq_lens=True)
        with envs.SGLANG_ENABLE_NGRAM_PRECOMPUTE.override(True):
            with self.assertRaisesRegex(ValueError, "two-batch overlap"):
                decide_needs_cpu_seq_lens(_server_args(enable_tbo=True), [backend])


if __name__ == "__main__":
    unittest.main()
