"""Unit tests for pinned weight staging: CPU checks, plus GPU copies when one is present."""

from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")

import threading
import unittest

import torch

from sglang.srt.layers.quantization.online_quantization import CopyNumelCounter
from sglang.srt.layers.utils.pinned_weight_staging import (
    MIN_STAGED_BYTES,
    PinnedWeightStager,
    staged_copy_,
    staging_weight_copies,
)
from sglang.test.test_utils import CustomTestCase


def _stager(num_threads: int = 1) -> PinnedWeightStager:
    # A 1 MiB buffer splits every staged copy below into several chunks, so
    # both buffers are refilled behind their events.
    return PinnedWeightStager(num_threads, buffer_bytes=MIN_STAGED_BYTES)


def _checkpoint(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(0)).to(dtype)


def _gpu(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.zeros(*shape, dtype=dtype, device="cuda")


def _row_half():
    """`_load_w13`: a TP rank's rows into one half of the fused w13."""
    w13 = _gpu(2048, 1024)
    return w13.narrow(0, 1024, 1024), _checkpoint(2048, 1024).narrow(0, 1024, 1024)


def _strided_source():
    """`_load_w2`: a row-parallel shard is a dim-1 slice of the checkpoint."""
    return _gpu(1024, 512), _checkpoint(1024, 2048).narrow(1, 512, 512)


def _padded_destination():
    """A destination narrowed for padding is not contiguous."""
    return _gpu(1024, 1536).narrow(1, 0, 1024), _checkpoint(1024, 1024)


STAGED_CASES = {
    "contiguous": lambda: (_gpu(2048, 1024), _checkpoint(2048, 1024)),
    "row_half": _row_half,
    "strided_source": _strided_source,
    "padded_destination": _padded_destination,
    "dtype_conversion": lambda: (
        _gpu(1024, 1024),
        _checkpoint(1024, 1024, dtype=torch.float32),
    ),
    "one_dim": lambda: (_gpu(1 << 20), _checkpoint(1 << 20)),
}

PASSTHROUGH_CASES = {
    "below_threshold": lambda: (_gpu(256, 1024), _checkpoint(256, 1024)),
    "pinned_source": lambda: (
        _gpu(2048, 1024),
        _checkpoint(2048, 1024).pin_memory(),
    ),
    "broadcast": lambda: (_gpu(2, 512, 1024), _checkpoint(512, 1024)),
    "row_larger_than_buffer": lambda: (_gpu(2, 1 << 20), _checkpoint(2, 1 << 20)),
}


class TestPinnedWeightStagingCpu(CustomTestCase):
    def test_threads_are_raised_only_while_staging(self):
        before = torch.get_num_threads()
        src = _checkpoint(2048, 1024)
        dst = torch.zeros_like(src)
        stager = _stager(num_threads=before + 1)
        with staging_weight_copies(stager):
            self.assertEqual(torch.get_num_threads(), before + 1)
            staged_copy_(dst, src)
        self.assertEqual(torch.get_num_threads(), before)
        # A copy to the CPU is not staged.
        self.assertEqual(stager.staged_bytes, 0)
        self.assertTrue(torch.equal(dst, src))


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA or ROCm GPU")
class TestPinnedWeightStagingGpu(CustomTestCase):
    def test_staged_copy_matches_plain_copy(self):
        cases = [(name, True) for name in STAGED_CASES]
        cases += [(name, False) for name in PASSTHROUGH_CASES]
        for name, staged in cases:
            with self.subTest(case=name):
                make = {**STAGED_CASES, **PASSTHROUGH_CASES}[name]
                expected_dst, src = make()
                expected_dst.copy_(src)
                dst, src = make()
                stager = _stager()
                with staging_weight_copies(stager):
                    staged_copy_(dst, src)
                self.assertEqual(stager.staged_bytes, src.nbytes if staged else 0)
                self.assertTrue(torch.equal(dst, expected_dst))

    def test_copies_on_other_threads_are_not_staged(self):
        dst, src = STAGED_CASES["contiguous"]()
        stager = _stager()
        with staging_weight_copies(stager):
            worker = threading.Thread(target=staged_copy_, args=(dst, src))
            worker.start()
            worker.join()
        self.assertEqual(stager.staged_bytes, 0)
        self.assertTrue(torch.equal(dst.cpu(), src))

    def test_copies_under_a_dispatch_mode_are_not_staged(self):
        # Online quantization counts copied elements with CopyNumelCounter;
        # staging would add copies to the count.
        dst, src = STAGED_CASES["contiguous"]()
        stager = _stager()
        counter = CopyNumelCounter()
        with staging_weight_copies(stager), counter:
            staged_copy_(dst, src)
        self.assertEqual(stager.staged_bytes, 0)
        self.assertEqual(counter.copied_numel, src.numel())
        self.assertTrue(torch.equal(dst.cpu(), src))


if __name__ == "__main__":
    unittest.main()
