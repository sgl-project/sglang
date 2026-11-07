"""The HIP FP4 prefill schedule compiles no Triton variant while serving: after warm_prefill_schedule, prefill calls
of any row count, table width and row-metadata offset reuse the warmed variants."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")


def _aiter_fp4_indexer_available() -> bool:
    if not (is_hip() and is_gfx95_supported()):
        return False
    try:
        from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4_prefill  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(
    _aiter_fp4_indexer_available(), "gfx950 with aiter's FlyDSL FP4 indexer"
)
class TestPrefillScheduleCompilesBeforeServing(CustomTestCase):
    def test_no_new_variant_after_warm_up(self):
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
            CTA_INFO_WIDTH,
            _prefill_cta_info_kernel,
        )

        from sglang.kernels.ops.attention.dsv4 import fp4_indexer_schedule_hip as sched

        kernels = (sched._prefill_schedule_prep_kernel, _prefill_cta_info_kernel)
        if not all(hasattr(kernel, "device_caches") for kernel in kernels):
            self.skipTest("this Triton keeps no per-device kernel cache")

        def variants():
            return [
                sum(len(entry[0]) for entry in kernel.device_caches.values())
                for kernel in kernels
            ]

        device = torch.device("cuda", torch.cuda.current_device())
        sched.warm_prefill_schedule(device)
        warm = variants()
        # every row-count residue, table widths and row metadata at unaligned offsets,
        # as a row share or a request group produces them
        for rows, width, start in (
            (17, 5, 0),
            (102, 37, 3),
            (333, 1440, 1),
            (1000, 23, 2),
            (2047, 64, 0),
            (2049, 9, 5),
            (3001, 730, 7),
            (3002, 12, 4),
            (4095, 4, 1),
            (4096, 1441, 0),
        ):
            table = torch.zeros(
                (start + rows, width), dtype=torch.int32, device=device
            )[start:]
            ends = torch.randint(
                1, width * 64, (start + rows,), dtype=torch.int32, device=device
            )[start:]
            units = max(rows, 1024)
            sched.build_prefill_schedule(
                page_table=table,
                local_ends=ends,
                cta_info_out=torch.empty(
                    (units, CTA_INFO_WIDTH), dtype=torch.int32, device=device
                ),
                parallel_unit_num=units,
                max_seq_len=width * 64,
            )
        torch.cuda.synchronize()
        self.assertEqual(variants(), warm)


if __name__ == "__main__":
    unittest.main()
