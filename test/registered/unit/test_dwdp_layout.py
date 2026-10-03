"""DWDP composite-VA layout arithmetic, on CPU with no accelerator or driver.

Every other DWDP test needs 2-4 real devices and runs nightly, so the byte math
that decides where each expert lands has no per-PR guard. These cases cover the
part that is pure integer arithmetic, plus the import contract that keeps the
package usable on non-CUDA hardware.
"""

import subprocess
import sys
import textwrap
import unittest

import torch

from sglang.srt.layers.moe.dwdp.layout import (
    DwdpExpertLayout,
    PageAlignedLayout,
    build_layer_weight_specs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# 2 MiB is what both drivers report for objects this size; 64 KiB is the finest either
# reports at all, so the pair brackets the arithmetic rather than pinning one device.
_GRANULARITIES = (2 * 1024 * 1024, 64 * 1024)
# PagePool.DEFAULT_PAGE_SIZE_MULTIPLIER, which WeightBuffer passes as both the
# pool's page size and the layout's pool_granularity
_POOL_MULTIPLIER = 8


def _awkward_expert_byte_sizes(granularity: int):
    # None is a multiple of the page size, so the local handle's first and last page
    # hold bytes of experts this rank does not own.
    return (
        6145 * 512,  # the intermediate size the multi-device prefetch test uses
        granularity + 1,
        granularity - 1,
        3 * granularity + 17,
        2654321,  # prime-ish, no shared factor with either granularity
    )


class TestPageAlignedLayoutArithmetic(unittest.TestCase):
    def test_experts_land_inside_the_regions_that_back_them(self):
        """A rank's own experts must land inside the window its own handle backs; a
        wrong padding or region size reads them out of memory the rank never mapped."""
        for granularity in _GRANULARITIES:
            pool_granularity = _POOL_MULTIPLIER * granularity
            for expert_bytes in _awkward_expert_byte_sizes(granularity):
                for num_experts, dwdp_size in (
                    (8, 2),
                    (8, 4),
                    (32, 4),
                    (128, 8),
                    (6, 3),
                ):
                    for dwdp_rank in range(dwdp_size):
                        layout = DwdpExpertLayout(
                            num_routed_experts=num_experts,
                            dwdp_size=dwdp_size,
                            dwdp_rank=dwdp_rank,
                        )
                        with self.subTest(
                            granularity=granularity,
                            expert_bytes=expert_bytes,
                            num_experts=num_experts,
                            dwdp_size=dwdp_size,
                            dwdp_rank=dwdp_rank,
                        ):
                            self._assert_layout_covers_every_expert(
                                expert_bytes=expert_bytes,
                                num_experts=num_experts,
                                local_start=layout.local_expert_start,
                                local_end=layout.local_expert_end,
                                granularity=granularity,
                                pool_granularity=pool_granularity,
                            )

    def _assert_layout_covers_every_expert(
        self,
        *,
        expert_bytes: int,
        num_experts: int,
        local_start: int,
        local_end: int,
        granularity: int,
        pool_granularity: int,
    ) -> None:
        total_expert_bytes = num_experts * expert_bytes
        layout = PageAlignedLayout.compute(
            expert_bytes=expert_bytes,
            num_experts=num_experts,
            local_start=local_start,
            local_end=local_end,
            granularity=granularity,
            # deliberately generous: compute() raises once the region outgrows the
            # physical object, and that guard is not what these cases exercise
            handle_phys_size=total_expert_bytes + granularity,
            pool_granularity=pool_granularity,
        )

        self.assertEqual(
            layout.pre_size + layout.mnnvl_size + layout.post_size, layout.total_size
        )

        # PagePool.map_binding walks num_pages fixed-size pages, so a region that
        # is not a whole number of pages would be left partly unmapped
        self.assertEqual(layout.pre_size % pool_granularity, 0)
        self.assertEqual(layout.post_size % pool_granularity, 0)
        self.assertEqual(layout.pre_pages * pool_granularity, layout.pre_size)
        self.assertEqual(layout.post_pages * pool_granularity, layout.post_size)

        tensor_start = layout.pre_padding
        self.assertGreaterEqual(tensor_start, 0)
        self.assertLessEqual(
            tensor_start + total_expert_bytes,
            layout.total_size,
            "the tail expert runs past the end of the composite VA",
        )

        # mnnvl_size is the field name for the window the rank's own handle backs,
        # whichever driver exported it; nothing here is fabric-specific.
        handle_start = layout.pre_size
        handle_end = layout.pre_size + layout.mnnvl_size
        for expert in range(local_start, local_end):
            begin = tensor_start + expert * expert_bytes
            end = begin + expert_bytes
            self.assertGreaterEqual(
                begin, handle_start, f"local expert {expert} starts before the handle"
            )
            self.assertLessEqual(
                end, handle_end, f"local expert {expert} ends past the handle"
            )

        # the first local byte sits leading_edge into the handle, and the bytes
        # before it belong to the peer that owns the previous expert
        self.assertEqual(
            tensor_start + local_start * expert_bytes,
            handle_start + layout.leading_edge,
        )
        self.assertEqual(
            handle_end - (tensor_start + local_end * expert_bytes),
            layout.trailing_edge,
        )

    def test_peer_ranges_partition_the_experts_without_a_hole(self):
        """Every expert must be owned by exactly one peer: an expert nobody owns is
        never prefetched and reads as whatever the pool page last held."""
        for num_experts, dwdp_size in ((8, 2), (8, 4), (32, 4), (128, 8), (6, 3)):
            layout = DwdpExpertLayout(
                num_routed_experts=num_experts, dwdp_size=dwdp_size, dwdp_rank=0
            )
            with self.subTest(num_experts=num_experts, dwdp_size=dwdp_size):
                owners = [0] * num_experts
                for start, end in layout.peer_ranges:
                    for expert in range(start, end):
                        owners[expert] += 1
                self.assertEqual(
                    owners,
                    [1] * num_experts,
                    f"peer_ranges={layout.peer_ranges} does not partition "
                    f"{num_experts} experts",
                )


class TestWeightSpecAdmission(unittest.TestCase):
    def test_padded_row_stride_is_rejected(self):
        """A narrowed view over a padded allocation keeps its logical shape but not
        its stride, while the layout addresses experts by logical byte size."""
        padded = torch.empty(4, 8, 16 + 2)
        narrowed = padded[:, :, :16]
        self.assertFalse(narrowed.is_contiguous())

        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            build_layer_weight_specs({(0, "w13_weight"): narrowed}, 8)

    def test_contiguous_shard_gets_the_global_expert_count(self):
        specs = build_layer_weight_specs({(3, "w2_weight"): torch.empty(2, 8, 16)}, 8)
        spec = specs[3]["w2_weight"]
        self.assertEqual(spec.chunk_shape, (2, 8, 16))
        self.assertEqual(spec.full_shape, (8, 8, 16))
        self.assertEqual(spec.expert_bytes, 8 * 16 * 4)


class TestImportContract(unittest.TestCase):
    def test_importing_dwdp_loads_no_per_driver_module(self):
        """dwdp must import no per-driver module (issue #31995). Asserted over
        sys.modules, not cuda.bindings, which torch.cuda imports on its own."""
        probe = textwrap.dedent(
            """
            import sys
            import sglang.srt.layers.moe.dwdp  # noqa: F401
            loaded = sorted(
                m for m in sys.modules
                if m.endswith(("cuda_vmm_utils", "xpu_vmm_utils"))
            )
            print(",".join(loaded))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            "",
            "importing dwdp loaded a per-driver module; get_vmm_backend imports "
            "cuda_vmm_utils / xpu_vmm_utils inside its methods to keep that out "
            "of import time",
        )


if __name__ == "__main__":
    unittest.main()
