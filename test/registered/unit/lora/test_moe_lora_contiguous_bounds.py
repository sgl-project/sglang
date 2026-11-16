"""Pure-math bounds of the contiguous row domain: the padded-row ceiling and
the CuTeDSL schedule capacities hold for every split of routed pairs."""

from __future__ import annotations

import pytest

from sglang.kernels.ops.lora.moe.dispatch_contiguous import contiguous_m_pad_ceiling
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestAlignedSegmentMath:
    def test_ceiling_bounds_every_aligned_prefix_total(self) -> None:
        import random

        rng = random.Random(0xA11C)
        for _ in range(64):
            alignment = rng.choice((8, 64, 128))
            num_experts = rng.randint(1, 32)
            num_pairs = rng.randint(0, 4096)
            counts = [0] * num_experts
            for _ in range(num_pairs):
                counts[rng.randrange(num_experts)] += 1
            total = sum(-(-count // alignment) * alignment for count in counts)
            ceiling = contiguous_m_pad_ceiling(num_pairs, num_experts, alignment)
            assert total <= ceiling, (counts, alignment, total, ceiling)


class TestCuteDslContiguousScheduleGeometry:
    def test_capacity_bounds_every_actual_schedule_total(self) -> None:
        import random

        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            dual_stage_schedule_capacities_contiguous,
        )

        rng = random.Random(0xC0DE)
        for _ in range(64):
            num_experts = rng.randint(1, 64)
            num_pairs = rng.randint(0, 4096)
            token_width = rng.choice((8, 64, 128))
            counts = [0] * num_experts
            for _ in range(num_pairs):
                counts[rng.randrange(num_experts)] += 1
            m_pad_ceiling = contiguous_m_pad_ceiling(num_pairs, num_experts, 128)
            n_gemm1, n_gemm2 = 512, 256
            capacity1, capacity2 = dual_stage_schedule_capacities_contiguous(
                num_experts=num_experts,
                m_pad_ceiling=m_pad_ceiling,
                max_expert_rows=max(num_pairs, 1),
                m_alignment=128,
                token_width=token_width,
                n_gemm1=n_gemm1,
                n_gemm2=n_gemm2,
                output_width=128,
            )
            clusters = sum(-(-count // token_width) for count in counts)
            assert clusters * -(-n_gemm1 // 128) <= capacity1, (counts, token_width)
            assert clusters * -(-n_gemm2 // 128) <= capacity2, (counts, token_width)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
