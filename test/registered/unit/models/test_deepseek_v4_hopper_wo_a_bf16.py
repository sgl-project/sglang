"""Hopper G8 target-verify dispatch and native token-major BMM numerics."""

import sys
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Hopper-only verify specialization",
)


@pytest.mark.parametrize(
    "rows,strided", [(1, False), (6, False), (6, True), (48, False), (49, False)]
)
def test_hopper_g8_verify(rows, strided):
    from sglang.srt.models import deepseek_v4 as dsv4

    generator = torch.Generator(device="cuda").manual_seed(66068407)
    weight = (
        torch.randn(
            (8, 1024, 4096), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        * 0.02
    )
    backing = (
        torch.randn(
            (rows, 16 if strided else 8, 4096),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    activation = backing[:, :8]
    expected = torch.einsum("tgd,grd->tgr", activation, weight)

    def apply():
        return dsv4._apply_wo_a_bf16_matmul(
            activation, weight, is_decode=False, is_target_verify=True, fast_path=True
        )

    with mock.patch.object(
        dsv4, "_wo_a_bf16_bmm_token_major", wraps=dsv4._wo_a_bf16_bmm_token_major
    ) as routed:
        actual = apply()
        assert routed.call_count == (1 if rows <= 48 else 0)
    torch.testing.assert_close(actual, expected, rtol=2**-6, atol=2**-10)
    if rows <= 48:
        assert actual.is_contiguous()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply()
    for _ in range(3):
        activation.copy_(
            torch.randn(
                activation.shape,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            captured,
            torch.einsum("tgd,grd->tgr", activation, weight),
            rtol=2**-6,
            atol=2**-10,
        )
    with mock.patch.object(
        dsv4, "_wo_a_bf16_bmm_token_major", wraps=dsv4._wo_a_bf16_bmm_token_major
    ) as routed:
        fallback = dsv4._apply_wo_a_bf16_matmul(
            activation, weight, is_decode=False, is_target_verify=True, fast_path=False
        )
        assert routed.call_count == 0
    torch.testing.assert_close(
        fallback, torch.einsum("tgd,grd->tgr", activation, weight), rtol=0, atol=0
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
