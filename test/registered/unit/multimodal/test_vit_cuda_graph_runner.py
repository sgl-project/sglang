import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

from sglang.srt.multimodal.internvl_vit_cuda_graph_runner import (
    InternViTCudaGraphRunner,
)
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner


class _Block:
    def forward(self, x):
        return x


def _runner(*, use_data_parallel: bool) -> ViTCudaGraphRunner:
    vit = SimpleNamespace(
        blocks=[_Block()],
        deepstack_visual_indexes=[],
        deepstack_merger_list=None,
        use_data_parallel=use_data_parallel,
    )
    return ViTCudaGraphRunner(vit)


def test_dp_vit_graph_capture_does_not_enter_tp_communication_capture():
    runner = _runner(use_data_parallel=True)
    with patch(
        "sglang.srt.multimodal.vit_cuda_graph_runner.get_tp_group",
        side_effect=AssertionError("DP capture must be rank-local"),
    ):
        with runner._capture_context():
            pass


def test_non_dp_vit_graph_capture_uses_tp_communication_capture():
    entered = []

    class Capture:
        def __enter__(self):
            entered.append(True)

        def __exit__(self, *args):
            return False

    group = SimpleNamespace(ca_comm=SimpleNamespace(capture=lambda: Capture()))
    runner = _runner(use_data_parallel=False)
    with patch(
        "sglang.srt.multimodal.vit_cuda_graph_runner.get_tp_group", return_value=group
    ):
        with runner._capture_context():
            pass
    assert entered == [True]


def test_vit_graph_runner_caches_resolved_backend_name():
    class Block:
        attn = SimpleNamespace(
            qkv_backend_name="fa3",
            qkv_backend=object(),
        )

        def forward(self, x, output_ws=None):
            return x

    vit = SimpleNamespace(blocks=[Block()])

    runner = ViTCudaGraphRunner(vit)

    assert runner._attn_backend == "fa3"


def test_internvl_graph_runner_caches_resolved_backend_name():
    attention = SimpleNamespace(
        qkv_backend_name="triton_attn",
        qkv_backend=object(),
    )
    layer = SimpleNamespace(attn=SimpleNamespace(attn=attention))
    encoder = SimpleNamespace(layers=[layer])

    runner = InternViTCudaGraphRunner(encoder)

    assert runner._attn_backend == "triton_attn"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
