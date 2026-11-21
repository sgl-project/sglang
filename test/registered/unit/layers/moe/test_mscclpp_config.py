from types import SimpleNamespace

import pytest

from sglang.srt.arg_groups.moe_hook import handle_a2a_moe
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_mscclpp_ep_spanning_accepts_matching_sizes(monkeypatch) -> None:
    monkeypatch.setattr(
        moe_utils,
        "get_parallel",
        lambda: SimpleNamespace(moe_ep_size=4, tp_size=4),
    )

    moe_utils._validate_mscclpp_ep_spanning(MoeA2ABackend.MSCCLPP)


def test_mscclpp_ep_spanning_rejects_mismatched_sizes(monkeypatch) -> None:
    monkeypatch.setattr(
        moe_utils,
        "get_parallel",
        lambda: SimpleNamespace(moe_ep_size=2, tp_size=4),
    )

    with pytest.raises(
        ValueError,
        match=r"requires --ep-size to equal --tp-size.*--ep-size 2.*--tp-size 4",
    ):
        moe_utils._validate_mscclpp_ep_spanning(MoeA2ABackend.MSCCLPP)


def test_non_mscclpp_backend_does_not_require_ep_spanning(monkeypatch) -> None:
    def unexpected_parallel_access():
        raise AssertionError("non-MSCCL++ validation should not read topology")

    monkeypatch.setattr(moe_utils, "get_parallel", unexpected_parallel_access)

    moe_utils._validate_mscclpp_ep_spanning(MoeA2ABackend.NONE)


@pytest.mark.parametrize(
    "overlap_flag",
    ["enable_two_batch_overlap", "enable_single_batch_overlap"],
)
def test_initialize_moe_config_rejects_mscclpp_batch_overlap(
    monkeypatch, overlap_flag
) -> None:
    overlap = SimpleNamespace(
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
    )
    setattr(overlap, overlap_flag, True)
    monkeypatch.setattr(
        moe_utils,
        "get_exec",
        lambda: SimpleNamespace(
            moe=SimpleNamespace(
                moe_a2a_backend="mscclpp",
                moe_runner_backend="triton",
                deepep_mode="auto",
                mscclpp_mode="latency",
                mscclpp_ep_layout="expert_major",
            ),
            overlap=overlap,
        ),
    )
    monkeypatch.setattr(
        moe_utils,
        "get_spec",
        lambda: SimpleNamespace(
            speculative_moe_runner_backend=None,
            speculative_moe_a2a_backend=None,
        ),
    )
    monkeypatch.setattr(
        moe_utils,
        "get_flags",
        lambda: SimpleNamespace(moe=SimpleNamespace()),
    )
    monkeypatch.setattr(
        moe_utils,
        "get_parallel",
        lambda: SimpleNamespace(moe_ep_size=1, tp_size=1),
    )

    with pytest.raises(
        ValueError,
        match=r"--enable-two-batch-overlap.*--enable-single-batch-overlap",
    ):
        moe_utils.initialize_moe_config()


@pytest.mark.parametrize(
    "overlap_flag",
    ["enable_two_batch_overlap", "enable_single_batch_overlap"],
)
def test_handle_a2a_moe_rejects_mscclpp_batch_overlap(overlap_flag) -> None:
    server_args = ServerArgs(
        model_path="dummy",
        moe_a2a_backend="mscclpp",
        **{overlap_flag: True},
    )

    with pytest.raises(
        ValueError,
        match=r"--enable-two-batch-overlap.*--enable-single-batch-overlap",
    ):
        handle_a2a_moe(server_args)
