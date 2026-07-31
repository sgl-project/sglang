from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.arg_groups.overrides import _kimi_k3_overrides
from sglang.srt.layers.dcp.comm import all_gather_q_for_mla_decode
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_direct_query_dispatch_bypasses_collective():
    q_nope = torch.randn(2, 3, 4)
    q_rope = torch.randn(2, 3, 1)
    expected = (q_nope + 1, q_rope + 2)
    calls = []

    def direct_gatherer(nope, rope):
        calls.append((nope, rope))
        return expected

    actual = all_gather_q_for_mla_decode(
        q_nope,
        q_rope,
        direct_gatherer=direct_gatherer,
    )

    assert actual == expected
    assert calls == [(q_nope, q_rope)]


def test_kimi_k3_direct_query_disables_replicated_projection():
    args = SimpleNamespace(
        dcp_size=4,
        enable_symm_mem=False,
        speculative_algorithm=None,
        dcp_direct_q_gather=True,
        dcp_replicate_q_proj=None,
        dcp_comm_backend="ag_rs",
    )
    with (
        patch(
            "sglang.srt.arg_groups.overrides.attention_backends_of",
            return_value=(None, None),
        ),
        patch(
            "sglang.srt.arg_groups.overrides.get_device_name",
            return_value="NVIDIA GB200",
        ),
        patch(
            "sglang.srt.arg_groups.overrides.is_mnnvl_fabric_device",
            return_value=False,
        ),
    ):
        overrides = _kimi_k3_overrides(args, hf_config=None)

    assert overrides["dcp_replicate_q_proj"] is False
    assert overrides["dcp_comm_backend"] == "a2a"


def test_direct_query_requires_dcp():
    args = ServerArgs(
        model_path="dummy",
        dcp_size=1,
        dcp_direct_q_gather=True,
    )
    with pytest.raises(ValueError, match="requires --dcp-size > 1"):
        args._handle_dcp_validation()


def test_direct_query_rejects_explicit_replication():
    args = ServerArgs(
        model_path="dummy",
        dcp_size=4,
        dcp_direct_q_gather=True,
        dcp_replicate_q_proj=True,
        dcp_comm_backend="a2a",
    )
    with (
        patch("sglang.srt.server_args.is_cuda", return_value=True),
        pytest.raises(ValueError, match="mutually exclusive"),
    ):
        args._handle_dcp_validation()


def test_direct_query_rejects_multi_node():
    args = ServerArgs(
        model_path="dummy",
        dcp_size=4,
        nnodes=2,
        dcp_direct_q_gather=True,
        dcp_comm_backend="a2a",
    )
    with pytest.raises(ValueError, match="single-node"):
        args._handle_dcp_validation()
