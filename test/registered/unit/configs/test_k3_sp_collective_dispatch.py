from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.kimi_k3 import sp_collective
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("device_name", "rs_tuning", "ag_tuning"),
    [
        (
            "NVIDIA_GB200",
            sp_collective.Tuning(32, 1024),
            sp_collective.Tuning(2, 512),
        ),
        (
            "NVIDIA_GB300",
            sp_collective.Tuning(32, 1024),
            sp_collective.Tuning(1, 1024),
        ),
    ],
)
def test_world16_dispatch_table(device_name, rs_tuning, ag_tuning):
    device = torch.device("cuda")
    with patch.object(sp_collective, "_device_name", return_value=device_name):
        sp_collective._TABLES.clear()

        assert sp_collective.get_dispatch(
            "reduce_scatter", 16, 7168, 512, device
        ) == sp_collective.Dispatch("pull", rs_tuning)
        assert sp_collective.get_dispatch(
            "all_gather", 16, 7168, 512, device
        ) == sp_collective.Dispatch("direct", ag_tuning)
        assert (
            sp_collective.get_fusion_dispatch(
                "reduce_scatter_attn_res", 16, 7168, 512, device
            )
            is None
        )
        assert (
            sp_collective.get_fusion_dispatch(
                "attn_res_all_gather", 16, 7168, 512, device
            )
            is None
        )

        table = sp_collective._table(16, 7168, device)
        assert table is not None
        for kind in ("reduce_scatter", "all_gather"):
            for num_tokens, config in table["configs"][kind].items():
                if config["strategy"] == "push":
                    local_bytes = int(num_tokens) * 7168 * 2 // 16
                    assert local_bytes <= 256 * 1024


def test_world16_dispatch_requires_exact_device_table():
    device = torch.device("cuda")
    with patch.object(sp_collective, "_device_name", return_value="NVIDIA_B200"):
        sp_collective._TABLES.clear()
        assert (
            sp_collective.get_dispatch("reduce_scatter", 16, 7168, 512, device) is None
        )
