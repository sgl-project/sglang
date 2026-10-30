"""DSA passes global IDs; PP HiSparse buffers are stage-local."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("start,count", [(0, 10), (10, 10), (70, 8)])
def test_global_dsa_layer_id_maps_once(start, count):
    owner = SimpleNamespace(
        is_dsv4_hisparse=False,
        is_m3_hisparse=False,
        mem_pool_device=SimpleNamespace(start_layer=start, layer_num=count),
        swap_in_selected_pages=Mock(return_value="result"),
    )
    for local in range(count):
        assert (
            HiSparseCoordinator.swap_in_selected_dsa_pages(
                owner, "reqs", "lengths", "topk", start + local
            )
            == "result"
        )
        owner.swap_in_selected_pages.assert_called_with(
            "reqs", "lengths", "topk", local
        )
    for invalid in (start - 1, start + count):
        with pytest.raises(ValueError, match="outside this PP stage"):
            HiSparseCoordinator.swap_in_selected_dsa_pages(
                owner, None, None, None, invalid
            )


@pytest.mark.parametrize("kind", ["is_dsv4_hisparse", "is_m3_hisparse"])
def test_other_pool_id_spaces_are_not_reinterpreted(kind):
    owner = SimpleNamespace(is_dsv4_hisparse=False, is_m3_hisparse=False)
    setattr(owner, kind, True)
    with pytest.raises(ValueError, match="requires a DSA pool"):
        HiSparseCoordinator.swap_in_selected_dsa_pages(owner, None, None, None, 0)
