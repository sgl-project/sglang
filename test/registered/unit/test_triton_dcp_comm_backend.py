from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.layers.attention.triton_backend import _reduce_dcp_mha_output


@pytest.mark.parametrize("backend", ["a2a", "fi_a2a"])
@patch("sglang.srt.layers.attention.triton_backend.cp_lse_ag_out_rs_mha")
@patch("sglang.srt.layers.attention.triton_backend.dcp_a2a_lse_reduce")
def test_dcp_mha_routes_a2a_backends(a2a_reduce, ag_rs_reduce, backend):
    attn_output = MagicMock()
    attn_lse = MagicMock()
    group = MagicMock()
    expected = MagicMock()
    a2a_reduce.return_value = expected

    result = _reduce_dcp_mha_output(attn_output, attn_lse, group, backend)

    assert result is expected
    a2a_reduce.assert_called_once_with(
        attn_output,
        attn_lse,
        group,
        is_lse_base_on_e=True,
        comm_backend=backend,
    )
    ag_rs_reduce.assert_not_called()


@patch("sglang.srt.layers.attention.triton_backend.cp_lse_ag_out_rs_mha")
@patch("sglang.srt.layers.attention.triton_backend.dcp_a2a_lse_reduce")
def test_dcp_mha_routes_ag_rs_by_default(a2a_reduce, ag_rs_reduce):
    attn_output = MagicMock()
    attn_lse = MagicMock()
    group = MagicMock()
    expected = MagicMock()
    ag_rs_reduce.return_value = expected

    result = _reduce_dcp_mha_output(attn_output, attn_lse, group, "ag_rs")

    assert result is expected
    ag_rs_reduce.assert_called_once_with(attn_output, attn_lse, group)
    a2a_reduce.assert_not_called()
