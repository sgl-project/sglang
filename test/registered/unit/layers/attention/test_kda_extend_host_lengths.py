"""KDA extend must trim padding without reading device lengths when host metadata exists."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class DeviceOffsets:
    """A stand-in that rejects any host read of GPU sequence boundaries."""

    def __getitem__(self, index):
        raise AssertionError("Host metadata path read device query_start_loc")


def run(lengths, padding, prefix, host_metadata, track=False):
    n = sum(lengths)
    seen = []
    offsets = (
        DeviceOffsets()
        if host_metadata
        else torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    )

    def conv(x, *args, **kw):
        assert x.shape == (6, n)
        assert kw["query_start_loc"] is offsets
        assert kw["seq_lens_cpu"] == (lengths if host_metadata else None)
        torch.testing.assert_close(kw["has_initial_state"], torch.tensor(prefix) > 0)
        seen.append("conv")
        return x

    def extend(**kw):
        assert kw["q"].shape == (1, n, 1, 2)
        assert kw["g"].shape == (1, n, 1, 2)
        assert kw["beta"].shape == (1, n, 1)
        assert kw["query_start_loc"] is offsets
        out = kw["q"] + kw["k"] + kw["v"] + kw["g"] + kw["beta"].unsqueeze(-1)
        seen.append("extend")
        return (out, torch.ones(1)) if track else out

    states = NS(conv=[torch.zeros(3, 2, 6)], temporal=torch.zeros(3, 1, 2, 2))
    meta = NS(
        query_start_loc=offsets,
        mamba_cache_indices=torch.tensor([2, 0, 1][: len(lengths)]),
        has_mamba_track_mask=track,
        conv_states_mask_indices=torch.tensor([1]),
        track_conv_indices=torch.tensor([[0, 1]]),
        track_ssm_h_src=torch.tensor([0]),
    )

    def track_state(*args):
        seen.append("track")

    backend = NS(
        forward_metadata=meta,
        req_to_token_pool=NS(mamba2_layer_cache=lambda _: states),
        kernel_dispatcher=NS(extend=extend),
        accept_lens_pool=None,
        _track_mamba_state_extend=track_state,
    )
    mode = NS(is_target_verify=lambda: False, is_draft_extend_v2=lambda: False)
    batch = NS(
        forward_mode=mode,
        extend_prefix_lens=torch.tensor(prefix),
        extend_seq_lens_cpu=list(lengths) if host_metadata else None,
    )
    layer = NS(
        layer_id=0,
        conv_weights=None,
        bias=None,
        q_dim=2,
        k_dim=2,
        v_dim=2,
        head_q_dim=2,
        head_k_dim=2,
        head_v_dim=2,
        A_log=None,
        dt_bias=None,
        lower_bound=None,
    )
    x = torch.arange((n + padding) * 6, dtype=torch.float32).reshape(n + padding, 6)
    a = torch.ones(1, n + padding, 2)
    b = torch.ones(1, n + padding, 1)
    x[n:] = float("nan")
    a[:, n:] = float("nan")
    b[:, n:] = float("nan")
    with patch.object(kda_backend, "causal_conv1d_fn", conv, create=True):
        out = KDAAttnBackend.forward_extend(backend, layer, batch, x, a, b)
    expected = x[:n, :2] + x[:n, 2:4] + x[:n, 4:] + 2
    torch.testing.assert_close(out[0, :n, 0], expected, rtol=0, atol=0)
    assert out.shape == (1, n + padding, 1, 2)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[:, n:]) == 0
    assert seen == ["conv", "extend"] + (["track"] if track else [])
    return out, states.conv[0]


@pytest.mark.parametrize(
    "lengths,prefix", [([5], [0]), ([3, 2], [4096, 0]), ([3, 2, 0], [8192, 7, 0])]
)
@pytest.mark.parametrize("padding", [0, 11])
@pytest.mark.parametrize("host_metadata", [False, True])
@pytest.mark.parametrize("track", [False, True])
def test_kda_extend_host_lengths_preserve_layout_and_tracking(
    lengths, prefix, padding, host_metadata, track
):
    run(lengths, padding, prefix, host_metadata, track=track)
