import sys
import types
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.layers.communicator_mhc import (
    MHCCommunicateSummableTensorPairFn,
    MHCLayerCommunicator,
    MHCPostPreResult,
    MHCState,
)
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _fake_communicator(mlp_mode=ScatterMode.TP_ATTN_FULL):
    return types.SimpleNamespace(
        _speculative_algo=None,
        layer_scatter_modes=types.SimpleNamespace(mlp_mode=mlp_mode),
        is_last_layer=False,
        _context=types.SimpleNamespace(tp_size=4),
    )


class TestFuseMlpAllReduceGate(CustomTestCase):
    """Hybrid EP+TP must not fuse the post-experts all-reduce away.

    The fused residual+LN reduces over a single group, but with moe_ep_size > 1
    and moe_tp_size > 1 the post-experts reduction spans two disjoint groups
    (_MOE_EP then _MOE_TP) and should_skip_post_experts_all_reduce() drops both
    once fusion is published. The result is activations reduced over only half
    the peers -- wrong output, no crash. Observed as garbage completions on
    Qwen3-30B-A3B with --tp-size 4 --ep-size 2.
    """

    def _should_fuse(
        self, *, moe_ep_size, moe_tp_size, mlp_mode=ScatterMode.TP_ATTN_FULL
    ):
        forward_batch = types.SimpleNamespace(
            input_ids=types.SimpleNamespace(shape=(8,))
        )
        with (
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=True),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=False),
            ),
            get_parallel().override(
                moe_ep_size=moe_ep_size, moe_tp_size=moe_tp_size, tp_size=4
            ),
        ):
            return LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                _fake_communicator(mlp_mode), forward_batch
            )

    def test_hybrid_ep_tp_does_not_fuse(self):
        self.assertFalse(self._should_fuse(moe_ep_size=2, moe_tp_size=2))

    def test_pure_tp_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=1, moe_tp_size=4))

    def test_pure_ep_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=4, moe_tp_size=1))

    def test_moe_full_layer_does_not_fuse(self):
        # Fusion skips postprocess_layer, which holds the CP scatter; a dense
        # MOE_FULL layer (moe_dp_size == attn_cp_size) is not caught by the
        # is_enable_moe_cp_allgather gate.
        self.assertFalse(
            self._should_fuse(
                moe_ep_size=1, moe_tp_size=4, mlp_mode=ScatterMode.MOE_FULL
            )
        )


@pytest.mark.parametrize(
    "mode,num_tokens,norm_fused",
    [
        ("fused", 2, True),
        ("fused", 2, False),
        ("decline", 2, False),
        ("fused", 0, False),
    ],
)
def test_mhc_boundary_preserves_residual_mix_and_norm(
    mode, num_tokens, norm_fused
):
    hidden = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4)
    residual = hidden.repeat(1, 2) + 1
    comb = torch.tensor([[0.7, 0.3], [0.2, 0.8]]).expand(num_tokens, 2, 2)
    post = torch.tensor([0.4, 0.9]).expand(num_tokens, 2)
    norm = torch.nn.RMSNorm(4, eps=1e-5)
    norm.variance_epsilon = norm.eps
    norm.weight.data.copy_(torch.tensor([1.0, 0.7, 1.3, 0.8]))

    def hc_post(x, r, c, p):
        mixed = torch.bmm(c.reshape(-1, 2, 2), r.reshape(-1, 2, 4))
        return (mixed + x[:, None, :] * p[:, :, None]).flatten(1)

    def hc_pre(r, weight, eps):
        return r.reshape(-1, 2, 4).sum(1), comb.flip(1).flatten(1), post.flip(1), False

    def unfused_pre(r, weight, eps):
        if mode == "fused" and num_tokens:
            raise AssertionError("Eligible fused boundary fell back")
        return hc_pre(r, weight, eps)

    def fused(x, r, c, p, weight, eps):
        if mode == "decline":
            return None
        if x.shape[0] == 0:
            raise AssertionError("Empty ranks must retain the zero-token fallback")
        r = hc_post(x, r, c, p)
        x, c, p, _ = hc_pre(r, weight, eps)
        if norm_fused:
            x = torch.nn.functional.rms_norm(x, (4,), weight, eps)
        return MHCPostPreResult(x, r, c, p, norm_fused)

    state = MHCState(2, hc_pre, unfused_pre, hc_post, hc_post_attn_pre=fused)
    state.h_res, state.h_post = comb.flatten(1), post
    expected_residual = hc_post(hidden, residual, state.h_res, post)
    expected_hidden, next_comb, next_post, _ = hc_pre(expected_residual, None, None)
    expected_hidden = torch.nn.functional.rms_norm(
        expected_hidden, (4,), norm.weight, norm.variance_epsilon
    )
    previous = state
    state = MHCState(2, unfused_pre, hc_pre, hc_post, hc_post_attn_pre=fused)
    actual_hidden, actual_residual = state.attn_split(
        hidden, norm, residual=residual, previous_mhc=previous
    )
    assert previous.h_res is previous.h_post is None
    torch.testing.assert_close(actual_hidden, expected_hidden)
    torch.testing.assert_close(actual_residual, expected_residual)
    torch.testing.assert_close(
        state.mlp_combine(actual_hidden * 2, actual_residual),
        hc_post(expected_hidden * 2, expected_residual, next_comb, next_post),
    )
    state.reset_aux()
    assert state.h_res is state.h_post is None


@pytest.mark.parametrize("failure", ["fused", "fallback"])
def test_mhc_cross_layer_failure_releases_previous_state(failure):
    def fail(*args):
        raise RuntimeError("boundary failure")

    previous = MHCState(2, None, None, fail, torch.ones(1, 4), torch.ones(1, 2))
    current = MHCState(
        2, None, None, None, hc_post_attn_pre=fail if failure == "fused" else None
    )
    with pytest.raises(RuntimeError, match="boundary failure"):
        current.attn_split(
            torch.ones(1, 4), residual=torch.ones(1, 8), previous_mhc=previous
        )
    assert previous.h_res is previous.h_post is None


@pytest.mark.parametrize(
    "mode",
    ["trivial", "last", "first", "reduce", "gather", "layout", "callback", "scattered"],
)
def test_mhc_cross_layer_preserves_communication_boundaries(mode):
    previous = MHCLayerCommunicator.__new__(MHCLayerCommunicator)
    current = MHCLayerCommunicator.__new__(MHCLayerCommunicator)
    previous.is_last_layer = mode == "last"
    current.is_first_layer = mode == "first"
    previous._communicate_summable_tensor_pair_fn = (
        MHCCommunicateSummableTensorPairFn._scatter
        if mode == "reduce"
        else MHCCommunicateSummableTensorPairFn._trivial
    )
    current._communicate_simple_fn = (
        comm.CommunicateSimpleFn._scattered_to_tp_attn_full
        if mode == "gather"
        else comm.CommunicateSimpleFn._trivial
    )
    previous.layer_scatter_modes = types.SimpleNamespace(
        layer_output_mode=ScatterMode.TP_ATTN_FULL
    )
    current.layer_scatter_modes = types.SimpleNamespace(
        layer_input_mode=ScatterMode.SCATTERED
        if mode == "layout"
        else ScatterMode.TP_ATTN_FULL
    )
    current.mhc = MHCState(
        2, None, None, None, hc_post_attn_pre=None if mode == "callback" else torch.add
    )
    with get_forward().scoped(attn_input_scattered=mode == "scattered"):
        assert previous.can_fuse_mhc_boundary(current) == (mode == "trivial")


def test_mhc_post_pre_result_fullgraph():
    @torch.compile(backend="eager", fullgraph=True)
    def boundary(x):
        result = MHCPostPreResult(x + 1, x + 2, x + 3, x + 4, True)
        return result.hidden_states + result.residual + result.h_res + result.h_post

    torch.testing.assert_close(boundary(torch.ones(3)), torch.full((3,), 14.0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
