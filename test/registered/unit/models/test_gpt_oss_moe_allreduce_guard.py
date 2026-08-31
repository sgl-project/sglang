"""
Unit tests for the post-experts all-reduce guard in
GptOssSparseMoeBlock.forward_normal.

Regression test for the GPT-OSS + DP-attention double reduction: with a2a
none and TP == attn DP == EP, forward_normal all-reduced the expert partials
and LayerCommunicator.postprocess_layer then reduce-scatterv'ed the already
summed result again, scaling the MoE output by tp_size (observed as garbage
completions on GPT-OSS-120B MXFP4, 4x GB200, TP4/DP4/EP4: GSM8K 0/128).
The guard must skip the model-side all-reduce exactly when
should_skip_post_experts_all_reduce() reports that a downstream component
owns the reduction.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models import gpt_oss as gpt_oss_module
from sglang.srt.models.gpt_oss import GptOssSparseMoeBlock
from sglang.srt.runtime_context import (
    get_context,
    get_flags,
    get_forward,
    get_parallel,
    reset_context,
)
from sglang.test.test_utils import CustomTestCase


def _make_block(hidden_size=8, tp_size=4):
    """A GptOssSparseMoeBlock skeleton with stubbed router/topk/experts.

    forward_normal only touches these attributes; the weight-bearing
    __init__ is irrelevant to the communication guard.
    """
    block = object.__new__(GptOssSparseMoeBlock)
    block.hidden_size = hidden_size
    block.tp_size = tp_size
    block.layer_id = 0
    block.router = lambda x: (torch.zeros(x.shape[0], 4), None)
    block.topk = lambda x, _logits: SimpleNamespace()
    block.experts = lambda hs, _topk: hs.clone()
    return block


class TestGptOssMoeAllReduceGuard(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.hidden_states = torch.randn(4, 8)
        self.block = _make_block()

    def _forward_normal_tracking_all_reduce(self):
        """Run forward_normal, recording post-experts all-reduce calls."""
        with (
            patch.object(
                gpt_oss_module, "is_in_tc_piecewise_cuda_graph", return_value=False
            ),
            patch.object(
                gpt_oss_module,
                "tensor_model_parallel_all_reduce",
                side_effect=lambda x: x,
            ) as all_reduce,
        ):
            out = self.block.forward_normal(self.hidden_states.clone())
        return out, all_reduce

    @contextmanager
    def _parallel_ctx(
        self,
        *,
        dpa_enabled,
        tp_size=4,
        attn_dp_size=4,
        moe_ep_size=4,
        fuse_mlp_allreduce=False,
    ):
        """Publish a minimal config and force one parallel/forward layout.

        Publishing ``dwdp_size`` matters because
        should_skip_post_experts_all_reduce() reads it as a parallel config
        leaf; the parallel topology sizes come from the scoped overrides.
        """
        with ExitStack() as stack:
            stack.enter_context(get_context().override_server_args(dwdp_size=1))
            stack.enter_context(
                get_parallel().override(
                    tp_size=tp_size,
                    attn_dp_size=attn_dp_size,
                    moe_ep_size=moe_ep_size,
                )
            )
            stack.enter_context(get_flags().dp.override(enabled=dpa_enabled))
            stack.enter_context(
                get_forward().scoped(fuse_mlp_allreduce=fuse_mlp_allreduce)
            )
            yield

    def test_dp_reduce_scatterv_config_skips_all_reduce(self):
        """TP4/DP4/EP4 + DPA: postprocess owns the reduction.

        All-reducing here would make the reduce-scatterv sum every rank's
        already-reduced copy again (tp_size-fold over-count).
        """
        with self._parallel_ctx(dpa_enabled=True):
            out, all_reduce = self._forward_normal_tracking_all_reduce()
        all_reduce.assert_not_called()
        self.assertTrue(torch.equal(out, self.hidden_states))

    def test_plain_tp_still_all_reduces(self):
        """No DPA: the plain TP all-reduce must be kept."""
        with self._parallel_ctx(dpa_enabled=False, attn_dp_size=1):
            _, all_reduce = self._forward_normal_tracking_all_reduce()
        all_reduce.assert_called_once()

    def test_fused_mlp_allreduce_still_skips(self):
        """Fusion published by the decoder still absorbs the all-reduce."""
        with self._parallel_ctx(
            dpa_enabled=False, attn_dp_size=1, fuse_mlp_allreduce=True
        ):
            _, all_reduce = self._forward_normal_tracking_all_reduce()
        all_reduce.assert_not_called()

    def test_partial_attention_dp_still_all_reduces(self):
        """DPA with TP != attn DP does not use reduce_scatterv.

        should_use_dp_reduce_scatterv() requires each attention-DP shard to
        be a single rank (tp_size == attn_dp_size == moe_ep_size); other DPA
        layouts fall back to all-reduce + dp_scatter in postprocess.
        """
        with self._parallel_ctx(dpa_enabled=True, attn_dp_size=2, moe_ep_size=2):
            _, all_reduce = self._forward_normal_tracking_all_reduce()
        all_reduce.assert_called_once()

    def test_partial_attention_dp_with_ep4_still_all_reduces(self):
        """TP4/DP2/EP4 keeps all-reduce because attention TP is two."""
        with self._parallel_ctx(dpa_enabled=True, attn_dp_size=2, moe_ep_size=4):
            _, all_reduce = self._forward_normal_tracking_all_reduce()
        all_reduce.assert_called_once()


if __name__ == "__main__":
    unittest.main()
