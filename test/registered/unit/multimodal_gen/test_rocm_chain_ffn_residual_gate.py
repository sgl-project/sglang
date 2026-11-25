# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.multimodal_gen.runtime.models.dits import wanvideo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Op:
    """A norm/residual op whose dispatch is bound to one forward variant."""

    def __init__(self, variant):
        self._forward_method = getattr(self, variant)

    def forward_hip(self):
        pass

    def forward_cuda(self):
        pass

    def forward_native(self):
        pass


def _block(norm1, ffn_residual_norm1, mlp_residual):
    return SimpleNamespace(
        norm1=_Op(norm1),
        ffn_residual_norm1=_Op(ffn_residual_norm1),
        mlp_residual=_Op(mlp_residual),
    )


def _chains(block):
    return wanvideo.WanTransformerBlock.chains_ffn_residual.fget(block)


class TestRocmChainFfnResidualGate(unittest.TestCase):
    """With SGLANG_ROCM_CHAIN_FFN_RESIDUAL on, the FFN residual chains into the next
    norm1 when both norms run their ROCm (HIP) variants, regardless of MulAdd."""

    def test_flag_on_chains_when_both_norms_run_hip(self):
        block = _block("forward_hip", "forward_hip", "forward_native")
        with mock.patch.object(wanvideo, "_use_rocm_chain_ffn_residual", True):
            self.assertTrue(_chains(block))

    def test_flag_on_does_not_chain_when_a_norm_is_not_hip(self):
        block = _block("forward_hip", "forward_cuda", "forward_native")
        with mock.patch.object(wanvideo, "_use_rocm_chain_ffn_residual", True):
            self.assertFalse(_chains(block))

    def test_flag_off_keeps_the_cuda_only_contract(self):
        block = _block("forward_hip", "forward_hip", "forward_native")
        with mock.patch.object(wanvideo, "_use_rocm_chain_ffn_residual", False):
            self.assertFalse(_chains(block))


if __name__ == "__main__":
    unittest.main()
