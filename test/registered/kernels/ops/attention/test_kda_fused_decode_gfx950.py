# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention import kda_fused_decode_gfx950
from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

H, D = 12, 128
C = 3 * H * D
LOWER_BOUND = -5.0
NORM_EPS = 1e-5


def _is_gfx950() -> bool:
    return bool(
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties().gcnArchName.split(":", 1)[0] == "gfx950"
    )


def _inputs(indices, slots=8) -> dict[str, torch.Tensor]:
    batch = len(indices)
    generator = torch.Generator(device="cuda").manual_seed(batch)

    def randn(shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=generator, device="cuda") * scale).to(
            dtype
        )

    # The model slices these from one projection; an odd row width leaves
    # the slices unaligned.
    fused = randn((batch, C + H * D + D + H + 1))
    return {
        "mixed_qkv": fused[:, :C],
        "output_gate": fused[:, C : C + H * D].view(batch, H, D),
        "f_a": fused[:, C + H * D : C + H * D + D],
        "raw_beta": fused[:, C + H * D + D : C + H * D + D + H].unsqueeze(0),
        "f_b_weight": randn((H, D, D), 0.05),
        "conv_weight": randn((C, 4), 0.3, torch.float32),
        "conv_state": randn((slots, 3, C)).transpose(-1, -2),
        "A_log": randn((H,), 0.5, torch.float32),
        "dt_bias": randn((H * D,), 0.1, torch.float32),
        "state": randn((slots, H, D, D), 0.05, torch.float32),
        "state_indices": torch.tensor(indices, device="cuda", dtype=torch.int32),
        "norm_weight": randn((D,)),
    }


def _clone_caches(inputs):
    cloned = dict(inputs)
    cloned["conv_state"] = inputs["conv_state"].transpose(-1, -2).clone()
    cloned["conv_state"] = cloned["conv_state"].transpose(-1, -2)
    cloned["state"] = inputs["state"].clone()
    return cloned


def _triton_chain(inputs):
    gate = F.linear(inputs["f_a"], inputs["f_b_weight"].view(H * D, D))
    qkv = causal_conv1d_update(
        inputs["mixed_qkv"],
        inputs["conv_state"],
        inputs["conv_weight"],
        None,
        activation="silu",
        conv_state_indices=inputs["state_indices"],
    )
    out = TritonKDAKernel().packed_decode(
        qkv,
        gate,
        inputs["raw_beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=D**-0.5,
        ssm_states=inputs["state"],
        cache_indices=inputs["state_indices"],
        num_v_heads=H,
        head_v_dim=D,
        lower_bound=LOWER_BOUND,
    )
    norm = FusedRMSNormGated(D, eps=NORM_EPS, activation="sigmoid").cuda()
    norm.weight.data = inputs["norm_weight"]
    return norm(out, inputs["output_gate"])


def _fused(inputs, out=None):
    return kda_fused_decode_gfx950.run(
        lower_bound=LOWER_BOUND, norm_eps=NORM_EPS, out=out, **inputs
    )


@unittest.skipUnless(_is_gfx950(), "gfx950 KDA decode requires MI35x")
class TestGfx950KDAFusedDecode(CustomTestCase):
    @torch.inference_mode()
    def test_matches_triton_chain_over_steps(self):
        inputs = _inputs([3, 1, -1, 6, 5])
        expected_inputs = _clone_caches(inputs)
        actual_inputs = _clone_caches(inputs)
        # the default backend
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_ROCM_K3_KDA_FUSED_BACKEND", None)
            self.assertTrue(
                kda_fused_decode_gfx950.covered(
                    *(
                        inputs[name]
                        for name in (
                            "f_a",
                            "f_b_weight",
                            "mixed_qkv",
                            "raw_beta",
                            "conv_state",
                            "state",
                            "state_indices",
                            "output_gate",
                            "norm_weight",
                        )
                    )
                )
            )
        for _ in range(4):
            expected = _triton_chain(expected_inputs)
            actual = _fused(actual_inputs)
            torch.testing.assert_close(
                actual.float(), expected.float(), rtol=1e-2, atol=1e-2
            )
        torch.testing.assert_close(
            actual_inputs["state"], expected_inputs["state"], rtol=1e-4, atol=1e-4
        )
        self.assertTrue(
            torch.equal(actual_inputs["conv_state"], expected_inputs["conv_state"])
        )

    @torch.inference_mode()
    def test_non_positive_slots_are_skipped(self):
        inputs = _inputs([0, -1, 4])
        actual_inputs = _clone_caches(inputs)
        actual = _fused(actual_inputs)
        self.assertEqual(torch.count_nonzero(actual[:, :2]).item(), 0)
        self.assertTrue(torch.equal(actual_inputs["state"][0], inputs["state"][0]))
        self.assertTrue(
            torch.equal(actual_inputs["conv_state"][0], inputs["conv_state"][0])
        )
        self.assertFalse(torch.equal(actual_inputs["state"][4], inputs["state"][4]))

    @torch.inference_mode()
    def test_graph_replay_matches_eager(self):
        inputs = _inputs([2, 7])
        eager_inputs = _clone_caches(inputs)
        graph_inputs = _clone_caches(inputs)
        expected = _fused(eager_inputs)
        out = torch.empty_like(expected)
        _fused(_clone_caches(inputs), out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _fused(graph_inputs, out=out)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, expected))
        self.assertTrue(torch.equal(graph_inputs["state"], eager_inputs["state"]))


if __name__ == "__main__":
    unittest.main()
