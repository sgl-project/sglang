"""Quark MXFP4 MoE W4A16 (triton_kernels) path on consumer RDNA.

The path must only replace the AITER path on consumer RDNA (gfx11 / gfx120);
other AMD GPUs, including gfx1250, keep AITER.
With EP > 1 the dispatcher maps experts owned by other ranks to -1; the W4A16
path must drop those slots instead of indexing with -1 (a GPU memory fault).
"""

import importlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4_moe
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
    QuarkW4A4MXFp4MoE,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=40, stage="stage-b", runner_config="1-gpu-small-amd")

try:
    import triton_kernels  # noqa: F401

    _has_triton_kernels = True
except ImportError:
    _has_triton_kernels = False

NUM_EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 8, 512, 256, 2
EP_SIZE = 2
LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE


def _random_weights(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def u8(shape, low=0, high=256):
        return torch.randint(
            low, high, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    # Scales stay near e8m0 bias 127 so outputs remain finite.
    return {
        "w13_weight": u8((NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2)),
        "w2_weight": u8((NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2)),
        "w13_weight_scale": u8((NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32), 118, 124),
        "w2_weight_scale": u8((NUM_EXPERTS, HIDDEN, INTERMEDIATE // 32), 118, 124),
    }


def _build_rank(weights, experts: slice, runner_config: MoeRunnerConfig):
    scheme = QuarkW4A4MXFp4MoE(
        weight_config={"qscheme": "per_group"},
        input_config={"qscheme": "per_group", "is_dynamic": True},
    )
    layer = torch.nn.Module()
    for name, tensor in weights.items():
        param = torch.nn.Parameter(tensor[experts].clone(), requires_grad=False)
        layer.register_parameter(name, param)
    scheme.create_moe_runner(layer, runner_config)
    scheme.process_weights_after_loading(layer)
    return scheme, layer


def _run(scheme, layer, hidden_states, topk_weights, topk_ids):
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None
        ),
    )
    return scheme.apply_weights(layer, dispatch_output).hidden_states


def _to_rank_local(topk_ids: torch.Tensor, rank: int) -> torch.Tensor:
    # Dispatcher contract under EP: local expert id, or -1 when another rank owns it.
    local = topk_ids - rank * LOCAL_EXPERTS
    owned = (local >= 0) & (local < LOCAL_EXPERTS)
    return torch.where(owned, local, -1).to(torch.int32)


def _w4a16_enabled_on(arch: str) -> bool:
    props = SimpleNamespace(gcnArchName=arch)
    with mock.patch("torch.cuda.get_device_properties", return_value=props):
        return importlib.reload(quark_w4a4_mxfp4_moe)._use_triton_w4a16


@unittest.skipUnless(is_hip(), "needs a ROCm build")
class TestQuarkMxfp4MoeW4A16Gate(unittest.TestCase):
    def setUp(self):
        # Re-import with the real device so later tests see the real gate.
        self.addCleanup(importlib.reload, quark_w4a4_mxfp4_moe)

    def test_gfx942_keeps_the_aiter_path(self):
        self.assertFalse(_w4a16_enabled_on("gfx942:sramecc+:xnack-"))

    def test_gfx1250_keeps_the_aiter_path(self):
        # gfx1250 (MI455) also starts with gfx12, so a gfx12 prefix would misfire.
        self.assertFalse(_w4a16_enabled_on("gfx1250"))

    def test_consumer_rdna_uses_the_w4a16_path(self):
        for arch in ("gfx1100", "gfx1151", "gfx1200", "gfx1201:sramecc-:xnack-"):
            with self.subTest(arch=arch):
                self.assertTrue(_w4a16_enabled_on(arch))

    def test_env_switch_disables_the_w4a16_path_on_consumer_rdna(self):
        with envs.SGLANG_ENABLE_QUARK_MXFP4_MOE_W4A16.override(False):
            for arch in ("gfx1151", "gfx1201"):
                with self.subTest(arch=arch):
                    self.assertFalse(_w4a16_enabled_on(arch))


@unittest.skipUnless(
    is_hip() and _has_triton_kernels, "needs a ROCm GPU and triton_kernels"
)
class TestQuarkMxfp4MoeW4A16ExpertParallel(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(quark_w4a4_mxfp4_moe, "_use_triton_w4a16", True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.weights = _random_weights(seed=0)
        self.runner_config = MoeRunnerConfig(activation="silu")

        g = torch.Generator(device="cuda").manual_seed(1)
        num_tokens = 33
        self.hidden_states = (
            torch.randn(
                num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g
            )
            * 0.1
        )
        logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda", generator=g)
        self.topk_weights, topk_ids = torch.topk(logits.softmax(-1), TOP_K, dim=-1)
        self.topk_ids = topk_ids.to(torch.int32)

    def test_ep_rank_outputs_sum_to_single_rank_output(self):
        full = _build_rank(self.weights, slice(None), self.runner_config)
        expected = _run(*full, self.hidden_states, self.topk_weights, self.topk_ids)

        combined = torch.zeros_like(expected, dtype=torch.float32)
        for rank in range(EP_SIZE):
            experts = slice(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS)
            scheme, layer = _build_rank(self.weights, experts, self.runner_config)
            local_ids = _to_rank_local(self.topk_ids, rank)
            self.assertTrue((local_ids < 0).any())
            out = _run(scheme, layer, self.hidden_states, self.topk_weights, local_ids)
            combined += out.float()

        torch.testing.assert_close(combined, expected.float(), rtol=2e-2, atol=2e-2)

    def test_tokens_routed_only_to_other_ranks_produce_zero_output(self):
        experts = slice(0, LOCAL_EXPERTS)
        scheme, layer = _build_rank(self.weights, experts, self.runner_config)
        all_remote = torch.full_like(self.topk_ids, -1)

        out = _run(scheme, layer, self.hidden_states, self.topk_weights, all_remote)

        self.assertEqual(out.shape, self.hidden_states.shape)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_clamped_swiglu_config_is_rejected(self):
        clamped = MoeRunnerConfig(activation="silu", gemm1_clamp_limit=7.0)

        with self.assertRaises(NotImplementedError):
            _build_rank(self.weights, slice(None), clamped)


if __name__ == "__main__":
    unittest.main()
