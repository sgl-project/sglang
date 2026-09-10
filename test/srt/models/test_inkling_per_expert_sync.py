"""CPU unit test for Inkling per-expert RL weight-sync loading.

Exercises ``_load_per_expert_param`` on a simulated EP x MoE-TP grid (parallel
helpers monkeypatched, no process groups) and checks every (ep_rank, tp_rank)
against a reference fused stack built directly from the full per-expert weights:

  - EP: global expert id remapped to the rank's contiguous local block,
    non-owned experts consumed without touching the stack
  - MoE-TP: w13 slices the intermediate dim (dim 0 of gate/up), w2 dim 1
  - w13 row layout: Inkling-interleaved vs contiguous [gate || up]
    (lora_compatible_layout_enabled() or inference_moe_w13_interleaved=False)
  - trtllm MoE layouts rejected loudly

Run: python3 test/srt/models/test_inkling_per_expert_sync.py
"""

import types
import unittest

import torch

import sglang.srt.models.inkling as inkling_mod

N_EXPERTS, I_FULL, H = 8, 6, 4


class _FakeModel:
    """Just enough of InklingForConditionalGeneration for _load_per_expert_param."""

    def __init__(self, interleaved: bool, moe=None):
        self.text_config = types.SimpleNamespace(
            n_routed_experts=N_EXPERTS,
            inference_moe_w13_interleaved=interleaved,
        )
        self._moe = moe if moe is not None else types.SimpleNamespace()

    def get_submodule(self, path):
        return self._moe

    _load_per_expert_param = (
        inkling_mod.InklingForConditionalGeneration._load_per_expert_param
    )


def _full_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        (e, proj): torch.randn(
            (H, I_FULL) if proj == "down_proj" else (I_FULL, H), generator=g
        )
        for e in range(N_EXPERTS)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }


def _expected_stacks(full, ep_size, ep_rank, tp_size, tp_rank, contiguous):
    """Reference: what the fused w13/w2 stacks must contain on this rank."""
    local = N_EXPERTS // ep_size
    i_tp = I_FULL // tp_size
    w13 = torch.empty(local, 2 * i_tp, H)
    w2 = torch.empty(local, H, i_tp)
    for e_local in range(local):
        e = ep_rank * local + e_local
        gate = full[(e, "gate_proj")][tp_rank * i_tp : (tp_rank + 1) * i_tp]
        up = full[(e, "up_proj")][tp_rank * i_tp : (tp_rank + 1) * i_tp]
        if contiguous:
            w13[e_local] = torch.cat([gate, up], dim=0)
        else:  # Inkling-interleaved rows [g0, u0, g1, u1, ...]
            w13[e_local, 0::2] = gate
            w13[e_local, 1::2] = up
        w2[e_local] = full[(e, "down_proj")][:, tp_rank * i_tp : (tp_rank + 1) * i_tp]
    return w13, w2


class TestPerExpertSync(unittest.TestCase):
    def setUp(self):
        self._saved = {
            n: getattr(inkling_mod, n)
            for n in (
                "get_moe_expert_parallel_world_size",
                "get_moe_expert_parallel_rank",
                "get_moe_tensor_parallel_rank",
                "lora_compatible_layout_enabled",
            )
        }

    def tearDown(self):
        for n, f in self._saved.items():
            setattr(inkling_mod, n, f)

    def _patch(self, ep_size, ep_rank, tp_rank, lora_layout=False):
        inkling_mod.get_moe_expert_parallel_world_size = lambda: ep_size
        inkling_mod.get_moe_expert_parallel_rank = lambda: ep_rank
        inkling_mod.get_moe_tensor_parallel_rank = lambda: tp_rank
        inkling_mod.lora_compatible_layout_enabled = lambda: lora_layout

    def _run_rank(
        self, full, ep_size, ep_rank, tp_size, tp_rank, *, interleaved, lora_layout
    ):
        self._patch(ep_size, ep_rank, tp_rank, lora_layout)
        model = _FakeModel(interleaved)
        local, i_tp = N_EXPERTS // ep_size, I_FULL // tp_size
        params_dict = {
            "model.layers.0.mlp.experts.w13_weight": torch.nn.Parameter(
                torch.full((local, 2 * i_tp, H), float("nan")), requires_grad=False
            ),
            "model.layers.0.mlp.experts.w2_weight": torch.nn.Parameter(
                torch.full((local, H, i_tp), float("nan")), requires_grad=False
            ),
        }
        loaded = set()
        for (e, proj), w in full.items():
            name = f"model.layers.0.mlp.experts.{e}.{proj}.weight"
            self.assertTrue(model._load_per_expert_param(params_dict, loaded, name, w))
        contiguous = lora_layout or not interleaved
        exp_w13, exp_w2 = _expected_stacks(
            full, ep_size, ep_rank, tp_size, tp_rank, contiguous
        )
        got_w13 = params_dict["model.layers.0.mlp.experts.w13_weight"].data
        got_w2 = params_dict["model.layers.0.mlp.experts.w2_weight"].data
        self.assertFalse(torch.isnan(got_w13).any(), "unwritten w13 slots")
        self.assertFalse(torch.isnan(got_w2).any(), "unwritten w2 slots")
        torch.testing.assert_close(got_w13, exp_w13, rtol=0, atol=0)
        torch.testing.assert_close(got_w2, exp_w2, rtol=0, atol=0)
        self.assertEqual(loaded, set(params_dict))

    def test_ep1_tp1_interleaved(self):
        # the validated RL rollout config (weight-checker <=1e-6 on 4layer + 951B)
        self._run_rank(_full_weights(), 1, 0, 1, 0, interleaved=True, lora_layout=False)

    def test_ep_tp_grid_interleaved(self):
        full = _full_weights(1)
        for ep_rank in range(4):
            for tp_rank in range(2):
                self._run_rank(
                    full, 4, ep_rank, 2, tp_rank, interleaved=True, lora_layout=False
                )

    def test_ep_tp_grid_contiguous_layouts(self):
        full = _full_weights(2)
        # contiguous via the LoRA-serving layout and via a non-interleaved config
        for interleaved, lora_layout in ((True, True), (False, False)):
            for ep_rank in range(2):
                self._run_rank(
                    full,
                    2,
                    ep_rank,
                    2,
                    1,
                    interleaved=interleaved,
                    lora_layout=lora_layout,
                )

    def test_trtllm_layout_rejected(self):
        self._patch(1, 0, 0)
        moe = types.SimpleNamespace(use_flashinfer_trtllm_moe=True)
        model = _FakeModel(True, moe=moe)
        params_dict = {
            "model.layers.0.mlp.experts.w13_weight": torch.nn.Parameter(
                torch.zeros(N_EXPERTS, 2 * I_FULL, H), requires_grad=False
            )
        }
        with self.assertRaises(NotImplementedError):
            model._load_per_expert_param(
                params_dict,
                set(),
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                torch.zeros(I_FULL, H),
            )


if __name__ == "__main__":
    unittest.main()
