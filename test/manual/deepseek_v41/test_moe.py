import unittest

import torch
from dsv41_moe import Gate, MoE
from ref_loader import (
    RefTestCase,
    assert_equal,
    randomize_,
    report,
    requires_ref,
    small_args,
)


@requires_ref
class TestMoE(RefTestCase):
    def _gate_kwargs(self, args):
        return dict(
            dim=args.dim,
            n_routed_experts=args.n_routed_experts,
            topk=args.n_activated_experts,
            score_func=args.score_func,
            gate_temp=args.gate_temp,
            norm_topk_prob=args.norm_topk_prob,
            route_scale=args.route_scale,
            has_vl_bias=args.vision_enabled,
        )

    def test_gate(self):
        model = self.model
        for score_func in ("sqrtsoftplus", "sigmoid", "softmax"):
            args = small_args(model, score_func=score_func, vision_n_layers=1)
            ref = model.Gate(0, args)
            randomize_(ref)
            ours = Gate(**self._gate_kwargs(args))
            ours.load_state_dict(ref.state_dict())
            x = torch.randn(16, args.dim)
            image_mask = torch.arange(16) % 3 == 0
            for mask in (None, image_mask):
                exp_w, exp_i = ref(x, mask)
                act_w, act_i = ours(x, mask)
                assert_equal(act_i, exp_i)
                torch.testing.assert_close(act_w, exp_w, rtol=1e-6, atol=1e-7)

    def test_moe(self):
        model = self.model
        args = small_args(model)
        ref = model.MoE(0, args)
        randomize_(ref)
        gate_kwargs = self._gate_kwargs(args)
        gate_kwargs["n_activated_experts"] = gate_kwargs.pop("topk")
        ours = MoE(
            moe_inter_dim=args.moe_inter_dim,
            swiglu_limit=args.swiglu_limit,
            expert_dtype=torch.float4_e2m1fn_x2,
            shared_expert_dtype=torch.float8_e4m3fn,
            **gate_kwargs,
        )
        ours.load_state_dict(ref.state_dict())
        x = torch.randn(2, 5, args.dim)
        expected = ref(x)
        actual = ours(x)
        report("moe", actual, expected)
        torch.testing.assert_close(
            actual, expected, rtol=2**-5, atol=2**-5 * expected.abs().amax().item()
        )


if __name__ == "__main__":
    unittest.main()
