"""The DFlash MoE FFN on CUDA against a torch reference of the routing the
config describes: biased or plain softmax top-k (SpecForge's Qwen3.5-MoE
drafts), sigmoid top-k with DeepSeek's router bias, gated and summed shared
experts, FusedMoE experts."""

import unittest

import torch
from transformers import Qwen3Config

from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.models.dflash import DFlashMoE
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import assert_output_close, init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="nightly", runner_config="1-gpu-large")

E, K, H, I, S, T = 8, 2, 64, 32, 48, 96

_BASE = dict(
    hidden_size=H,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=64,
    num_experts_per_tok=K,
    moe_intermediate_size=I,
    dflash_config={"block_size": 4, "target_layer_ids": [0]},
)


def _config(style, **overrides):
    if style == "qwen":
        fields = dict(
            _BASE,
            num_experts=E,
            shared_expert_intermediate_size=S,
            norm_topk_prob=False,
        )
    else:
        fields = dict(
            _BASE,
            n_routed_experts=E,
            n_shared_experts=1,
            scoring_func="sigmoid",
            norm_topk_prob=True,
        )
    fields.update(overrides)
    return Qwen3Config(**fields)


def _weights(generator, style, *, shared, expert_bias):
    """A layer's FFN under the family's export names, in the served bf16."""

    def rand(*shape):
        return (torch.randn(*shape, generator=generator, device="cuda") * 0.1).to(
            torch.bfloat16
        )

    weights = {
        "gate.weight": rand(E, H),
        "experts.gate_up_proj": rand(E, 2 * I, H),
        "experts.down_proj": rand(E, H, I),
    }
    bias_key = "gate.expert_bias" if style == "qwen" else "gate.e_score_correction_bias"
    if expert_bias:
        weights[bias_key] = torch.linspace(-0.2, 0.2, E, device="cuda")
    if shared and style == "qwen":
        weights.update(
            {
                "shared_expert.gate_proj.weight": rand(S, H),
                "shared_expert.up_proj.weight": rand(S, H),
                "shared_expert.down_proj.weight": rand(H, S),
                "shared_expert_gate.weight": rand(1, H),
            }
        )
    elif shared:
        weights.update(
            {
                "shared_experts.gate_proj.weight": rand(I, H),
                "shared_experts.up_proj.weight": rand(I, H),
                "shared_experts.down_proj.weight": rand(H, I),
            }
        )
    return weights


def _reference(x, w, *, scoring_func, renormalize):
    """The routing in fp32: scores (softmax or sigmoid), picks shifted by the
    router bias when the checkpoint has one, unbiased weights, optional
    renormalization; SwiGLU experts; a gated or summed shared expert."""
    x = x.float()
    logits = x @ w["gate.weight"].float().T
    scores = (
        torch.softmax(logits, dim=-1)
        if scoring_func == "softmax"
        else torch.sigmoid(logits)
    )
    bias = w.get("gate.expert_bias", w.get("gate.e_score_correction_bias"))
    choice = scores if bias is None else scores + bias
    ids = torch.topk(choice, k=K, dim=-1).indices
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    gate_up, down = w["experts.gate_up_proj"].float(), w["experts.down_proj"].float()
    out = torch.zeros_like(x)
    for slot in range(K):
        for expert in range(E):
            rows = ids[:, slot] == expert
            if not rows.any():
                continue
            gate, up = (x[rows] @ gate_up[expert].T).chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
            out[rows] += weights[rows, slot, None] * (hidden @ down[expert].T)
    for prefix in ("shared_expert.", "shared_experts."):
        if prefix + "gate_proj.weight" not in w:
            continue
        gate = x @ w[prefix + "gate_proj.weight"].float().T
        up = x @ w[prefix + "up_proj.weight"].float().T
        shared = (torch.nn.functional.silu(gate) * up) @ w[
            prefix + "down_proj.weight"
        ].float().T
        if "shared_expert_gate.weight" in w:
            shared = (
                torch.sigmoid(x @ w["shared_expert_gate.weight"].float().T) * shared
            )
        out += shared
    return out, ids


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestDFlashMoEForward(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29635)

    def setUp(self):
        config = get_context().override_server_args(model_path="dummy")
        config.install()
        self.addCleanup(config.restore)
        initialize_moe_config()
        parallel = get_parallel().override(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            tp_size=1,
            tp_rank=0,
        )
        parallel.__enter__()
        self.addCleanup(parallel.__exit__, None, None, None)

    @staticmethod
    def _block(style, **overrides):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                return DFlashMoE(_config(style, **overrides), layer_id=0)
        finally:
            torch.set_default_dtype(previous)

    def _check(self, style, *, expert_bias, shared, renormalize):
        overrides = dict(norm_topk_prob=renormalize)
        if not shared:
            overrides.update(shared_expert_intermediate_size=0, n_shared_experts=0)
        block = self._block(style, **overrides)
        generator = torch.Generator(device="cuda").manual_seed(1)
        weights = _weights(generator, style, shared=shared, expert_bias=expert_bias)
        for key, tensor in weights.items():
            block.load_weight(key, tensor)
        block.check_loaded("test")
        block.experts.quant_method.process_weights_after_loading(block.experts)
        self.assertEqual(block.has_expert_bias, expert_bias)
        self.assertIn(f"expert_bias={expert_bias}", block.describe())

        x = (torch.randn(T, H, generator=generator, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        with torch.no_grad():
            picks = block.topk(x, block.gate(x)[0]).topk_ids
            out = block(x.clone())  # the runner may overwrite its input
        ref, ids = _reference(
            x, weights, scoring_func=block.spec.scoring_func, renormalize=renormalize
        )
        self.assertTrue(
            torch.equal(picks.long().sort(dim=-1).values, ids.sort(dim=-1).values),
            "the served router picks other experts than the reference",
        )
        assert_output_close(self, out, ref, cos_threshold=0.995, rtol=2e-2, atol=1e-2)

    def test_specforge_biased_softmax_with_a_gated_shared_expert(self):
        self._check("qwen", expert_bias=True, shared=True, renormalize=False)

    def test_plain_softmax_renormalized(self):
        self._check("qwen", expert_bias=False, shared=False, renormalize=True)

    def test_biased_softmax_renormalized(self):
        self._check("qwen", expert_bias=True, shared=True, renormalize=True)

    def test_deepseek_biased_sigmoid_with_summed_shared_experts(self):
        self._check("deepseek", expert_bias=True, shared=True, renormalize=True)


if __name__ == "__main__":
    unittest.main()
