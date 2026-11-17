"""DraftMoE on CUDA against a torch reference of the routing and experts the spec describes."""

import unittest

import torch

from sglang.srt.layers.draft.ffn import DraftMoE
from sglang.srt.layers.draft.ffn_spec import DraftMixFFNSpec
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import assert_output_close, init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

E, H, I, S, T = 8, 64, 32, 48, 96

QWEN = {  # SpecForge's qwen3_5_moe family: softmax, optional bias, gated shared expert
    "num_hidden_layers": 1,
    "hidden_size": H,
    "hidden_act": "silu",
    "num_experts": E,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": I,
    "norm_topk_prob": False,
    "shared_expert_intermediate_size": S,
    "dflash_config": {"moe_router_enable_expert_bias": True},
}
DEEPSEEK_V3 = {  # sigmoid scores, bias-shifted group-limited top-k, scaled, summed shared
    "num_hidden_layers": 1,
    "hidden_size": H,
    "hidden_act": "silu",
    "n_routed_experts": E,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": I,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "n_group": 4,
    "topk_group": 2,
    "topk_method": "noaux_tc",
    "routed_scaling_factor": 2.5,
    "n_shared_experts": 1,
}
DEEPSEEK_V4 = {  # sqrtsoftplus scores with bias, clamped SwiGLU, summed shared
    "num_hidden_layers": 1,
    "hidden_size": H,
    "hidden_act": "silu",
    "n_routed_experts": E,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": I,
    "norm_topk_prob": True,
    "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,  # the fused kernel supports only DeepSeek-V4's limit
    "n_shared_experts": 1,
    "dflash_config": {"moe_router_enable_expert_bias": True},
}


def _weights(generator, spec):
    """A layer's MoE FFN under SpecForge's export names, in the served bf16."""

    def rand(*shape, scale=0.1):
        return (torch.randn(*shape, generator=generator, device="cuda") * scale).to(
            torch.bfloat16
        )

    w = {  # wide experts when clamped, so the clamp is exercised
        "gate.weight": rand(E, H, scale=0.5),
        "experts.gate_up_proj": rand(
            E, 2 * I, H, scale=3.0 if spec.swiglu_limit else 0.1
        ),
        "experts.down_proj": rand(E, H, I),
    }
    if spec.expert_bias:
        w["gate.expert_bias"] = torch.linspace(-0.2, 0.2, E, device="cuda")
    if spec.shared_expert_intermediate_size:
        size = spec.shared_expert_intermediate_size
        w["shared_expert.gate_proj.weight"] = rand(size, H)
        w["shared_expert.up_proj.weight"] = rand(size, H)
        w["shared_expert.down_proj.weight"] = rand(H, size)
    if spec.shared_expert_gated:
        w["shared_expert_gate.weight"] = rand(1, H, scale=0.5)
    return w


def _scores(logits, scoring_func):
    if scoring_func == "softmax":
        return torch.softmax(logits, dim=-1)
    if scoring_func == "sigmoid":
        return torch.sigmoid(logits)
    return torch.nn.functional.softplus(logits).sqrt()


def _select(choice, spec):
    """Top-k over the bias-shifted scores, restricted to the best groups when grouped."""
    if spec.num_expert_group is None:
        return torch.topk(choice, k=spec.top_k, dim=-1).indices
    groups = choice.view(choice.shape[0], spec.num_expert_group, -1)
    group_score = groups.topk(2, dim=-1).values.sum(
        -1
    )  # DeepSeek-V3: top-2 sum per group
    keep = torch.topk(group_score, k=spec.topk_group, dim=-1).indices
    mask = torch.zeros_like(group_score, dtype=torch.bool).scatter_(1, keep, True)
    masked = groups.masked_fill(~mask.unsqueeze(-1), float("-inf")).view_as(choice)
    return torch.topk(masked, k=spec.top_k, dim=-1).indices


def _act(gate, up, spec):
    if spec.swiglu_limit is None:
        return torch.nn.functional.silu(gate) * up
    limit = spec.swiglu_limit
    return torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


def _reference(x, w, spec):
    """SpecForge's MoELayer in fp32: scores, bias-shifted picks, unshifted weights, experts, shared."""
    x = x.float()
    scores = _scores(x @ w["gate.weight"].float().T, spec.scoring_func)
    bias = w.get("gate.expert_bias")
    ids = _select(scores if bias is None else scores + bias, spec)
    weights = scores.gather(-1, ids)
    if spec.norm_topk_prob:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    weights = weights * spec.routed_scaling_factor
    gate_up, down = w["experts.gate_up_proj"].float(), w["experts.down_proj"].float()
    out = torch.zeros_like(x)
    for slot in range(spec.top_k):
        for expert in range(E):
            rows = ids[:, slot] == expert
            if rows.any():
                gate, up = (x[rows] @ gate_up[expert].T).chunk(2, dim=-1)
                out[rows] += weights[rows, slot, None] * (
                    _act(gate, up, spec) @ down[expert].T
                )
    if spec.shared_expert_intermediate_size:
        gate = x @ w["shared_expert.gate_proj.weight"].float().T
        up = x @ w["shared_expert.up_proj.weight"].float().T
        shared = _act(gate, up, spec) @ w["shared_expert.down_proj.weight"].float().T
        if spec.shared_expert_gated:
            shared = (
                torch.sigmoid(x @ w["shared_expert_gate.weight"].float().T) * shared
            )
        out = out + shared
    return out, ids


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestDraftMoEForward(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29636)

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

    def _block(self, config):
        spec = DraftMixFFNSpec.from_config(config).moe
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                block = DraftMoE(spec, H, layer_id=0)
        finally:
            torch.set_default_dtype(previous)
        generator = torch.Generator(device="cuda").manual_seed(1)
        weights = _weights(generator, spec)
        for name, tensor in weights.items():
            block.load_weight(name, tensor)
        block.check_loaded()
        block.experts.quant_method.process_weights_after_loading(block.experts)
        x = (torch.randn(T, H, generator=generator, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        return block, spec, weights, x

    def _check(self, config):
        block, spec, weights, x = self._block(config)
        with torch.no_grad():
            picks = block.topk(x, block.gate(x)[0].float()).topk_ids
            out = block(x.clone())  # the runner may overwrite its input
        ref, ids = _reference(x, weights, spec)
        self.assertTrue(
            torch.equal(picks.long().sort(dim=-1).values, ids.sort(dim=-1).values),
            "the served router picks other experts than the reference",
        )
        atol = 0.02 * ref.abs().max().item()  # bf16 experts against the fp32 reference
        assert_output_close(self, out, ref, cos_threshold=0.995, rtol=2e-2, atol=atol)

    def test_qwen_biased_softmax_with_gated_shared_expert(self):
        self._check(QWEN)

    def test_qwen_plain_softmax_renormalized_no_shared(self):
        config = dict(QWEN, norm_topk_prob=True, shared_expert_intermediate_size=0)
        config["dflash_config"] = {"moe_router_enable_expert_bias": False}
        self._check(config)

    def test_deepseek_v3_sigmoid_grouped_with_bias_and_summed_shared(self):
        self._check(DEEPSEEK_V3)

    def test_deepseek_v4_sqrtsoftplus_with_clamp(self):
        self._check(DEEPSEEK_V4)

    def test_swiglu_limit_other_than_10_is_refused(self):
        spec = DraftMixFFNSpec.from_config(dict(DEEPSEEK_V4, swiglu_limit=1.0)).moe
        with self.assertRaisesRegex(ValueError, "clamps SwiGLU at 10"):
            DraftMoE(spec, H, layer_id=0)

    def test_cuda_graph_replay_matches_eager(self):
        block, _, _, x = self._block(QWEN)
        static = x[:8].clone()
        with torch.no_grad():
            eager = block(static.clone())
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):  # warm up kernels and allocator before capture
                    block(static.clone())
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = block(static)
            graph.replay()
            torch.cuda.synchronize()
        torch.testing.assert_close(captured, eager, rtol=2e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
