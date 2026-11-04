"""Which TRT-LLM-gen FP4 MoE tactics a tuning file may pin: every non-split-K tactic
FlashInfer offers at a token bucket must be bit-identical to its default tactic.

Each call writes into a NaN-poisoned output, the test records the tactics it
compared, and a seeded positive control (one perturbed hidden-state row) must
change the output. A tactic that differs from the default is reported as split-K
by kernel name or the test fails.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="nightly", runner_config="4-gpu-b200")

flashinfer = pytest.importorskip("flashinfer")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="needs an SM100-class GPU (TRT-LLM-gen FP4 MoE)",
    ),
]

# Qwen3.8-Flash-Next routed MoE at TP1.
HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 2560, 640, 512, 10
MOE_OP = "flashinfer::trtllm_fp4_block_scale_moe"


@pytest.fixture(scope="module")
def experts():
    gen = torch.Generator(device="cuda").manual_seed(0)
    dev = torch.device("cuda")

    def fp4_bytes(*shape):
        return torch.randint(0, 256, shape, generator=gen, device=dev, dtype=torch.uint8)

    def block_scales(*shape):
        exponent = -6.0 + 12.0 * torch.rand(shape, generator=gen, device=dev)
        return torch.exp2(exponent).to(torch.float8_e4m3fn)

    gate_alpha = 0.01 * (0.5 + torch.rand(EXPERTS, generator=gen, device=dev))
    return dict(
        gemm1_weights=fp4_bytes(EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
        gemm1_weights_scale=block_scales(EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16),
        gemm2_weights=fp4_bytes(EXPERTS, HIDDEN, INTERMEDIATE // 2),
        gemm2_weights_scale=block_scales(EXPERTS, HIDDEN, INTERMEDIATE // 16),
        output1_scale_scalar=(gate_alpha * 270.0).float(),
        output1_scale_gate_scalar=gate_alpha.float(),
        output2_scale_scalar=(
            3.7e-3 * (0.5 + torch.rand(EXPERTS, generator=gen, device=dev))
        ).float(),
    )


def _moe_kwargs(experts, tokens, seed):
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType

    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, generator=gen, device="cuda")
    global_scale = torch.tensor(
        [448.0 * 6.0 / float(x.abs().max())], dtype=torch.float32, device="cuda"
    )
    x_fp4, x_sf = flashinfer.fp4_quantize(x.to(torch.bfloat16), global_scale, 16, False, False)
    return dict(
        routing_logits=torch.randn(tokens, EXPERTS, generator=gen, device="cuda").to(
            torch.bfloat16
        ),
        routing_bias=None,
        hidden_states=x_fp4.reshape(tokens, HIDDEN // 2),
        hidden_states_scale=x_sf.view(torch.float8_e4m3fn).reshape(tokens, HIDDEN // 16),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_bias=None,
        per_token_scale=None,
        num_experts=EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTERMEDIATE,
        local_expert_offset=0,
        local_num_experts=EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.RenormalizeNaive,
        do_finalize=True,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=1 << (tokens - 1).bit_length(),
        enable_pdl=False,
        output=torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda"),
        **experts,
    )


class _ForcedTactic:
    """Routes the MoE op to one tactic and captures its runner and inputs."""

    def __init__(self, monkeypatch):
        from flashinfer.autotuner import AutoTuner

        self.force, self.runner, self.inputs = -1, None, None
        original = AutoTuner.choose_one

        def choose_one(tuner, custom_op, runners, tuning_config, inputs, **kwargs):
            if custom_op != MOE_OP:
                return original(tuner, custom_op, runners, tuning_config, inputs, **kwargs)
            self.runner, self.inputs = runners[0], inputs
            return runners[0], self.force

        monkeypatch.setattr(AutoTuner, "choose_one", choose_one)

    def run(self, kwargs, tactic):
        self.force = tactic
        kwargs["output"].fill_(float("nan"))
        out = flashinfer.trtllm_fp4_block_scale_moe(**kwargs)[0]
        torch.cuda.synchronize()
        return out.clone()


def _kernel_names(run):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        run()
    return {event.name for event in prof.events() if event.device_type.name == "CUDA"}


# Decode buckets (pinned after the decode-shaped autotune) and prefill buckets.
@pytest.mark.parametrize(
    "tokens", [4, 8, 16, 32, 64, 128, 256, 512, 768, 2048, 8192, 16384]
)
def test_non_split_k_tactics_match_the_default(experts, tokens, monkeypatch, record_property):
    forced = _ForcedTactic(monkeypatch)
    kwargs = _moe_kwargs(experts, tokens, seed=tokens)
    default = forced.run(kwargs, -1)
    assert not torch.isnan(default).any(), "the default tactic left rows unwritten"
    tactics = [tuple(t) for t in forced.runner.get_valid_tactics(forced.inputs, None)]

    equal, split_k = [], []
    for tactic in tactics:
        out = forced.run(kwargs, list(tactic))
        if torch.equal(out.view(torch.int16), default.view(torch.int16)):
            equal.append(tactic)
            continue
        names = _kernel_names(lambda: forced.run(kwargs, list(tactic)))
        assert any("splitK" in name for name in names), (
            f"non-split-K tactic {tactic} differs from the default at {tokens} tokens"
        )
        split_k.append(tactic)
    record_property("tactics", {"bitwise": equal, "split_k": split_k})
    assert equal, "no tactic was compared"

    perturbed = dict(kwargs, hidden_states=kwargs["hidden_states"].clone())
    perturbed["hidden_states"][0] ^= 0x11
    control = forced.run(perturbed, list(equal[0]))
    assert not torch.equal(control.view(torch.int16), default.view(torch.int16))
