"""Unit test for the SM100 trtllm-gen MXFP4 plumbing in :class:`Mxfp4MoEMethod`.

Drives the real chain -- ``create_moe_runner`` -> ``MoeRunner`` -> the fused func
registered for ``("none", "flashinfer_mxfp4")`` -> the SM100 helper -- with only
``trtllm_fp4_block_scale_moe`` faked, so it runs without a GPU. Numerical
equivalence needs Blackwell and lives in ``test_mxfp4_trtllm_gen.py``.

Run anywhere (CPU-only, no flashinfer):

    python -m pytest test/registered/unit/layers/quantization/test_mxfp4_sm100_wiring.py -v
"""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# All distinct, so transposing any two is observable at the kernel call.
NUM_EXPERTS = 32
NUM_LOCAL_EXPERTS = 8
MOE_EP_RANK = 2
EXPECTED_LOCAL_EXPERT_OFFSET = MOE_EP_RANK * NUM_LOCAL_EXPERTS  # 16
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 512
TOP_K = 4
NUM_TOKENS = 6
MXFP4_BLOCK = 32  # MXFP4 block size

# tune_max_num_tokens and output are passed by keyword.
KERNEL_POSITIONAL_ARITY = 26


class _MockLayer:
    """``FusedMoE`` stand-in. Each tensor carries a distinct fill value so an
    assertion can prove *which* tensor reached a given kernel position."""

    def __init__(self) -> None:
        def marked(value, *shape):
            return torch.full(shape, value, dtype=torch.float32)

        self.w13_weight = marked(1.0, NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, 4)
        self.w2_weight = marked(2.0, NUM_EXPERTS, HIDDEN_SIZE, 4)
        self.w13_weight_scale = marked(3.0, NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, 4)
        self.w2_weight_scale = marked(4.0, NUM_EXPERTS, HIDDEN_SIZE, 4)
        self.w13_weight_bias = marked(5.0, NUM_EXPERTS, 2 * INTERMEDIATE_SIZE)
        self.w2_weight_bias = marked(6.0, NUM_EXPERTS, HIDDEN_SIZE)
        self.gemm1_alpha = marked(7.0, NUM_EXPERTS)
        self.gemm1_beta = marked(8.0, NUM_EXPERTS)
        self.gemm1_clamp_limit = marked(9.0, NUM_EXPERTS)
        self.num_experts = NUM_EXPERTS
        self.num_local_experts = NUM_LOCAL_EXPERTS
        self.moe_ep_rank = MOE_EP_RANK


class _MockDispatchOutput:
    """Stand-in for StandardDispatchOutput. ``topk_output`` is a real
    ``BypassedTopKOutput`` so ``TopKOutputChecker.format_is_bypassed``
    (an isinstance check) returns True without distributed init."""

    def __init__(self, hidden_states, router_logits):
        from sglang.srt.layers.moe.topk import BypassedTopKOutput, TopKConfig

        self.hidden_states = hidden_states
        self.topk_output = BypassedTopKOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=TopKConfig(top_k=TOP_K, renormalize=True),
        )


@pytest.fixture(autouse=True)
def _cpu_runnable(monkeypatch):
    """Pin ``_is_cpu`` False so ``apply`` takes the SM100 branch rather than the
    AMX one, and neutralize the symmetric-memory / TP-group allocation wrapper."""
    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as fi_cutlass_mod
    import sglang.srt.layers.quantization.mxfp4 as mxfp4_mod

    monkeypatch.setattr(mxfp4_mod, "_is_cpu", False)
    monkeypatch.setattr(
        fi_cutlass_mod, "use_symmetric_memory", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(fi_cutlass_mod, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(fi_cutlass_mod, "get_tp_group", lambda: None)


def _install_fake_flashinfer(monkeypatch, record):
    """Install recorder functions on ``flashinfer``.

    The SM100 helper imports its two entry points inside the function body, so
    the attributes must be patched on the root module; patching
    flashinfer_cutlass attributes would not intercept the call. Preserve a real
    package when one is installed so imports such as ``flashinfer.comm`` keep
    working, and create a stand-in root module only on CPU-only environments.
    """

    def fake_trtllm_fp4_block_scale_moe(*args, **kwargs):
        record["args"] = args
        record["kwargs"] = kwargs
        return (kwargs["output"],)  # real kernel returns a sequence

    def fake_mxfp8_quantize(x, sf_use_ue8m0, alignment):
        record["mxfp8_quantize"] = {
            "alignment": alignment,
            "sf_use_ue8m0": sf_use_ue8m0,
        }
        x_quant = torch.zeros(x.shape[0], alignment, dtype=torch.uint8)
        # uint8 so the caller's .view(float8_e4m3fn) has a matching itemsize.
        x_scale = torch.zeros(
            x.shape[0] * (alignment // MXFP4_BLOCK), dtype=torch.uint8
        )
        return x_quant, x_scale

    try:
        import flashinfer as flashinfer_mod
    except ImportError:
        flashinfer_mod = types.ModuleType("flashinfer")
        monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_mod)

    monkeypatch.setattr(
        flashinfer_mod, "mxfp8_quantize", fake_mxfp8_quantize, raising=False
    )
    monkeypatch.setattr(
        flashinfer_mod,
        "trtllm_fp4_block_scale_moe",
        fake_trtllm_fp4_block_scale_moe,
        raising=False,
    )


def _build_method(monkeypatch, precision, backend_name="FLASHINFER_MXFP4"):
    """Build an ``Mxfp4MoEMethod`` without ``__init__`` (it reads global server
    args and probes the GPU arch), then let ``create_moe_runner`` wire the real
    runner so the fused-func lookup is part of what the test covers."""
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_flags
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    monkeypatch.setattr(
        get_flags().moe, "runner_backend", getattr(MoeRunnerBackend, backend_name)
    )
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method._fi_kernel = "trtllm_sm100"
    method.use_flashinfer = True
    method.use_marlin = False
    method.hidden_size = HIDDEN_SIZE
    method.intermediate_size_per_partition = INTERMEDIATE_SIZE
    method.flashinfer_mxfp4_moe_precision = precision
    method.create_moe_runner(
        None,
        MoeRunnerConfig(
            num_experts=NUM_EXPERTS,
            num_local_experts=NUM_LOCAL_EXPERTS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size_per_partition=INTERMEDIATE_SIZE,
            top_k=None,
            activation="silu",
            is_gated=True,
        ),
    )
    return method


@pytest.mark.parametrize("precision", ["default", "bf16"])
@pytest.mark.parametrize("model_hidden", [HIDDEN_SIZE, HIDDEN_SIZE - MXFP4_BLOCK])
def test_apply_calls_the_kernel_with_the_expected_args(
    monkeypatch, precision, model_hidden
):
    """``apply`` must reach ``trtllm_fp4_block_scale_moe`` with the same 26
    positional args the pre-refactor inline call used.

    Many args are interchangeably typed (three ``None`` scale scalars, two expert
    counts, several packed tensors), so a value packed into the wrong quant_info
    field or a shifted kernel position would compute the wrong MoE rather than
    raise. Also covers ``trtllm_sm100`` staying in ``create_moe_runner``'s
    accepted set, the fused func resolving from the pool, and the type dispatch
    picking the SM100 helper, since all three sit on this path.

    ``model_hidden`` below ``HIDDEN_SIZE`` exercises the mxfp4 padding path: the
    activations are widened but the output keeps the model's width.
    """
    record = {}
    _install_fake_flashinfer(monkeypatch, record)
    method = _build_method(monkeypatch, precision)
    layer = _MockLayer()
    x = torch.randn(NUM_TOKENS, model_hidden, dtype=torch.bfloat16)
    router_logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32)

    out = method.apply(layer, _MockDispatchOutput(x, router_logits))

    args = record["args"]
    assert len(args) == KERNEL_POSITIONAL_ARITY, f"kernel arity changed: {len(args)}"

    assert args[0].dtype == torch.bfloat16, "router_logits must be cast to bf16"
    assert torch.equal(args[0], router_logits.to(torch.bfloat16))
    assert args[1] is None, "routing_bias"
    if precision == "bf16":
        # No input quant: TRT-LLM quantizes inside the kernel.
        assert "mxfp8_quantize" not in record
        assert args[3] is None, "x_scale"
        if model_hidden == HIDDEN_SIZE:
            assert args[2] is x
        else:
            assert args[2].shape == (NUM_TOKENS, HIDDEN_SIZE)
            assert torch.equal(args[2][:, :model_hidden], x)
            assert torch.all(args[2][:, model_hidden:] == 0), "pad must be zeros"
    else:
        assert record["mxfp8_quantize"]["alignment"] == HIDDEN_SIZE
        assert record["mxfp8_quantize"]["sf_use_ue8m0"] is False
        assert args[3].dtype == torch.float8_e4m3fn
        assert args[3].shape[0] == NUM_TOKENS, "x_scale must stay token-major"
    assert args[4] is layer.w13_weight
    assert args[5] is layer.w13_weight_scale
    assert args[6] is layer.w13_weight_bias
    assert args[7] is layer.gemm1_alpha
    assert args[8] is layer.gemm1_beta
    assert args[9] is layer.gemm1_clamp_limit
    assert args[10] is layer.w2_weight
    assert args[11] is layer.w2_weight_scale
    assert args[12] is layer.w2_weight_bias
    assert args[13] is None, "output1_scale_scalar"
    assert args[14] is None, "output1_scale_gate_scalar"
    assert args[15] is None, "output2_scale_scalar"
    assert args[16] == NUM_EXPERTS, "global_num_experts"
    assert args[17] == TOP_K
    assert args[18] is None, "n_group"
    assert args[19] is None, "topk_group"
    assert args[20] == INTERMEDIATE_SIZE
    assert args[21] == EXPECTED_LOCAL_EXPERT_OFFSET, "local_expert_offset"
    assert args[22] == NUM_LOCAL_EXPERTS, "local_num_experts"
    assert args[23] is None, "routed_scaling_factor"
    assert args[24] == 1, "routing_method_type, renormalize"
    assert args[25] is True, "do finalize"

    kwargs = record["kwargs"]
    assert kwargs["tune_max_num_tokens"] == 8, "next_power_of_2(NUM_TOKENS)"
    output = kwargs["output"]
    assert output.shape == (NUM_TOKENS, model_hidden), "output keeps model width"
    assert output.dtype == torch.bfloat16
    assert out.hidden_states is output


def test_create_moe_runner_rejects_an_unsupported_backend(monkeypatch):
    """An unhandled backend must fail at construction. Red if the else-branch
    goes back to silently passing, which leaves ``self.runner`` unset and defers
    the failure to an ``AttributeError`` inside ``apply``."""
    with pytest.raises(NotImplementedError, match="has no MoeRunner"):
        _build_method(monkeypatch, "bf16", backend_name="FLASHINFER_CUTLASS")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
