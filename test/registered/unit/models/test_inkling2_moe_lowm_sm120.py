"""The SM12x low-M MoE path is bitwise equal to the general SM120 expert path."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.inkling_common import inkling2_moe as moe
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.inkling2_test_utils import precision_group

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="Requires an SM12x GPU",
)

lowm = moe.moe_lowm_sm120
EXPERTS, TOP_K = 128, 8
# (hidden, intermediate): Inkling-2-Mini's widths, and one whose K is not a
# multiple of 384 so both GEMMs also run their single-K-block tail loop.
SHAPES = [(2304, 1152), (640, 256)]
# Both sides of the fused-route (16) and low-M (64) limits, and of the 8-, 16-
# and 32-row GEMM tiles.
TOKENS = (1, 2, 8, 9, 16, 17, 33, 64)


def _bitwise(actual, expected):
    return torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _mlp(monkeypatch, precision, hidden, intermediate):
    """An Inkling2MoE with random weights of one precision on the current GPU."""
    parallel = SimpleNamespace(
        moe_dp_size=1,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=SimpleNamespace(all_reduce=lambda x: x),
    )
    monkeypatch.setattr(moe, "get_parallel", lambda: parallel)
    prefix = "llm.layers.0.mlp"
    names = [f"model.{prefix}.experts.{name}_weight" for name in ("w13", "w2")]
    quant = Inkling2QuantConfig(
        {"model": {"parameter_groups": [precision_group(precision, names)]}}
    )
    config = SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=intermediate,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        num_experts_per_tok=TOP_K,
        n_routed_experts=EXPERTS,
        route_scale=8.0,
        use_gate_bias=True,
    )
    generator = torch.Generator(device="cuda").manual_seed(hidden)
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        mlp = moe.Inkling2MoE(config, 0, prefix, quant)
        for name, param in mlp.experts.named_parameters():
            shape = tuple(param.shape)
            if name.endswith("outer_scale_denominator"):
                data = torch.full(shape, 2688.0)
            elif name.endswith("amax"):
                data = torch.rand(shape, generator=generator) + 0.5
            elif param.dtype == torch.uint8:
                # MXFP8 scales are UE8M0 exponents, kept near 2^0; NVFP4 weights
                # are packed E2M1 nibbles, where any byte is valid.
                low, high = (121, 130) if name.endswith("_scale") else (0, 256)
                data = torch.randint(
                    low, high, shape, dtype=torch.uint8, generator=generator
                )
            else:
                data = torch.rand(shape, generator=generator) * 2 - 1
            param.weight_loader(param, data.to(param.dtype))
        mlp.experts.swizzle_nvfp4_scales()
        mlp.gate.weight.data.copy_(
            torch.randn(mlp.gate.weight.shape, generator=generator) * 0.05
        )
        mlp.gate.bias.data.copy_(
            torch.randn(mlp.gate.bias.shape, generator=generator) * 0.01
        )
    assert mlp.experts.native_sm120
    return mlp


def _combined(out, hidden):
    """[tokens, hidden] rows of a tensor or a RoutedOutput, combined as the routed
    epilogue does: the FP32 sum of the rows in top-k order, rounded once."""
    if isinstance(out, torch.Tensor):
        return out
    if out.top_k == 0:
        return out.rows
    rows = out.rows[out.src2dst.long()].view(-1, out.top_k, hidden).float()
    total = torch.zeros_like(rows[:, 0])
    for k in range(out.top_k):
        total = total + rows[:, k]
    return total.to(out.rows.dtype)


def _count_low_m_routes(monkeypatch):
    """Calls of the two low-M route kernels, so a closed gate cannot pass as parity."""
    calls = []
    fused, sort_only = lowm.route_topk_meta_lowm, lowm.route_metadata_lowm

    def counted_fused(*args, **kwargs):
        calls.append("fused")
        return fused(*args, **kwargs)

    def counted_sort(*args, **kwargs):
        calls.append("sort")
        return sort_only(*args, **kwargs)

    monkeypatch.setattr(lowm, "route_topk_meta_lowm", counted_fused)
    monkeypatch.setattr(lowm, "route_metadata_lowm", counted_sort)
    return calls


@pytest.mark.parametrize("hidden,intermediate", SHAPES)
@pytest.mark.parametrize("precision", ["nvfp4", "mxfp8"])
def test_low_m_forward_matches_general_path(
    monkeypatch, precision, hidden, intermediate
):
    mlp = _mlp(monkeypatch, precision, hidden, intermediate)
    low_m_routes = _count_low_m_routes(monkeypatch)
    forwards = 0
    generator = torch.Generator(device="cuda").manual_seed(intermediate)
    for tokens in TOKENS:
        x = torch.randn(tokens, hidden, device="cuda", generator=generator).to(
            torch.bfloat16
        )
        # Identical rows send every token to the same 8 experts: `tokens` rows per
        # expert, so several GEMM tiles per expert and a partial last tile.
        same_experts = x[:1].expand(tokens, hidden).contiguous()
        for rows in (x, same_experts):
            for combine in (True, False):
                actual = _combined(mlp(rows, combine=combine), hidden)
                forwards += 1
                with monkeypatch.context() as general:
                    general.setattr(lowm, "can_use_lowm_moe", lambda *a, **k: False)
                    expected = _combined(mlp(rows, combine=combine), hidden)
                assert torch.isfinite(expected.float()).all()
                assert _bitwise(actual, expected), (tokens, combine)
    assert len(low_m_routes) == forwards


def test_forward_hooks_on_gate_or_experts_still_fire(monkeypatch):
    """The fused route calls neither module, so a hooked one takes the module path."""
    mlp = _mlp(monkeypatch, "nvfp4", 640, 256)
    x = torch.randn(4, 640, device="cuda").to(torch.bfloat16)
    expected = mlp(x)
    for module in (mlp.gate, mlp.experts):
        low_m_routes = _count_low_m_routes(monkeypatch)
        fired = []
        handle = module.register_forward_hook(lambda *_: fired.append(1))
        actual = mlp(x)
        handle.remove()
        assert fired == [1]
        assert "fused" not in low_m_routes
        assert _bitwise(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
