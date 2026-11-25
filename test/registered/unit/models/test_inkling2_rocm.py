"""Inkling2's ROCm kernels against float64 / torch references, and their run-to-run
determinism (two ROCm Triton miscompilations showed up only as nondeterminism)."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

# backend-specific: the ROCm kernels and dispatch Inkling2 runs on MI35x
register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires a ROCm GPU",
)

H, HKV, D, EXT = 6, 1, 128, 1024
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


@pytest.fixture(autouse=True)
def _no_grad():
    with torch.no_grad():
        yield


def test_fused_kv_write_declines_translated_locs(monkeypatch):
    """A unified (translating) KV pool hands the batch virtual write locs, which the
    prologue must not scatter into: decode graph capture raised on them."""
    from types import SimpleNamespace

    from sglang.srt.layers.attention import triton_backend

    monkeypatch.setattr(triton_backend, "_is_hip", True)
    k_buffer = torch.zeros(4, 1, 8, dtype=torch.bfloat16)
    v_buffer = torch.zeros(4, 1, 8, dtype=torch.bfloat16)
    kv_layers = SimpleNamespace(
        read=lambda layer_id: SimpleNamespace(kv=lambda: (k_buffer, v_buffer)),
        route=lambda layer_id: SimpleNamespace(sub_pool="full"),
    )
    layer = SimpleNamespace(
        layer_id=0, is_cross_attention=False, k_scale=None,
        tp_k_head_num=1, tp_v_head_num=1, qk_head_dim=8, v_head_dim=8,
    )  # fmt: skip
    batch = SimpleNamespace(
        out_cache_loc=torch.zeros(2, dtype=torch.int64), out_cache_loc_is_physical=True
    )
    for translating in (True, False):
        backend = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(dtype=torch.bfloat16),
            kv_layers=kv_layers,
            use_mla=False,
            window_write_sub_pools=(),
            dcp_size=1,
            runner_ids=SimpleNamespace(is_translating=translating),
            _window_write_locs=lambda: None,
        )
        buffers = triton_backend.TritonAttnBackend.get_fused_kv_write_buffers(
            backend, layer, batch
        )
        if translating:
            assert buffers is None
        else:
            assert buffers[0] is k_buffer and buffers[1] is v_buffer
            assert torch.equal(buffers[2], batch.out_cache_loc)


@pytest.mark.parametrize("rows", [1, 64, 128])
def test_apply_wo_ud_matches_hipblaslt(rows):
    """Attention's wo_ud on ROCm at TP8 (K 768): aiter's Triton GEMM where it routes, up to
    64 rows, is bitwise F.linear (hipBLASLt); larger batches keep the layer's own path."""
    from sglang.srt.models.inkling_common import inkling2_attn_hip

    if inkling2_attn_hip._aiter_a16w16 is None:
        pytest.skip("aiter's Triton GEMM needs gfx95")

    class OutProjection(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight, self.bias, self.calls = weight, None, 0

        def forward(self, x):
            self.calls += 1
            return torch.nn.functional.linear(x, self.weight), None

    torch.manual_seed(rows)
    wo_ud = OutProjection((torch.randn(4608, 768, device="cuda") * 0.03).bfloat16())
    for scale in (1, 3):
        x = (torch.randn(rows, 768, device="cuda") * scale).bfloat16()
        out = inkling2_attn_hip.apply_wo_ud(wo_ud, x)
        assert torch.equal(out, torch.nn.functional.linear(x, wo_ud.weight))
    assert wo_ud.calls == (0 if rows <= 64 else 2)


@pytest.mark.parametrize("precision", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [3, 40, 600])
def test_experts_load_in_mfma_order(monkeypatch, precision, tokens):
    """ROCm quantized experts load in MFMA order: their forward (Gluon at 6 and 80
    routed rows, the Triton MFMA-order tile at 1,200) is bitwise the same experts
    written in the checkpoint layout, and a reload rewrites the buffers in place."""
    from types import SimpleNamespace

    from sglang.kernels.ops.moe import inkling2_experts_hip as experts_hip
    from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.inkling_common import inkling2_moe as moe
    from sglang.test.inkling2_test_utils import precision_group

    if not experts_hip.MFMA_ORDER:
        pytest.skip("MFMA-order experts need gfx950 and Gluon")
    parallel = SimpleNamespace(
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=SimpleNamespace(all_reduce=lambda x: x),
    )
    monkeypatch.setattr(moe, "get_parallel", lambda: parallel)
    prefix = "model.llm.layers.0.mlp.experts."
    names = [prefix + name + "_weight" for name in ("w13", "w2")]
    quant = Inkling2QuantConfig(
        {"model": {"parameter_groups": [precision_group(precision, names)]}}
    )
    config = SimpleNamespace(
        hidden_size=256,
        intermediate_size=256,
        mlp_gate_softcap=4,
        mlp_up_softcap=25,
        num_experts_per_tok=2,
        route_scale=8.0,
    )
    torch.manual_seed(tokens)
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        plain = moe.Inkling2Experts(config, 8, prefix, quant)
        loaded = moe.Inkling2Experts(config, 8, prefix, quant)
        x = torch.randn(tokens, 256)
        ids = torch.randint(0, 8, (tokens, 2), dtype=torch.int32)
        gammas = torch.rand(tokens, 2, dtype=torch.float32)
        buffers = [p.data_ptr() for p in loaded.parameters()]
        for _ in range(2):
            checkpoint = {}
            for name, value in plain.named_parameters():
                if name.endswith("outer_scale_denominator"):
                    checkpoint[name] = torch.full_like(value, 2688)
                elif value.dtype == torch.uint8:
                    # NVFP4 weights; MXFP8 ue8m0 scales near 1
                    checkpoint[name] = (
                        torch.randint_like(value, 0, 256)
                        if precision == "nvfp4"
                        else torch.randint_like(value, 120, 130)
                    )
                else:
                    checkpoint[name] = (torch.rand(value.shape) + 0.125).to(value.dtype)
                value.copy_(checkpoint[name])
            for name, value in loaded.named_parameters():
                value.weight_loader(value, checkpoint[name])
            plain.swizzle_nvfp4_scales()
            loaded.swizzle_nvfp4_scales()
            assert [p.data_ptr() for p in loaded.parameters()] == buffers
            assert loaded.in_mfma_order("w13") and loaded.in_mfma_order("w2")
            assert not plain.in_mfma_order("w13") and not plain.in_mfma_order("w2")
            assert not torch.equal(loaded.w13_weight, plain.w13_weight)
            expected = plain(x, gammas, ids)
            actual = loaded(x, gammas, ids)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
