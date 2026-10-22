import pytest
import torch

pytest.importorskip("aiter")
from aiter.jit.utils.chip_info import get_gfx_runtime  # noqa: E402

from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip as ptpc  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_gfx_runtime() != "gfx950"
    or not ptpc.available(),
    reason="PTPC FP8 GEMM needs AITER on gfx950",
)

HIDDEN = 7168
LATENT = 3584
# Not a multiple of 64, so it exercises the zero-padded weight rows.
KDA_ROWS = 6284


def _bf16(shape, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn(shape, generator=generator)
        .mul_(shape[-1] ** -0.5)
        .bfloat16()
        .cuda()
    )


def _assert_close_direction(actual, expected):
    # FP8 keeps about two decimal digits; the model depends on per-row direction.
    cosine = torch.nn.functional.cosine_similarity(
        actual.float(), expected.float(), dim=-1
    )
    assert cosine.min().item() > 0.995, cosine.min().item()


@pytest.mark.parametrize("num_tokens", [1, 2, 8, 64])
@pytest.mark.parametrize(
    "out_features,in_features", [(HIDDEN, LATENT), (KDA_ROWS, HIDDEN)]
)
@torch.inference_mode()
def test_matches_bf16_reference(num_tokens, out_features, in_features):
    x = _bf16((num_tokens, in_features), seed=7 + num_tokens)
    weight = _bf16((out_features, in_features), seed=11 + out_features)
    packed, scale, logical_n = ptpc.pack(weight)
    actual = ptpc.run(x, packed, scale, logical_n)
    assert actual.shape == (num_tokens, out_features)
    _assert_close_direction(actual, x @ weight.t())


@torch.inference_mode()
def test_prequantized_weight_keeps_checkpoint_scales():
    x = _bf16((4, HIDDEN), seed=3)
    weight = _bf16((KDA_ROWS, HIDDEN), seed=5)
    channel_scale = weight.float().abs().amax(dim=1) / 448.0
    fp8_weight = (weight.float() / channel_scale[:, None]).to(torch.float8_e4m3fn)
    packed, scale, logical_n = ptpc.pack_prequantized(fp8_weight, channel_scale)
    assert logical_n == KDA_ROWS
    # Padded rows carry scale 1 and zero weights, so they never leak into outputs.
    assert torch.all(scale[KDA_ROWS:] == 1)
    dequantized = fp8_weight.float() * channel_scale[:, None]
    _assert_close_direction(
        ptpc.run(x, packed, scale, logical_n), x.float() @ dequantized.t()
    )


@torch.inference_mode()
def test_prequantized_input_matches_bf16_input():
    from aiter import dtypes
    from aiter.ops.quant import per_token_quant_hip

    x = _bf16((8, LATENT), seed=13)
    packed, scale, logical_n = ptpc.pack(_bf16((HIDDEN, LATENT), seed=17))
    x_q, x_scale = per_token_quant_hip(x, quant_dtype=dtypes.fp8)
    assert ptpc.covered_prequant(x_q, x_scale)
    from_bf16 = ptpc.run(x, packed, scale, logical_n)
    from_fp8 = ptpc.run(x_q, packed, scale, logical_n, x_scale=x_scale)
    torch.testing.assert_close(from_fp8, from_bf16, rtol=0, atol=0)


@torch.inference_mode()
def test_covered_prequant_rejects_scale_count_mismatch():
    from aiter import dtypes

    x_q = torch.zeros((4, LATENT), dtype=dtypes.fp8, device="cuda")
    assert not ptpc.covered_prequant(x_q, torch.ones((8, 1), device="cuda"))
    assert not ptpc.covered_prequant(
        x_q, torch.ones((4, 1), dtype=torch.bfloat16, device="cuda")
    )


@torch.inference_mode()
def test_run_writes_into_out_and_replays_under_graph_capture():
    x = _bf16((32, LATENT), seed=19)
    packed, scale, logical_n = ptpc.pack(_bf16((HIDDEN, LATENT), seed=23))
    eager = ptpc.run(x, packed, scale, logical_n)
    out = torch.empty_like(eager)
    assert ptpc.run(x, packed, scale, logical_n, out=out) is out
    torch.testing.assert_close(out, eager, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ptpc.run(x, packed, scale, logical_n)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
