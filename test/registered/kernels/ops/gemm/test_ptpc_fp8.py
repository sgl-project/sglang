import pytest
import torch
from aiter.jit.utils.chip_info import get_gfx_runtime

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="Kimi-K3 PTPC FP8 requires gfx950",
)

HIDDEN = 7168
LATENT = 3584
# The merged KDA input projection is not a multiple of 64, so it exercises the
# zero-padded path; 7168 exercises the unpadded one.
KDA_ROWS = 6288


def _ops():
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    return ptpc_fp8_aiter_hip


@pytest.mark.parametrize("num_tokens", [1, 2, 8, 32, 64])
@pytest.mark.parametrize(
    "out_features,in_features", [(HIDDEN, LATENT), (KDA_ROWS, HIDDEN)]
)
@torch.inference_mode()
def test_ptpc_fp8_matches_bf16_reference(num_tokens, out_features, in_features):
    ptpc = _ops()
    if not ptpc.available():
        pytest.skip("aiter PTPC FP8 GEMM unavailable")
    generator = torch.Generator(device="cpu").manual_seed(7 + num_tokens + out_features)
    x = torch.randn((num_tokens, in_features), generator=generator).bfloat16().cuda()
    weight = (
        torch.randn((out_features, in_features), generator=generator)
        .mul_(in_features**-0.5)
        .bfloat16()
        .cuda()
    )
    packed, scale, logical_n = ptpc.pack(weight)
    assert logical_n == out_features

    actual = ptpc.run(x, packed, scale, logical_n)
    torch.cuda.synchronize()
    assert actual.shape == (num_tokens, out_features)
    assert actual.dtype == torch.bfloat16

    expected = torch.mm(x, weight.t())
    # FP8 e4m3 carries ~2 decimal digits, so compare direction rather than
    # bitwise magnitude: per-row cosine similarity is the property the model
    # actually depends on.
    cosine = torch.nn.functional.cosine_similarity(
        actual.float(), expected.float(), dim=-1
    )
    assert cosine.min().item() > 0.995, cosine.min().item()
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel.item() < 0.06, rel.item()


@torch.inference_mode()
def test_ptpc_fp8_replays_under_graph_capture():
    ptpc = _ops()
    if not ptpc.available():
        pytest.skip("aiter PTPC FP8 GEMM unavailable")
    generator = torch.Generator(device="cpu").manual_seed(11)
    x = torch.randn((32, LATENT), generator=generator).bfloat16().cuda()
    weight = (
        torch.randn((HIDDEN, LATENT), generator=generator)
        .mul_(LATENT**-0.5)
        .bfloat16()
        .cuda()
    )
    packed, scale, logical_n = ptpc.pack(weight)
    eager = ptpc.run(x, packed, scale, logical_n).clone()

    ptpc.warmup(packed, scale, logical_n, LATENT, token_buckets=(32,))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ptpc.run(x, packed, scale, logical_n)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@torch.inference_mode()
def test_pack_rejects_non_2d_weight():
    ptpc = _ops()
    if not ptpc.available():
        pytest.skip("aiter PTPC FP8 GEMM unavailable")
    with pytest.raises(ValueError):
        ptpc.pack(torch.zeros((2, 3, 4), dtype=torch.bfloat16, device="cuda"))
