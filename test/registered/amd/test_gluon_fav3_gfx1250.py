import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="nightly-amd-1-gpu-mi45x", nightly=True)


def _is_gfx1250() -> bool:
    if not torch.cuda.is_available() or not torch.version.hip:
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx1250" in arch


def test_persistent_launch_selection_cpu():
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        select_fav3_launch_config,
    )

    config = select_fav3_launch_config(
        2,
        40,
        176_400,
        seqlen_k=512,
        num_cus=1,
    )
    assert config.schedule == "persistent"
    assert (config.block_m, config.block_n, config.num_warps) == (128, 256, 4)
    assert config.tdm_output
    assert config.llvm_fn_attrs == "amdgpu-sched-strategy=iterative-ilp"

    non_wan = select_fav3_launch_config(
        1,
        40,
        32_768,
        seqlen_k=511,
        num_cus=256,
    )
    assert non_wan.schedule == "persistent"

    underfilled = select_fav3_launch_config(
        1,
        8,
        4_096,
        seqlen_k=512,
        num_cus=256,
    )
    assert underfilled.schedule != "persistent"

    for unsupported_k in (256, 513):
        unsupported = select_fav3_launch_config(
            2,
            40,
            176_400,
            seqlen_k=unsupported_k,
            num_cus=256,
        )
        assert unsupported.schedule != "persistent"

    stable_self_config = select_fav3_launch_config(
        2,
        40,
        176_400,
        seqlen_k=176_400,
        num_cus=256,
    )
    assert stable_self_config.schedule == "pingpong"
    single_video = select_fav3_launch_config(
        1, 40, 176_400, seqlen_k=176_400, num_cus=256
    )
    assert single_video.schedule == "pingpong"


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize(
    "batch,seqlen_q,seqlen_k,num_heads,expected_schedule",
    [
        (1, 129, 65, 8, "pipeline"),
        (1, 300, 513, 8, "pipeline"),
        (1, 2050, 512, 40, "pingpong"),
        (4, 176_400, 513, 40, "pingpong"),
        (1, 131_073, 257, 40, "persistent"),
        (1, 176_400, 511, 40, "persistent"),
        (2, 176_400, 512, 40, "persistent"),
        (1, 176_400, 176_400, 40, "pingpong"),
    ],
    ids=[
        "short-k-tail",
        "partial-rectangular-pipeline",
        "wide-pingpong-tdm",
        "wan-b4-large-batch-offset",
        "persistent-balanced-grid-q-kv-tails",
        "non-wan-long-persistent-short-kv",
        "wan-cross-persistent-large-offset",
        "wan-b1-self-stable",
    ],
)
def test_gluon_fav3_matches_aiter(
    batch, seqlen_q, seqlen_k, num_heads, expected_schedule
):
    aiter = pytest.importorskip("aiter")
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        select_fav3_launch_config,
    )
    from sglang.multimodal_gen.runtime.layers.attention.layer import (
        LocalAttention,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    torch.manual_seed(0)
    query = torch.randn(
        batch,
        seqlen_q,
        num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        batch,
        seqlen_k,
        num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)

    expected = aiter.flash_attn_func(
        query,
        key,
        value,
        dropout_p=0.0,
        causal=False,
        return_attn_probs=False,
        return_lse=False,
    )
    attention = LocalAttention(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_size=128,
        softmax_scale=128**-0.5,
        required_attention_backend=AttentionBackendEnum.GLUON_FAV3,
        compute_dtype=torch.bfloat16,
        is_cross_attention=seqlen_q != seqlen_k,
    )
    assert attention.backend is AttentionBackendEnum.GLUON_FAV3
    with set_forward_context(current_timestep=0, attn_metadata=None):
        actual = attention(query, key, value)
    torch.cuda.synchronize()

    config = select_fav3_launch_config(
        batch,
        num_heads,
        seqlen_q,
        seqlen_k=seqlen_k,
    )
    assert config.schedule == expected_schedule
    assert config.tdm_output is (expected_schedule in ("pingpong", "persistent"))
    if expected_schedule == "persistent":
        assert config.llvm_fn_attrs == "amdgpu-sched-strategy=iterative-ilp"
    elif seqlen_k <= 512:
        assert config.llvm_fn_attrs == "amdgpu-sched-strategy=max-ilp"
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=0.04,
        atol=0.04,
    )


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
def test_gluon_fav3_rejects_unsafe_options():
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        FAv3LaunchConfig,
        gluon_fav3_attention,
    )

    query = torch.randn(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="softmax_scale"):
            gluon_fav3_attention(query, key, value, softmax_scale=scale)

    invalid_config = FAv3LaunchConfig("pipeline", 64, 64, 4, False, "")
    with pytest.raises(ValueError, match="requires.*block_m"):
        gluon_fav3_attention(
            query,
            key,
            value,
            launch_config=invalid_config,
        )


def _forced_config(schedule):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import FAv3LaunchConfig

    geometry = {
        "pipeline": (128, 64, 4, False),
        "pingpong": (256, 64, 8, True),
        "persistent": (128, 256, 4, True),
    }
    return FAv3LaunchConfig(schedule, *geometry[schedule], "")


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("schedule", ["pipeline", "pingpong", "persistent"])
@pytest.mark.parametrize("key_value", [66048.0, -66048.0])
def test_large_uniform_logits_remain_finite(schedule, key_value):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import gluon_fav3_attention

    query = torch.full((1, 129, 1, 128), 66048.0, device="cuda", dtype=torch.bfloat16)
    key = torch.full((1, 257, 1, 128), key_value, device="cuda", dtype=torch.bfloat16)
    value = torch.ones_like(key)
    actual = gluon_fav3_attention(
        query, key, value, launch_config=_forced_config(schedule)
    )
    # Every logit is equal; the exact output is the mean of V, regardless of
    # logit magnitude/sign. This also exercises a masked second KV tile.
    torch.testing.assert_close(actual, torch.ones_like(query), rtol=0, atol=0)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("schedule", ["pipeline", "pingpong", "persistent"])
def test_late_distinct_outliers_preserve_softmax_weights(schedule):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import gluon_fav3_attention

    query = torch.ones((1, 129, 1, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.zeros((1, 257, 1, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    # Separate high logits across KV tiles. Clipping them to the same exponent
    # incorrectly returns 0.5 instead of allowing the second key to dominate.
    key[:, 128].fill_(8.0)
    key[:, 256].fill_(16.0)
    value[:, 256].fill_(1.0)
    actual = gluon_fav3_attention(
        query, key, value, launch_config=_forced_config(schedule)
    )
    torch.testing.assert_close(actual, torch.ones_like(query), rtol=0, atol=0)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("schedule", ["pipeline", "pingpong"])
def test_query_span_beyond_4gib(schedule):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import gluon_fav3_attention

    free_memory, _ = torch.cuda.mem_get_info()
    if free_memory < 12 * 1024**3:
        pytest.skip("requires 12 GiB free for the large query and output")
    query = torch.zeros((1, 262_145, 64, 128), device="cuda", dtype=torch.bfloat16)
    # This row starts exactly 4 GiB after the first query row.
    query[:, 262_144:].fill_(1.0)
    key = torch.zeros((1, 65, 64, 128), device="cuda", dtype=torch.bfloat16)
    key[:, 32:].fill_(1.0)
    value = key.clone()
    actual = gluon_fav3_attention(
        query, key, value, launch_config=_forced_config(schedule)
    )
    expected_late = torch.ones_like(actual[:, -1])
    torch.testing.assert_close(actual[:, -1], expected_late, rtol=0, atol=0)
    expected_early = torch.full_like(actual[:, 0], 33 / 65)
    torch.testing.assert_close(actual[:, 0], expected_early, rtol=0.01, atol=0.001)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
def test_grouped_query_fallback_preserves_custom_scale():
    from sglang.multimodal_gen.runtime.layers.attention.backends.gluon_fav3 import (
        GluonFAv3Impl,
    )

    query = torch.ones((1, 64, 2, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.zeros((1, 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    key[:, 32:].fill_(1.0)
    value = key.clone()
    impl = GluonFAv3Impl(
        num_heads=2, num_kv_heads=1, head_size=128, softmax_scale=0.005
    )
    actual = impl.forward(query, key, value)
    expected = torch.full_like(actual, torch.sigmoid(torch.tensor(0.64)).item())
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.001)


@pytest.mark.parametrize("shape", [(0, 1, 1, 128), (1, 1, 0, 128), (1, 0, 1, 128)])
def test_persistent_rejects_empty_dimensions_cpu(shape):
    from sglang.kernels.ops.attention.gluon_fav3_persistent_gfx1250 import (
        gluon_fav3_persistent_attention,
    )

    query = torch.empty(shape, dtype=torch.bfloat16)
    key = torch.empty((shape[0], 257, shape[2], 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="non-empty batches, heads, and query"):
        gluon_fav3_persistent_attention(query, key, key)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("schedule", ["pipeline", "pingpong"])
def test_strided_query_tile_span_beyond_4gib(schedule):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import gluon_fav3_attention

    free_memory, _ = torch.cuda.mem_get_info()
    if free_memory < 6 * 1024**3:
        pytest.skip("requires 6 GiB free for sparse query storage")
    row_stride = 2**25
    storage = torch.empty(64 * row_stride + 128, device="cuda", dtype=torch.bfloat16)
    query = storage.as_strided((1, 65, 1, 128), (65 * row_stride, row_stride, 128, 1))
    # Initialize only the logical rows, leaving the large gaps untouched.
    query.zero_()
    query[:, -1].fill_(1.0)
    key = torch.zeros((1, 65, 1, 128), device="cuda", dtype=torch.bfloat16)
    key[:, 32:].fill_(1.0)
    value = key.clone()
    actual = gluon_fav3_attention(
        query, key, value, launch_config=_forced_config(schedule)
    )

    # Row 64 has a 4-GiB byte offset within the same query tile as row zero.
    # A wrapped buffer offset incorrectly reloads row zero instead.
    expected = torch.full_like(actual, 33 / 65)
    expected[:, -1].fill_(1.0)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.001)


def _fp64_attention(query, key, value, scale):
    query = query.transpose(1, 2).double()
    key = key.transpose(1, 2).double()
    value = value.transpose(1, 2).double()
    probabilities = torch.softmax(query @ key.transpose(-1, -2) * scale, dim=-1)
    return (probabilities @ value).transpose(1, 2)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("schedule", ["pipeline", "pingpong", "persistent"])
@pytest.mark.parametrize("case", ["random", "small-scale", "sharp", "permuted-kv"])
def test_gluon_fav3_matches_independent_fp64(schedule, case):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import gluon_fav3_attention

    torch.manual_seed(20261015)
    seqlen_k = 511 if schedule == "persistent" else 769
    if case == "permuted-kv" and schedule != "persistent":
        seqlen_k = 16_385
    query = torch.randn(2, 257, 3, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, seqlen_k, 3, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    scale = 0.005 if case == "small-scale" else 128**-0.5
    if case == "sharp":
        query.mul_(4)
        key.mul_(4)
    expected = _fp64_attention(query, key, value, scale)
    if case == "permuted-kv":
        # Attention is invariant to jointly permuting K and V.
        order = torch.randperm(seqlen_k, device="cuda")
        key = key.index_select(1, order)
        value = value.index_select(1, order)
    actual = gluon_fav3_attention(
        query, key, value, softmax_scale=scale, launch_config=_forced_config(schedule)
    ).double()

    assert torch.isfinite(actual).all()
    reference_rms = expected.square().mean().sqrt()
    error_rms = (actual - expected).square().mean().sqrt()
    # Include final BF16 rounding, while rejecting the systematic probability
    # truncation bias. A separate pointwise bound catches isolated bad outputs.
    assert error_rms <= 0.003 * reference_rms
    torch.testing.assert_close(
        actual, expected, rtol=0.01, atol=0.01 * reference_rms.item()
    )


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("schedule", ["pipeline", "pingpong", "persistent"])
def test_random_logits_preserve_constant_values(schedule):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import gluon_fav3_attention

    torch.manual_seed(41969)
    seqlen_k = 511 if schedule == "persistent" else 513
    query = torch.randn(2, 129, 3, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, seqlen_k, 3, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.ones_like(key)
    actual = gluon_fav3_attention(
        query, key, value, launch_config=_forced_config(schedule)
    )
    # Any probability distribution has a weighted mean of one for V=1.
    # Truncating every probability toward zero biased these rows by one BF16 ULP.
    torch.testing.assert_close(actual, torch.ones_like(query), rtol=0, atol=0)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("seqlen_k", [191, 192, 193, 255, 256, 257])
@pytest.mark.parametrize("scheduler", ["", "max-ilp"], ids=["default", "max-ilp"])
def test_pingpong_short_kv_boundary_matches_fp64(seqlen_k, scheduler):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        FAv3LaunchConfig,
        gluon_fav3_attention,
    )

    torch.manual_seed(41969)
    query = torch.randn(2, 257, 3, 128, device="cuda", dtype=torch.bfloat16)
    # Keep the batch stride larger than the visible KV length.
    key = torch.randn(2, 769, 3, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key, value = key[:, :seqlen_k], value[:, :seqlen_k]
    expected = _fp64_attention(query, key, value, 128**-0.5)
    reference_rms = expected.square().mean().sqrt()
    config = FAv3LaunchConfig(
        "pingpong",
        256,
        64,
        8,
        True,
        f"amdgpu-sched-strategy={scheduler}" if scheduler else "",
    )
    # The previous one-iteration pipeline could race; repeat identical inputs.
    for _ in range(3):
        actual = gluon_fav3_attention(query, key, value, launch_config=config).double()
        assert torch.isfinite(actual).all()
        assert (actual - expected).square().mean().sqrt() <= 0.003 * reference_rms
        torch.testing.assert_close(
            actual, expected, rtol=0.01, atol=0.01 * reference_rms.item()
        )


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("seqlen_k", [192, 193, 256])
def test_auto_short_kv_boundary_matches_fp64(seqlen_k):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        gluon_fav3_attention,
        select_fav3_launch_config,
    )

    torch.manual_seed(173)
    query = torch.randn(1, 4097, 40, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, seqlen_k, 40, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    config = select_fav3_launch_config(1, 40, 4097, seqlen_k=seqlen_k)
    assert config.schedule == "pingpong"
    rows = torch.tensor([0, 127, 128, 255, 256, 2048, 4096], device="cuda")
    expected = _fp64_attention(query.index_select(1, rows), key, value, 128**-0.5)
    reference_rms = expected.square().mean().sqrt()
    actual = gluon_fav3_attention(query, key, value)
    assert torch.isfinite(actual).all()
    actual = actual.index_select(1, rows).double()
    assert (actual - expected).square().mean().sqrt() <= 0.003 * reference_rms
    torch.testing.assert_close(
        actual, expected, rtol=0.01, atol=0.01 * reference_rms.item()
    )


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("seqlen_k", [193, 769])
@pytest.mark.parametrize(
    "case",
    [
        "random",
        "below",
        "above",
        "large-positive",
        "large-negative",
        "strided",
        "single-outlier",
    ],
)
def test_guarded_exponent_matches_fp64(seqlen_k, case):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        _guard_bound,
        _launch_guarded_attention,
    )

    torch.manual_seed(8459)
    stride = 2 if case == "strided" else 1
    query = torch.randn(2, 257 * stride, 3, 128, device="cuda", dtype=torch.bfloat16)[
        :, ::stride
    ]
    key = torch.randn(
        2, seqlen_k * stride, 3, 128, device="cuda", dtype=torch.bfloat16
    )[:, ::stride]
    value = torch.randn_like(key)
    scale = 128**-0.5
    bound = _guard_bound(scale)
    low = torch.tensor(bound, dtype=torch.bfloat16)
    if low.item() > bound:
        low = torch.nextafter(low, torch.tensor(0.0, dtype=torch.bfloat16))
    high = torch.nextafter(low, torch.tensor(float("inf"), dtype=torch.bfloat16))
    if case in ("below", "above"):
        query.fill_(low.item() if case == "below" else high.item())
        key.fill_(low.item() if case == "below" else high.item())
    elif case in ("large-positive", "large-negative"):
        query.fill_(66048)
        key.fill_(66048 if case == "large-positive" else -66048)
    elif case == "single-outlier":
        query[-1, -1, -1, -1] = 66048
    expected = (
        value.double().mean(1, keepdim=True).expand(query.shape)
        if case in ("below", "above", "large-positive", "large-negative")
        else _fp64_attention(query, key, value, scale)
    )
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    invalid = torch.empty((), device=query.device, dtype=torch.int32)
    _launch_guarded_attention(query, key, value, output, invalid, scale)
    fallback = case in ("above", "large-positive", "large-negative", "single-outlier")
    assert invalid.item() == int(fallback)
    actual = output.double()
    reference_rms = expected.square().mean().sqrt()
    assert torch.isfinite(actual).all()
    assert (actual - expected).square().mean().sqrt() <= 0.003 * reference_rms
    torch.testing.assert_close(
        actual, expected, rtol=0.01, atol=0.01 * reference_rms.item()
    )


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
def test_guard_recomputed_on_graph_replay():
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        _launch_guarded_attention,
    )

    torch.manual_seed(20261015)
    query = torch.randn(2, 257, 3, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 769, 3, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    original_query, original_key = query.clone(), key.clone()
    output = torch.empty_like(query)
    invalid = torch.empty((), device=query.device, dtype=torch.int32)
    scale = 128**-0.5

    def launch():
        _launch_guarded_attention(query, key, value, output, invalid, scale)

    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for fallback in (False, True, False, True, False):
        if fallback:
            query.fill_(66048)
            key.fill_(-66048)
            expected = value.double().mean(1, keepdim=True).expand(query.shape)
        else:
            query.copy_(original_query)
            key.copy_(original_key)
            expected = _fp64_attention(query, key, value, scale)
        invalid.fill_(int(not fallback))
        graph.replay()
        assert invalid.item() == int(fallback)
        reference_rms = expected.square().mean().sqrt()
        actual = output.double()
        assert torch.isfinite(actual).all()
        assert (actual - expected).square().mean().sqrt() <= 0.003 * reference_rms
        torch.testing.assert_close(
            actual, expected, rtol=0.01, atol=0.01 * reference_rms.item()
        )


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_exponent_guard_rejects_nonfinite_inputs(nonfinite):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import _check_bounded_inputs

    query = torch.zeros((1, 257, 1, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.zeros((1, 769, 1, 128), device="cuda", dtype=torch.bfloat16)
    key[0, -1, 0, -1] = nonfinite
    invalid = torch.empty((), device=query.device, dtype=torch.int32)
    _check_bounded_inputs(query, key, invalid, 128**-0.5)
    assert invalid.item() == 1


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
def test_exponent_guard_checks_values_beyond_4gib():
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import _check_bounded_inputs

    free_memory, _ = torch.cuda.mem_get_info()
    if free_memory < 12 * 1024**3:
        pytest.skip("requires 12 GiB free for the large query")
    query = torch.zeros((1, 262_145, 64, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.zeros((1, 65, 64, 128), device="cuda", dtype=torch.bfloat16)
    invalid = torch.empty((), device=query.device, dtype=torch.int32)
    # This element is past both the 4 GiB byte boundary and signed int32 indexing.
    query[0, -1, -1, -1] = 16
    _check_bounded_inputs(query, key, invalid, 128**-0.5)
    assert invalid.item() == 1
    query[0, -1, -1, -1] = 0
    _check_bounded_inputs(query, key, invalid, 128**-0.5)
    assert invalid.item() == 0


def _assert_norm_attention_fp64(query, key, value, output):
    expected = _fp64_attention(query, key, value, 128**-0.5)
    actual = output.double()
    rms = expected.square().mean().sqrt()
    assert torch.isfinite(actual).all()
    assert (actual - expected).square().mean().sqrt() <= 0.003 * rms
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01 * rms.item())


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("seqlen_k", [193, 769])
@pytest.mark.parametrize(
    "case,expected_bits",
    [
        ("random", 0),
        ("strided", 0),
        ("below", 0),
        ("above", 1),
        ("sparse", 2),
        ("negative", 0),
        ("tiny-value", 1),
        ("large-value", 1),
        ("value-low", 0),
        ("value-high", 0),
        ("extreme", 3),
        ("constant", 0),
    ],
)
def test_norm_guarded_attention_matches_fp64(seqlen_k, case, expected_bits):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        _launch_norm_guarded_attention,
        _norm_guard_bound,
    )

    torch.manual_seed(8459)
    step = 2 if case == "strided" else 1
    query = torch.randn(2, 257 * step, 3, 128, device="cuda", dtype=torch.bfloat16)[
        :, ::step
    ]
    key = torch.randn(2, seqlen_k * step, 3, 128, device="cuda", dtype=torch.bfloat16)[
        :, ::step
    ]
    value = torch.randn_like(key)
    limit = _norm_guard_bound(128**-0.5)
    low = torch.tensor((limit / 128) ** 0.5, dtype=torch.bfloat16)
    if low.item() ** 2 * 128 > limit:
        low = torch.nextafter(low, torch.tensor(0.0, dtype=torch.bfloat16))
    high = torch.nextafter(low, torch.tensor(float("inf"), dtype=torch.bfloat16))
    if case in ("below", "above", "negative"):
        query.fill_(high.item() if case == "above" else low.item())
        key.fill_(-low.item() if case == "negative" else query[0, 0, 0, 0].item())
    elif case == "sparse":
        query.zero_()
        key.zero_()
        query[..., 0] = 15.0
        key[..., 0] = 15.0
    elif case == "extreme":
        query[-1, -1, -1, -1] = 66048
    constants = {
        "tiny-value": 2**-60,
        "large-value": float(2**70),
        "value-low": 2**-32,
        "value-high": float(2**64),
        "constant": 1.0,
    }
    if case in constants:
        value.fill_(constants[case])
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    flags = torch.empty((), device=query.device, dtype=torch.int32)
    _launch_norm_guarded_attention(query, key, value, output, flags, 128**-0.5)
    assert flags.item() == expected_bits
    _assert_norm_attention_fp64(query, key, value, output)
    if case in constants:
        assert torch.all(output == constants[case])


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("seqlen_k", [193, 769])
def test_norm_guard_routes_recomputed_on_graph_replay(seqlen_k):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        _launch_norm_guarded_attention,
    )

    torch.manual_seed(71723)
    query = torch.randn(2, 257, 3, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, seqlen_k, 3, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    original_query, original_key = query.clone(), key.clone()
    output = torch.empty_like(query)
    flags = torch.empty((), device=query.device, dtype=torch.int32)

    def launch():
        _launch_norm_guarded_attention(query, key, value, output, flags, 128**-0.5)

    for _ in range(2):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for bits in (0, 1, 2, 3, 0):
        query.copy_(original_query)
        key.copy_(original_key)
        if bits == 1:
            query.fill_(2)
            key.fill_(2)
        elif bits == 2:
            query.zero_()
            key.zero_()
            query[..., 0] = 15
            key[..., 0] = 15
        elif bits == 3:
            query.fill_(66048)
            key.fill_(-66048)
        graph.replay()
        assert flags.item() == bits
        _assert_norm_attention_fp64(query, key, value, output)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("operand", ["query", "key", "value"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_norm_guard_rejects_nonfinite_inputs(operand, nonfinite):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        _check_norm_bounded_inputs,
    )

    query = torch.zeros((1, 257, 1, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.zeros((1, 769, 1, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    {"query": query, "key": key, "value": value}[operand][0, -1, 0, -1] = nonfinite
    flags = torch.empty((), device=query.device, dtype=torch.int32)
    _check_norm_bounded_inputs(query, key, value, flags, 128**-0.5)
    assert flags.item() == (1 if operand == "value" else 3)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
def test_norm_guard_checks_values_beyond_4gib():
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        _check_norm_bounded_inputs,
    )

    free_memory, _ = torch.cuda.mem_get_info()
    if free_memory < 12 * 1024**3:
        pytest.skip("requires 12 GiB free for the large query")
    query = torch.zeros((1, 262_145, 64, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.zeros((1, 65, 64, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.ones_like(key)
    flags = torch.empty((), device=query.device, dtype=torch.int32)
    for magnitude, bits in ((15, 2), (16, 3), (0, 0)):
        query[0, -1, -1, -1] = magnitude
        _check_norm_bounded_inputs(query, key, value, flags, 128**-0.5)
        assert flags.item() == bits


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
def test_norm_guarded_attention_with_large_query_stride():
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        _launch_norm_guarded_attention,
    )

    free_memory, _ = torch.cuda.mem_get_info()
    if free_memory < 12 * 1024**3:
        pytest.skip("requires 12 GiB free for the strided query")
    torch.manual_seed(763)
    query = torch.empty_strided(
        (1, 2, 2, 128), (2**33, 2**31, 128, 1), device="cuda", dtype=torch.bfloat16
    )
    query.normal_()
    key = torch.randn(1, 769, 2, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    flags = torch.empty((), device=query.device, dtype=torch.int32)
    for bits in (0, 3):
        if bits == 3:
            query[0, 1, 1, -1] = 66048
        _launch_norm_guarded_attention(query, key, value, output, flags, 128**-0.5)
        assert flags.item() == bits
        _assert_norm_attention_fp64(query, key, value, output)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("block_m", [128, 256])
def test_norm_probability_rounding_boundaries(num_warps, block_m):
    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        _round_norm_probabilities,
    )

    @gluon.jit
    def convert(
        values,
        probabilities,
        rounded,
        count: gl.constexpr,
        BLOCK_M: gl.constexpr,
        NUM_WARPS: gl.constexpr,
    ):
        warp_bases: gl.constexpr = (
            [[1, 0], [2, 0]] if NUM_WARPS == 4 else [[1, 0], [2, 0], [4, 0]]
        )
        layout: gl.constexpr = gl.amd.AMDWMMALayout(
            3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, 32]
        )
        rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout))
        cols = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
        offsets = gl.program_id(0) * (BLOCK_M * 64) + rows[:, None] * 64 + cols[None, :]
        values = gl.load(values + offsets, offsets < count, other=0.0)
        p, r = _round_norm_probabilities(values)
        gl.store(probabilities + offsets, p, offsets < count)
        gl.store(rounded + offsets, r, offsets < count)

    # Cover every BF16 bin in the fixed-shift probability interval, with FP32
    # values immediately below, at and above RTNE ties, plus exact values and zero.
    exponents = torch.arange(63, 128, dtype=torch.int32)[:, None, None]
    mantissas = torch.arange(128, dtype=torch.int32)[None, :, None]
    tails = torch.tensor([0, 0x7FFF, 0x8000, 0x8001, 0xFFFF], dtype=torch.int32)[
        None, None, :
    ]
    bits = ((exponents << 23) | (mantissas << 16) | tails).flatten()
    bits = bits[bits <= 0x3F800000]
    values = torch.cat((torch.zeros(1), bits.view(torch.float32)))
    permutation = torch.randperm(
        values.numel(), generator=torch.Generator().manual_seed(7183)
    )
    values = values[permutation].cuda()
    probabilities = torch.empty_like(values, dtype=torch.bfloat16)
    rounded = torch.empty_like(values)
    convert[(triton.cdiv(values.numel(), block_m * 64),)](
        values,
        probabilities,
        rounded,
        values.numel(),
        block_m,
        num_warps,
        num_warps=num_warps,
    )
    expected = values.bfloat16()
    torch.testing.assert_close(probabilities, expected, rtol=0, atol=0)
    torch.testing.assert_close(rounded, expected.float(), rtol=0, atol=0)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("layout", ["sliced", "transposed"])
@pytest.mark.parametrize(
    "route,expected_bits", [("norm", 0), ("bounded", 1), ("raw", 3)]
)
def test_guarded_attention_independent_layout_strides(layout, route, expected_bits):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        _launch_norm_guarded_attention,
    )

    torch.manual_seed(20261016)
    device = "cuda"
    dtype = torch.bfloat16
    if layout == "sliced":
        query = torch.randn(4, 514, 6, 128, device=device, dtype=dtype)[::2, ::2, ::2]
        key = torch.randn(4, 769, 6, 128, device=device, dtype=dtype)[::2, :, ::2]
        value = torch.randn(2, 1538, 3, 128, device=device, dtype=dtype)[:, ::2]
        storage = torch.full((4, 514, 6, 128), 7.0, device=device, dtype=dtype)
        output = storage[::2, ::2, ::2]
        untouched = torch.ones_like(storage, dtype=torch.bool)
        untouched[::2, ::2, ::2] = False
    else:
        query = torch.randn(2, 3, 257, 128, device=device, dtype=dtype).transpose(1, 2)
        key = torch.randn(2, 769, 6, 128, device=device, dtype=dtype)[:, :, 1::2]
        value = torch.randn(2, 3, 769, 128, device=device, dtype=dtype).transpose(1, 2)
        storage = torch.full((2, 3, 514, 128), 7.0, device=device, dtype=dtype)
        output = storage[:, :, ::2].transpose(1, 2)
        untouched = torch.ones_like(storage, dtype=torch.bool)
        untouched[:, :, ::2] = False

    if route != "norm":
        scale = 1.5 if route == "bounded" else 4.0
        query.clamp_(-4, 4).mul_(scale)
        key.clamp_(-4, 4).mul_(scale)
    invalid = torch.empty((), device=device, dtype=torch.int32)
    _launch_norm_guarded_attention(query, key, value, output, invalid, 128**-0.5)
    assert invalid.item() == expected_bits
    _assert_norm_attention_fp64(query, key, value, output)
    assert torch.all(storage[untouched] == 7.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_extended_launch_selection_cpu(dtype, head_size):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        select_fav3_launch_config,
    )

    config = select_fav3_launch_config(
        1,
        40,
        32768,
        seqlen_k=511,
        num_cus=256,
        head_size=head_size,
        dtype=dtype,
    )
    if dtype == torch.bfloat16 and head_size == 128:
        assert config.schedule == "persistent"
    elif head_size == 256:
        assert config.schedule == "pipeline"
        assert not config.tdm_output
    else:
        assert config.schedule == "pingpong"
        assert config.tdm_output is (head_size == 128)


def _assert_extended_attention_fp64(query, key, value, output, scale):
    expected = _fp64_attention(query, key, value, scale)
    actual = output.double()
    rms = expected.square().mean().sqrt()
    assert output.dtype == query.dtype
    assert torch.isfinite(actual).all()
    tolerance = 0.003 if query.dtype == torch.bfloat16 else 0.001
    assert (actual - expected).square().mean().sqrt() <= tolerance * rms
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01 * rms.item())


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("schedule", ["pipeline", "pingpong"])
@pytest.mark.parametrize("case", ["tail-scale", "strided", "large-uniform", "late-key"])
def test_gluon_fav3_extended_contract_fp64(dtype, head_size, schedule, case):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import gluon_fav3_attention

    torch.manual_seed(20261016)
    batch, sq, sk, heads = 2, 129, 769, 3
    scale = head_size**-0.5
    if case == "strided":
        query = torch.randn(
            batch, heads, sq * 2, head_size, device="cuda", dtype=dtype
        )[:, :, ::2].transpose(1, 2)
        key = torch.randn(batch, sk, heads * 2, head_size, device="cuda", dtype=dtype)[
            :, :, 1::2
        ]
        value = torch.randn(
            batch, heads, sk * 2, head_size, device="cuda", dtype=dtype
        )[:, :, ::2].transpose(1, 2)
    else:
        query = torch.randn(batch, sq, heads, head_size, device="cuda", dtype=dtype)
        key = torch.randn(batch, sk, heads, head_size, device="cuda", dtype=dtype)
        value = torch.randn_like(key)
        if case == "tail-scale":
            scale = 0.017
        elif case == "large-uniform":
            magnitude = 66048 if dtype == torch.bfloat16 else 60000
            query.fill_(magnitude)
            query[1].neg_()
            key.fill_(magnitude)
            value.fill_(1)
        elif case == "late-key":
            query.fill_(1)
            key.zero_()
            key[:, 128].fill_(8)
            key[:, 129].fill_(16)
            value.zero_()
            value[:, 129].fill_(1)

    output = gluon_fav3_attention(
        query,
        key,
        value,
        softmax_scale=scale,
        launch_config=_forced_config(schedule),
    )
    _assert_extended_attention_fp64(query, key, value, output, scale)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_gluon_fav3_extended_auto_short_kv_fp64(dtype, head_size):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import gluon_fav3_attention

    torch.manual_seed(20261017)
    query = torch.randn(1, 32768, 40, head_size, device="cuda", dtype=dtype)
    key = torch.randn(1, 511, 40, head_size, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    output = gluon_fav3_attention(query, key, value)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    rows = torch.tensor([0, 127, 128, 255, 256, 16384, 32766, 32767], device="cuda")
    heads = torch.tensor([0, 20, 39], device="cuda")
    _assert_extended_attention_fp64(
        query.index_select(1, rows).index_select(2, heads),
        key.index_select(2, heads),
        value.index_select(2, heads),
        output.index_select(1, rows).index_select(2, heads),
        head_size**-0.5,
    )


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize(
    "dtype,head_size",
    [
        (torch.float16, 64),
        (torch.float16, 128),
        (torch.float16, 256),
        (torch.bfloat16, 64),
        (torch.bfloat16, 256),
    ],
)
def test_extended_long_attention_avoids_bf16_d128_norm_route(dtype, head_size):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import gluon_fav3_attention

    # The BF16 fixed shift yields probabilities of 2**-32 for these zero logits;
    # they would underflow in FP16. Other dimensions also need their own guards.
    query = torch.zeros(1, 65536, 1, head_size, device="cuda", dtype=dtype)
    key = torch.zeros_like(query)
    value = torch.ones_like(query)
    output = gluon_fav3_attention(query, key, value)
    torch.testing.assert_close(output, value, rtol=0, atol=0)


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize(
    "dtype,head_size", [(torch.float16, 128), (torch.bfloat16, 64)]
)
def test_forced_persistent_rejects_other_dtypes_and_dimensions(dtype, head_size):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import gluon_fav3_attention

    query = torch.zeros(1, 129, 1, head_size, device="cuda", dtype=dtype)
    key = torch.zeros(1, 257, 1, head_size, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match="persistent requires BF16 with head_size=128"):
        gluon_fav3_attention(
            query, key, key, launch_config=_forced_config("persistent")
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
