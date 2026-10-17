import pytest
import torch

from sglang.kernels.ops.attention.qwen3_next_prologue import (
    fused_qwen3_next_qk_norm_rope,
    gluon_available,
    prologue_kind,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=180, stage="jit-kernel-unit", runner_config="amd")

HEADS = 4
HEAD_DIM = 256
ROTARY_DIM = 64
WIDTH = (HEADS + 1) * 512
EPS = 1.0e-6
DEVICE = "cuda"
DTYPE = torch.bfloat16
ROWS = [1, 8192, 8193, 24576, 24577, 32768]


def _cos_sin_cache(length: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10_000.0 ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM)
    )
    positions = torch.arange(length, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _reference(packed, q_weight, k_weight, positions, cos_sin):
    rows = packed.shape[0]
    queries = []
    gates = []
    for head in range(HEADS):
        base = head * 512
        queries.append(packed[:, base : base + HEAD_DIM])
        gates.append(packed[:, base + HEAD_DIM : base + 512])
    query = torch.stack(queries, dim=1)
    gate = torch.cat(gates, dim=-1)
    key = packed[:, HEADS * 512 : HEADS * 512 + HEAD_DIM]
    value = packed[:, HEADS * 512 + HEAD_DIM :]

    def gemma(values, weight):
        fp32 = values.float()
        variance = fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = fp32 * torch.rsqrt(variance + EPS) * (1.0 + weight.float())
        return normalized.to(DTYPE)

    def rope(values):
        fp32 = values.float()
        half = ROTARY_DIM // 2
        rotated = fp32[..., :ROTARY_DIM]
        kept = fp32[..., ROTARY_DIM:]
        table = cos_sin[positions.long()]
        cosine = table[..., :half].float()
        sine = table[..., half:].float()
        while cosine.ndim < rotated[..., :half].ndim:
            cosine = cosine.unsqueeze(-2)
            sine = sine.unsqueeze(-2)
        first = rotated[..., :half] * cosine - rotated[..., half:] * sine
        second = rotated[..., half:] * cosine + rotated[..., :half] * sine
        return torch.cat((first, second, kept), dim=-1).to(DTYPE)

    return (
        rope(gemma(query, q_weight)).reshape(rows, -1),
        rope(gemma(key, k_weight)).reshape(rows, -1),
        value.reshape(rows, -1),
        gate.reshape(rows, -1),
    )


def _inputs(rows: int):
    generator = torch.Generator().manual_seed(rows)
    packed = torch.randn(rows, WIDTH, dtype=DTYPE, generator=generator, device="cpu")
    q_weight = torch.randn(HEAD_DIM, dtype=DTYPE, generator=generator, device="cpu")
    k_weight = torch.randn(HEAD_DIM, dtype=DTYPE, generator=generator, device="cpu")
    positions = torch.randint(0, 128, (rows,), generator=generator)
    cos_sin = _cos_sin_cache(256)
    return (
        packed.to(DEVICE),
        q_weight.to(DEVICE),
        k_weight.to(DEVICE),
        positions.to(DEVICE),
        cos_sin.to(DEVICE),
    )


def test_prologue_kind_follows_the_benchmark_winners():
    assert prologue_kind(1) == "m64"
    assert prologue_kind(8192) == "m64"
    if gluon_available():
        assert prologue_kind(8193) == "m128"
        assert prologue_kind(24576) == "m128"
        assert prologue_kind(24577) == "prefill"
        assert prologue_kind(32768) == "prefill"
    else:
        assert prologue_kind(8193) is None
        assert prologue_kind(32768) is None
    assert prologue_kind(32769) is None


@pytest.mark.parametrize("rows", ROWS)
def test_fused_qk_norm_rope_matches_reference(rows: int):
    if rows > 8192 and not gluon_available():
        pytest.skip("Gluon is unavailable")
    packed, q_weight, k_weight, positions, cos_sin = _inputs(rows)
    key_cache = torch.full((rows, HEAD_DIM), 7, dtype=DTYPE, device=DEVICE)
    value_cache = torch.full((rows, HEAD_DIM), 9, dtype=DTYPE, device=DEVICE)
    locations = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    key_before = key_cache.clone()
    value_before = value_cache.clone()

    result = fused_qwen3_next_qk_norm_rope(
        packed,
        q_weight,
        k_weight,
        positions,
        cos_sin,
        eps=EPS,
        rotary_dim=ROTARY_DIM,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_locations=locations,
    )
    assert result is not None
    query, key, value, gate = result
    expected = _reference(packed, q_weight, k_weight, positions, cos_sin)
    torch.testing.assert_close(query, expected[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(key, expected[1], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(value, expected[2], atol=0, rtol=0)
    torch.testing.assert_close(gate, expected[3], atol=0, rtol=0)
    assert torch.equal(key_cache, key_before)
    assert torch.equal(value_cache, value_before)


def _assert_store_flag_preserves_outputs(implementation, rows: int, scales: bool):
    packed, q_weight, k_weight, positions, cos_sin = _inputs(rows)
    locations = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    sentinel_key = torch.full((rows, HEAD_DIM), 3, dtype=DTYPE, device=DEVICE)
    sentinel_value = torch.full((rows, HEAD_DIM), 5, dtype=DTYPE, device=DEVICE)
    key_before = sentinel_key.clone()
    value_before = sentinel_value.clone()
    scale = torch.tensor(2.0, dtype=torch.float32, device=DEVICE)
    k_scale = scale if scales else None
    v_scale = scale if scales else None

    without = implementation(
        packed,
        q_weight,
        k_weight,
        positions,
        cos_sin,
        locations,
        sentinel_key,
        sentinel_value,
        eps=EPS,
        rotary_dim=ROTARY_DIM,
        k_scale=k_scale,
        v_scale=v_scale,
        store_kv=False,
    )
    assert torch.equal(sentinel_key, key_before)
    assert torch.equal(sentinel_value, value_before)

    written_key = torch.empty(rows * HEAD_DIM, dtype=DTYPE, device=DEVICE)
    written_value = torch.empty(rows * HEAD_DIM, dtype=DTYPE, device=DEVICE)
    with_store = implementation(
        packed,
        q_weight,
        k_weight,
        positions,
        cos_sin,
        locations,
        written_key,
        written_value,
        eps=EPS,
        rotary_dim=ROTARY_DIM,
        k_scale=k_scale,
        v_scale=v_scale,
        store_kv=True,
    )
    for kept, stored in zip(without[:3], with_store[:3]):
        torch.testing.assert_close(kept, stored, atol=0, rtol=0)


def test_store_kv_false_matches_store_kv_true():
    from sglang.kernels.ops.attention.qwen3_next_prologue.attention_prologue_m64 import (
        fused_attention_qk_norm_rope_kv_cache,
    )

    _assert_store_flag_preserves_outputs(
        fused_attention_qk_norm_rope_kv_cache, rows=4, scales=False
    )
    if not gluon_available():
        return
    from sglang.kernels.ops.attention.qwen3_next_prologue.attention_prologue_m128_m256 import (
        fused_attention_qk_norm_rope_kv_cache as m128,
    )
    from sglang.kernels.ops.attention.qwen3_next_prologue.attention_prologue_prefill_1024_32768 import (
        fused_attention_qk_norm_rope_kv_cache as prefill,
    )

    _assert_store_flag_preserves_outputs(m128, rows=4, scales=True)
    _assert_store_flag_preserves_outputs(prefill, rows=4, scales=False)
