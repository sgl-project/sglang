# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytest.importorskip("aiter")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.models.kimi_k3_rocm_mla import (  # noqa: E402
    create_q_cache_writer,
    fused_q_cache_available,
)
from sglang.srt.runtime_context import get_resources  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(est_time=15, stage="base-b", runner_config="1-gpu-small-amd-mi35x")

with envs.SGLANG_USE_AITER.override(True):
    _AVAILABLE = fused_q_cache_available()

pytestmark = pytest.mark.skipif(
    not _AVAILABLE, reason="Kimi-K3 fused MLA Q/cache write runs on gfx950 only"
)


@pytest.fixture
def writer():
    with envs.SGLANG_USE_AITER.override(True):
        return create_q_cache_writer(kv_lora_rank=512, qk_rope_head_dim=64)


def _inputs(tokens, cache_dtype=torch.bfloat16, cache_rows=None):
    torch.manual_seed(23)
    heads = 12
    return {
        "q_nope": torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16),
        "q_pe": torch.randn(tokens, heads, 64, device="cuda", dtype=torch.bfloat16),
        "k_nope": torch.randn(tokens, 1, 512, device="cuda", dtype=torch.bfloat16),
        "k_pe": torch.randn(tokens, 1, 64, device="cuda", dtype=torch.bfloat16),
        "kv_cache": torch.zeros(
            cache_rows or tokens + 4, 1, 576, device="cuda", dtype=cache_dtype
        ),
        "slot_mapping": torch.arange(tokens, device="cuda", dtype=torch.int64),
    }


def _write(writer, values):
    return writer.write(
        **values, k_scale_float=1.0, attention_backend="aiter", is_target_verify=False
    )


def _split_chain(values):
    """The split chain: Q is the BF16 concat, the cache a plain cast at scale 1."""
    q = torch.cat((values["q_nope"], values["q_pe"]), dim=-1)
    k = torch.cat((values["k_nope"], values["k_pe"]), dim=-1)
    return q, k.to(values["kv_cache"].dtype)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("tokens", [1, 33, 256])
def test_writer_matches_split_chain_bytes(writer, cache_dtype, tokens):
    """K3's KV scale is 1, so the fused write must store the split chain's bytes.
    A slot < 0 token skips only its cache write; its Q row is still written."""
    values = _inputs(tokens, cache_dtype=cache_dtype, cache_rows=tokens + 8)
    slots = torch.randperm(tokens + 8, device="cuda")[:tokens]
    slots[-1] = -1
    values["slot_mapping"] = slots
    q, k = _write(writer, values)
    torch.cuda.synchronize()
    expected_q, expected_k = _split_chain(values)
    assert torch.equal(q, expected_q)
    assert k.shape == expected_k.shape
    cache = values["kv_cache"].view(torch.uint8)
    written = slots[:-1]
    assert torch.equal(cache[written], expected_k[:-1].view(torch.uint8))
    untouched = torch.ones(cache.shape[0], dtype=torch.bool, device="cuda")
    untouched[written] = False
    assert torch.count_nonzero(cache[untouched]).item() == 0


def test_writer_graph_survives_a_larger_eager_batch(writer):
    """A captured step keeps reading the zero positions it saw,
    even after a larger eager batch replaces the shared buffer."""
    # Start from no shared buffer so the larger batch below has to replace it.
    get_resources().buffers.pop("kimi_k3_mla_identity_rope", None)
    small = _inputs(4)
    _write(writer, small)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, _ = _write(writer, small)
    _write(writer, _inputs(1000))
    torch.cuda.synchronize()
    # Refill freed small blocks so a dropped buffer would hold non-zero positions.
    junk = [  # noqa: F841
        torch.full((64,), 1, dtype=torch.int64, device="cuda") for _ in range(256)
    ]
    for name in ("q_nope", "q_pe", "k_nope", "k_pe"):
        small[name].copy_(torch.randn_like(small[name]))
    small["kv_cache"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    expected_q, expected_k = _split_chain(small)
    assert torch.equal(q, expected_q)
    assert torch.equal(small["kv_cache"][:4], expected_k)


def test_capture_never_allocates_the_zero_positions(writer):
    """A capture the eager warmup did not size must not allocate the shared buffers
    inside the graph; the writer declines and that step keeps the split chain."""
    get_resources().buffers.pop("kimi_k3_mla_identity_rope", None)
    values = _inputs(4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = _write(writer, values)
    assert result is None
    rope = get_resources().buffers["kimi_k3_mla_identity_rope"]
    assert rope.positions is None and rope.tables is None


def test_writer_raises_on_a_strided_slot_mapping(writer):
    """The kernel reads slot_mapping at stride 1, so a strided one must fail loudly
    rather than land tokens in the wrong slots; the caller makes it contiguous."""
    values = _inputs(16, cache_rows=32)
    values["slot_mapping"] = torch.arange(32, device="cuda", dtype=torch.int64)[::2]
    with pytest.raises(ValueError):
        _write(writer, values)
    torch.cuda.synchronize()
    assert torch.count_nonzero(values["kv_cache"]).item() == 0


@pytest.mark.parametrize("cache", ["fp8_e5m2", "paged_4d"])
def test_writer_falls_back_on_caches_the_kernel_does_not_take(writer, cache):
    """Caches the kernel asserts on must reach the split chain unlaunched,
    not fail the decode step or graph capture."""
    values = _inputs(4)
    if cache == "fp8_e5m2":
        values["kv_cache"] = torch.zeros(
            8, 1, 576, device="cuda", dtype=torch.float8_e5m2
        )
    else:
        values["kv_cache"] = torch.zeros(
            8, 1, 1, 576, device="cuda", dtype=torch.bfloat16
        )
    assert _write(writer, values) is None
    torch.cuda.synchronize()
    assert torch.count_nonzero(values["kv_cache"].view(torch.uint8)).item() == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
