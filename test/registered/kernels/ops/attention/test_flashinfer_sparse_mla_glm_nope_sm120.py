"""SM120 tests for the native FlashInfer GLM NoPE sparse-MLA adapter.

These run the real kernels through ``flashinfer_sparse_mla_forward`` and
compare against a dequantized PyTorch reference, replay a captured CUDA graph
after the candidate metadata changes, and check that variable eager token
counts do not retain per-shape LSE allocations.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    create_flashinfer_sparse_mla_lse_buffer,
    create_flashinfer_sparse_mla_runner,
    flashinfer_sparse_mla_forward,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _has_compact_glm_nope() -> bool:
    try:
        from flashinfer import mla
    except Exception:
        return False
    configs = getattr(mla, "supported_sparse_mla_sm120_configs", None)
    config = configs().get("glm53_nope") if configs is not None else None
    return getattr(config, "bytes_per_token", None) == 528


# Marks rather than module-level skips keep `python3 file.py` (how CI runs
# registered files) a clean pytest run that reports skips.
pytestmark = [
    pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12),
        reason="Native GLM NoPE sparse MLA requires CUDA SM 12.x.",
    ),
    pytest.mark.skipif(
        not _has_compact_glm_nope(),
        reason="Installed FlashInfer lacks compact glm53_nope rows (needs 0.7.1+).",
    ),
]

D_LATENT = 512
TILE = 128
PAGE_SIZE = 64
NUM_PAGES = 64
NUM_SLOTS = NUM_PAGES * PAGE_SIZE
TOPK = 2051
SM_SCALE = D_LATENT**-0.5
WORKSPACE_BYTES = 128 * 1024 * 1024
# Unit-variance queries and KV give logits with unit standard deviation, so the
# softmax is far from uniform and a wrong scale changes the output by tens of
# percent, well beyond the kernel's FP8 rounding.
KV_CLAMP = 4.0
MAX_REL_ERROR = 0.04
# Rows with a few candidates mix large KV values and are the most sensitive.
MAX_ROW_REL_ERROR = 0.1
MIN_WRONG_REL_ERROR = 0.15


def _quantize_rows(kv: torch.Tensor, row_bytes: int) -> torch.Tensor:
    """512 FP8 values followed by four FP32 scales, one per 128 values."""
    rows = torch.zeros(kv.shape[0], row_bytes, dtype=torch.uint8, device=kv.device)
    for tile in range(D_LATENT // TILE):
        values = kv[:, tile * TILE : (tile + 1) * TILE].float()
        scale = values.abs().amax(dim=-1).clamp(min=1e-4) / 448.0
        fp8 = (values / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        rows[:, tile * TILE : (tile + 1) * TILE] = fp8.view(torch.uint8)
        rows[:, D_LATENT + 4 * tile : D_LATENT + 4 * (tile + 1)] = (
            scale.contiguous().view(torch.uint8).view(-1, 4)
        )
    if row_bytes > 528:
        # Unused bytes of the padded ABI must never be read.
        rows[:, 528:] = 0xFF
    return rows


def _dequantize_rows(rows: torch.Tensor) -> torch.Tensor:
    out = torch.empty(rows.shape[0], D_LATENT, dtype=torch.float32, device=rows.device)
    for tile in range(D_LATENT // TILE):
        values = rows[:, tile * TILE : (tile + 1) * TILE].view(torch.float8_e4m3fn)
        scale = (
            rows[:, D_LATENT + 4 * tile : D_LATENT + 4 * (tile + 1)]
            .contiguous()
            .view(torch.float32)
        )
        out[:, tile * TILE : (tile + 1) * TILE] = values.float() * scale
    return out


def _make_cache(row_bytes: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kv = (
        torch.randn(NUM_SLOTS, D_LATENT, device="cuda", generator=generator)
        .clamp(-KV_CLAMP, KV_CLAMP)
        .to(torch.bfloat16)
    )
    rows = _quantize_rows(kv, row_bytes)
    reference = _dequantize_rows(rows)
    # Slot 0 is poisoned: masked candidates must not read it.
    rows[0] = 0xFF
    return rows.view(NUM_SLOTS, 1, row_bytes), reference


def _make_indices(tokens: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    indices = torch.randint(
        1,
        NUM_SLOTS,
        (tokens, TOPK),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    # Interior holes followed by valid candidates, including the K-pool tail.
    indices[:, 100:300] = -1
    indices[::3, 1000:2048] = -1
    if tokens > 1:
        # A row with only tail candidates and a row with no candidates.
        indices[1, :2048] = -1
        indices[-1] = -1
    return indices


def _make_q(tokens: int, heads: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(tokens, heads, D_LATENT, device="cuda", generator=generator).to(
        torch.bfloat16
    )


def _lengths(indices: torch.Tensor) -> torch.Tensor:
    """One past the last valid column of each row (0 for an empty row)."""
    columns = torch.arange(1, indices.shape[-1] + 1, device=indices.device)
    return torch.where(indices >= 0, columns, 0).amax(dim=-1)


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float = SM_SCALE,
    lengths: torch.Tensor | None = None,
):
    valid = indices >= 0
    if lengths is not None:
        columns = torch.arange(indices.shape[-1], device=indices.device)
        valid &= columns[None, :] < lengths[:, None]
    gathered = kv[indices.clamp(min=0).long()]
    logits = torch.einsum("thd,tkd->thk", q.float(), gathered) * sm_scale
    logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
    weights = torch.softmax(logits, dim=-1).nan_to_num(0.0)
    return torch.einsum("thk,tkd->thd", weights, gathered)


def _make_runner(heads: int, max_tokens: int):
    runner = create_flashinfer_sparse_mla_runner(
        qk_rope_head_dim=0,
        kv_lora_rank=D_LATENT,
        max_num_tokens=max_tokens,
        max_num_heads=heads,
        device="cuda",
    )
    lse = create_flashinfer_sparse_mla_lse_buffer(
        runner, max_num_tokens=max_tokens, max_num_heads=heads, device="cuda"
    )
    return runner, lse


def _forward(q, cache, indices, runner, lse, workspace):
    return flashinfer_sparse_mla_forward(
        q=q,
        kv_cache=cache,
        indices=indices,
        seq_lens=torch.full((q.shape[0],), TOPK, dtype=torch.int32, device=q.device),
        workspace_buffer=workspace,
        runner=runner,
        out_lse=lse,
        page_size=PAGE_SIZE,
        kv_cache_dim=cache.shape[-1],
        qk_nope_head_dim=256,
        kv_lora_rank=D_LATENT,
        qk_rope_head_dim=0,
        sm_scale=SM_SCALE,
        skip_softmax_threshold_scale_factor=None,
    )


def _rel_error(out: torch.Tensor, expected: torch.Tensor) -> float:
    return ((out.float() - expected).norm() / expected.norm()).item()


def _assert_matches(out, q, kv, indices):
    assert not out.isnan().any()
    expected = _reference(q, kv, indices)
    error = _rel_error(out, expected)
    assert error < MAX_REL_ERROR, error
    empty = (indices < 0).all(dim=-1)
    assert torch.all(out[empty] == 0)
    rows = (out[~empty].float() - expected[~empty]).norm(dim=-1)
    rows /= expected[~empty].norm(dim=-1)
    assert rows.max().item() < MAX_ROW_REL_ERROR, rows.max().item()
    # The comparison must be able to tell a wrong softmax scale apart.
    for factor in (0.0, 0.5, 2.0):
        wrong = _rel_error(out, _reference(q, kv, indices, SM_SCALE * factor))
        assert wrong > max(MIN_WRONG_REL_ERROR, 5 * error), (factor, wrong)


@pytest.mark.parametrize("row_bytes", [528, 656])
@pytest.mark.parametrize("tokens,heads", [(4, 8), (4, 64), (80, 8), (80, 64)])
def test_native_glm_nope_matches_reference(row_bytes, tokens, heads):
    cache, kv = _make_cache(row_bytes, seed=0)
    q = _make_q(tokens, heads, seed=1)
    indices = _make_indices(tokens, seed=1)
    runner, lse = _make_runner(heads, max_tokens=256)
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")

    out = _forward(q, cache, indices, runner, lse, workspace)

    _assert_matches(out, q, kv, indices)


@pytest.mark.parametrize("heads", [8, 64])
def test_graph_replay_uses_updated_candidates(heads):
    tokens = 4
    cache, kv = _make_cache(528, seed=2)
    q = _make_q(tokens, heads, seed=3)
    indices = _make_indices(tokens, seed=3)
    # Captured with a short row 0, a tail-only row 1 and an empty row 3.
    indices[0, 50:] = -1
    captured_lengths = _lengths(indices)
    runner, lse = _make_runner(heads, max_tokens=256)
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _forward(q, cache, indices, runner, lse, workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _forward(q, cache, indices, runner, lse, workspace)

    # Extend the short and empty rows, fill row 1's holes, leave only the
    # tail in row 2, then replay. Lengths frozen at capture would drop the new
    # candidates of rows 0 and 3.
    updated = _make_indices(tokens, seed=4)
    updated[0, 300:] = torch.arange(1, TOPK - 299, device="cuda")
    updated[1, :2048] = torch.arange(1, 2049, device="cuda")
    updated[2, :2048] = -1
    updated[3] = torch.arange(1, TOPK + 1, device="cuda")
    assert torch.all(_lengths(updated)[[0, 3]] > captured_lengths[[0, 3]])
    indices.copy_(updated)
    q.copy_(_make_q(tokens, heads, seed=5))
    graph.replay()
    torch.cuda.synchronize()

    _assert_matches(out, q, kv, updated)
    stale = _reference(q, kv, updated, lengths=captured_lengths)
    assert _rel_error(out, stale) > MIN_WRONG_REL_ERROR


def test_variable_eager_lengths_do_not_retain_lse():
    heads, max_tokens = 64, 1024
    cache, _ = _make_cache(528, seed=5)
    runner, lse = _make_runner(heads, max_tokens=max_tokens)
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")

    def sweep(lengths):
        for tokens in lengths:
            q = torch.zeros(
                tokens, heads, D_LATENT, device="cuda", dtype=torch.bfloat16
            )
            indices = _make_indices(tokens, seed=tokens)
            _forward(q, cache, indices, runner, lse, workspace)
            del q, indices
        torch.cuda.synchronize()

    # Warm the runner's shared split-K scratch over the whole range first.
    sweep(range(max_tokens, 64, -16))
    baseline = torch.cuda.memory_allocated()
    lengths = range(max_tokens - 8, 64, -16)
    sweep(lengths)
    growth = torch.cuda.memory_allocated() - baseline

    retained_if_owned = sum(lengths) * heads * 4
    assert growth < retained_if_owned // 4, (growth, retained_if_owned)
    prepared = getattr(runner, "_prepared_calls", {})
    assert all(getattr(call, "lse", None) is None for call in prepared.values())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
