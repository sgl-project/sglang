"""SM120 tests for the native FlashInfer GLM NoPE sparse-MLA adapter.

These run the real kernels through ``flashinfer_sparse_mla_forward`` and
compare against a dequantized PyTorch reference, replay a captured CUDA graph
after the candidate metadata changes, and check that variable eager token
counts do not retain per-shape LSE allocations.
"""

import pytest
import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    create_flashinfer_sparse_mla_lse_buffer,
    create_flashinfer_sparse_mla_runner,
    flashinfer_sparse_mla_forward,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-small")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12):
    pytest.skip(
        "Native GLM NoPE sparse MLA requires CUDA SM 12.x.", allow_module_level=True
    )


def _has_compact_glm_nope() -> bool:
    try:
        from flashinfer import mla
    except ImportError:
        return False
    configs = getattr(mla, "supported_sparse_mla_sm120_configs", None)
    config = configs().get("glm53_nope") if configs is not None else None
    return getattr(config, "bytes_per_token", None) == 528


if not _has_compact_glm_nope():
    pytest.skip(
        "Installed FlashInfer lacks compact glm53_nope support.",
        allow_module_level=True,
    )

D_LATENT = 512
TILE = 128
PAGE_SIZE = 64
NUM_PAGES = 64
NUM_SLOTS = NUM_PAGES * PAGE_SIZE
TOPK = 2051
SM_SCALE = D_LATENT**-0.5
WORKSPACE_BYTES = 128 * 1024 * 1024


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
        torch.randn(NUM_SLOTS, D_LATENT, device="cuda", generator=generator).to(
            torch.bfloat16
        )
        / 10
    ).clamp(-1, 1)
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


def _reference(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor):
    valid = indices >= 0
    gathered = kv[indices.clamp(min=0).long()]
    logits = torch.einsum("thd,tkd->thk", q.float(), gathered) * SM_SCALE
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


def _assert_matches(out, q, kv, indices):
    assert not out.isnan().any()
    expected = _reference(q, kv, indices)
    torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=5e-2)
    empty = (indices < 0).all(dim=-1)
    assert torch.all(out[empty] == 0)


@pytest.mark.parametrize("row_bytes", [528, 656])
@pytest.mark.parametrize("tokens,heads", [(4, 8), (4, 64), (80, 8), (80, 64)])
def test_native_glm_nope_matches_reference(row_bytes, tokens, heads):
    cache, kv = _make_cache(row_bytes, seed=0)
    q = (
        torch.randn(tokens, heads, D_LATENT, device="cuda").to(torch.bfloat16) / 10
    ).clamp(-1, 1)
    indices = _make_indices(tokens, seed=1)
    runner, lse = _make_runner(heads, max_tokens=256)
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")

    out = _forward(q, cache, indices, runner, lse, workspace)

    _assert_matches(out, q, kv, indices)


@pytest.mark.parametrize("heads", [8, 64])
def test_graph_replay_uses_updated_candidates(heads):
    tokens = 4
    cache, kv = _make_cache(528, seed=2)
    q = (
        torch.randn(tokens, heads, D_LATENT, device="cuda").to(torch.bfloat16) / 10
    ).clamp(-1, 1)
    indices = _make_indices(tokens, seed=3)
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

    # Change candidates and their valid lengths, then replay.
    updated = _make_indices(tokens, seed=4)
    updated[0, 50:] = -1
    updated[2, :2048] = -1
    indices.copy_(updated)
    q.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()

    _assert_matches(out, q, kv, updated)


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
