"""Tests for context-scanning KDA draft attention (``linear_attn_config.context_state: scan``)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dflash import (
    DFlashDraftModel,
    DFlashKDAAttention,
    reference_dflash_kda,
)
from sglang.srt.speculative.dflash_utils import parse_dflash_kda_config
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN, HEADS, HEAD_DIM, BLOCK, CONV = 16, 2, 8, 3, 3


def _config(
    context_state="scan",
    hidden=HIDDEN,
    heads=HEADS,
    head_dim=HEAD_DIM,
    **dflash_overrides,
):
    dflash_config = {
        "attention_modes": ["kda"],
        "block_size": BLOCK,
        **dflash_overrides,
    }
    return SimpleNamespace(
        hidden_size=hidden,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
        dflash_config=dflash_config,
        linear_attn_config={
            "head_dim": head_dim,
            "num_heads": heads,
            "short_conv_kernel_size": CONV,
            "use_full_rank_gate": False,
            "gate_lower_bound": -5.0,
            "backend": "fla",
            "context_state": context_state,
        },
    )


@pytest.fixture
def tp1(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel", lambda: SimpleNamespace(tp_size=1)
    )


def _attention(
    context_state="scan", device="cpu", dtype=torch.float32, slots=4, **dims
):
    torch.manual_seed(7)
    attention = DFlashKDAAttention(_config(context_state, **dims), layer_id=0).to(
        device=device, dtype=dtype
    )
    attention.A_log.data = attention.A_log.data.float()
    attention.dt_bias.data = attention.dt_bias.data.float()
    attention.init_context_state(slots, torch.device(device), dtype)
    return attention.eval()


def _oracle(attention, context, block):
    """KDA over the contiguous sequence ``[context ; block]``, block rows only."""
    sequence = torch.cat((context, block), dim=1)
    length = sequence.shape[1]
    heads, head_dim, hidden = (
        attention.num_heads,
        attention.head_dim,
        attention.hidden_size,
    )
    shape = (1, length, heads, head_dim)
    q = attention.q_conv1d(attention.q_proj(sequence)).reshape(shape)
    k = attention.k_conv1d(attention.k_proj(sequence)).reshape(shape)
    v = attention.v_conv1d(attention.v_proj(sequence)).reshape(shape)
    raw_gate, beta = attention._gates(sequence)
    out = reference_dflash_kda(
        q,
        k,
        v,
        raw_gate,
        beta,
        attention.A_log,
        attention.dt_bias,
        attention.lower_bound,
    )[:, context.shape[1] :]
    gate = attention._output_gate(block).reshape(1, BLOCK, heads, head_dim)
    out = attention.o_norm(out, gate)
    return attention.o_proj(out.flatten(-2)).reshape(BLOCK, hidden)


def test_context_state_parses_and_validates():
    assert parse_dflash_kda_config(_config("reset")).scans_context is False
    assert parse_dflash_kda_config(_config("scan")).scans_context is True
    cfg = _config("scan")
    del cfg.linear_attn_config["context_state"]
    assert parse_dflash_kda_config(cfg).context_state == "reset"
    with pytest.raises(ValueError, match="context_state"):
        parse_dflash_kda_config(_config("sliding"))


def test_reset_policy_keeps_block_local_behaviour(tp1):
    attention = _attention("reset")
    assert attention.is_dflash_kda_scan is False
    hidden = torch.randn(2 * BLOCK, HIDDEN)
    combined = attention(None, hidden, None)
    separate = torch.cat(
        [attention(None, hidden[:BLOCK], None), attention(None, hidden[BLOCK:], None)]
    )
    torch.testing.assert_close(combined, separate)


@torch.no_grad()
def test_scan_matches_full_sequence_oracle_cpu(tp1):
    attention = _attention("scan")
    context = torch.randn(1, 7, HIDDEN)
    block = torch.randn(1, BLOCK, HIDDEN)
    slots = torch.tensor([2])
    # Two verified slices (3 rows, then 4 rows); the first starts the request.
    attention.advance_context_state(
        slots, context[0, :3], torch.tensor([3]), torch.tensor([True])
    )
    attention.advance_context_state(
        slots, context[0, 3:], torch.tensor([4]), torch.tensor([False])
    )
    out = attention(None, block[0], SimpleNamespace(req_pool_indices=slots))
    torch.testing.assert_close(
        out, _oracle(attention, context, block), rtol=1e-4, atol=1e-4
    )
    # The block forward must not advance the running state.
    state_before = attention._ctx_state[2].clone()
    attention(None, block[0], SimpleNamespace(req_pool_indices=slots))
    torch.testing.assert_close(attention._ctx_state[2], state_before)


@torch.no_grad()
def test_scan_batches_independent_requests_cpu(tp1):
    attention = _attention("scan")
    ctx_a, ctx_b = torch.randn(1, 5, HIDDEN), torch.randn(1, 2, HIDDEN)
    blocks = torch.randn(2, BLOCK, HIDDEN)
    slots = torch.tensor([3, 0])
    rows = torch.cat([ctx_a[0], ctx_b[0]])
    attention.advance_context_state(
        slots, rows, torch.tensor([5, 2]), torch.tensor([True, True])
    )
    out = attention(
        None, blocks.reshape(-1, HIDDEN), SimpleNamespace(req_pool_indices=slots)
    )
    torch.testing.assert_close(
        out[:BLOCK], _oracle(attention, ctx_a, blocks[:1]), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        out[BLOCK:], _oracle(attention, ctx_b, blocks[1:]), rtol=1e-4, atol=1e-4
    )


@torch.no_grad()
def test_reset_mask_starts_a_fresh_request(tp1):
    attention = _attention("scan")
    stale = torch.randn(1, 6, HIDDEN)
    context = torch.randn(1, 4, HIDDEN)
    block = torch.randn(1, BLOCK, HIDDEN)
    slots = torch.tensor([1])
    attention.advance_context_state(
        slots, stale[0], torch.tensor([6]), torch.tensor([True])
    )
    attention.advance_context_state(
        slots, context[0], torch.tensor([4]), torch.tensor([True])
    )
    out = attention(None, block[0], SimpleNamespace(req_pool_indices=slots))
    torch.testing.assert_close(
        out, _oracle(attention, context, block), rtol=1e-4, atol=1e-4
    )


def test_worker_groups_rows_per_request():
    calls = []
    fake = SimpleNamespace(
        _has_scan_kda=True,
        _kda_state_ready=True,
        draft_model=SimpleNamespace(
            advance_kda_context=lambda slots, rows, lens, reset: calls.append(
                (slots.tolist(), rows.shape[0], lens.tolist(), reset.tolist())
            )
        ),
    )
    ctx_hidden = torch.arange(12.0).reshape(12, 1)
    # Packed prefill layout: request 7 owns rows 0..4 (positions from 0), request 9 owns rows 5..11 (from 3).
    positions = torch.tensor([0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 9])
    DFlashWorkerV2._advance_kda_context(
        fake,
        ctx_hidden=ctx_hidden,
        positions=positions,
        req_pool_indices=torch.tensor([7, 9]),
        row_lens=torch.tensor([5, 7]),
        row_stride=None,
    )
    assert calls == [([7, 9], 12, [5, 7], [True, False])]
    calls.clear()
    # Dense verify layout: block of 4 rows per request, commit 2 / 0 / 4 rows.
    ctx_hidden = torch.arange(12.0).reshape(12, 1)
    positions = torch.tensor([10, 11, 12, 13, 0, 1, 2, 3, 0, 1, 2, 3])
    DFlashWorkerV2._advance_kda_context(
        fake,
        ctx_hidden=ctx_hidden,
        positions=positions,
        req_pool_indices=torch.tensor([1, 2, 3]),
        row_lens=torch.tensor([2, 0, 4]),
        row_stride=4,
    )
    assert calls == [([1, 3], 6, [2, 4], [False, True])]


def test_draft_model_iterates_scan_layers_only():
    scan = SimpleNamespace(
        self_attn=SimpleNamespace(is_dflash_kda=True, is_dflash_kda_scan=True)
    )
    local = SimpleNamespace(self_attn=SimpleNamespace(is_dflash_kda=True))
    gqa = SimpleNamespace(self_attn=SimpleNamespace())
    model = SimpleNamespace(layers=[scan, local, gqa])
    assert list(DFlashDraftModel.iter_scan_kda_layers(model)) == [scan]
    assert list(DFlashDraftModel.iter_context_attention_layers(model)) == [gqa]


CUDA_DIMS = dict(
    hidden=64, heads=2, head_dim=128
)  # Triton kernels need kernel-sized heads


def _reference_state_after(attention, context):
    """Reference recurrent state ([1, H, K, V]) after scanning ``context`` from zero."""
    length = context.shape[1]
    shape = (1, length, attention.num_heads, attention.head_dim)
    k = attention.k_conv1d(attention.k_proj(context)).reshape(shape)
    v = attention.v_conv1d(attention.v_proj(context)).reshape(shape)
    raw_gate, beta = attention._gates(context)
    _, state = reference_dflash_kda(
        k,
        k,
        v,
        raw_gate,
        beta,
        attention.A_log,
        attention.dt_bias,
        attention.lower_bound,
        output_final_state=True,
    )
    return state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_reset_policy_cuda_kernel_matches_reference(tp1):
    attention = _attention("reset", device="cuda", **CUDA_DIMS)
    hidden = torch.randn(2 * BLOCK, CUDA_DIMS["hidden"], device="cuda")
    out = attention(None, hidden, None)
    expected = torch.cat(
        [
            _oracle(
                attention,
                hidden.new_zeros(1, 0, CUDA_DIMS["hidden"]),
                hidden[i * BLOCK : (i + 1) * BLOCK].unsqueeze(0),
            )
            for i in range(2)
        ]
    )
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_scan_cuda_state_matches_reference(tp1):
    attention = _attention("scan", device="cuda", **CUDA_DIMS)
    context = torch.randn(1, 70, CUDA_DIMS["hidden"], device="cuda")
    slots = torch.tensor([1], device="cuda")
    attention.advance_context_state(
        slots, context[0, :66], torch.tensor([66]), torch.tensor([True])
    )
    attention.advance_context_state(
        slots, context[0, 66:], torch.tensor([4]), torch.tensor([False])
    )
    expected = _reference_state_after(attention, context)[0]  # [H, K, V]
    kernel_state = attention._ctx_state[1]  # pool layout [H, V, K]
    torch.testing.assert_close(
        kernel_state.transpose(-1, -2), expected, rtol=2e-3, atol=2e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_scan_cuda_kernels_match_cpu_oracle(tp1):
    attention = _attention("scan", device="cuda", **CUDA_DIMS)
    hidden = CUDA_DIMS["hidden"]
    context = torch.randn(1, 70, hidden, device="cuda")
    block = torch.randn(1, BLOCK, hidden, device="cuda")
    slots = torch.tensor([1], device="cuda")
    attention.advance_context_state(
        slots, context[0, :66], torch.tensor([66]), torch.tensor([True])
    )
    attention.advance_context_state(
        slots, context[0, 66:], torch.tensor([4]), torch.tensor([False])
    )
    out = attention(None, block[0], SimpleNamespace(req_pool_indices=slots))
    expected = _oracle(attention, context, block)
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)
    # Batched advance with two requests of different lengths.
    attention.reset_context_state(torch.tensor([1, 2], device="cuda"))
    ctx_b = torch.randn(1, 9, hidden, device="cuda")
    attention.advance_context_state(
        torch.tensor([1, 2], device="cuda"),
        torch.cat([context[0], ctx_b[0]]),
        torch.tensor([70, 9]),
        torch.tensor([True, True]),
    )
    out = attention(
        None,
        torch.cat([block[0], block[0]]),
        SimpleNamespace(req_pool_indices=torch.tensor([1, 2], device="cuda")),
    )
    torch.testing.assert_close(out[:BLOCK], expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        out[BLOCK:], _oracle(attention, ctx_b, block), rtol=2e-3, atol=2e-3
    )
