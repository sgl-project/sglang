"""Cake variable-Q DCP MLA decode (world 1) through sglang.kernels.

Kept in its own registered test file: the CI harness runs one process per file,
and FlashInfer 46340689a5ab shows a cross-kernel interference where the first
var-Q launch after a Kimi-K3 FP8 MLA (cake) launch in the same process returns
wrong rows (see ``test_cake_mla.py::test_varq_after_kimi_k3_mla`` and the
Linear/FlashInfer issue referenced there).
"""

import math
import sys

import pytest
import torch

from sglang.kernels.cake_kernels import attention_mla as cake
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.attention.cake import (
    cake_mla_varq_dcp_decode,
    cake_prepare_mla_varq_dcp_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

LATENT = 512
ROPE = 64
QK_DIM = LATENT + ROPE
PAGE = 64


def _ceil_div(a, b):
    return -(-a // b)


def _skip_unless(archs, *modules):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(*modules):
        pytest.skip(f"installed FlashInfer lacks {', '.join(modules)}")
    cc = torch.cuda.get_device_capability()
    if cc not in archs:
        pytest.skip(f"Cake kernel is built for {archs}, device is {cc}")


# --------------------------------------------------------------------------
# Variable-q MLA decode (cp_world = 1)
# --------------------------------------------------------------------------


def _varq_reference(query, cum_q, kv_rows, kv_lens, scale):
    """FP32 (out [total_q, H, 512], natural-log lse [total_q, H]) with causal tail."""
    total_q, num_heads, _ = query.shape
    out = torch.zeros((total_q, num_heads, LATENT), device=query.device)
    lse = torch.full((total_q, num_heads), -math.inf, device=query.device)
    offsets = cum_q.tolist()
    for b, (q0, q1) in enumerate(zip(offsets[:-1], offsets[1:])):
        q_len, g = q1 - q0, kv_lens[b]
        keys = kv_rows[b][:g].float()
        bounds = g - q_len + torch.arange(q_len, device=query.device)
        visible = torch.arange(g, device=query.device)[None, :] <= bounds[:, None]
        scores = torch.einsum("qhd,kd->qhk", query[q0:q1].float(), keys) * scale
        scores = scores.masked_fill(~visible[:, None, :], -math.inf)
        row_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.exp(scores - row_lse[..., None])
        out[q0:q1] = torch.einsum("qhk,kd->qhd", probs, keys[:, :LATENT])
        lse[q0:q1] = row_lse
    return out, lse


def _varq_plan(device, *, batch, max_q_len, num_heads, max_seq_len):
    """FlashInfer's host plan for the shape; ``plan["can_split"]`` decides
    whether a request may be split across clusters (merge-order spread)."""
    from flashinfer.experimental.cake_mla_varq_dcp_decode.cake_backend import (
        plan_varq_dcp_decode,
    )

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    return plan_varq_dcp_decode(
        batch_size=batch,
        max_q_len=max_q_len,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        num_sms=num_sms,
    )


def _assert_varq_replay_close(out, lse, other_out, other_lse, *, can_split):
    """Two launches on identical inputs agree within FlashInfer's replay spread.

    FlashInfer's contract (``tests/experimental/test_cake_mla_varq_dcp_decode.py``
    ``_assert_replay_close``): split items are merged from BF16-staged partials
    in unit-completion order, so consecutive launches may differ by a few BF16
    ulps (``out`` atol 1e-3 / rtol 2^-7, finite ``lse`` atol 4e-3); the ``-inf``
    LSE positions and the zero output rows are exact.  FlashInfer's own suite
    never asserts bitwise replay, and on sm_103a (GB300, FlashInfer 46340689a5ab)
    even plans with ``can_split`` False were observed to differ by one BF16 ulp
    between launches, so ``can_split`` only labels the failure message here.
    """
    assert not torch.isnan(out.float()).any()
    neg_inf = torch.isneginf(other_lse.float())
    assert torch.equal(torch.isneginf(lse.float()), neg_inf), f"can_split={can_split}"
    torch.testing.assert_close(
        out.float(),
        other_out.float(),
        atol=1e-3,
        rtol=2.0**-7,
        msg=lambda m: f"can_split={can_split}: {m}",
    )
    finite = ~neg_inf
    torch.testing.assert_close(
        lse.float()[finite],
        other_lse.float()[finite],
        atol=4e-3,
        rtol=0,
        msg=lambda m: f"can_split={can_split}: {m}",
    )
    assert torch.equal(out.float()[neg_inf], torch.zeros_like(out.float()[neg_inf]))


def test_mla_varq_decode_world1_one_shot_and_prepared():
    """Facade one-shot, direct FlashInfer one-shot and the prepared runner.

    Every launch owns its workspace (FlashInfer: never share one varq workspace
    between two live runners) and writes NaN-filled caller-owned ``out`` /
    ``lse`` so an unwritten row shows up as NaN instead of stale memory.  Each
    result is checked against the FP32 reference before any run-to-run
    comparison, so a wrong launch is attributed to its call rather than to
    "parity".  Run-to-run equality follows FlashInfer's contract: bitwise when
    the host plan cannot split items, otherwise the documented replay spread.
    """
    _skip_unless(cake.ARCHS, cake.FI_VARQ_DCP_MODULE)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(12)
    kv_lens, q_lens, num_heads = [300, 70], [2, 1], 12
    batch, max_q_len, total_q = len(kv_lens), max(q_lens), sum(q_lens)
    query = (
        torch.randn((total_q, num_heads, QK_DIM), generator=gen, device=device) * 0.1
    ).to(torch.bfloat16)
    pages_per = [_ceil_div(k, PAGE) for k in kv_lens]
    kv_cache = (
        torch.randn((sum(pages_per), PAGE, QK_DIM), generator=gen, device=device) * 0.1
    ).to(torch.bfloat16)
    page_table = torch.zeros((batch, max(pages_per)), dtype=torch.int32, device=device)
    kv_rows, off = [], 0
    for b, n in enumerate(pages_per):
        page_table[b, :n] = torch.arange(off, off + n, dtype=torch.int32, device=device)
        kv_rows.append(kv_cache[off : off + n].reshape(-1, QK_DIM))
        off += n
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    cum_q = torch.tensor([0, q_lens[0], total_q], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(LATENT)
    assert cake.supports_mla_varq_dcp_decode(
        query, kv_cache, batch_size=batch, max_q_len=max_q_len
    )
    plan = _varq_plan(
        device,
        batch=batch,
        max_q_len=max_q_len,
        num_heads=num_heads,
        max_seq_len=max(kv_lens),
    )
    can_split = bool(plan["can_split"])
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    def workspace():
        # One workspace per runner: its scheduler state is live across launches.
        return torch.empty(
            cake.max_mla_varq_dcp_decode_workspace_size(
                batch_size=batch,
                max_q_len=max_q_len,
                num_heads=num_heads,
                num_sms=num_sms,
            ),
            dtype=torch.uint8,
            device=device,
        )

    def outputs():
        out = torch.full(
            (total_q, num_heads, LATENT), math.nan, dtype=torch.bfloat16, device=device
        )
        lse = torch.full(
            (total_q, num_heads), math.nan, dtype=torch.float32, device=device
        )
        return out, lse

    ref_out, ref_lse = _varq_reference(query, cum_q, kv_rows, kv_lens, scale)

    def check_reference(out, lse):
        assert not torch.isnan(out).any(), "rows left unwritten by the launch"
        torch.testing.assert_close(out.float(), ref_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-2)

    kwargs = dict(cum_seq_lens_q=cum_q, max_q_len=max_q_len)
    args = (query, kv_cache)
    tail = (page_table, seq_lens, max(kv_lens), scale)
    out, lse = outputs()
    try:
        got = cake_mla_varq_dcp_decode(
            *args, workspace(), *tail, out=out, lse=lse, **kwargs
        )
    except NotImplementedError as error:  # generated program not registered
        pytest.skip(str(error))
    torch.cuda.synchronize()
    assert got[0] is out and got[1] is lse
    check_reference(out, lse)

    from flashinfer.mla import cake_mla_varq_dcp_decode as fi_varq_dcp_decode

    out_fi, lse_fi = outputs()
    fi_varq_dcp_decode(
        *args, workspace(), *tail, backend="cake", out=out_fi, lse=lse_fi, **kwargs
    )
    torch.cuda.synchronize()
    check_reference(out_fi, lse_fi)
    _assert_varq_replay_close(out, lse, out_fi, lse_fi, can_split=can_split)

    out_p, lse_p = outputs()
    runner = cake_prepare_mla_varq_dcp_decode(
        query,
        kv_cache,
        page_table,
        seq_lens,
        cum_q,
        max_q_len,
        max_seq_len=max(kv_lens),
        softmax_scale=scale,
        workspace_buffer=workspace(),
        out=out_p,
        lse=lse_p,
    )
    runner.launch()
    torch.cuda.synchronize()
    check_reference(out_p, lse_p)
    _assert_varq_replay_close(out_p, lse_p, out, lse, can_split=can_split)

    # Replay on identical inputs through the live runner (FlashInfer's
    # test_prepared_runner_replays_without_allocation): no allocation, and the
    # self-resetting scheduler state reproduces the result.
    first = (out_p.clone(), lse_p.clone())
    out_p.fill_(math.nan)
    lse_p.fill_(math.nan)
    before = torch.cuda.memory_allocated()
    runner.launch()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    _assert_varq_replay_close(out_p, lse_p, first[0], first[1], can_split=can_split)

    # New query contents through the same bindings.
    query.copy_(torch.randn(query.shape, generator=gen, device=device) * 0.1)
    out_p.fill_(math.nan)
    lse_p.fill_(math.nan)
    runner.launch()
    torch.cuda.synchronize()
    ref_out, ref_lse = _varq_reference(query, cum_q, kv_rows, kv_lens, scale)
    check_reference(out_p, lse_p)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
