"""T40 parity: the draft LM head sharded over the DP-attention ranks (SGLANG_OPT_SPEC_DRAFT_TP_LM_HEAD).

Drives the real ``LogitsProcessor._get_argmax_partials_dp`` for every rank of a DP4 (GB300 tray) and
DP8 (Rubin decode) group on one GPU: the all-gather and the sum all-reduce are emulated in two
passes, everything else (row gather layout, vocab shard slicing, partials kernels) is the
production code. The selected draft token must equal ``torch.argmax`` (draft extend: NaN first,
lowest index on ties) and ``draft_topk1_postprocess`` (draft decode: NaN ranks as -1e30) on the
concatenated shard logits, including NaN rows, NaN vocab columns and ties; against the replicated
head the base tree runs, tokens may differ only where the top two logits are within GEMM
rounding of each other.
"""

import contextlib
import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_sm10 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SM 10.x only",
)

from sglang.kernels.ops.speculative.topk1 import (  # noqa: E402
    draft_topk1_argmax_from_partials,
    draft_topk1_postprocess,
    draft_topk1_postprocess_from_partials,
)
from sglang.srt.layers import logits_processor as lp  # noqa: E402
from sglang.srt.layers import dp_attention  # noqa: E402

DEV = "cuda"
HIDDEN = 6144
VOCAB = 154880  # GLM-5.2 vocab_size; the padded head has the same rows (multiple of 64)
ROWS = (1, 4, 8, 16, 24, 96)  # draft decode bs, draft extend 4 * bs


def _record(**fields) -> None:
    path = os.environ.get("T40_PARITY_JSONL")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"test": "dp_draft_argmax", **fields}) + "\n")


@pytest.fixture(scope="module")
def head():
    gen = torch.Generator(device=DEV).manual_seed(7)
    w = (torch.randn(VOCAB, HIDDEN, device=DEV, generator=gen) * 0.02).to(torch.bfloat16)
    return w


class _Group:
    def __init__(self, tp_size, rank, rows_by_rank, reduce_to=None):
        self.world_size, self.rank_in_group = tp_size, rank
        self.unique_name = f"tp:{tp_size}"
        self.rows_by_rank, self.reduce_to, self.stashed = rows_by_rank, reduce_to, None

    def all_gather_into_tensor(self, out, inp):
        out.copy_(torch.cat(self.rows_by_rank, dim=0))

    def device_all_reduce_method(self, _):
        return "emulated"

    def all_reduce(self, x):
        self.stashed = x.clone()
        return x if self.reduce_to is None else self.reduce_to


def _dp_partials(w, hidden_by_rank, tp_size):
    """Per-rank outputs of the production method; returns [rank rows] of the all-reduced partials."""
    self_ = SimpleNamespace(
        logit_scale=None,
        vocab_size=VOCAB,
        _argmax_exchange_methods={},
        _log_argmax_exchange_once=lambda **_: None,
    )
    meta = SimpleNamespace(
        global_num_tokens_for_logprob_cpu=None,  # graph replay: equal padded rows
        global_num_tokens_for_logprob_gpu=torch.zeros(tp_size, device=DEV),
    )
    head_ = SimpleNamespace(weight=w)

    def run(rank, reduce_to):
        group = _Group(tp_size, rank, hidden_by_rank, reduce_to)
        parallel = SimpleNamespace(tp_group=group, attn_dp_rank=rank)
        with mock.patch.object(lp, "get_parallel", lambda: parallel), mock.patch.object(
            lp, "world_dp_gather_enabled", lambda: False
        ), mock.patch.object(lp, "_is_replicated_plain_head", lambda *_a, **_k: True):
            out = lp.LogitsProcessor._get_argmax_partials_dp(
                self_, hidden_by_rank[rank], head_, meta
            )
        return out, group.stashed

    stashed = [run(r, None)[1] for r in range(tp_size)]
    total = torch.stack(stashed).sum(0)  # the sum all-reduce
    return [run(r, total)[0] for r in range(tp_size)], total


def _shard_logits(w, gathered, tp_size):
    width = VOCAB // tp_size
    return torch.cat(
        [gathered @ w[r * width : (r + 1) * width].T for r in range(tp_size)], dim=1
    )


def _check(w, hidden_by_rank, tp_size, case):
    rows = hidden_by_rank[0].shape[0]
    per_rank, _ = _dp_partials(w, hidden_by_rank, tp_size)
    gathered = torch.cat(hidden_by_rank, dim=0)
    shard_logits = _shard_logits(w, gathered, tp_size).float()[:, :VOCAB]
    agree_replicated = []
    for r in range(tp_size):
        partials = per_rank[r]
        assert partials.shape[0] == rows
        logits_r = shard_logits[r * rows : (r + 1) * rows]
        # Draft extend: torch.argmax order.
        _, idx_ext = draft_topk1_argmax_from_partials(partials)
        ref_ext = torch.argmax(logits_r, dim=-1, keepdim=True)
        assert torch.equal(idx_ext, ref_ext), (case, r)
        # Draft decode: draft_topk1_postprocess order on the same logits, positions advanced.
        pos_a = torch.arange(rows, device=DEV, dtype=torch.long)
        pos_b = pos_a.clone()
        _, idx_dec = draft_topk1_postprocess_from_partials(partials, pos_a)
        _, ref_dec = draft_topk1_postprocess(logits_r.contiguous(), pos_b)
        assert torch.equal(idx_dec, ref_dec) and torch.equal(pos_a, pos_b), (case, r)
        # The replicated head of the base tree (one [rows, VOCAB] GEMM per rank).
        repl = (hidden_by_rank[r] @ w.T).float()
        if torch.isfinite(repl).all():
            repl_idx = torch.argmax(repl, dim=-1, keepdim=True)
            same = (repl_idx == ref_ext).squeeze(1)
            top2 = repl.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            # Disagreements only at near-ties (bf16 GEMM rounding of the logits).
            assert (same | (margin <= top2[:, 0].abs() * 2.0**-7 + 1e-3)).all(), (case, r)
            agree_replicated.append(same.float().mean().item())
    _record(case=case, tp=tp_size, rows=rows,
            agree_replicated=min(agree_replicated) if agree_replicated else None)


@pytest.mark.parametrize("tp_size", [4, 8])
@pytest.mark.parametrize("rows", ROWS)
@_sm10
def test_dp_shard_argmax_random(head, tp_size, rows):
    gen = torch.Generator(device=DEV).manual_seed(100 * tp_size + rows)
    hidden = [
        torch.randn(rows, HIDDEN, device=DEV, generator=gen).to(torch.bfloat16)
        for _ in range(tp_size)
    ]
    _check(head, hidden, tp_size, "random")


@pytest.mark.parametrize("tp_size", [4, 8])
@_sm10
def test_dp_shard_argmax_ties_and_nan_row(head, tp_size):
    w = head.clone()
    width = VOCAB // tp_size
    # Four vocab rows share the top logit for the probe row: two inside shard 0, one at the
    # end of shard 0, one at the start of shard 1. The lowest index must win in both orders.
    tied = (5, 9, width - 3, width + 11)
    for v in tied:
        w[v] = 0
        w[v, :8] = 1.0
    rows = 8
    gen = torch.Generator(device=DEV).manual_seed(3)
    hidden = [
        (torch.randn(rows, HIDDEN, device=DEV, generator=gen) * 0.01).to(torch.bfloat16)
        for _ in range(tp_size)
    ]
    hidden[0][1] = 0
    hidden[0][1, :8] = 4.0
    hidden[1][2] = float("nan")  # a whole NaN row: every logit NaN
    _check(w, hidden, tp_size, "ties_and_nan_row")
    per_rank, _ = _dp_partials(w, hidden, tp_size)
    _, idx0 = draft_topk1_argmax_from_partials(per_rank[0])
    assert idx0[1].item() == tied[0]
    _, idx1 = draft_topk1_argmax_from_partials(per_rank[1])
    assert idx1[2].item() == 0  # torch.argmax: the first NaN
    pos = torch.zeros(rows, device=DEV, dtype=torch.long)
    _, dec1 = draft_topk1_postprocess_from_partials(per_rank[1], pos)
    assert dec1[2].item() == 0  # NaN ranks as -1e30: all equal, lowest index


@pytest.mark.parametrize("tp_size", [4, 8])
@_sm10
def test_dp_shard_argmax_nan_vocab_column(head, tp_size):
    """A NaN logit column in the last shard: draft extend selects it (torch.argmax order),
    draft decode skips it (NaN ranks as -1e30)."""
    w = head.clone()
    nan_col = VOCAB - 17
    w[nan_col, 100] = float("nan")
    rows = 4
    gen = torch.Generator(device=DEV).manual_seed(5)
    hidden = [
        torch.randn(rows, HIDDEN, device=DEV, generator=gen).to(torch.bfloat16)
        for _ in range(tp_size)
    ]
    _check(w, hidden, tp_size, "nan_vocab_column")
    per_rank, _ = _dp_partials(w, hidden, tp_size)
    _, idx = draft_topk1_argmax_from_partials(per_rank[0])
    assert (idx == nan_col).all()
    pos = torch.zeros(rows, device=DEV, dtype=torch.long)
    _, dec = draft_topk1_postprocess_from_partials(per_rank[0], pos)
    assert (dec != nan_col).all()


def test_dp_rows_unequal_reason_eager_and_graph_counts():
    """Eager batches with uneven or DP-local counts keep the replicated head; a graph
    padded to the DP-max bucket (GPU counts only) shards."""
    reason = lp._dp_rows_unequal_reason
    meta = SimpleNamespace(global_num_tokens_for_logprob_cpu=[8, 8, 4, 8],
                           global_num_tokens_for_logprob_gpu=None)
    assert reason(meta, num_rows=8, tp_size=4) is not None
    meta.global_num_tokens_for_logprob_cpu = [8, 8, 8, 8]
    assert reason(meta, num_rows=8, tp_size=4) is None
    meta.global_num_tokens_for_logprob_cpu = [0, 0, 0, 0]
    assert reason(meta, num_rows=0, tp_size=4) is not None
    meta.global_num_tokens_for_logprob_cpu = [8]  # DP-local counts: draft not DP-synced
    assert reason(meta, num_rows=8, tp_size=4) is not None
    meta.global_num_tokens_for_logprob_cpu = None
    meta.global_num_tokens_for_logprob_gpu = torch.zeros(8)
    assert reason(meta, num_rows=24, tp_size=8) is None


def test_dp_shard_decline_is_logged_not_silent():
    """Rubin 38961: a draft graph padded per rank (no DP row counts) made the shard
    return without any line, so the startup gate saw neither engage nor decline."""
    logged = []
    self_ = SimpleNamespace(
        _log_argmax_exchange_once=lambda **kw: logged.append(kw),
        _argmax_exchange_methods={},
    )
    meta = SimpleNamespace(global_num_tokens_for_logprob_cpu=None,
                           global_num_tokens_for_logprob_gpu=None)
    group = _Group(8, 3, rows_by_rank=None)
    with mock.patch.object(lp, "get_parallel", lambda: SimpleNamespace(tp_group=group)):
        out = lp.LogitsProcessor._get_argmax_partials_dp(
            self_, torch.zeros(24, 16), SimpleNamespace(), meta
        )
    assert out is None
    assert len(logged) == 1 and logged[0]["all_ranks"]
    assert logged[0]["outcome"].startswith("keeps the replicated LM head: ")


def _require_mlp_tp_gather(*, draft_a2a, f3, speculative_context, **kwargs):
    """require_mlp_tp_gather at the Rubin 38961 decode layout (2-node DEP8: tp 8, dp 8,
    attention DP, --enable-dp-lm-head --moe-dense-tp-size 1, EAGLE 3/1/4, decode role)."""
    parallel = SimpleNamespace(
        attn_dp_enabled=True, num_dp_ranks=8, moe_dense_tp_size=1,
        enable_dp_lm_head=True, tp_size=8, attn_dp_size=8, attn_tp_size=1,
        attn_cp_size=1,
    )
    with contextlib.ExitStack() as stack:
        for p in (
            mock.patch.object(dp_attention, "get_parallel", lambda: parallel),
            mock.patch.object(lp, "get_parallel", lambda: parallel),
            mock.patch.object(dp_attention, "get_exec", lambda: SimpleNamespace(
                moe=SimpleNamespace(elastic_ep_backend=None))),
            mock.patch.object(dp_attention, "get_flags", lambda: SimpleNamespace(
                moe=SimpleNamespace(speculative_context=speculative_context))),
            mock.patch.object(lp, "get_disagg", lambda: SimpleNamespace(
                disaggregation_mode="decode")),
            mock.patch.object(lp, "get_spec", lambda: SimpleNamespace(
                speculative_use_rejection_sampling=False)),
            lp.envs.SGLANG_OPT_SPEC_DRAFT_TP_LM_HEAD.override(f3),
        ):
            stack.enter_context(p)
        return dp_attention.require_mlp_tp_gather(moe_a2a_backend=draft_a2a, **kwargs)


@pytest.mark.parametrize("draft_a2a", ["deepep", "none", "flashinfer"])
def test_rubin_dep8_draft_rows_dp_synced_when_f3_on(draft_a2a):
    """Rubin 38961: with the draft MoE on DeepEP (--speculative-moe-a2a-backend deepep)
    the draft padded its graphs per rank, so the shard never had equal DP rows and F3
    never engaged; GB300's draft A2A none was DP-synced. With F3 on, both the
    scheduler's draft counts and the draft runners must ask for DP-synced rows."""
    # The scheduler (explicit) and the draft runners (speculative_moe_a2a_backend_context).
    assert _require_mlp_tp_gather(draft_a2a=draft_a2a, f3=True, speculative_context=False,
                                  speculative_draft=True)
    assert _require_mlp_tp_gather(draft_a2a=draft_a2a, f3=True, speculative_context=True)


def test_rubin_dep8_default_draft_and_target_unchanged():
    """F3 off keeps the DeepEP draft rank-local, and F3 never changes a non-draft answer."""
    assert not _require_mlp_tp_gather(draft_a2a="deepep", f3=False, speculative_context=True)
    assert not _require_mlp_tp_gather(draft_a2a="deepep", f3=False, speculative_context=False,
                                      speculative_draft=True)
    assert not _require_mlp_tp_gather(draft_a2a="deepep", f3=True, speculative_context=False)
