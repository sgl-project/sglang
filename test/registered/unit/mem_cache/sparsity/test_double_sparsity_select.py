"""Selection-pipeline correctness for Double Sparsity (M3).

Pure-python numpy oracle vs the implementation. Covers:
- Top-k correctness for each GQA reduction (max_abs / mean / soq).
- Recency window forced into the selected set; sink tokens forced too.
- Logical-order preservation (selected positions ascending).
- Permuted physical KV (radix prefix reuse): logical positions stay correct
  even when `req_to_token` row is shuffled.
- Decode self-token invariant: the current decode position (seq_len - 1) is
  retained even though K_label has not been written for it yet.
- Device-side dense fallback when seq_len < min_seq_len.
- CUDA Triton path == CPU torch reference.
"""

import unittest

import numpy as np
import torch

from sglang.srt.mem_cache.sparsity.triton_ops.select_kernels import (
    GQA_REDUCTION_MAX_ABS,
    GQA_REDUCTION_MEAN,
    GQA_REDUCTION_SOQ,
    ds_select_tokens_torch_ref,
    ds_select_tokens_triton,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


def _numpy_oracle(
    *,
    queries: np.ndarray,  # [bs, H_q, D]
    channel_idx: np.ndarray,  # [H_kv, S]
    k_label: np.ndarray,  # [T, H_kv, S]
    req_to_token: np.ndarray,  # [max_reqs, max_ctx]
    req_pool_indices: np.ndarray,  # [bs]
    seq_lens: np.ndarray,  # [bs]
    num_kv_heads: int,
    token_budget: int,
    recent_tokens: int,
    sink_tokens: int,
    min_seq_len: int,
    max_selected: int,
    gqa_reduction_id: int,
):
    """Independent reference. Same algorithm, written from scratch in numpy."""
    bs, h_q, d = queries.shape
    s = channel_idx.shape[1]
    group = h_q // num_kv_heads

    out = np.full((bs, max_selected), -1, dtype=np.int32)
    valid = np.zeros(bs, dtype=np.int32)

    # Q_label = reduce_group(Q[:, :, channel_idx[h, :]])
    q_grouped = queries.reshape(bs, num_kv_heads, group, d)
    chans = channel_idx.astype(np.int64)  # [H_kv, S]
    q_gathered = np.take_along_axis(
        q_grouped,
        np.broadcast_to(chans[None, :, None, :], (bs, num_kv_heads, group, s)),
        axis=3,
    )  # [bs, kv, group, S]

    if gqa_reduction_id == GQA_REDUCTION_MAX_ABS:
        idx = np.argmax(np.abs(q_gathered), axis=2, keepdims=True)
        q_label = np.take_along_axis(q_gathered, idx, axis=2).squeeze(2)
    elif gqa_reduction_id == GQA_REDUCTION_MEAN:
        q_label = q_gathered.mean(axis=2)
    elif gqa_reduction_id == GQA_REDUCTION_SOQ:
        sign = np.sign(q_gathered.mean(axis=2))
        energy = np.sqrt((q_gathered**2).sum(axis=2))
        q_label = sign * energy
    else:
        raise ValueError(gqa_reduction_id)

    for b in range(bs):
        sl = int(seq_lens[b])
        if sl <= 0:
            continue
        if sl < min_seq_len:
            n = min(sl, max_selected)
            out[b, :n] = np.arange(n, dtype=np.int32)
            valid[b] = n
            continue
        history_len = max(sl - 1, 0)
        if history_len == 0:
            out[b, 0] = 0
            valid[b] = 1
            continue

        phys = req_to_token[req_pool_indices[b], :history_len].astype(np.int64)
        kl = k_label[phys]  # [hist, kv, S]
        scores = (kl * q_label[b][None, :, :]).sum(axis=-1).sum(axis=-1)  # [hist]

        k = min(token_budget, history_len)
        topk = (
            np.argpartition(-scores, k - 1)[:k]
            if k > 0
            else np.array([], dtype=np.int64)
        )
        rec_start = max(sl - recent_tokens, 0)
        recent = np.arange(rec_start, sl, dtype=np.int64)
        sink_n = min(sink_tokens, sl)
        sink = np.arange(sink_n, dtype=np.int64)

        combined = np.unique(np.concatenate([sink, topk, recent]))

        if combined.size > max_selected:
            is_sink = combined < sink_n
            is_recent = combined >= rec_start
            keep_mask = is_sink | is_recent
            forced = combined[keep_mask]
            history_only = combined[~keep_mask]
            slack = max_selected - forced.size
            if slack <= 0:
                combined = np.sort(forced[:max_selected])
            elif history_only.size:
                hist_scores = scores[history_only.clip(max=history_len - 1)]
                top = np.argpartition(-hist_scores, min(slack, hist_scores.size) - 1)[
                    :slack
                ]
                combined = np.sort(np.concatenate([forced, history_only[top]]))
            else:
                combined = np.sort(forced)

        n = combined.size
        out[b, :n] = combined.astype(np.int32)
        valid[b] = n
    return out, valid


def _alloc(seed, *, bs, h_q, h_kv, d, s, max_ctx, T, dtype, device, seq_lens=None):
    g = torch.Generator(device=device).manual_seed(seed)
    queries = torch.randn(bs, h_q, d, generator=g, device=device, dtype=dtype)
    channel_idx = torch.stack(
        [torch.randperm(d, generator=g, device=device)[:s] for _ in range(h_kv)]
    ).to(torch.int32)
    k_label = torch.randn(T, h_kv, s, generator=g, device=device, dtype=dtype)
    # Build a permuted req_to_token row to exercise the radix-prefix-reuse case.
    req_to_token = torch.zeros(4, max_ctx, dtype=torch.int32, device=device)
    perm = torch.randperm(T, generator=g, device=device)[:max_ctx]
    req_to_token[0] = perm.to(torch.int32)
    if bs > 1:
        perm2 = torch.randperm(T, generator=g, device=device)[:max_ctx]
        req_to_token[1] = perm2.to(torch.int32)
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device)
    if seq_lens is None:
        seq_lens = torch.tensor([max_ctx] * bs, dtype=torch.int64, device=device)
    return queries, channel_idx, k_label, req_to_token, req_pool_indices, seq_lens


class TestSelectionAgainstNumpy(CustomTestCase):
    """Compare implementation vs an independent numpy oracle, for each GQA reduction."""

    BS = 2
    H_Q = 8
    H_KV = 4
    D = 32
    S = 8
    MAX_CTX = 64
    T = 256
    TOKEN_BUDGET = 8
    RECENT = 4
    SINK = 2
    MIN_SEQ = 16
    MAX_SEL = 24

    def _run(self, gqa_id, dtype):
        device = torch.device("cpu")
        q, ch, kl, r2t, rpi, sl = _alloc(
            seed=42 + gqa_id,
            bs=self.BS,
            h_q=self.H_Q,
            h_kv=self.H_KV,
            d=self.D,
            s=self.S,
            max_ctx=self.MAX_CTX,
            T=self.T,
            dtype=dtype,
            device=device,
        )
        out_t, val_t = ds_select_tokens_torch_ref(
            queries=q,
            channel_idx=ch,
            k_label_layer=kl,
            req_to_token=r2t,
            req_pool_indices=rpi,
            seq_lens=sl,
            num_kv_heads=self.H_KV,
            token_budget=self.TOKEN_BUDGET,
            recent_tokens=self.RECENT,
            sink_tokens=self.SINK,
            min_seq_len=self.MIN_SEQ,
            max_selected=self.MAX_SEL,
            gqa_reduction_id=gqa_id,
        )

        out_n, val_n = _numpy_oracle(
            queries=q.float().numpy(),
            channel_idx=ch.numpy(),
            k_label=kl.float().numpy(),
            req_to_token=r2t.numpy(),
            req_pool_indices=rpi.numpy(),
            seq_lens=sl.numpy(),
            num_kv_heads=self.H_KV,
            token_budget=self.TOKEN_BUDGET,
            recent_tokens=self.RECENT,
            sink_tokens=self.SINK,
            min_seq_len=self.MIN_SEQ,
            max_selected=self.MAX_SEL,
            gqa_reduction_id=gqa_id,
        )

        # Compare valid lengths
        self.assertTrue(np.array_equal(val_t.numpy(), val_n))

        # Compare selected sets per row (allowing for tied-score ambiguity by
        # comparing as sets, not order-sensitive on ties — but order should be
        # ascending logical when no ties).
        for b in range(self.BS):
            n = val_n[b]
            tset = set(int(x) for x in out_t[b, :n].tolist())
            nset = set(int(x) for x in out_n[b, :n].tolist())
            self.assertEqual(
                tset,
                nset,
                f"row {b}: torch_ref={sorted(tset)} numpy_ref={sorted(nset)}",
            )

    def test_max_abs(self):
        self._run(GQA_REDUCTION_MAX_ABS, torch.float32)

    def test_mean(self):
        self._run(GQA_REDUCTION_MEAN, torch.float32)

    def test_soq(self):
        self._run(GQA_REDUCTION_SOQ, torch.float32)


class TestRecencyAndSink(CustomTestCase):
    def test_recency_includes_current_token(self):
        device = torch.device("cpu")
        q, ch, kl, r2t, rpi, sl = _alloc(
            seed=1,
            bs=1,
            h_q=2,
            h_kv=1,
            d=8,
            s=4,
            max_ctx=32,
            T=128,
            dtype=torch.float32,
            device=device,
            seq_lens=torch.tensor([32], dtype=torch.int64),
        )
        out, valid = ds_select_tokens_torch_ref(
            queries=q,
            channel_idx=ch,
            k_label_layer=kl,
            req_to_token=r2t,
            req_pool_indices=rpi,
            seq_lens=sl,
            num_kv_heads=1,
            token_budget=4,
            recent_tokens=4,
            sink_tokens=2,
            min_seq_len=8,
            max_selected=20,
            gqa_reduction_id=GQA_REDUCTION_MAX_ABS,
        )
        sel = set(int(x) for x in out[0, : valid[0]].tolist())
        # Current decode position (seq_len-1 = 31) must be in selected.
        self.assertIn(31, sel, f"current decode token missing; selected={sorted(sel)}")
        # Recency window [28, 32) all included.
        for t in (28, 29, 30, 31):
            self.assertIn(t, sel)
        # Sink tokens [0, 2) all included.
        self.assertIn(0, sel)
        self.assertIn(1, sel)

    def test_logical_order_ascending(self):
        device = torch.device("cpu")
        q, ch, kl, r2t, rpi, sl = _alloc(
            seed=2,
            bs=1,
            h_q=2,
            h_kv=1,
            d=8,
            s=4,
            max_ctx=64,
            T=256,
            dtype=torch.float32,
            device=device,
            seq_lens=torch.tensor([64], dtype=torch.int64),
        )
        out, valid = ds_select_tokens_torch_ref(
            queries=q,
            channel_idx=ch,
            k_label_layer=kl,
            req_to_token=r2t,
            req_pool_indices=rpi,
            seq_lens=sl,
            num_kv_heads=1,
            token_budget=8,
            recent_tokens=4,
            sink_tokens=2,
            min_seq_len=8,
            max_selected=32,
            gqa_reduction_id=GQA_REDUCTION_MAX_ABS,
        )
        sel = out[0, : valid[0]].tolist()
        self.assertEqual(
            sel, sorted(sel), "selected logical positions must be ascending"
        )


class TestDenseFallback(CustomTestCase):
    def test_short_seq_falls_back_to_dense(self):
        device = torch.device("cpu")
        q, ch, kl, r2t, rpi, sl = _alloc(
            seed=3,
            bs=2,
            h_q=2,
            h_kv=1,
            d=8,
            s=4,
            max_ctx=16,
            T=64,
            dtype=torch.float32,
            device=device,
            seq_lens=torch.tensor(
                [16, 4], dtype=torch.int64
            ),  # row 1 below min_seq_len
        )
        out, valid = ds_select_tokens_torch_ref(
            queries=q,
            channel_idx=ch,
            k_label_layer=kl,
            req_to_token=r2t,
            req_pool_indices=rpi,
            seq_lens=sl,
            num_kv_heads=1,
            token_budget=4,
            recent_tokens=2,
            sink_tokens=1,
            min_seq_len=8,
            max_selected=16,
            gqa_reduction_id=GQA_REDUCTION_MAX_ABS,
        )
        # Row 1: seq_len=4 < min_seq_len=8 → dense fallback emits [0,1,2,3].
        self.assertEqual(int(valid[1]), 4)
        self.assertEqual(out[1, :4].tolist(), [0, 1, 2, 3])
        # Row 0: full sparse path, valid > 0.
        self.assertGreater(int(valid[0]), 0)


class TestPermutedPhysical(CustomTestCase):
    """Radix-cache prefix reuse case: req_to_token is permuted, but selected
    LOGICAL positions still match a numpy oracle. The adaptor (M4) handles
    logical→physical mapping; we never sort by physical id."""

    def test_logical_correctness_under_permuted_physical(self):
        device = torch.device("cpu")
        # Build a deliberately permuted req_to_token row.
        max_ctx, T = 32, 128
        torch.manual_seed(99)
        perm_a = torch.randperm(T)[:max_ctx]
        r2t = torch.zeros(2, max_ctx, dtype=torch.int32)
        r2t[0] = perm_a.to(torch.int32)

        q = torch.randn(1, 2, 8, dtype=torch.float32)
        ch = torch.stack([torch.randperm(8)[:4]]).to(torch.int32)
        kl = torch.randn(T, 1, 4, dtype=torch.float32)
        rpi = torch.tensor([0], dtype=torch.int64)
        sl = torch.tensor([max_ctx], dtype=torch.int64)

        out, valid = ds_select_tokens_torch_ref(
            queries=q,
            channel_idx=ch,
            k_label_layer=kl,
            req_to_token=r2t,
            req_pool_indices=rpi,
            seq_lens=sl,
            num_kv_heads=1,
            token_budget=4,
            recent_tokens=2,
            sink_tokens=1,
            min_seq_len=8,
            max_selected=16,
            gqa_reduction_id=GQA_REDUCTION_MAX_ABS,
        )

        # Numpy oracle on the same permuted layout.
        out_n, val_n = _numpy_oracle(
            queries=q.numpy(),
            channel_idx=ch.numpy(),
            k_label=kl.numpy(),
            req_to_token=r2t.numpy(),
            req_pool_indices=rpi.numpy(),
            seq_lens=sl.numpy(),
            num_kv_heads=1,
            token_budget=4,
            recent_tokens=2,
            sink_tokens=1,
            min_seq_len=8,
            max_selected=16,
            gqa_reduction_id=GQA_REDUCTION_MAX_ABS,
        )

        self.assertEqual(int(valid[0]), int(val_n[0]))
        n = int(val_n[0])
        self.assertEqual(
            set(int(x) for x in out[0, :n].tolist()),
            set(int(x) for x in out_n[0, :n].tolist()),
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTritonRecallVsCpuRef(CustomTestCase):
    """v1.1: ds_select_tokens_triton is the two-stage block-topk + score-
    aware union pipeline. It is APPROXIMATE by construction (per-block
    K_block cap), so it must NOT match the CPU exact-topk reference.
    The block-topk-vs-exact-topk recall floors are verified per-stage in
    test_double_sparsity_stage1.py / stage2 / union; this test only
    pins overlap >= 50% between the CUDA pipeline and the CPU oracle on
    a high-capacity config — a sanity floor that catches catastrophic
    regressions without demanding exact equality."""

    def test_cuda_path_overlaps_cpu_ref(self):
        # Pick BLOCK_T and K_BLOCK so num_blocks * K_BLOCK >= token_budget
        # (capacity-sufficient regime), then the two-stage pipeline should
        # find most of the same top picks the CPU exact-topk does.
        bs, h_q, h_kv, d, s, max_ctx, T = 2, 8, 4, 32, 8, 64, 256
        cpu_args = _alloc(
            seed=7,
            bs=bs,
            h_q=h_q,
            h_kv=h_kv,
            d=d,
            s=s,
            max_ctx=max_ctx,
            T=T,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        cuda_args = tuple(t.to("cuda") if torch.is_tensor(t) else t for t in cpu_args)

        kw = dict(
            num_kv_heads=h_kv,
            token_budget=8,
            recent_tokens=4,
            sink_tokens=2,
            min_seq_len=16,
            max_selected=24,
            gqa_reduction_id=GQA_REDUCTION_MAX_ABS,
        )

        out_cpu, val_cpu = ds_select_tokens_torch_ref(
            queries=cpu_args[0],
            channel_idx=cpu_args[1],
            k_label_layer=cpu_args[2],
            req_to_token=cpu_args[3],
            req_pool_indices=cpu_args[4],
            seq_lens=cpu_args[5],
            **kw,
        )
        # 16 blocks of BLOCK_T=4, K_BLOCK=4 → 64 candidates >> token_budget=8.
        out_cuda, val_cuda = ds_select_tokens_triton(
            queries=cuda_args[0],
            channel_idx=cuda_args[1],
            k_label_layer=cuda_args[2],
            req_to_token=cuda_args[3],
            req_pool_indices=cuda_args[4],
            seq_lens=cuda_args[5],
            block_t=4,
            k_block=4,
            **kw,
        )
        # Sanity floor — both paths find at least half the same logical tokens.
        for b in range(bs):
            cpu_set = set(int(x) for x in out_cpu[b].tolist() if int(x) >= 0)
            cuda_set = set(int(x) for x in out_cuda[b].cpu().tolist() if int(x) >= 0)
            if cpu_set:
                overlap = len(cpu_set & cuda_set) / len(cpu_set)
                self.assertGreaterEqual(
                    overlap,
                    0.5,
                    f"row {b}: cuda picks overlap with cpu ref < 50%; "
                    f"cpu={sorted(cpu_set)} cuda={sorted(cuda_set)} overlap={overlap:.2f}",
                )


if __name__ == "__main__":
    unittest.main()
