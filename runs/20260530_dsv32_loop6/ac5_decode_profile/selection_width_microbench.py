#!/usr/bin/env python3
"""DS decode-selection cost vs scanned width (max_seq_len) microbench.

The graph-safe decode selection (`retrieve_topk_graph_safe`) scores + runs two
torch.topk over `max_seq_len` columns PER LAYER (x61) PER DECODE STEP. At the
lifted point `max_seq_len = req_to_token.shape[1] = model_config.context_len =
163840`, even though a client request is only ~4096 tokens — a ~35x over-scan.

This times the full 61-layer selection (= one decode step's DS selection cost) at
bs=16, seq_len=4096, across scan widths, on one GPU. It isolates the over-scan as
the dominant DS decode overhead and quantifies the savings from tightening the
width to the actual batch-max sequence length. Pure-compute; no served model.

Run on a free GPU (DS server killed):  python3 selection_width_microbench.py
"""
import json, os, time, torch

from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    retrieve_topk_graph_safe,
)

DEV = "cuda"
BS, L, H, LABEL_DIM, HEAD_DIM = 16, 61, 16, 16, 128
SEQ_LEN, MAX_TOP_K, T_PHYS = 4096, 2048, 8192
WIDTHS = [4608, 8192, 16384, 32768, 65536, 131072, 163840]  # 163840 = the real lifted-point width
ITERS, WARMUP = 30, 8


def mk_layer_inputs():
    q = torch.randn(BS, H, HEAD_DIM, dtype=torch.float32, device=DEV)
    ch_sel = torch.randint(0, HEAD_DIM, (L, H, LABEL_DIM), dtype=torch.int32, device=DEV)
    ch_w = torch.randn(L, H, LABEL_DIM, dtype=torch.float32, device=DEV)
    sig = torch.randint(-127, 127, (L, T_PHYS, H, LABEL_DIM), dtype=torch.int8, device=DEV)
    scales = torch.rand(L, T_PHYS, H, dtype=torch.float16, device=DEV) * 0.1 + 0.01
    written = torch.zeros(L, T_PHYS, dtype=torch.bool, device=DEV)
    written[:, :SEQ_LEN] = True
    return q, ch_sel, ch_w, sig, scales, written


def time_width(width, q, ch_sel, ch_w, sig, scales, written, topk_only=False):
    rpi = torch.arange(BS, dtype=torch.int32, device=DEV)
    r2t = torch.arange(width, dtype=torch.int32, device=DEV).clamp_max(T_PHYS - 1)
    r2t = r2t.unsqueeze(0).repeat(BS, 1).contiguous()
    seq = torch.full((BS,), SEQ_LEN, dtype=torch.int32, device=DEV)
    out_idx = torch.full((BS, MAX_TOP_K), -1, dtype=torch.int32, device=DEV)
    out_len = torch.zeros(BS, dtype=torch.int32, device=DEV)
    sc = {
        "scratch_scores": torch.zeros(BS, width, dtype=torch.float32, device=DEV),
        "scratch_topk_values": torch.zeros(BS, MAX_TOP_K, dtype=torch.float32, device=DEV),
        "scratch_topk_indices": torch.zeros(BS, MAX_TOP_K, dtype=torch.int64, device=DEV),
        "scratch_invalid_mask": torch.zeros(BS, MAX_TOP_K, dtype=torch.bool, device=DEV),
        "scratch_sorted_vals": torch.zeros(BS, MAX_TOP_K, dtype=torch.int64, device=DEV),
        "scratch_boundary": torch.zeros(BS, 1, dtype=torch.int64, device=DEV),
        "scratch_valid_i64": torch.zeros(BS, 1, dtype=torch.int64, device=DEV),
        "scratch_throwaway_idx": torch.zeros(BS, MAX_TOP_K, dtype=torch.int64, device=DEV),
    }

    def one_step():
        if topk_only:
            # isolate just the two topk passes over `width` (no scoring)
            scores = sc["scratch_scores"]
            for _ in range(L):
                torch.topk(scores, MAX_TOP_K, dim=-1, largest=True, sorted=False,
                           out=(sc["scratch_topk_values"], sc["scratch_topk_indices"]))
                torch.topk(sc["scratch_topk_indices"], MAX_TOP_K, dim=-1, largest=False,
                           sorted=True, out=(sc["scratch_sorted_vals"], sc["scratch_throwaway_idx"]))
        else:
            for lid in range(L):
                retrieve_topk_graph_safe(
                    queries=q, token_signatures=sig, written=written,
                    channel_selection=ch_sel, channel_weights=ch_w, layer_id=lid,
                    req_pool_indices=rpi, req_to_token=r2t, seq_lens=seq,
                    max_seq_len=width, max_top_k=MAX_TOP_K,
                    out_indices=out_idx, out_lengths=out_len,
                    token_scales=scales, **sc)

    for _ in range(WARMUP):
        one_step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        torch.cuda.synchronize(); t0 = time.time()
        one_step()
        torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]  # median ms per decode step (61 layers)


def main():
    q, ch_sel, ch_w, sig, scales, written = mk_layer_inputs()
    rows = []
    for w in WIDTHS:
        full = time_width(w, q, ch_sel, ch_w, sig, scales, written, topk_only=False)
        tk = time_width(w, q, ch_sel, ch_w, sig, scales, written, topk_only=True)
        rows.append({"max_seq_len": w, "selection_ms_per_step": round(full, 3),
                     "topk_only_ms_per_step": round(tk, 3),
                     "score_etc_ms_per_step": round(full - tk, 3)})
        print(f"width={w:7d}  selection={full:7.2f} ms/step  (topk={tk:7.2f}  score+misc={full-tk:7.2f})")
    out = {
        "_purpose": "DS decode-selection cost (61-layer, bs=16, seq_len=4096) vs scanned max_seq_len. "
                    "The lifted-point real width is 163840 (=context_len=req_to_token.shape[1]); the "
                    "actual sequence is ~4096, so tightening the scan to ~4608 is the candidate lever.",
        "bs": BS, "layers": L, "seq_len": SEQ_LEN, "max_top_k": MAX_TOP_K,
        "H_local": H, "label_dim": LABEL_DIM, "iters": ITERS,
        "note": "selection_ms_per_step = full retrieve_topk_graph_safe x61; per-step is added to every "
                "decode step. DS closed-batch-16 measured step ~57.6 ms (17.4 TPS/req); target <=33.3 ms (30 TPS/req).",
        "rows": rows,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(out, open(os.path.join(here, "selection_width_microbench.json"), "w"), indent=1)
    print("wrote selection_width_microbench.json")


if __name__ == "__main__":
    main()
