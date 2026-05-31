#!/usr/bin/env python3
"""AC-5 conc-16 lever: which graph-safe blocked-top-k design actually reduces the
residual merge over-scan at FULL context (no context cap)?

The decode-step selection's residual cost (after the R17 score early-exit) is the first
`torch.topk(scores, 2048)` over the FULL captured width `max_seq_len = req_to_token.shape[1]
= context_len = 163840`, even though a client request is only ~4096 tokens. This times the
candidate designs on GPU (61 layers, bs=16, the per-decode-step amortization) so the kernel
is built against a design that wins, not a guessed one.

Designs (all return the global top-2048; the deterministic tie-break is a separate correctness
layer — these time the dominant top-k cost with torch.topk):
  A monolithic           : topk over 163840            (current production merge)
  B skip-ideal (CAPPED)  : topk over the live region 4096 only (best case if dead blocks are
                           skipped AND the merge width is capped to the live region -> caps
                           servable context for the graph path)
  C blocked bw=8192/pk=2048, SKIP : Stage1 within-block top-2048 of the live 8192-block (1 live
                           block for seq=4096) + Stage2 merge over num_blocks*2048 = 40960
                           (NO context cap: all 20 blocks represented; dead blocks sentinel)
  C' blocked bw=8192/pk=2048, torch-FULL (no skip): Stage1 topk over reshaped [bs,20,8192] (full
                           163840) + Stage2 merge 40960 -> shows torch-blocked WITHOUT a skip kernel
                           is strictly worse than monolithic.
"""
import time, torch

DEV = "cuda"
BS, L = 16, 61
MAXLEN = 163840
SEQ = 4096
K = 2048
ITERS, WARMUP = 30, 8


def _mk(width):
    s = torch.full((BS, width), float("-inf"), device=DEV)
    s[:, :min(SEQ, width)] = torch.randn(BS, min(SEQ, width), device=DEV)
    return s


def _time(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        torch.cuda.synchronize(); t0 = time.time()
        fn()
        torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    full = _mk(MAXLEN)            # [bs, 163840], live [:4096]
    live = full[:, :SEQ].contiguous()      # [bs, 4096]
    nb = MAXLEN // 8192           # 20 blocks of 8192
    # C: skip => only the live block(s) feed Stage1. seq=4096 -> 1 live 8192-block.
    live_block = full[:, :8192].contiguous()        # the single live 8192-block
    merge_buf_C = torch.full((BS, nb * K), float("-inf"), device=DEV)   # 40960
    # C' torch-full reshape
    full_blocks = full.view(BS, nb, 8192)

    def A():  # monolithic over full width
        for _ in range(L):
            torch.topk(full, K, dim=-1)

    def B():  # skip-ideal: topk over live region only (CAPS context)
        for _ in range(L):
            torch.topk(live, K, dim=-1)

    def C():  # bw=8192/pk=2048 with skip: within-block top-2048 of the live block + merge 40960
        for _ in range(L):
            v, _i = torch.topk(live_block, K, dim=-1)   # Stage1 (1 live block)
            merge_buf_C[:, :K] = v                      # place live candidates; rest stay -inf
            torch.topk(merge_buf_C, K, dim=-1)          # Stage2 merge over 40960 (fixed)

    def Cprime():  # bw=8192/pk=2048 torch-FULL (no skip kernel)
        for _ in range(L):
            v, _i = torch.topk(full_blocks, K, dim=-1)  # Stage1 over all 163840 (reshaped)
            torch.topk(v.reshape(BS, nb * K), K, dim=-1)  # Stage2 merge 40960

    res = {"A_monolithic_163840": _time(A), "B_skipideal_live4096_CAPPED": _time(B),
           "C_blocked_bw8192_pk2048_SKIP_nocap": _time(C), "Cprime_blocked_torch_full_noskip": _time(Cprime)}
    import json, os
    here = os.path.dirname(os.path.abspath(__file__))
    # implied conc-16 decode step: R17 measured step 36.9ms with selection 12.5ms, of which the
    # merge topk ~ A. new_step = 36.9 - (A - design); per-req TPS = 1000/new_step.
    base_step, base_merge = 36.9, res["A_monolithic_163840"]
    rows = {}
    for k, v in res.items():
        new_step = base_step - (base_merge - v)
        rows[k] = {"selection_topk_ms_per_step_61L": round(v, 3),
                   "implied_conc16_step_ms": round(new_step, 2),
                   "implied_conc16_TPS": round(1000.0 / new_step, 1)}
    out = {"_purpose": "AC-5 conc-16 lever: graph-safe blocked-top-k design timing (61L, bs16, seq4096, "
                       "maxlen163840). Picks the design that reduces the residual full-width topk over-scan.",
           "note": "A = current production merge (full width). B caps context (live-only). C is the no-context-cap "
                   "win but needs a Triton within-block top-2048 + skip kernel. Cprime shows torch-blocked w/o a "
                   "skip kernel is worse than monolithic. implied step uses R17's measured 36.9ms step / 12.5ms selection.",
           "timings_ms_per_step": {k: round(v, 3) for k, v in res.items()}, "implied_conc16": rows}
    json.dump(out, open(os.path.join(here, "topk_design_microbench.json"), "w"), indent=1)
    for k in res:
        print(f"{k:38s}: {res[k]:7.2f} ms/step  -> implied conc16 step {rows[k]['implied_conc16_step_ms']:6.2f} ms "
              f"= {rows[k]['implied_conc16_TPS']:5.1f} TPS/req")


if __name__ == "__main__":
    main()
