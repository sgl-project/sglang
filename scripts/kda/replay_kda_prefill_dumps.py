"""Replay dumped real KDA prefill activations through Triton and the prepared export.

Each dump (written by SGLANG_KDA_PREFILL_DUMP_DIR) holds one packed prefill call:
q/k/v/g/beta, gate params, cu_seqlens, host sequence lengths, the FP32 initial
states of the touched slots and the checkpoint plan. For every dump this script
runs SGLang's Triton KDA prefill and the Cake prepared BF16 export on identical
FP32 state copies and reports output, final-state and per-64-token-chunk
checkpoint-state errors (max abs and relative L2), grouped by sequence length.

Usage: python scripts/kda/replay_kda_prefill_dumps.py --dumps DIR [--limit N] [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

import torch


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import CakeKDAKernel
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

    os.environ["SGLANG_KDA_CAKE_PREFILL_API"] = "prepared"
    triton, cake = TritonKDAKernel(), CakeKDAKernel()
    files = sorted(glob.glob(os.path.join(args.dumps, "kda_prefill_*.pt")))
    if args.limit:
        files = files[: args.limit]
    rows = []
    for path in files:
        d = torch.load(path, map_location="cuda")
        q, k, v, g, beta = (d[n] for n in ("q", "k", "v", "g", "beta"))
        if beta.is_contiguous():
            # torch.save flattens the fused-projection slice into a contiguous copy;
            # the serving call hands the export a row-strided view (pitch > H) and the
            # export's schedule selection depends on that layout, so restore it.
            num_heads = beta.shape[-1]
            wide = torch.zeros(
                (*beta.shape[:-1], num_heads + 24), device=beta.device, dtype=beta.dtype
            )
            wide[..., 8 : 8 + num_heads] = beta
            beta = wide[..., 8 : 8 + num_heads]
        cu = d["cu_seqlens"].to(torch.int64)
        lengths = list(d["sequence_lengths"]) or (cu[1:] - cu[:-1]).tolist()
        n_seq = len(lengths)
        pool = torch.zeros(
            (n_seq + 1, *d["initial_state"].shape[1:]),
            device="cuda",
            dtype=torch.float32,
        )
        pool[:n_seq] = d["initial_state"].float()
        idx = torch.arange(n_seq, device="cuda", dtype=torch.int32)
        common = dict(
            cache_indices=idx,
            query_start_loc=cu.to(torch.int32),
            A_log=d["A_log"],
            dt_bias=d["dt_bias"],
            lower_bound=None if d["lower_bound"] is None else float(d["lower_bound"]),
            extend_seq_lens_cpu=lengths,
        )
        ncp = int(d.get("num_state_checkpoints") or 0)
        cp_kwargs = {}
        if ncp:
            cp_kwargs = dict(
                return_intermediate_states=True,
                track_ssm_h_src=torch.zeros(1, device="cuda", dtype=torch.int64),
                state_checkpoint_cu_starts=d["checkpoint_cu_starts"].to("cuda"),
                num_state_checkpoints=ncp,
                state_checkpoint_every_n_tokens=int(d["checkpoint_every_n_tokens"]),
            )
        pool_t, pool_c = pool.clone(), pool.clone()
        q_norm = q.float().norm(dim=-1)
        stats = dict(
            g_min=float(g.float().min()),
            g_mean=float(g.float().mean()),
            beta_logit_min=float(beta.float().min()),
            beta_logit_max=float(beta.float().max()),
            q_norm_cv=float(q_norm.std() / q_norm.mean().clamp_min(1e-12)),
            init_state_abs_max=float(d["initial_state"].float().abs().max()),
        )
        out_t = triton.extend(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            torch.sigmoid(beta),
            ssm_states=pool_t,
            **common,
            **cp_kwargs,
        )
        try:
            out_c = cake.extend(
                q.clone(),
                k.clone(),
                v.clone(),
                g.clone(),
                beta,
                ssm_states=pool_c,
                layer_id=int(d["layer_id"]),
                **common,
                **cp_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            err = dict(
                file=os.path.basename(path),
                layer=int(d["layer_id"]),
                lengths=lengths,
                heads=int(q.shape[2]),
                error=str(exc)[:300],
            )
            rows.append(err)
            print(json.dumps(err), flush=True)
            continue
        h_t = h_c = None
        if ncp:
            out_t, h_t = out_t
            out_c, h_c = out_c
        row = dict(
            file=os.path.basename(path),
            layer=int(d["layer_id"]),
            lengths=lengths,
            heads=int(q.shape[2]),
            output_max_abs=max_abs(out_c, out_t),
            output_rel_l2=rel_l2(out_c, out_t),
            final_state_max_abs=max_abs(pool_c[:n_seq], pool_t[:n_seq]),
            final_state_rel_l2=rel_l2(pool_c[:n_seq], pool_t[:n_seq]),
            **stats,
        )
        if ncp:
            starts = d["checkpoint_cu_starts"].tolist()
            per_chunk = []
            for s_idx, length in enumerate(lengths):
                for j in range(starts[s_idx], starts[s_idx + 1]):
                    per_chunk.append(
                        dict(
                            seq=s_idx,
                            chunk=j - starts[s_idx],
                            tokens=64 * (j - starts[s_idx] + 1),
                            rel_l2=rel_l2(h_c[0, j], h_t[0, j]),
                            max_abs=max_abs(h_c[0, j], h_t[0, j]),
                        )
                    )
            row["checkpoints"] = per_chunk
            row["checkpoint_rel_l2_max"] = (
                max(c["rel_l2"] for c in per_chunk) if per_chunk else None
            )
        rows.append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k != "checkpoints"}), flush=True
        )
    by_len = {}
    for r in rows:
        key = (
            "short<=256"
            if max(r["lengths"]) <= 256
            else ("mid<=2048" if max(r["lengths"]) <= 2048 else "long")
        )
        by_len.setdefault(key, []).append(r)
    print(
        "\n| Bucket | Calls | output rel L2 (median/max) | final state rel L2 (median/max) | worst chunk rel L2 |"
    )
    print("|---|---:|---|---|---:|")
    for key, rs in by_len.items():
        o = [r["output_rel_l2"] for r in rs]
        f = [r["final_state_rel_l2"] for r in rs]
        c = [
            r["checkpoint_rel_l2_max"]
            for r in rs
            if r.get("checkpoint_rel_l2_max") is not None
        ]
        print(
            f"| {key} | {len(rs)} | {statistics.median(o):.3g} / {max(o):.3g} | {statistics.median(f):.3g} / {max(f):.3g} | {max(c) if c else float('nan'):.3g} |"
        )
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
