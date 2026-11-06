"""Capture actual Top-K IDs outside timing for the matched global workload."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from bench_moe_tp_ep import (
    build_block,
    finalize_weights,
    load_real_weights,
    make_forward_batch,
)
from workload import make_input


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--backend", choices=["none", "deepep"], required=True)
    p.add_argument("--ep-size", type=int, required=True)
    p.add_argument("--global-tokens", type=int, required=True)
    p.add_argument("--phase", choices=["prefill", "decode"], required=True)
    p.add_argument("--input-seed", type=int, default=20260922)
    p.add_argument("--skew-experts", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    block, cfg, sa = build_block(args.model_path, args.ep_size, args.backend, "auto")
    load_real_weights(block, cfg, args.model_path, skew_experts=args.skew_experts)
    finalize_weights(block)
    x, counts, input_hash = make_input(
        args.global_tokens,
        cfg.hidden_size,
        args.backend,
        args.input_seed,
        torch.device("npu"),
    )
    captured = []

    def hook(module, inputs, output):
        captured.append(output.topk_ids.detach())

    handle = block.topk.register_forward_hook(hook)
    with torch.inference_mode():
        block(x, make_forward_batch(x.shape[0], is_extend=args.phase == "prefill"))
        ids = (
            captured[0]
            if captured
            else torch.empty(
                (0, cfg.num_experts_per_tok), device=x.device, dtype=torch.int64
            )
        )
        # Empty source ranks use empty_topk_output directly, so the hook does not fire.
        hist = torch.bincount(
            ids[ids >= 0].to(torch.int64).flatten(), minlength=cfg.num_experts
        ).to(torch.int32)
        ranks = [torch.empty_like(hist) for _ in counts]
        dist.all_gather(ranks, hist)
        global_hist = (
            torch.stack(ranks).sum(0) if args.backend == "deepep" else ranks[0]
        )
        assert int(global_hist.sum()) == args.global_tokens * cfg.num_experts_per_tok
    handle.remove()
    if dist.get_rank() == 0:
        record = dict(
            backend=args.backend,
            phase=args.phase,
            global_tokens=args.global_tokens,
            tokens_per_rank=counts,
            input_sha256=input_hash,
            global_expert_tokens=global_hist.cpu().tolist(),
            source_rank_expert_tokens=[r.cpu().tolist() for r in ranks],
            active_experts=int((global_hist > 0).sum()),
            max_expert_tokens=int(global_hist.max()),
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
