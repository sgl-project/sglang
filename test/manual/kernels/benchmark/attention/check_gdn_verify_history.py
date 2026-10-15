"""Exercise accepted-prefix commits through repeated circular verify windows."""

import argparse
import json

import torch

from bench_gdn_verify_tiles import make_case

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_circular,
    commit_gdn_replayssm_spec,
    gdn_replayssm_spec_decode,
)


def commit(case, accepted):
    commit_gdn_replayssm_spec(
        write_pos=case["write_pos"],
        cache_base=case["cache_base"],
        is_flush=case["is_flush"],
        num_accepted=accepted,
        replay_indices=case["replay_indices"],
        max_cache_len=16,
        max_spec_len=7,
        fold_every_commit=True,
        null_block_id=-1,
    )
    commit_gdn_replayssm_circular(
        checkpoint_state=case["checkpoint_state"].unsqueeze(0),
        d_cache=case["d_cache"].unsqueeze(0),
        k_cache=case["k_cache"].unsqueeze(0),
        g_cache=case["g_cache"].unsqueeze(0),
        d_residual_cache=case["rawv_cache"].unsqueeze(0),
        k_residual_cache=case["rawk_cache"].unsqueeze(0),
        state_batch_indices=case["ssm_state_indices"],
        replay_indices=case["replay_indices"],
        write_pos=case["write_pos"],
        cache_base=case["cache_base"],
        is_flush=case["is_flush"],
        accept_lens=accepted,
        null_block_id=-1,
    )


def reference_step(case, state, accepted):
    batch = accepted.numel()
    length = case["max_spec_len"]
    h, hv, kdim = case["k"].shape[1], case["v"].shape[1], case["k"].shape[-1]
    heads = torch.arange(hv, device=state.device) // (hv // h)
    q = case["q"].view(batch, length, h, kdim).double()
    k = case["k"].view(batch, length, h, kdim).double()
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) / kdim**0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q, k = q[:, :, heads], k[:, :, heads]
    v = case["v"].view(batch, length, hv, kdim).double()
    a = case["a"].view(batch, length, hv).double()
    b = case["b"].view(batch, length, hv).double()
    gates = (
        -case["A_log"].double().exp()
        * torch.nn.functional.softplus(a + case["dt_bias"].double())
    ).exp()
    beta = b.sigmoid()
    committed = state
    outputs = []
    for token in range(length):
        state = gates[:, token, :, None, None] * state
        key = k[:, token, :, None, :]
        delta = beta[:, token, :, None] * (v[:, token] - (state * key).sum(-1))
        state = state + delta[..., None] * key
        outputs.append((state * q[:, token, :, None, :]).sum(-1))
        committed = torch.where(
            (accepted == token + 1)[:, None, None, None], state, committed
        )
    return torch.stack(outputs, dim=1).reshape_as(case["out"]), committed


def error(actual, expected):
    diff = actual.double() - expected
    return (
        diff.norm() / expected.norm().clamp_min(1e-30)
    ).item(), diff.abs().max().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--fp64", action="store_true")
    args = parser.parse_args()
    stock = make_case(args.tp, 2, 7, 0)
    # Checkpoints and compact histories begin in the same state for both arms.
    arm = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in stock.items()
    }
    inputs = [make_case(args.tp, 2, 7, 0, seed) for seed in range(16)]
    accepts = [
        torch.tensor([1 + i % 7, 1 + (i * 3) % 7], device="cuda", dtype=torch.int32)
        for i in range(7)
    ]
    max_output_abs, max_state_abs, max_state_rel, mismatched_steps = 0.0, 0.0, 0.0, 0
    stats = torch.zeros(2, device="cuda", dtype=torch.int64) if args.guarded else None
    reference_state = stock["checkpoint_state"][:2].double().clone()
    errors = {
        name: [] for name in ("stock_output", "arm_output", "stock_state", "arm_state")
    }
    for step in range(args.steps):
        for key in ("q", "k", "v", "a", "b"):
            stock[key] = arm[key] = inputs[step % 16][key]
        gdn_replayssm_spec_decode(**stock)
        gdn_replayssm_spec_decode(
            **arm,
            block_v=16 if args.direct else 32,
            num_warps=2,
            direct_solve=args.direct,
            history_stats=stats,
        )
        output_error = (stock["out"].float() - arm["out"].float()).abs().max().item()
        max_output_abs = max(max_output_abs, output_error)
        if args.fp64:
            ref_out, reference_state = reference_step(
                stock, reference_state, accepts[step % 7]
            )
            errors["stock_output"].append(error(stock["out"], ref_out))
            errors["arm_output"].append(error(arm["out"], ref_out))
        commit(stock, accepts[step % 7])
        commit(arm, accepts[step % 7])
        if args.fp64:
            errors["stock_state"].append(
                error(stock["checkpoint_state"][:2], reference_state)
            )
            errors["arm_state"].append(
                error(arm["checkpoint_state"][:2], reference_state)
            )
        diff = stock["checkpoint_state"].float() - arm["checkpoint_state"].float()
        max_state_abs = max(max_state_abs, diff.abs().max().item())
        max_state_rel = max(
            max_state_rel,
            (diff.norm() / stock["checkpoint_state"].float().norm()).item(),
        )
        exact = bool(
            torch.equal(
                stock["checkpoint_state"].view(torch.uint8),
                arm["checkpoint_state"].view(torch.uint8),
            )
        )
        mismatched_steps += not exact
        assert torch.equal(stock["write_pos"], torch.zeros_like(stock["write_pos"]))
        assert torch.equal(stock["write_pos"], arm["write_pos"])
        assert torch.equal(stock["cache_base"], arm["cache_base"])
        if (step + 1) % 256 == 0:
            print(
                json.dumps(
                    dict(
                        step=step + 1,
                        direct=args.direct,
                        max_output_abs=max_output_abs,
                        max_state_abs=max_state_abs,
                        max_state_rel=max_state_rel,
                        mismatched_steps=mismatched_steps,
                    )
                ),
                flush=True,
            )
        if not args.direct:
            assert exact and output_error == 0, (
                f"state or output changed at step {step}"
            )
    fp64_summary = {}
    for name, rows in errors.items():
        if rows:
            fp64_summary[name] = dict(
                mean_rel_l2=sum(r[0] for r in rows) / len(rows),
                max_abs=max(r[1] for r in rows),
            )
    if args.fp64:
        for kind in ("output", "state"):
            ref, candidate = fp64_summary[f"stock_{kind}"], fp64_summary[f"arm_{kind}"]
            assert candidate["mean_rel_l2"] <= 1.05 * ref["mean_rel_l2"], fp64_summary
            assert candidate["max_abs"] <= 2 * ref["max_abs"], fp64_summary
    if stats is not None:
        assert stats.cpu().tolist() == [2 * args.steps, 0], stats.cpu().tolist()
    print(
        json.dumps(
            dict(
                steps=args.steps,
                accepted_tokens_per_row=sum(1 + i % 7 for i in range(args.steps)),
                guarded=args.guarded,
                history_counts=stats.cpu().tolist() if stats is not None else None,
                fp64=fp64_summary,
                direct=args.direct,
                max_output_abs=max_output_abs,
                max_state_abs=max_state_abs,
                max_state_rel=max_state_rel,
                mismatched_steps=mismatched_steps,
                verdict="PASS"
                if args.fp64 or (not args.direct and not mismatched_steps)
                else "REPORT_ONLY",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
