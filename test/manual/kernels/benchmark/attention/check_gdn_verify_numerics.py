"""Compare verify output and compact deltas to an FP64 recurrence."""

import json
import argparse

import torch

from bench_gdn_verify_tiles import make_case, snapshot

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)


def reference(case):
    batch = case["query_start_loc"].numel() - 1
    length = case["max_spec_len"]
    h, hv, kdim = case["k"].shape[1], case["v"].shape[1], case["k"].shape[-1]
    head_ids = torch.arange(hv, device="cuda") // (hv // h)
    outputs, deltas = [], []
    for row in range(batch):
        slot = int(case["ssm_state_indices"][row])
        replay = int(case["replay_indices"][row])
        state = case["checkpoint_state"][slot].double().clone()
        wp, base = int(case["write_pos"][replay]), int(case["cache_base"][replay])
        for i in range(wp):
            pos = (base + i) % 16
            decay = case["g_cache"][replay, :, pos].double().exp()
            delta = (
                case["d_cache"][replay, :, pos].double()
                + case["rawv_cache"][replay, :, pos].double()
            )
            key = (
                case["k_cache"][replay, :, pos].double()
                + case["rawk_cache"][replay, :, pos].double()
            )
            state = (
                decay[:, None, None] * state
                + delta[:, :, None] * key[head_ids, None, :]
            )
        for i in range(length):
            token = row * length + i
            key, query = case["k"][token].double(), case["q"][token].double()
            key = key / torch.sqrt((key * key).sum(-1, keepdim=True) + 1e-6)
            query = (
                query
                / torch.sqrt((query * query).sum(-1, keepdim=True) + 1e-6)
                / (kdim**0.5)
            )
            key, query = key[head_ids], query[head_ids]
            gate = -case["A_log"].double().exp() * torch.nn.functional.softplus(
                case["a"][token].double() + case["dt_bias"].double()
            )
            beta = torch.sigmoid(case["b"][token].double())
            state = gate.exp()[:, None, None] * state
            delta = beta[:, None] * (
                case["v"][token].double() - (state * key[:, None, :]).sum(-1)
            )
            state = state + delta[:, :, None] * key[:, None, :]
            outputs.append((state * query[:, None, :]).sum(-1))
            deltas.append(delta)
    return torch.stack(outputs), torch.stack(deltas)


def effective_deltas(case):
    rows = []
    for row in range(case["query_start_loc"].numel() - 1):
        replay = int(case["replay_indices"][row])
        start = int(case["write_pos"][replay])
        base = int(case["cache_base"][replay])
        for i in range(case["max_spec_len"]):
            pos = (base + start + i) % 16
            rows.append(
                case["d_cache"][replay, :, pos].double()
                + case["rawv_cache"][replay, :, pos].double()
            )
    return torch.stack(rows)


def error(actual, expected):
    diff = actual.double() - expected
    return dict(
        rel_l2=(diff.norm() / expected.norm().clamp_min(1e-30)).item(),
        max_abs=diff.abs().max().item(),
        signed_mean=diff.mean().item(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--guarded", action="store_true")
    args = parser.parse_args()
    failures = 0
    for tp in (2, 4, 8):
        for batch in (1, 2):
            for history in (0, 1, 7, 9):
                for seed in (0, 1, 42):
                    for length, outliers in (
                        [(n, o) for n in (1, 2, 7) for o in (False, True)]
                        if args.expanded
                        else [(7, False)]
                    ):
                        case = make_case(tp, batch, length, history, seed)
                        if outliers:
                            case["q"][:, :, 0].mul_(128)
                            case["k"][:, :, 0].mul_(128)
                            case["v"][:, :, 0].mul_(64)
                        # Low components have the magnitude of BF16 rounding residuals.
                        case["rawv_cache"].mul_(1e-3)
                        case["rawk_cache"].mul_(1e-3)
                        expected_out, expected_delta = reference(case)
                        gdn_replayssm_spec_decode(**case)
                        stock = snapshot(case)
                        stock_delta = effective_deltas(case)
                        floor = dict(
                            out=error(stock["out"], expected_out),
                            delta=error(stock_delta, expected_delta),
                        )
                        for direct in (False, True):
                            opts = dict(
                                block_v=16 if direct else 32,
                                num_warps=2,
                                direct_solve=direct,
                            )
                            if args.guarded:
                                opts["history_stats"] = torch.zeros(
                                    2, device="cuda", dtype=torch.int64
                                )
                            gdn_replayssm_spec_decode(**case, **opts)
                            actual = dict(
                                out=error(case["out"], expected_out),
                                delta=error(effective_deltas(case), expected_delta),
                            )
                            passed = all(
                                actual[key][metric]
                                <= max(
                                    (1.05 if metric == "rel_l2" else 2)
                                    * floor[key][metric],
                                    1e-12,
                                )
                                for key in floor
                                for metric in ("rel_l2", "max_abs")
                            )
                            failures += not passed
                            print(
                                json.dumps(
                                    dict(
                                        tp=tp,
                                        batch=batch,
                                        length=length,
                                        outliers=outliers,
                                        guarded=args.guarded,
                                        history=history,
                                        seed=seed,
                                        direct=direct,
                                        floor=floor,
                                        actual=actual,
                                        bitwise={
                                            key: bool(
                                                torch.equal(
                                                    case[key].view(torch.uint8),
                                                    ref.view(torch.uint8),
                                                )
                                            )
                                            for key, ref in stock.items()
                                        },
                                        verdict="PASS" if passed else "FAIL",
                                    )
                                ),
                                flush=True,
                            )
    print(json.dumps(dict(failures=failures)), flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
