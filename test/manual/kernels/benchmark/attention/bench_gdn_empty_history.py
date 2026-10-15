"""Measure guarded empty-history verify and exercise its generic fallback."""

import hashlib
import inspect
import json
import argparse

import torch

from bench_gdn_verify_tiles import benchmark, make_case, snapshot

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)
from sglang.kernels.ops.attention.fla.gdn_replayssm_verify_direct import (
    gdn_replayssm_verify_direct_kernel,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    paths = [
        inspect.getsourcefile(getattr(fn, "fn", fn))
        for fn in (gdn_replayssm_spec_decode, gdn_replayssm_verify_direct_kernel)
    ]
    print(
        json.dumps(
            dict(
                source_md5={
                    p.rsplit("/", 1)[-1]: hashlib.md5(open(p, "rb").read()).hexdigest()
                    for p in paths
                },
                torch=torch.__version__,
                mode="static microbenchmark",
                accuracy="PENDING",
            )
        ),
        flush=True,
    )
    for tp in (4, 8):
        for history in (0, 7):
            case = make_case(tp, 1, 7, history)
            gdn_replayssm_spec_decode(**case)
            expected = snapshot(case)
            for direct, guarded in (
                (False, False),
                (False, True),
                (True, False),
                (True, True),
            ):
                stats = torch.zeros(2, device="cuda", dtype=torch.int64)
                opts = dict(
                    block_v=16 if direct else 32, num_warps=2, direct_solve=direct
                )
                if guarded:
                    opts["history_stats"] = stats

                def run():
                    return gdn_replayssm_spec_decode(**case, **opts)

                run()
                counts = stats.cpu().tolist()
                if guarded:
                    assert counts == ([1, 0] if history == 0 else [0, 1]), counts
                equal = {
                    key: bool(
                        torch.equal(case[key].view(torch.uint8), ref.view(torch.uint8))
                    )
                    for key, ref in expected.items()
                }
                diffs = {
                    key: (case[key].float() - ref.float()).abs().max().item()
                    for key, ref in expected.items()
                }
                timing = {} if args.check_only else benchmark(run)
                print(
                    json.dumps(
                        dict(
                            tp=tp,
                            history=history,
                            direct=direct,
                            guarded=guarded,
                            bitwise=equal,
                            max_abs_vs_stock=diffs,
                            counts=counts,
                            **timing,
                        )
                    ),
                    flush=True,
                )

    for flush in (False, True):
        case = make_case(4, 1, 7, 0)
        case["launch_mode"] = "both"
        case["beta_cache"] = torch.zeros_like(case["g_cache"])
        case["is_flush"].fill_(flush)
        stats = torch.zeros(2, device="cuda", dtype=torch.int64)
        gdn_replayssm_spec_decode(
            **case, direct_solve=True, history_stats=stats, block_v=16, num_warps=2
        )
        assert stats.sum().item() == 1, stats.cpu().tolist()
    case = make_case(4, 1, 7, 0)
    case["ssm_state_indices"].fill_(-1)
    case["replay_indices"].fill_(2**30)
    stats.zero_()
    gdn_replayssm_spec_decode(
        **case, direct_solve=True, history_stats=stats, block_v=16, num_warps=2
    )
    assert stats.sum().item() == 0
    assert torch.count_nonzero(case["out"]).item() == 0
    print(json.dumps(dict(routing_and_padding="PASS")), flush=True)


if __name__ == "__main__":
    main()
