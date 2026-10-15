"""Compare dependent-launch verify outputs and compact state with ordinary launches."""

import argparse
import json

import torch
from bench_gdn_verify_tiles import make_case
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    checks = 0
    for tp in (4, 8, 2):
        for batch in (1, 2):
            for length in (7,) if args.quick else (1, 2, 7, 8):
                for history in (0, 7) if args.quick else (0, 1, 7, 9):
                    if length + history > 16:
                        continue
                    for variant in ("stock", "tiled", "direct", "empty", "guarded"):
                        base = make_case(tp, batch, length, history)
                        base["out"].zero_()
                        base["ssm_state_indices"].copy_(
                            torch.tensor([3, 1][:batch], device="cuda")
                        )
                        base["replay_indices"].copy_(
                            torch.tensor([1, 2][:batch], device="cuda")
                        )
                        arm = {
                            k: v.clone() if isinstance(v, torch.Tensor) else v
                            for k, v in base.items()
                        }
                        opts = (
                            {}
                            if variant == "stock"
                            else dict(
                                block_v=16 if variant in ("direct", "guarded") else 32,
                                num_warps=2,
                            )
                        )
                        if variant in ("direct", "guarded"):
                            opts["direct_solve"] = True
                        stats0 = stats1 = None
                        if variant in ("empty", "guarded"):
                            stats0 = torch.zeros(2, device="cuda", dtype=torch.int64)
                            stats1 = torch.zeros_like(stats0)
                        gdn_replayssm_spec_decode(**base, history_stats=stats0, **opts)
                        gdn_replayssm_spec_decode(
                            **arm, pdl=True, history_stats=stats1, **opts
                        )
                        names = (
                            "out",
                            "checkpoint_state",
                            "d_cache",
                            "k_cache",
                            "g_cache",
                            "rawv_cache",
                            "rawk_cache",
                            "write_pos",
                            "cache_base",
                            "is_flush",
                        )
                        equal = {
                            name: torch.equal(
                                base[name].contiguous().reshape(-1).view(torch.uint8),
                                arm[name].contiguous().reshape(-1).view(torch.uint8),
                            )
                            for name in names
                        }
                        if stats0 is not None:
                            equal["counts"] = torch.equal(stats0, stats1)
                        assert all(equal.values()), (
                            tp,
                            batch,
                            length,
                            history,
                            variant,
                            equal,
                        )
                        checks += 1
    for variant in ("stock", "tiled", "direct", "empty", "guarded"):
        for scenario in ("padding", "inactive", "flush"):
            base = make_case(4, 2, 7, 7)
            base["out"].zero_()
            if scenario == "padding":
                base["ssm_state_indices"][1] = -1
                base["replay_indices"][1] = 1 << 30
            elif scenario == "inactive":
                base["is_flush"].fill_(1)
            else:
                base["is_flush"][0] = 1
                base["beta_cache"] = torch.zeros_like(base["g_cache"])
                base["launch_mode"] = "both"
            arm = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in base.items()
            }
            opts = (
                {}
                if variant == "stock"
                else {
                    "block_v": 16 if variant in ("direct", "guarded") else 32,
                    "num_warps": 2,
                }
            )
            if variant in ("direct", "guarded"):
                opts["direct_solve"] = True
            counts0 = (
                torch.zeros(2, device="cuda", dtype=torch.int64)
                if variant in ("empty", "guarded")
                else None
            )
            counts1 = counts0.clone() if counts0 is not None else None
            gdn_replayssm_spec_decode(**base, history_stats=counts0, **opts)
            gdn_replayssm_spec_decode(**arm, pdl=True, history_stats=counts1, **opts)
            for key, expected in base.items():
                if isinstance(expected, torch.Tensor):
                    assert torch.equal(
                        expected.contiguous().reshape(-1).view(torch.uint8),
                        arm[key].contiguous().reshape(-1).view(torch.uint8),
                    ), (variant, scenario, key)
            if counts0 is not None:
                assert torch.equal(counts0, counts1), (variant, scenario, "counts")
            checks += 1
    print(
        json.dumps(
            dict(
                checks=checks,
                bitwise="PASS",
                scope="PDL off/on at identical tiling and arithmetic",
            )
        )
    )


if __name__ == "__main__":
    main()
