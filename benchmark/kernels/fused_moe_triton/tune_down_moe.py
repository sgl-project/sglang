# SPDX-License-Identifier: Apache-2.0
"""Tune a separate ``*_down.json`` config for the fused MoE down GEMM.

``tuning_fused_moe_triton.py`` times the fused pipeline under one config, so
both GEMMs ship the gate/up winner even though their aspect ratios are
opposite (gate/up: N = 2 x shard, K = hidden; down: N = hidden, K = shard).
The runtime already prefers an ``E=...,N=..._down.json`` when one exists and
only pins BLOCK_SIZE_M across the pair (one moe_align sort feeds both).

This script pins the tuned gate/up config per batch size and sweeps down
candidates that share its BLOCK_SIZE_M, timing the same fused call the main
tuner times: the gate/up term is constant, so the argmin isolates the down
kernel. Run it after the main tuner, pointing --gate-up-config at its output.
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tuning_fused_moe_triton as tuner  # noqa: E402
from common_utils import (  # noqa: E402
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    sort_config,
)

from sglang.srt.layers.moe.moe_runner.triton_utils import (  # noqa: E402
    fused_moe as fused_moe_mod,
)
from sglang.srt.layers.moe.moe_runner.triton_utils import (  # noqa: E402
    fused_moe_triton_config as cfg_mod,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (  # noqa: E402
    get_config_file_name,
)
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)

_PIN = {"gate_up": None, "down": None}


def _patched_try_get_optimal(*args, **kwargs):
    if kwargs.get("return_down_config") or (len(args) >= 7 and args[6]):
        return dict(_PIN["gate_up"]), (dict(_PIN["down"]), _PIN["down"]["BLOCK_SIZE_M"])
    return dict(_PIN["gate_up"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--dtype", default="fp8_w8a8")
    parser.add_argument("--gate-up-config", required=True,
                        help="the main tuner's output json for this geometry")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    args = parser.parse_args()

    server_args = ServerArgs(
        model_path=args.model, tp_size=args.tp_size, ep_size=args.ep_size
    )
    # Mirror the main tuner's BenchmarkWorker.__init__: benchmark_config
    # allocates tensors without device=, so the default device must be cuda.
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    set_global_server_args_for_scheduler(server_args)
    model_config = get_model_config(
        args.model, args.tp_size, args.ep_size, args.disable_shared_experts_fusion
    )
    E = model_config["num_experts"]
    topk = model_config["topk"]
    hidden_size = model_config["hidden_size"]
    shard_intermediate_size = model_config["shard_intermediate_size"]
    dtype = model_config["dtype"]
    block_shape = model_config["block_shape"]
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"

    gate_up = {int(k): v for k, v in json.load(open(args.gate_up_config)).items()}
    batch_sizes = args.batch_sizes or get_default_batch_sizes()
    space = get_configs_compute_bound()

    # The fused call must resolve configs through our pin, not the override.
    # benchmark_config also reads the tuner's module-global ``args.model``.
    tuner.args = args
    tuner.override_config = lambda *_a, **_k: nullcontext()
    cfg_mod.try_get_optimal_moe_config = _patched_try_get_optimal
    fused_moe_mod.try_get_optimal_moe_config = _patched_try_get_optimal

    results = {}
    for M in batch_sizes:
        pinned = dict(gate_up[min(gate_up, key=lambda k: abs(k - M))])
        block_m = pinned["BLOCK_SIZE_M"]
        cands, seen = [], set()
        for c in space:
            c = dict(c)
            c["BLOCK_SIZE_M"] = block_m  # one sort feeds both GEMMs
            key = tuple(sorted(c.items()))
            if key not in seen:
                seen.add(key)
                cands.append(c)
        _PIN["gate_up"] = pinned
        best, best_t = None, float("inf")
        first_err = None
        for cand in cands:
            _PIN["down"] = cand
            try:
                t = tuner.benchmark_config(
                    cand, M, E, shard_intermediate_size, hidden_size, topk,
                    dtype, use_fp8_w8a8, False, False, False, False,
                    block_shape, num_iters=10,
                )
            except Exception as e:
                if first_err is None:
                    first_err = repr(e)[:300]
                continue
            if t < best_t:
                best_t, best = t, cand
        assert best is not None, (
            f"no down candidate compiled at M={M}; first error: {first_err}"
        )
        results[M] = sort_config(best)
        print(f"M={M}: best down {results[M]} ({best_t*1e3:.1f}us)", flush=True)

    name = get_config_file_name(
        E, shard_intermediate_size // 2,
        cfg_mod.get_config_dtype_str(dtype, use_fp8_w8a8=use_fp8_w8a8),
        block_shape, down_moe=True,
    )
    path = os.path.join(args.out_dir, name)
    json.dump(results, open(path, "w"), indent=4)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
