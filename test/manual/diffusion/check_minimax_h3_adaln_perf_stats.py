# SPDX-License-Identifier: Apache-2.0
"""Tiny CUDA/TP acceptance: torchrun --standalone --nproc-per-node {1,2} FILE."""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln_cache import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3AdalnCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    _record_adaln_cache_stats,
)
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    PerformanceLogger,
    RequestMetrics,
)


def main():
    tp, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    maybe_init_distributed_environment_and_model_parallel(tp_size=tp, sp_size=1)
    arch = MiniMaxH3DiTArchConfig(num_layers=2, hidden_size=4, time_embed_dim=3)
    width = 6 * MINIMAX_H3_ADALN_MODALITY_NUM * arch.hidden_size
    prefixes = [(f"blocks.{i}.adaln_proj.linear", width) for i in range(2)] + [
        ("final_layer.adaln_proj.linear", 8)
    ]
    weights = {}
    for prefix, n in prefixes:
        weights[prefix + ".weight"] = (
            torch.arange(n * 3).reshape(n, 3) % 11 - 5
        ).float() / 16
        weights[prefix + ".bias"] = torch.arange(n).float() / 32

    def embed(t):
        return torch.stack((t, t + 0.25, t * 2), dim=-1)

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "tiny.safetensors"
        save_file(weights, str(path))
        cache = MiniMaxH3AdalnCache(
            arch,
            weight_files=[str(path)],
            max_plans=2,
            max_plan_width=2,
            host_cache_bytes=8 * (2 * width + 8) * 2,
            precision="fp32",
        )
        cache.load(device)
        assert cache._host_tier.pinned
        a = [torch.tensor([0.0]), torch.tensor([0.25, 0.5])]
        b = [torch.tensor([0.625]), torch.tensor([0.75, 0.875])]
        records = []
        for name, plans, expected in [
            ("cold", a * 2, (0, 0, 2)),
            ("device", a, (2, 0, 0)),
            ("other", b, (0, 0, 2)),
            ("host", a, (0, 2, 0)),
        ]:
            metrics = RequestMetrics(name)
            with _record_adaln_cache_stats(
                SimpleNamespace(adaln_cache=cache),
                SimpleNamespace(metrics=metrics, is_warmup=False),
            ):
                cache.build(plans, embed=embed)
            for plan in plans:
                slot = int(cache.lookup(plan.to(device)))
                torch.testing.assert_close(
                    cache.plan_timesteps[slot, : len(plan)].cpu(), plan
                )
                assert int(cache.plan_lengths[slot]) == len(plan)
                # CPU full-width oracle is independent of TP-sharded CUDA GEMMs.
                for i, (prefix, _) in enumerate(prefixes):
                    want = (
                        embed(plan) @ weights[prefix + ".weight"].T
                        + weights[prefix + ".bias"]
                    ).bfloat16()
                    got = (
                        cache.block_params[slot, : len(plan), i]
                        if i < 2
                        else cache.final_params[slot, : len(plan)]
                    )
                    torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
            counts = metrics.cache_stats["minimax_h3_adaln"]["request"]
            assert (
                tuple(
                    counts[k]
                    for k in ("gpu_hit_plans", "host_hit_plans", "built_plans")
                )
                == expected
            )
            assert counts["host_pressure_skips"] == counts["host_evicted_groups"] == 0
            records.append(metrics.to_dict())
            if rank == 0:
                report = Path(temp) / "perf.json"
                PerformanceLogger.dump_benchmark_report(str(report), metrics)
                assert (
                    json.loads(report.read_text())["cache_stats"] == metrics.cache_stats
                )
        assert (
            records[0]["cache_stats"]["minimax_h3_adaln"]["cumulative"]["built_plans"]
            == 2
        )
        assert (
            records[-1]["cache_stats"]["minimax_h3_adaln"]["cumulative"]["built_plans"]
            == 4
        )
        print(
            json.dumps(
                {
                    "tp": tp,
                    "rank": rank,
                    "projection": "exact",
                    "host_pinned": True,
                    "records": records,
                }
            ),
            flush=True,
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
