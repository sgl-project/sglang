#!/usr/bin/env python3
"""Distributed BF16 MoonEP PoC validation.

Run with torchrun on a single NVLink/NVSwitch node, for example:

  SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 \
  torchrun --standalone --nproc-per-node=4 \
    scripts/moonep/validate_moonep_bf16_poc.py --tokens 128 --hidden-size 1024

The script validates SGLang's MoonEP BF16 path:
MoonEPDispatcher.dispatch -> MoonEPBuffer.prefetch_weight -> BF16 segment runner
-> MoonEPDispatcher.combine.  It compares the final token-major output against a
rank-local PyTorch reference using the same top-k and expert weights.
"""

from __future__ import annotations

import argparse
from enum import IntEnum, auto
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import NamedTuple, Protocol, TypeGuard, runtime_checkable

import torch
import torch.distributed as dist
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--prefetch-slots", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    return parser.parse_args()


def setup_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def make_topk(tokens: int, top_k: int, num_experts: int, device: torch.device):
    # Deterministic but rank-local routing.  Keep weights normalized so the
    # reference magnitude remains bounded.
    topk_ids = torch.randint(
        0,
        num_experts,
        (tokens, top_k),
        device=device,
        dtype=torch.int64,
    )
    raw_weights = torch.rand(tokens, top_k, device=device, dtype=torch.float32)
    topk_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
    return topk_ids, topk_weights


class LightweightTopKOutputFormat(IntEnum):
    STANDARD = auto()


@runtime_checkable
class LightweightTopKOutput(Protocol):
    @property
    def format(self) -> LightweightTopKOutputFormat: ...


class LightweightStandardTopKOutput(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def format(self) -> LightweightTopKOutputFormat:
        return LightweightTopKOutputFormat.STANDARD


class LightweightTopKOutputChecker:
    @staticmethod
    def format_is_standard(
        topk_output: LightweightTopKOutput,
    ) -> TypeGuard[LightweightStandardTopKOutput]:
        return isinstance(topk_output, LightweightStandardTopKOutput)


class LightweightDeepEPMode(IntEnum):
    NORMAL = 1
    LOW_LATENCY = 2
    AUTO = 3


def install_lightweight_sglang_imports() -> None:
    """Install validation-only SGLang package stubs.

    Full SGLang environments should use the normal imports.  Minimal remote
    validation images often do not carry every frontend/model dependency that
    SGLang's package ``__init__`` imports, though this script only needs the
    MoonEP dispatcher, its base protocol, envs, and runtime buffer registry.
    These stubs keep the package search path pointed at the checked-out source
    tree while bypassing heavyweight parent ``__init__`` files.
    """

    repo_root = Path(__file__).resolve().parents[2]
    sglang_root = repo_root / "python" / "sglang"

    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            del sys.modules[name]

    def add_package(name: str, path: Path) -> None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    add_package("sglang", sglang_root)
    add_package("sglang.srt", sglang_root / "srt")
    add_package("sglang.srt.layers", sglang_root / "srt" / "layers")
    add_package("sglang.srt.layers.moe", sglang_root / "srt" / "layers" / "moe")
    add_package(
        "sglang.srt.layers.moe.token_dispatcher",
        sglang_root / "srt" / "layers" / "moe" / "token_dispatcher",
    )

    topk_module = ModuleType("sglang.srt.layers.moe.topk")
    topk_module.TopKOutputFormat = LightweightTopKOutputFormat
    topk_module.TopKOutput = LightweightTopKOutput
    topk_module.StandardTopKOutput = LightweightStandardTopKOutput
    topk_module.TopKOutputChecker = LightweightTopKOutputChecker
    sys.modules[topk_module.__name__] = topk_module

    utils_module = ModuleType("sglang.srt.layers.moe.utils")
    utils_module.DeepEPMode = LightweightDeepEPMode
    sys.modules[utils_module.__name__] = utils_module


def import_moonep_validation_symbols():
    try:
        from sglang.srt.layers.moe.token_dispatcher.moonep import (
            MoonEPBuffer,
            MoonEPDispatcher,
            MoonEPExpertWeightLayout,
            run_moonep_bf16_expert,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        return (
            MoonEPBuffer,
            MoonEPDispatcher,
            MoonEPExpertWeightLayout,
            run_moonep_bf16_expert,
            StandardTopKOutput,
        )
    except ModuleNotFoundError as exc:
        print(
            "Normal SGLang import failed; retrying with lightweight validation "
            f"imports: {exc}",
            file=sys.stderr,
            flush=True,
        )
        install_lightweight_sglang_imports()
        from sglang.srt.layers.moe.token_dispatcher.moonep import (
            MoonEPBuffer,
            MoonEPDispatcher,
            MoonEPExpertWeightLayout,
            run_moonep_bf16_expert,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        return (
            MoonEPBuffer,
            MoonEPDispatcher,
            MoonEPExpertWeightLayout,
            run_moonep_bf16_expert,
            StandardTopKOutput,
        )


def expert_mlp(x, expert_id: int, gate, up, down):
    return F.linear(
        F.silu(F.linear(x, gate[expert_id])) * F.linear(x, up[expert_id]),
        down[expert_id],
    )


def reference_output(hidden, topk_ids, topk_weights, gate, up, down):
    out = torch.zeros_like(hidden)
    tokens, top_k = topk_ids.shape
    for token_idx in range(tokens):
        x = hidden[token_idx : token_idx + 1]
        acc = torch.zeros_like(x)
        for k in range(top_k):
            expert_id = int(topk_ids[token_idx, k].item())
            acc += expert_mlp(x, expert_id, gate, up, down) * topk_weights[
                token_idx, k
            ].to(hidden.dtype)
        out[token_idx] = acc[0]
    return out


def main() -> None:
    args = parse_args()
    os.environ["SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = str(args.tokens)
    if args.prefetch_slots > 0:
        os.environ["SGLANG_MOONEP_NUM_PREFETCH_SLOTS"] = str(args.prefetch_slots)

    rank, world_size, local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)

    (
        MoonEPBuffer,
        MoonEPDispatcher,
        MoonEPExpertWeightLayout,
        run_moonep_bf16_expert,
        StandardTopKOutput,
    ) = import_moonep_validation_symbols()

    num_experts = world_size * args.experts_per_rank
    hidden = torch.randn(
        args.tokens,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    topk_ids, topk_weights = make_topk(args.tokens, args.top_k, num_experts, device)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=torch.empty(0, device=device),
    )

    dispatcher = MoonEPDispatcher(
        group=dist.group.WORLD,
        router_topk=args.top_k,
        num_experts=num_experts,
        num_local_experts=args.experts_per_rank,
        hidden_size=args.hidden_size,
        params_dtype=torch.bfloat16,
    )

    dispatch_output = dispatcher.dispatch(hidden, topk_output)
    num_prefetch_slots = int(dispatch_output.expert_ids.numel()) - num_experts

    # Full global rows are deliberately replicated for this correctness PoC.
    # The production path should replace this with true symmetric expert-row
    # mappings whose physical storage is owned by each expert's home rank.
    torch.manual_seed(args.seed)
    gate = torch.randn(
        num_experts + num_prefetch_slots,
        args.intermediate_size,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    ) / 8
    up = torch.randn_like(gate) / 8
    down = torch.randn(
        num_experts + num_prefetch_slots,
        args.hidden_size,
        args.intermediate_size,
        device=device,
        dtype=torch.bfloat16,
    ) / 8
    gate[num_experts:].zero_()
    up[num_experts:].zero_()
    down[num_experts:].zero_()
    layout = MoonEPExpertWeightLayout(
        gate.contiguous(),
        up.contiguous(),
        down.contiguous(),
        num_prefetch_slots,
    )

    dispatcher.prefetch_weight(dispatch_output.plan, layout)
    combine_input = run_moonep_bf16_expert(dispatch_output, layout)
    output = dispatcher.combine(combine_input)

    expected = reference_output(hidden, topk_ids, topk_weights, gate, up, down)
    max_abs_err = (output.float() - expected.float()).abs().max()
    rel_err = max_abs_err / expected.float().abs().max().clamp_min(1e-6)
    local_ok = bool(
        torch.allclose(output.float(), expected.float(), atol=args.atol, rtol=args.rtol)
    )
    ok_tensor = torch.tensor([1 if local_ok else 0], device=device, dtype=torch.int32)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)

    result = {
        "rank": rank,
        "world_size": world_size,
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "top_k": args.top_k,
        "num_experts": num_experts,
        "num_prefetch_slots": num_prefetch_slots,
        "max_abs_err": float(max_abs_err.item()),
        "relative_err": float(rel_err.item()),
        "local_ok": local_ok,
        "global_ok": bool(ok_tensor.item()),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    dist.barrier(device_ids=[local_rank])
    MoonEPBuffer.destroy_all_buffers()
    dist.destroy_process_group()
    if rank == 0 and not bool(ok_tensor.item()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
