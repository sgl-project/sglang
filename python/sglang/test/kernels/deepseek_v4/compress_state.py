"""DeepSeek-V4 compress cases with an fp32 and a bf16 state pool, built from one draw.

Shared by the bf16-state test and its benchmark.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch

from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
)
from sglang.test.kernels.deepseek_v4.common import (
    make_legacy_context,
    to_seq_extend,
)

Mode = Literal["decode", "prefill"]


class CompressSpec(NamedTuple):
    shape_name: str
    ratio: Literal[4, 128]
    mode: Mode
    batch_size: int
    head_dim: int
    tokens_per_req: int


class CompressCase(NamedTuple):
    state_pool: torch.Tensor
    kv_score_input: torch.Tensor
    ape: torch.Tensor
    out: torch.Tensor
    plan: CompressorDecodePlan | CompressorPrefillPlan
    spec: CompressSpec

    @property
    def effective_bytes(self) -> int:
        tensors = (self.state_pool, self.kv_score_input, self.ape, self.out)
        return sum(t.numel() * t.element_size() for t in tensors)

    def run(self) -> torch.Tensor:
        return compress_forward(
            kv_score_buffer=self.state_pool,
            kv_score_input=self.kv_score_input,
            ape=self.ape,
            plan=self.plan,
            head_dim=self.spec.head_dim,
            compress_ratio=self.spec.ratio,
            out=self.out,
        )


def _last_dim(ratio: int, head_dim: int) -> int:
    return head_dim * (4 if ratio == 4 else 2)


def _ape_len(ratio: int) -> int:
    return 8 if ratio == 4 else 128


def _make_plan(spec: CompressSpec):
    ctx = make_legacy_context(
        bs=spec.batch_size,
        compress_ratio=spec.ratio,
        head_dim=spec.head_dim,
    )
    if spec.mode == "decode":
        seq_lens = torch.full(
            (spec.batch_size,),
            spec.tokens_per_req,
            dtype=torch.int64,
            device="cuda",
        )
        plan = ctx.make_decode_plan(seq_lens)
        num_output_tokens = spec.batch_size
    else:
        seq_lens_cpu, extend_lens_cpu, num_q_tokens = to_seq_extend(
            [(spec.tokens_per_req, spec.tokens_per_req)] * spec.batch_size
        )
        plan = ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, num_q_tokens)
        num_output_tokens = int(plan.plan_c.shape[0])
    return ctx, plan, num_output_tokens


def _make_case(
    spec: CompressSpec,
    state_dtype: torch.dtype,
    base_state: torch.Tensor,
    base_input: torch.Tensor,
    base_ape: torch.Tensor,
    plan: CompressorDecodePlan | CompressorPrefillPlan,
    num_q_tokens: int,
) -> CompressCase:
    state_pool = base_state.to(state_dtype)
    kv_score_input = base_input.clone()
    ape = base_ape.clone()
    out = torch.empty(
        (num_q_tokens, spec.head_dim),
        dtype=kv_score_input.dtype,
        device=kv_score_input.device,
    )
    return CompressCase(
        state_pool=state_pool,
        kv_score_input=kv_score_input,
        ape=ape,
        out=out,
        plan=plan,
        spec=spec,
    )


def make_compress_pair(
    spec: CompressSpec, seed: int
) -> tuple[CompressCase, CompressCase]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ctx, plan, num_q_tokens = _make_plan(spec)
    last_dim = _last_dim(spec.ratio, spec.head_dim)

    if spec.mode == "decode":
        base_state = torch.randn(
            (ctx.num_pages, spec.ratio, last_dim),
            dtype=torch.float32,
            device="cuda",
        )
        input_rows = spec.batch_size
    else:
        base_state = torch.zeros(
            (ctx.num_pages, spec.ratio, last_dim),
            dtype=torch.float32,
            device="cuda",
        )
        input_rows = spec.batch_size * spec.tokens_per_req

    base_input = torch.randn(
        (input_rows, last_dim),
        dtype=torch.float32,
        device="cuda",
    )
    base_ape = torch.randn(
        (_ape_len(spec.ratio), spec.head_dim),
        dtype=torch.float32,
        device="cuda",
    )

    return (
        _make_case(
            spec, torch.float32, base_state, base_input, base_ape, plan, num_q_tokens
        ),
        _make_case(
            spec, torch.bfloat16, base_state, base_input, base_ape, plan, num_q_tokens
        ),
    )


# Compress shapes captured from DeepSeek-V4-Flash serving with EAGLE enabled; the
# service used prefill-style compress plans for both EXTEND and TARGET_VERIFY.
CAPTURED_FLASH_RUNTIME_SPECS = [
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 1),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 4),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 6),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 28),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 29),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 30),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 284),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 285),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 286),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 287),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 288),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 289),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 512),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 2, 128, 4),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 2, 128, 142),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 2, 128, 143),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 2, 128, 144),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 2, 128, 145),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 3, 128, 4),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 4, 128, 4),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 5, 128, 4),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 6, 128, 4),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 7, 128, 4),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 8, 128, 4),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 10, 128, 4),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 12, 128, 4),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 14, 128, 4),
    CompressSpec("flash-runtime-indexer", 4, "prefill", 16, 128, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 1),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 6),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 28),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 29),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 30),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 284),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 285),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 286),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 287),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 288),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 289),
    CompressSpec("flash-runtime-core", 4, "prefill", 1, 512, 512),
    CompressSpec("flash-runtime-core", 4, "prefill", 2, 512, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 2, 512, 142),
    CompressSpec("flash-runtime-core", 4, "prefill", 2, 512, 143),
    CompressSpec("flash-runtime-core", 4, "prefill", 2, 512, 144),
    CompressSpec("flash-runtime-core", 4, "prefill", 2, 512, 145),
    CompressSpec("flash-runtime-core", 4, "prefill", 3, 512, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 4, 512, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 5, 512, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 6, 512, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 7, 512, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 8, 512, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 10, 512, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 12, 512, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 14, 512, 4),
    CompressSpec("flash-runtime-core", 4, "prefill", 16, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 1),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 6),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 28),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 29),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 30),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 284),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 285),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 286),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 287),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 288),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 289),
    CompressSpec("flash-runtime-core", 128, "prefill", 1, 512, 512),
    CompressSpec("flash-runtime-core", 128, "prefill", 2, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 2, 512, 142),
    CompressSpec("flash-runtime-core", 128, "prefill", 2, 512, 143),
    CompressSpec("flash-runtime-core", 128, "prefill", 2, 512, 144),
    CompressSpec("flash-runtime-core", 128, "prefill", 2, 512, 145),
    CompressSpec("flash-runtime-core", 128, "prefill", 3, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 4, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 5, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 6, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 7, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 8, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 10, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 12, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 14, 512, 4),
    CompressSpec("flash-runtime-core", 128, "prefill", 16, 512, 4),
]
