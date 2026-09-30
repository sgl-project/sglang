#!/usr/bin/env python3
"""GPU correctness smoke for FusedDecodePlan on B200.

For decode (next_n 1) and verify (next_n > 1) batches, every token's selected physical slots must equal the exact
top-2048 of DeepGEMM's own scores, ties going to the lower physical slot. Also checks that the histogram and the
candidate buffer return to zero and the hand-off counters balance, that a stale histogram and an overflowing
candidate buffer still select exactly, and CUDA
graph replay after in-place input updates. Random finite FP8 inputs; valid lengths and page tables only.
"""

from __future__ import annotations

import argparse

import deep_gemm
import torch

from sglang.kernels.experimental.litetopk_decode.fused import TOPK, FusedDecodePlan

PAGE = 64


class Inputs:
    """Fixed-address inputs of one batch shape; `fill` rewrites their contents."""

    def __init__(self, batch, next_n, max_len, device):
        self.batch, self.next_n, self.max_len = batch, next_n, max_len
        self.pages = (max_len + PAGE - 1) // PAGE
        self.physical_pages = self.pages + 7
        self.q = torch.empty(
            (batch, next_n, 32, 128), dtype=torch.float8_e4m3fn, device=device
        )
        self.weights = torch.empty(
            (batch * next_n, 32), dtype=torch.float32, device=device
        )
        self.cache = torch.empty(
            (self.physical_pages, PAGE, 1, 132), dtype=torch.uint8, device=device
        )
        self.table = torch.empty((batch, self.pages), dtype=torch.int32, device=device)
        self.lengths = torch.zeros((batch, next_n), dtype=torch.int32, device=device)
        self.schedule = self.metadata()

    def metadata(self):
        return deep_gemm.get_paged_mqa_logits_metadata(
            self.lengths, PAGE, deep_gemm.get_num_sms()
        )

    def fill(self, base_lengths, gen):
        device = self.q.device

        def fp8(shape):
            # Magnitudes below 127 exclude the FP8 NaN encodings
            magnitude = torch.randint(
                0, 127, shape, dtype=torch.uint8, device=device, generator=gen
            )
            return magnitude | (
                torch.randint(
                    0, 2, shape, dtype=torch.uint8, device=device, generator=gen
                )
                << 7
            )

        self.q.view(torch.uint8).copy_(fp8(self.q.shape))
        self.weights.normal_(generator=gen).mul_(0.05)
        packed = self.cache.view(self.physical_pages, -1)
        packed[:, : PAGE * 128].copy_(fp8((self.physical_pages, PAGE * 128)))
        scales = (
            torch.randint(
                -12, -5, (self.physical_pages, PAGE), device=device, generator=gen
            )
            .float()
            .exp2()
        )
        packed[:, PAGE * 128 :].copy_(scales.view(torch.uint8))
        for row in range(self.batch):
            self.table[row].copy_(
                torch.randperm(self.physical_pages, device=device, generator=gen)[
                    : self.pages
                ]
            )
        # Token j of a verify step sees j more keys than its request's first token
        live = [
            [min(n + j, self.max_len) if n else 0 for j in range(self.next_n)]
            for n in base_lengths
        ]
        self.lengths.copy_(torch.tensor(live, dtype=torch.int32))
        self.schedule.copy_(self.metadata())

    def call(self, plan):
        return plan(
            self.q,
            self.cache,
            self.weights,
            self.lengths,
            self.table,
            self.schedule,
            self.max_len,
        )

    def expected(self):
        """Exact top-K physical slots per token over DeepGEMM's scores, sorted, padded with -1."""
        scores = deep_gemm.fp8_fp4_paged_mqa_logits(
            (self.q, None),
            self.cache,
            self.weights,
            self.lengths,
            self.table,
            self.schedule,
            self.max_len,
            clean_logits=False,
        )
        table = self.table.long()
        result = torch.full(
            (scores.shape[0], TOPK), -1, dtype=torch.int64, device=scores.device
        )
        for row, n in enumerate(self.lengths.view(-1).tolist()):
            index = torch.arange(n, device=scores.device)
            slots = table[row // self.next_n, index // PAGE] * PAGE + index % PAGE
            if n > TOPK:
                bits = scores[row, :n].view(torch.int32).long() & 0xFFFFFFFF
                key = torch.where(bits >= 1 << 31, bits ^ 0xFFFFFFFF, bits | 1 << 31)
                slots = slots[
                    torch.topk((key << 31) | (0x7FFFFFFF - slots), TOPK).indices
                ]
            result[row, : slots.numel()] = slots.sort().values
        return result


def check(plan, inputs, output, label):
    got = output.long()
    got = torch.where(got < 0, torch.iinfo(torch.int64).max, got).sort(dim=1).values
    got = torch.where(got == torch.iinfo(torch.int64).max, -1, got)
    want = inputs.expected()
    bad = (got != want).any(dim=1).nonzero().flatten().tolist()
    assert not bad, f"{label}: rows {bad[:8]} differ from the exact top-{TOPK}"
    rows = plan.histogram.shape[0]
    state = plan.workspace[: rows * 16].view(torch.int32).view(rows, 4)
    assert not plan.histogram.any(), f"{label}: histogram not returned to zero"
    assert not state[:, :2].any() and torch.equal(state[:, 2], state[:, 3]), (
        f"{label}: hand-off state not at rest"
    )
    assert not plan.workspace[rows * 16 :].any(), (
        f"{label}: candidate buffer not returned to zero"
    )


def patterns(batch, max_len):
    boundary = [0, 1, 63, 2047, 2048, 2049, 4097]
    return {
        "long": [max_len - 3] * batch,
        "mixed": [
            (boundary + [max_len // 3, max_len - 3])[r % 9] for r in range(batch)
        ],
    }


def run(batch, next_n, max_len, gen):
    device = torch.device("cuda")
    inputs = Inputs(batch, next_n, max_len, device)
    plan = FusedDecodePlan(batch, next_n, device)
    inputs.fill(patterns(batch, max_len)["long"], gen)
    check(plan, inputs, inputs.call(plan), "eager")
    # Counts left in the histogram make it disagree with the scores: the whole-row path must still be exact
    plan.histogram[:, 5] += 1
    check(plan, inputs, inputs.call(plan), "stale histogram")
    small = FusedDecodePlan(batch, next_n, device, candidate_capacity=16)
    check(small, inputs, inputs.call(small), "candidate overflow")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = inputs.call(plan)
    replay_patterns = patterns(batch, max_len)
    for name, lengths in [
        *replay_patterns.items(),
        ("long again", replay_patterns["long"]),
    ]:
        inputs.fill(lengths, gen)
        graph.replay()
        check(plan, inputs, output, f"graph {name}")
    print(f"ok batch={batch} next_n={next_n} max_len={max_len}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,3,33,128")
    parser.add_argument("--next-n", default="1,2,3,4")
    parser.add_argument("--max-len", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument(
        "--pdl",
        action="store_true",
        help="Enable DeepGEMM PDL before eager calls and graph capture",
    )
    args = parser.parse_args()
    deep_gemm.set_pdl(args.pdl)
    assert deep_gemm.get_pdl() == args.pdl
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    for next_n in map(int, args.next_n.split(",")):
        for batch in map(int, args.batches.split(",")):
            run(batch, next_n, args.max_len, gen)
    print("all passed")


if __name__ == "__main__":
    main()
