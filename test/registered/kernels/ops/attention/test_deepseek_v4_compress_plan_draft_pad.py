"""Kernel-level tests for the DSV4 compress write-plan (`plan_prefill`).

The write plan (`plan_w`) decides which tokens' raw KV get persisted into the
compress-state ring so that a *future* compression window can still read them.
On a speculative verify batch the plan is built from the optimistic
`seq_len = prefix + num_draft_tokens`, but the sequence tail rolls back to
`prefix + accept_len`. Every committed token must therefore stay resident in the
ring whatever the accept length turns out to be, which requires the plan to
write all of `[prefix, seq_len)`.

`c_plan.cuh` used to hard-code that pad at 4 (`kMaxMTPDraftTokens`), silently
under-writing the ring for `num_draft_tokens > 4` — no IMA, no NaN, just wrong
compressed state. These tests pin the invariant for both planner paths (CPU host
loop and GPU `plan_compress_prefill_kernel0`) and both compress ratios.
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.common import make_paged_context, to_seq_extend
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

C4_RING_SIZE = 16  # get_compress_state_ring_size(4, is_speculative=True)
C128_RING_SIZE = 256  # get_compress_state_ring_size(128, is_speculative=True)


def _window_size(compress_ratio: int) -> int:
    """Tokens read by one compression: c4 overlaps two chunks, c128 does not."""
    return compress_ratio * (2 if compress_ratio == 4 else 1)


def _max_draft_tokens(compress_ratio: int, ring_size: int) -> int:
    """Largest `num_draft_tokens` the ring can serve; mirrors the C++ RuntimeCheck."""
    return ring_size - _window_size(compress_ratio) + 2


def _written_positions(plan_w: torch.Tensor, prefix_len: int) -> set[int]:
    """Decode `plan_w` into the set of positions written, for a bs=1 plan.

    `plan_w` is `[n, 8]` uint8 = (uint32 ragged_id, int32 write_loc). Stage 1
    overwrites `write_loc` with the final state slot, but `ragged_id` survives and
    equals the token's index within the ragged layout, so for a single request
    `position = prefix_len + ragged_id`.
    """
    words = plan_w.cpu().view(torch.uint32).view(-1, 2)
    ragged_ids = words[:, 0]
    valid = ragged_ids != 0xFFFFFFFF
    return {prefix_len + int(r) for r in ragged_ids[valid]}


class TestCompressWritePlanDraftPad(CustomTestCase):
    def _make_plan_positions(
        self,
        *,
        compress_ratio: int,
        ring_size: int,
        prefix_len: int,
        num_draft_tokens: int,
        on_gpu: bool = False,
    ) -> set[int]:
        """Build a bs=1 verify plan and return the positions it writes.

        `on_gpu=True` moves the planner inputs to device, which routes
        `plan_prefill` to `plan_compress_prefill_kernel0` instead of the host loop.
        """
        ctx = make_paged_context(
            bs=1, compress_ratio=compress_ratio, ring_size=ring_size
        )
        seq_lens, extend_lens, num_q = to_seq_extend(
            [(prefix_len + num_draft_tokens, num_draft_tokens)]
        )
        if on_gpu:
            seq_lens = seq_lens.to(ctx.req_to_token.device)
            extend_lens = extend_lens.to(ctx.req_to_token.device)
        plan = ctx.make_prefill_plan(
            seq_lens, extend_lens, num_q, num_draft_tokens=num_draft_tokens
        )
        return _written_positions(plan.plan_w, prefix_len)

    def _assert_ring_residency(self, compress_ratio: int, ring_size: int):
        """Every committed token must be written, for each (D, sl mod cr) combo.

        This is the sufficient condition: if a step writes all of
        `[prefix, prefix + D)`, then whatever the accept length, the tokens the
        next compression window needs are either from this step (written here) or
        older (written by an earlier step, same invariant by induction).
        """
        max_d = _max_draft_tokens(compress_ratio, ring_size)
        # Cover every residue of `seq_len % compress_ratio` — that residue is what
        # decides whether the unpadded rule alone would have sufficed. Bases start
        # page-aligned so the swa-page-boundary write clause does not mask the pad.
        bases = [512 + off for off in range(compress_ratio)]
        draft_counts = sorted(
            {1, 2, 3, 4, 5, max_d - 1, max_d} & set(range(1, max_d + 1))
        )
        for num_draft_tokens in draft_counts:
            for prefix_len in bases:
                with self.subTest(
                    cr=compress_ratio,
                    D=num_draft_tokens,
                    prefix=prefix_len,
                ):
                    written = self._make_plan_positions(
                        compress_ratio=compress_ratio,
                        ring_size=ring_size,
                        prefix_len=prefix_len,
                        num_draft_tokens=num_draft_tokens,
                    )
                    seq_len = prefix_len + num_draft_tokens
                    missing = set(range(prefix_len, seq_len)) - written
                    self.assertEqual(
                        missing,
                        set(),
                        f"plan_w skipped committed positions {sorted(missing)}; "
                        f"a later compression would read stale ring slots",
                    )

    def test_c4_ring_residency(self):
        self._assert_ring_residency(4, C4_RING_SIZE)

    def test_c128_ring_residency(self):
        self._assert_ring_residency(128, C128_RING_SIZE)

    def _assert_window_covered_over_decode_chain(
        self, compress_ratio: int, ring_size: int, num_draft_tokens: int
    ):
        """Replay a verify chain and check each compression window is resident.

        Simulates the worst accept length (a=1) at every step, accumulating the
        positions each step's `plan_w` writes. Whenever a step's tail crosses a
        compression point, the whole window that compression reads must already be
        in the accumulated set.
        """
        window = _window_size(compress_ratio)
        for start_prefix in (512, 512 + compress_ratio // 2 + 1):
            with self.subTest(
                cr=compress_ratio,
                D=num_draft_tokens,
                start=start_prefix,
            ):
                resident: set[int] = set(range(start_prefix))  # prior prefill
                prefix_len = start_prefix
                for _ in range(2 * compress_ratio + 4):
                    resident |= self._make_plan_positions(
                        compress_ratio=compress_ratio,
                        ring_size=ring_size,
                        prefix_len=prefix_len,
                        num_draft_tokens=num_draft_tokens,
                    )
                    prefix_len += 1  # accept exactly one token
                    p = prefix_len - 1
                    if (p + 1) % compress_ratio == 0:
                        need = {q for q in range(p - window + 1, p + 1) if q >= 0}
                        self.assertTrue(
                            need <= resident,
                            f"compression at p={p} reads unwritten positions "
                            f"{sorted(need - resident)}",
                        )

    def test_c4_window_covered_over_decode_chain(self):
        max_d = _max_draft_tokens(4, C4_RING_SIZE)
        for num_draft_tokens in (1, 4, 5, max_d):
            self._assert_window_covered_over_decode_chain(
                4, C4_RING_SIZE, num_draft_tokens
            )

    def test_c128_window_covered_over_decode_chain(self):
        for num_draft_tokens in (1, 4, 8):
            self._assert_window_covered_over_decode_chain(
                128, C128_RING_SIZE, num_draft_tokens
            )

    def test_cpu_and_gpu_planner_agree(self):
        """Both planner paths must emit the same write set.

        The residency invariants above are checked on the host-loop plan; this
        pins the GPU `plan_compress_prefill_kernel0` plan to it, so the pad fix
        has to hold on both paths.
        """
        for compress_ratio, ring_size in ((4, C4_RING_SIZE), (128, C128_RING_SIZE)):
            max_d = _max_draft_tokens(compress_ratio, ring_size)
            for num_draft_tokens in (1, 4, max_d):
                for prefix_len in (512, 513, 515):
                    with self.subTest(
                        cr=compress_ratio, D=num_draft_tokens, prefix=prefix_len
                    ):
                        kwargs = dict(
                            compress_ratio=compress_ratio,
                            ring_size=ring_size,
                            prefix_len=prefix_len,
                            num_draft_tokens=num_draft_tokens,
                        )
                        self.assertEqual(
                            self._make_plan_positions(**kwargs, on_gpu=False),
                            self._make_plan_positions(**kwargs, on_gpu=True),
                        )

    def test_plain_prefill_unaffected(self):
        """`num_draft_tokens=0` must reproduce the non-speculative write rule."""
        for compress_ratio, ring_size in ((4, C4_RING_SIZE), (128, C128_RING_SIZE)):
            is_overlap = compress_ratio == 4
            for seq_len in (512, 600, 777):
                with self.subTest(cr=compress_ratio, sl=seq_len):
                    ctx = make_paged_context(
                        bs=1, compress_ratio=compress_ratio, ring_size=ring_size
                    )
                    seq_lens, extend_lens, num_q = to_seq_extend([(seq_len, seq_len)])
                    plan = ctx.make_prefill_plan(
                        seq_lens, extend_lens, num_q, num_draft_tokens=0
                    )
                    written = _written_positions(plan.plan_w, 0)

                    last_c_pos = seq_len // compress_ratio * compress_ratio
                    first_w_pos = last_c_pos - (compress_ratio if is_overlap else 0)
                    sps = ctx.swa_page_size
                    expected = {
                        p
                        for p in range(seq_len)
                        if p >= first_w_pos
                        or (is_overlap and p % sps >= sps - compress_ratio)
                    }
                    self.assertEqual(written, expected)

    def test_over_capacity_raises(self):
        """`num_draft_tokens` beyond `ring_size - window_size` must fail loudly."""
        for compress_ratio, ring_size in ((4, C4_RING_SIZE), (128, C128_RING_SIZE)):
            too_many = _max_draft_tokens(compress_ratio, ring_size) + 1
            with self.subTest(cr=compress_ratio, D=too_many):
                with self.assertRaises(Exception):
                    self._make_plan_positions(
                        compress_ratio=compress_ratio,
                        ring_size=ring_size,
                        prefix_len=512,
                        num_draft_tokens=too_many,
                    )


if __name__ == "__main__":
    unittest.main()
