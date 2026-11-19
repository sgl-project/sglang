"""GPU producer/plan contract: virtual bookkeeping and shared physical writes."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.memory.virtual_slot import translate_token_ids
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="extra-a", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestUniformExtendPlan(CustomTestCase):
    def test_dflash_draft_only_window(self):
        ps, width = 64, 5
        device = "cuda"
        full = torch.tensor([0, 3, 2, -1], device=device)
        window = torch.tensor([0, 1, 7, -1], device=device)
        rows = torch.tensor([1, 9, 0], device=device)[::2]
        starts = torch.tensor([0, 9, 3], device=device)[::2]
        bonus = torch.tensor([81, 9, 93], device=device)[::2]
        slots = torch.tensor(
            [[0, 0, 0, 128, -1, 99], [63, 64, 129, 192, 300, 99]], device=device
        )[:, :5]
        source = RunnerIds.__new__(RunnerIds)
        source.page_size, source.req_to_token = ps, slots
        source._role_sub_pools = {}
        source._spaces = {
            name: IdSpace(
                key=name,
                read_v2p=table,
                write=lambda _: self.fail("separate write translation"),
            )
            for name, table in ((IdSpaceKind.FULL, full), ("window_2047", window))
        }
        draft = RunnerIds.__new__(RunnerIds)
        draft._role_sub_pools = {IdSpaceKind.SLIDING_WINDOW: "window_2047"}
        draft._spaces = source._spaces
        positions = torch.empty((2, width), device=device, dtype=torch.int64)
        block = torch.empty_like(positions)
        plan = source.plan_uniform_extend(
            req_pool_indices=rows,
            seq_lens=starts,
            seq_lens_cpu=None,
            width=width,
            positions_out=positions,
            anchor_src=bonus,
            anchor_out=block,
            mask_token_id=77,
            window_sub_pool=draft.sub_pool_name(IdSpaceKind.SLIDING_WINDOW),
        )
        for result, expected in (
            (block, [[81, 77, 77, 77, 77], [93, 77, 77, 77, 77]]),
            (positions, [[0, 1, 2, 3, 4], [3, 4, 5, 6, 7]]),
            (plan.write_virtual, [63, 64, 129, 192, 300, 128, -1, 0, 0, 0]),
            (plan.write_ids(source), [63, 192, 129, 0, 0, 128, 0, 0, 0, 0]),
            (
                plan.write_ids(draft, kind=IdSpaceKind.SLIDING_WINDOW),
                [63, 64, 449, 0, 0, 448, 0, 0, 0, 0],
            ),
        ):
            torch.testing.assert_close(result, result.new_tensor(expected))

        # DSPARK supplies its own proposal ids and only updates the anchor.
        block[0].fill_(42)
        source.plan_uniform_extend(
            req_pool_indices=rows[:1],
            seq_lens=starts[:1],
            seq_lens_cpu=None,
            width=width,
            anchor_src=bonus[:1],
            anchor_out=block[:1],
        )
        torch.testing.assert_close(block[0], block.new_tensor([81, 42, 42, 42, 42]))

    def test_shared_writes_preserve_virtual_ids_and_page_offsets(self):
        """A fused gather must retain per-page offsets and every reader's space.

        Changing v2p after plan creation must not retranslate that iteration's
        writes; the next iteration must pick up the new map. This catches
        losing the producer's outputs or incorrectly sharing FULL with SWA.
        """
        for ps in (1, 128):
            for width in (7, 8, 65):
                for swa_enabled in (False, True):
                    with self.subTest(page_size=ps, width=width, swa=swa_enabled):
                        self._check_plan(ps, width, swa_enabled)

    def _check_plan(self, ps, width, swa_enabled):
        full_kind, swa_kind = IdSpaceKind.FULL, IdSpaceKind.SLIDING_WINDOW
        row_tokens = ((2 * ps + width + ps - 1) // ps) * ps
        page_count = 3 * row_tokens // ps
        req_to_token = torch.arange(3 * row_tokens, device="cuda").view(3, row_tokens)
        req_to_token[0].zero_()
        full = (
            torch.arange(page_count, device="cuda", dtype=torch.int64)
            .flip(0)
            .contiguous()
        )
        full[0] = 0
        swa = torch.arange(page_count, device="cuda", dtype=torch.int64)
        swa[row_tokens // ps + 1] = -1

        def space(key, table):
            return IdSpace(
                key=key,
                read_v2p=table,
                write=lambda ids: translate_token_ids(loc=ids, v2p=table, page_size=ps),
            )

        source = RunnerIds.__new__(RunnerIds)
        source._role_sub_pools = {}
        source.page_size = ps
        source.req_to_token = req_to_token
        source._spaces = {full_kind: space("full", full)}
        if swa_enabled:
            source._spaces[swa_kind] = space("swa", swa)
        fused = RunnerIds.__new__(RunnerIds)
        fused._role_sub_pools = {}
        fused._spaces = {full_kind: source.space(full_kind)}
        private = RunnerIds.__new__(RunnerIds)
        private._role_sub_pools = {}
        private._spaces = {full_kind: IdSpace(key="private")}
        rows = torch.tensor([2, 1, 0], device="cuda")
        lengths = torch.tensor([ps - 1, ps + 1, 0], device="cuda")
        expected_virtual = req_to_token[
            rows[:, None], lengths[:, None] + torch.arange(width, device="cuda")
        ].flatten()

        for iteration in range(2):
            if iteration:
                full[1:] = 2
                swa[1:] = 3
            plan = source.plan_uniform_extend(
                req_pool_indices=rows,
                seq_lens=lengths,
                seq_lens_cpu=lengths.cpu(),
                width=width,
            )
            expected = {}
            for kind, mapping in ((full_kind, full), (swa_kind, swa)):
                if source.space(kind) is not None:
                    expected[kind] = (
                        mapping[expected_virtual // ps] * ps + expected_virtual % ps
                    ).clamp_min(0)
            torch.testing.assert_close(plan.write_virtual, expected_virtual)
            # Retain this iteration's ids even if a later allocator update
            # changes the table before another consumer asks for its view.
            full[1:] = 4
            swa[1:] = 5
            torch.testing.assert_close(plan.write_ids(source), expected[full_kind])
            self.assertIs(plan.write_ids(fused), plan.write_ids(source))
            torch.testing.assert_close(plan.write_ids(private), expected_virtual)
            if swa_enabled:
                torch.testing.assert_close(
                    plan.write_ids(source, kind=swa_kind), expected[swa_kind]
                )
            fb = SimpleNamespace()
            plan.bind(fb, fused, cols=slice(0, width - 1))
            torch.testing.assert_close(
                fb.out_cache_loc, expected[full_kind].view(3, width)[:, :-1].reshape(-1)
            )
            torch.testing.assert_close(
                fb.out_cache_loc_virtual,
                expected_virtual.view(3, width)[:, :-1].reshape(-1),
            )
            packed = torch.tensor([width + 1, -1, 0], device="cuda")
            torch.testing.assert_close(
                plan.write_ids(source, cols=packed),
                torch.stack(
                    (
                        expected[full_kind][width + 1],
                        expected[full_kind].new_zeros(()),
                        expected[full_kind][0],
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
