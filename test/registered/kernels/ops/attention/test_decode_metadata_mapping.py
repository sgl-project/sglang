"""Decode page addressing and row-tail policy are independent contracts."""

import itertools
import unittest

import torch

from sglang.kernels.ops.attention.metadata import normal_decode_set_metadata
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDecodeMetadataMapping(CustomTestCase):
    def test_mapping_and_tail_policy(self):
        # Cross a whole column block of dead pages, so zero_tail must also
        # control the early exit rather than just the last block's store mask.
        for ps, mapped_full, swa_kind, zero_tail in itertools.product(
            (1, 128), (False, True), (None, "token", "page"), (False, True)
        ):
            with self.subTest(
                ps=ps, mapped_full=mapped_full, swa=swa_kind, zero_tail=zero_tail
            ):
                self._check_tables(ps, mapped_full, swa_kind, zero_tail)

    def _check_tables(self, ps, mapped_full, swa_kind, zero_tail):
        device = "cuda"
        bs, width, delta, sentinel = 3, 300, 3, -17
        # Use strided inputs/outputs and mappings: graph replay must read the
        # original storage, without materializing a contiguous mapping copy.
        virtual_pages = torch.arange(bs * width, device=device).view(bs, width) % 31
        tokens = (
            virtual_pages[:, :, None] * ps + torch.arange(ps, device=device)
        ).reshape(bs, width * ps)
        req_to_token = torch.stack((tokens, tokens), dim=-1)[:, :, 0]
        rows = torch.tensor([1, 9, 2, 9, 0, 9], device=device)[::2]
        lengths = torch.tensor([2 * ps + 1, 1, 0], device=device, dtype=torch.int32)
        seq_lens = torch.stack((lengths - delta, lengths), dim=1)[:, 0]
        full = ((torch.arange(64, device=device, dtype=torch.int32) * 7) % 31)[::2]
        swa = ((torch.arange(64, device=device, dtype=torch.int32) * 11) % 31)[::2]
        full[21] = -1
        swa[22] = -1
        slots = torch.arange(32 * ps, device=device)
        token_map = ((slots // ps * 3 + 5) % 31) * ps + slots % ps
        token_map = torch.stack((token_map, token_map), dim=1)[:, 0]
        full_map = full if mapped_full else None
        swa_map = swa if swa_kind == "page" else None
        token_map = token_map if swa_kind == "token" else None
        full_storage = torch.full(
            (bs, 2 * (width + 2)), sentinel, device=device, dtype=torch.int32
        )
        swa_storage = torch.full_like(full_storage, sentinel)
        page_table = full_storage[:, ::2]
        swa_table = swa_storage[:, ::2] if swa_kind is not None else None
        cache_lens = torch.empty_like(lengths)
        cu_lens = torch.empty(bs + 1, device=device, dtype=torch.int32)

        normal_decode_set_metadata(
            cache_lens,
            cu_lens,
            page_table,
            req_to_token,
            rows,
            width,
            seq_lens,
            delta,
            ps,
            swa_page_table=swa_table,
            full_to_swa_index_mapping=token_map,
            full_v2p=full_map,
            swa_v2p=swa_map,
            zero_tail=zero_tail,
        )

        torch.testing.assert_close(cache_lens, lengths)
        torch.testing.assert_close(cu_lens[1:], lengths.cumsum(0).int())
        self.assertEqual(cu_lens[0].item(), 0)
        gathered = tokens[rows, ::ps]
        live = (
            torch.arange(width, device=device)[None, :]
            < ((lengths + ps - 1) // ps)[:, None]
        )
        expected_full = gathered // ps
        if mapped_full:
            expected_full = full[expected_full].clamp_min(0)
        expected_swa = None
        if swa_kind == "token":
            expected_swa = token_map[gathered] // ps
        elif swa_kind == "page":
            expected_swa = swa[gathered // ps].clamp_min(0)
        for storage, expected in (
            (full_storage, expected_full),
            (swa_storage, expected_swa),
        ):
            if expected is None:
                continue
            want = torch.full_like(storage, sentinel)
            want[:, : 2 * width : 2] = torch.where(
                live, expected, 0 if zero_tail else sentinel
            )
            torch.testing.assert_close(storage, want)

    def test_lengths_only_leaves_tables_untouched(self):
        for ps, skip in itertools.product((1, 128), (False, True)):
            with self.subTest(ps=ps, skip=skip):
                seq_lens = torch.tensor([0, 9, 130], device="cuda")
                cache_lens = torch.empty(3, device="cuda", dtype=torch.int32)
                cu_lens = torch.empty(4, device="cuda", dtype=torch.int32)
                table = torch.full((3, 4), 23, device="cuda", dtype=torch.int32)
                swa_table = table.clone()
                normal_decode_set_metadata(
                    cache_lens,
                    cu_lens,
                    table,
                    torch.zeros((3, 4 * ps), device="cuda", dtype=torch.int64),
                    torch.arange(3, device="cuda"),
                    4 if skip else 0,
                    seq_lens,
                    3,
                    ps,
                    swa_page_table=swa_table,
                    full_to_swa_index_mapping=torch.arange(4 * ps, device="cuda"),
                    skip_page_table=skip,
                    zero_tail=True,
                )
                torch.testing.assert_close(cache_lens, (seq_lens + 3).int())
                torch.testing.assert_close(
                    cu_lens,
                    torch.tensor([0, 3, 15, 148], device="cuda", dtype=torch.int32),
                )
                torch.testing.assert_close(table, torch.full_like(table, 23))
                torch.testing.assert_close(swa_table, table)


if __name__ == "__main__":
    unittest.main()
