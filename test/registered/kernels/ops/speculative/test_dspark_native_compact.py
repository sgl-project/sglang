"""Differential checks of the actual kernels, on CPU interpreter or CUDA."""

import os
import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dspark import native_compact as kernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def test_native_compact_kernels_and_replay():
    if os.environ.get("TRITON_INTERPRET") == "1":
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        pytest.skip("requires CUDA, or TRITON_INTERPRET=1 for CPU differential checks")
    generator = torch.Generator().manual_seed(412)
    tested = 0
    for bs in (1, 2, 3, 8, 32, 64, 65, 96, 128):
        for variant in range(5):
            width = 6
            lengths = torch.randint(1, 7, (bs,), generator=generator)
            if variant == 0:
                lengths.fill_(6)
            if variant == 1:
                lengths.fill_(1)
            if variant == 2:
                lengths[::3] = 0
            total = int(lengths.sum())
            n = max(8, (total + 7) // 8 * 8)
            lens = lengths.to(device=device, dtype=torch.int64)
            ptr = torch.cat(
                [torch.zeros(1, dtype=torch.int64, device=device), lens.cumsum(0)]
            ).int()
            req = torch.arange(1, bs + 1, device=device, dtype=torch.int64)
            prefix = torch.arange(bs, device=device, dtype=torch.int64) % 10 + 60
            table = torch.arange((bs + 1) * 96, device=device).view(bs + 1, 96)
            anchors = torch.arange(bs * 10, device=device).view(bs, 10)[:, ::2]
            drafts = torch.randint(0, 37, (bs, 10), generator=generator).to(device)[
                :, ::2
            ]
            ids = torch.full((n,), -999, dtype=torch.int64, device=device)
            pos, loc, window = [torch.full_like(ids, -999) for _ in range(3)]
            kernel.pack_verify_inputs(
                indptr=ptr,
                reqs=req,
                prefixes=prefix,
                req_to_token=table,
                anchors=anchors,
                drafts=drafts,
                width=width,
                ids=ids,
                positions=pos,
                locations=loc,
                window_index=window,
            )
            dense_ids = torch.cat([anchors[:, :1], drafts], dim=1)
            mask = torch.arange(6, device=device)[None, :] < lens[:, None]
            expected = [torch.zeros_like(ids) for _ in range(3)]
            expected_window = torch.full_like(ids, -1)
            expected[0][:total] = dense_ids[mask]
            dense_pos = prefix[:, None] + torch.arange(6, device=device)
            expected[1][:total] = dense_pos[mask]
            expected[2][:total] = table[req[:, None].expand(-1, 6), dense_pos][mask]
            expected_window[:total] = torch.arange(bs * 6, device=device).view(bs, 6)[
                mask
            ]
            for actual, reference in zip(
                (ids, pos, loc, window), (*expected, expected_window)
            ):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

            prediction = torch.randint(0, 37, (n,), generator=generator).to(device)
            dense_prediction = torch.zeros((bs, 6), dtype=torch.int64, device=device)
            padded_candidates = torch.where(mask, dense_ids, 0)
            if variant in (0, 3):
                # All draft tokens match; last-row prediction zero exercises cap_trim.
                dense_prediction[:, :-1] = padded_candidates[:, 1:]
                prediction[:total] = dense_prediction[mask]
            dense_prediction[mask] = prediction[:total]
            raw_correct = (
                (dense_prediction[:, :-1] == padded_candidates[:, 1:])
                .int()
                .cumprod(1)
                .sum(1)
            )
            correct = torch.minimum(raw_correct, lens - 1)
            bonus = dense_prediction.gather(1, correct.clamp(min=0)[:, None]).flatten()
            bonus = torch.where(lens > 0, bonus, 0)
            outputs = [torch.empty_like(lens) for _ in range(3)]
            kernel.accept_ragged(
                ids=ids,
                prediction=prediction,
                indptr=ptr,
                lens=lens,
                width=6,
                correct=outputs[0],
                bonus=outputs[1],
                trim=outputs[2],
            )
            for actual, reference in zip(
                outputs, (correct, bonus, raw_correct - correct)
            ):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            small = torch.empty_like(dense_ids)
            kernel.ragged_candidates(ids=ids, indptr=ptr, lens=lens, out=small)
            torch.testing.assert_close(small, padded_candidates, rtol=0, atol=0)

            commits = correct + 1
            mapping = torch.arange(table.numel(), device=device, dtype=torch.int64) + 17
            hidden = torch.arange(n * 7, device=device).view(n, 7).float()
            dense_hidden = torch.zeros((bs, 6, 7), device=device)
            dense_hidden[mask] = hidden[:total]
            for ring in (None, 64):
                for armed in (0, 1):
                    gate = torch.tensor([armed], device=device, dtype=torch.int32)
                    out_pos = torch.full_like(ids, -999)
                    out_loc = torch.full((n,), -999, device=device, dtype=torch.int32)
                    kernel.compact_commit_layout(
                        indptr=ptr,
                        lens=lens,
                        reqs=req,
                        prefixes=prefix,
                        commits=commits,
                        gate=gate,
                        req_to_token=table,
                        full_to_swa=mapping,
                        ring=ring,
                        positions=out_pos,
                        locations=out_loc,
                    )
                    committed = (
                        mask
                        & (torch.arange(6, device=device)[None, :] < commits[:, None])
                        & bool(armed)
                    )
                    dense_loc = (
                        req[:, None] * ring + dense_pos % ring
                        if ring
                        else mapping[table[req[:, None].expand(-1, 6), dense_pos]]
                    )
                    expected_loc = torch.full_like(out_loc, -1)
                    expected_loc[:total] = torch.where(committed, dense_loc, -1)[
                        mask
                    ].int()
                    torch.testing.assert_close(out_loc, expected_loc, rtol=0, atol=0)
                    torch.testing.assert_close(out_pos, expected[1], rtol=0, atol=0)
                    cache_a = torch.zeros((int(mapping.max()) + 128, 7), device=device)
                    cache_b = torch.zeros_like(cache_a)
                    valid = out_loc >= 0
                    cache_a[out_loc[valid].long()] = hidden[valid]
                    cache_b[dense_loc[committed]] = dense_hidden[committed]
                    torch.testing.assert_close(cache_a, cache_b, rtol=0, atol=0)
            tested += 1
    if device == "cuda":
        # Replay must consume new layouts and gates at the same addresses.
        def operation():
            kernel.pack_verify_inputs(
                indptr=ptr,
                reqs=req,
                prefixes=prefix,
                req_to_token=table,
                anchors=anchors,
                drafts=drafts,
                width=6,
                ids=ids,
                positions=pos,
                locations=loc,
                window_index=window,
            )
            kernel.accept_ragged(
                ids=ids,
                prediction=prediction,
                indptr=ptr,
                lens=lens,
                width=6,
                correct=outputs[0],
                bonus=outputs[1],
                trim=outputs[2],
            )
            kernel.ragged_candidates(ids=ids, indptr=ptr, lens=lens, out=small)
            kernel.compact_commit_layout(
                indptr=ptr,
                lens=lens,
                reqs=req,
                prefixes=prefix,
                commits=outputs[0] + 1,
                gate=gate,
                req_to_token=table,
                full_to_swa=mapping,
                ring=64,
                positions=out_pos,
                locations=out_loc,
            )

        operation()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        for index, value in enumerate((9, 13, 0)):
            changed = lengths.roll(index)
            if index == 2:
                changed[::4] = 0
            lens.copy_(changed)
            ptr.copy_(
                torch.cat(
                    [torch.zeros(1, dtype=torch.int64, device=device), lens.cumsum(0)]
                )
            )
            drafts.fill_(value)
            prediction.fill_(value)
            gate.fill_(index != 1)
            graph.replay()
            torch.cuda.synchronize()
            dense_ids[:, 1:] = value
            mask = torch.arange(6, device=device)[None, :] < lens[:, None]
            total = int(changed.sum())
            packed = torch.zeros_like(ids)
            packed[:total] = dense_ids[mask]
            torch.testing.assert_close(ids, packed, rtol=0, atol=0)
            padded_candidates = torch.where(mask, dense_ids, 0)
            dense_prediction = torch.where(mask, value, 0)
            raw_correct = (
                (dense_prediction[:, :-1] == padded_candidates[:, 1:])
                .int()
                .cumprod(1)
                .sum(1)
            )
            correct = torch.minimum(raw_correct, lens - 1)
            bonus = torch.where(lens > 0, value, 0)
            for actual, reference in zip(
                outputs, (correct, bonus, raw_correct - correct)
            ):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(small, padded_candidates, rtol=0, atol=0)
            committed = (
                mask
                & (torch.arange(6, device=device)[None, :] < (correct + 1)[:, None])
                & (index != 1)
            )
            expected_loc = torch.full_like(out_loc, -1)
            expected_loc[:total] = torch.where(
                committed, req[:, None] * 64 + dense_pos % 64, -1
            )[mask].int()
            torch.testing.assert_close(out_loc, expected_loc, rtol=0, atol=0)
            expected_pos = torch.zeros_like(out_pos)
            expected_pos[:total] = dense_pos[mask]
            torch.testing.assert_close(out_pos, expected_pos, rtol=0, atol=0)
    assert tested == 45


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
