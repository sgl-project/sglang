"""NPU Graph replay test for the pre-LIM launch (functional only; no timer)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_npu")

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.lim import (
    prepare_lim_rows,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_lim_input_metadata import (
    fused_prepare_lim_metadata,
)


@pytest.mark.parametrize("active_rows", [16, 9])
@pytest.mark.parametrize("capacity", [16384, 12288, 8192])
@pytest.mark.parametrize(
    "slot_dtype,length_dtype", [(torch.int32, torch.int32), (torch.int64, torch.int64)]
)
@pytest.mark.parametrize("src_cols,dst_cols", [(32, 512), (500, 513), (513, 513)])
def test_npu_graph_replay_matches_torch_reference(
    active_rows, capacity, slot_dtype, length_dtype, src_cols, dst_cols
):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("This test requires an available Ascend NPU")
    device = torch.device("npu:0")
    batch, rows = 16, 64
    slots = torch.zeros(batch, dtype=slot_dtype, device=device)
    lengths = torch.zeros(batch, dtype=length_dtype, device=device)
    last_prefix = torch.full((rows + 1,), -1, dtype=torch.int32, device=device)
    src_table = torch.full((batch, src_cols), -777777, dtype=torch.int32, device=device)
    out_table = torch.empty((batch, dst_cols), dtype=torch.int32, device=device)
    safe = torch.empty(batch, dtype=torch.int32, device=device)
    out_len = torch.empty_like(safe)
    prefix = torch.empty_like(safe)
    sparse = torch.empty(batch, dtype=torch.bool, device=device)
    valid = torch.empty_like(sparse)
    state = torch.empty_like(safe)
    next_prefix = torch.empty_like(safe)
    cache_size = torch.empty_like(safe)

    def launch():
        fused_prepare_lim_metadata(
            seq_lens=lengths,
            request_slots=slots,
            last_prefix=last_prefix,
            block_table=src_table,
            out_block_table=out_table,
            safe_slots=safe,
            lengths=out_len,
            prefixes=prefix,
            sparse_rows=sparse,
            valid_rows=valid,
            request_states=state,
            next_prefixes=next_prefix,
            cache_sizes=cache_size,
            capacity=capacity,
            request_rows=rows,
            full_visible=True,
        )

    # Compile and initialize outside capture; a successful import is not a
    # successful graph test.
    for _ in range(3):
        launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    for replay in range(5):
        # Different values on every replay exercise graph input addresses and
        # request-slot reuse. PAD rows retain slot/length zero; source PAD table
        # entries are deliberately dirty and must be copied verbatim.
        slot_values = [i + 1 + replay for i in range(active_rows)] + [0] * (
            batch - active_rows
        )
        length_values = [
            capacity + ((i + replay) % 3) * 128 for i in range(active_rows)
        ] + [0] * (batch - active_rows)
        slots.copy_(torch.tensor(slot_values, dtype=slot_dtype, device=device))
        lengths.copy_(torch.tensor(length_values, dtype=length_dtype, device=device))
        if replay == 3:
            # Shrink a reused request, enter the nonsparse path, and reject
            # negative/out-of-range request slots without reading outside state.
            lengths[0] = capacity - 1
            slots[1] = -1
            slots[2] = rows
            slots[3] = rows + 1
        elif replay == 4:
            lengths[0] = 0
        elif replay == 2 and src_cols == 500:
            # 500 pages cover exactly the native fixture's 64,000-token
            # context; destination columns 500..512 are stride padding.
            lengths[0] = src_cols * 128
        src_values = (
            torch.arange(batch * src_cols, dtype=torch.int32, device=device).view(
                batch, src_cols
            )
            + replay * 10000
        )
        src_table.copy_(src_values)
        # Seed state independently of the current slot inputs, which may be
        # invalid in the boundary replay above.
        for i, slot in enumerate(slot_values[:active_rows]):
            last_prefix[slot] = length_values[i] - 128 if i % 2 == 0 else -1
        if replay == 3:
            last_prefix[slot_values[4]] = length_values[4] + 128
        (
            old_safe,
            old_lengths,
            old_prefix,
            old_sparse,
            old_valid,
            old_state,
            old_next,
        ) = prepare_lim_rows(
            lengths, slots, last_prefix, capacity, rows, full_visible=True
        )
        old_table = torch.zeros_like(out_table)
        old_table[:, :src_cols].copy_(src_table)
        graph.replay()
        torch.npu.synchronize()
        # Preserve LIMDecodeAdapter's existing mixed/short CPU fallback rule:
        # a positive length below C forces every next-prefix entry to -1.
        cpu_lengths = lengths.cpu().tolist()
        mixed_short = any(0 < int(length) < capacity for length in cpu_lengths)
        if mixed_short:
            old_next.fill_(-1)
            next_prefix.fill_(-1)

        for actual, old in (
            (safe, old_safe),
            (out_len, old_lengths),
            (prefix, old_prefix),
            (sparse, old_sparse),
            (valid, old_valid),
            (state, old_state),
            (next_prefix, old_next),
            (cache_size, old_sparse.to(torch.int32) * capacity),
            (out_table, old_table),
        ):
            torch.testing.assert_close(actual.cpu(), old.cpu())

        host_slots, host_lengths, host_src = (
            slots.cpu().tolist(),
            lengths.cpu().tolist(),
            src_table.cpu(),
        )
        expected_safe, expected_len, expected_prefix, expected_sparse = [], [], [], []
        expected_valid, expected_state, expected_next, expected_cache = [], [], [], []
        for slot, length in zip(host_slots, host_lengths):
            ok = slot > 0 and slot < rows and length > 0
            ln = int(length) if ok else 0
            pre = ln
            sp = ok and pre >= capacity
            old = int(last_prefix[slot if ok else rows].item())
            warm = sp and old >= 0 and pre >= old
            expected_safe.append(slot if ok else rows)
            expected_len.append(ln)
            expected_prefix.append(pre)
            expected_sparse.append(sp)
            expected_valid.append(ok)
            expected_state.append(-1 if warm else (-2 if sp else -3))
            expected_next.append(pre if sp and not mixed_short else -1)
            expected_cache.append(capacity if sp else 0)
        pairs = [
            (safe, expected_safe),
            (out_len, expected_len),
            (prefix, expected_prefix),
            (sparse, expected_sparse),
            (valid, expected_valid),
            (state, expected_state),
            (next_prefix, expected_next),
            (cache_size, expected_cache),
        ]
        for actual, expected in pairs:
            torch.testing.assert_close(
                actual.cpu(), torch.tensor(expected, dtype=actual.dtype)
            )
        actual_table = out_table.cpu()
        assert torch.equal(actual_table[:, :src_cols], host_src)
        if src_cols >= 128:
            # Check values on both sides of the 128-page tile boundary and
            # at the final source page, including the non-tile-aligned tail.
            for boundary in (127, 128, src_cols - 1):
                assert torch.equal(actual_table[:, boundary], host_src[:, boundary])
        assert torch.count_nonzero(actual_table[:, src_cols:]) == 0
