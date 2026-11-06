"""Derived page/ring bounds and the independent P/D suffix-address contract."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.mem_cache.draft_window import (
    DraftWindowLayout,
    draft_window_size,
    draft_window_transfer_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_page_aligned_history_and_proposals_never_alias():
    # The oracle is an absolute interval, not the implementation's capacity formula.
    for page in (1, 4, 256):
        layout = DraftWindowLayout(2048, page, 4, 8)
        for prefix in (0, 1, 2047, 2048, 2049, 2303, 2304, 4097, 9999):
            slots = torch.tensor([0, 1, 3])
            reads, lengths, writes = layout.prepare(slots, torch.full((3,), prefix))
            assert not reads[0].any() and not writes[0].any()
            occupied = []
            start = max(0, prefix - 2048) // page * page
            for lane, slot in ((1, 1), (2, 3)):
                count = prefix - start + 8
                ids = reads[lane, :count].tolist()
                assert len(set(ids)) == count
                assert min(ids) >= page and max(ids) < layout.storage_tokens
                assert lengths[lane].item() == prefix - start
                assert ids[-8:] == writes[lane].tolist()
                for offset in range(0, count, page):
                    assert ids[offset] % page == 0
                    assert ids[offset : offset + page] == list(
                        range(ids[offset], ids[offset] + min(page, count - offset))
                    )
                occupied.append(set(ids))
            assert occupied[0].isdisjoint(occupied[1])


def test_transfer_restores_full_suffix_across_wrap_and_partial_commit():
    layout = DraftWindowLayout(7, 4, 3, 4)
    source = SimpleNamespace(draft_window_size=7, page_size=4)
    dest = SimpleNamespace(draft_window_layout=layout)
    # Different request ids and physical addresses on P/D. Target cache-hit
    # length deliberately has no role in selecting the draft suffix.
    table = torch.zeros((4, 101), dtype=torch.int32)
    table[3] = torch.arange(100, 201)
    backing = torch.full((layout.storage_tokens,), -1, dtype=torch.int64)
    for prefix in (19, 21, 41, 43):
        src = draft_window_transfer_rows(source, table, 3, prefix)
        dst = draft_window_transfer_rows(dest, table, 1, prefix)
        start = max(0, prefix - 7) // 4 * 4
        np.testing.assert_array_equal(src, np.arange(100 + start, 100 + prefix))
        backing[torch.from_numpy(dst).long()] = torch.arange(start, prefix)
        reads, lengths, writes = layout.prepare(
            torch.tensor([1]), torch.tensor([prefix])
        )
        torch.testing.assert_close(
            backing[reads[0, : lengths[0]].long()], torch.arange(start, prefix)
        )
        # Candidate writes followed by acceptance of only two rows; rejected
        # candidates must not enter the next iteration's historical prefix.
        backing[writes[0]] = -2
        backing[writes[0, :2]] = torch.tensor([prefix, prefix + 1])
        nxt_reads, nxt_lengths, _ = layout.prepare(
            torch.tensor([1]), torch.tensor([prefix + 2])
        )
        nxt_start = max(0, prefix + 2 - 7) // 4 * 4
        torch.testing.assert_close(
            backing[nxt_reads[0, : nxt_lengths[0]].long()],
            torch.arange(nxt_start, prefix + 2),
        )


def test_only_uniform_native_sliding_drafts_are_eligible():
    config = dict(
        num_hidden_layers=2, layer_types=["sliding_attention"] * 2, sliding_window=2048
    )
    assert draft_window_size(config) == 2048
    for change in (
        {"layer_types": ["sliding_attention", "full_attention"]},
        {"sliding_window": [2048, 4096]},
        {"sliding_window": 0},
        {"attention_sink_bias": True},
        {"dflash_config": {"attention_sink_bias": True}},
        {"use_sliding_window": False},
    ):
        with pytest.raises(ValueError):
            draft_window_size(config | change)

    # ModelConfig stores DFlash extras on hf_config; text-only fields can be
    # separate. Keep the guard consistent with the draft model's parser.
    for sink in (False, True):
        wrapped = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"attention_sink_bias": sink}),
            hf_text_config=SimpleNamespace(**config),
        )
        if sink:
            with pytest.raises(ValueError):
                draft_window_size(wrapped)
        else:
            assert draft_window_size(wrapped) == 2048


if __name__ == "__main__":
    import sys

    args = [arg for arg in sys.argv[1:] if arg != "-f"]
    sys.exit(pytest.main([__file__, "-v", "-x", *args]))
