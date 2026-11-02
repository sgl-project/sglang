"""Preserve allocator inputs and event ordering with NumPy CPU staging."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative import dflash_info_v2 as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


def run(monkeypatch, fast, carried, committed=(62, 130, 5), block=4):
    events = []
    monkeypatch.setattr(
        module, "_get_overlap_plan_stream", lambda device: (None, nullcontext())
    )
    monkeypatch.setattr(module, "is_pin_memory_available", lambda *args: False)
    monkeypatch.setattr(
        module,
        "get_global_server_args",
        lambda: SimpleNamespace(speculative_num_draft_tokens=block),
    )
    draft = module.DFlashDraftInputV2.create_idle_input("cpu")
    draft.verify_done = SimpleNamespace(
        synchronize=lambda: events.append("verify_done")
    )
    draft.cur_allocated_seq_lens_cpu = torch.tensor(carried, dtype=torch.int32)
    reqs = [
        SimpleNamespace(
            kv_committed_len=n,
            kv_allocated_len=max(0, n - 2),
            decode_batch_idx=5,
            sampling_params=SimpleNamespace(top_k=k),
        )
        for n, k in zip(committed, [1, 7, 3])
    ]
    batch = SimpleNamespace(
        device="cpu",
        reqs=reqs,
        batch_size=lambda: len(reqs),
        maybe_evict_swa=lambda: events.append("evict"),
        tree_cache=object(),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(len(reqs), 512, dtype=torch.int32)
        ),
        req_pool_indices=torch.arange(len(reqs), dtype=torch.int64),
    )
    calls = []
    monkeypatch.setattr(
        module, "get_last_loc", lambda *args: torch.zeros(len(reqs), dtype=torch.int64)
    )

    def allocate(tree, cur, cur_cpu, nxt, nxt_cpu, last, count):
        events.append("allocate")
        calls.append(
            (
                "allocate",
                cur.tolist(),
                cur_cpu.tolist(),
                nxt.tolist(),
                nxt_cpu.tolist(),
                count,
            )
        )
        return torch.arange(count, dtype=torch.int64) + 1000

    def assign(indices, table, cur, nxt, locations, bs):
        events.append("assign")
        calls.append(("assign", cur.tolist(), nxt.tolist(), locations.tolist(), bs))

    monkeypatch.setattr(module, "alloc_paged_token_slots_extend", allocate)
    monkeypatch.setattr(module, "assign_req_to_token_pool_func", assign)
    with module.envs.SGLANG_DFLASH_FAST_CPU_STAGING.override(fast):
        draft.prepare_for_decode(batch)
    return {
        "calls": calls,
        "events": events,
        "seq": batch.seq_lens_cpu.tolist(),
        "sum": batch.seq_lens_sum,
        "planning": draft.planning_seq_lens_cpu.tolist(),
        "planning_sum": draft.planning_seq_lens_sum,
        "reserved": draft.reserved_seq_lens_cpu.tolist(),
        "reserved_sum": draft.reserved_seq_lens_sum,
        "reqs": [
            (r.kv_committed_len, r.kv_allocated_len, r.decode_batch_idx) for r in reqs
        ],
        "topk": (draft.max_top_k, draft.uniform_top_k_value),
    }


@pytest.mark.parametrize(
    "carried,block",
    [([64, 200, 9], 4), ([70, 200, 20], 8), ([300, 300, 300], 4)],
)
def test_cpu_staging_preserves_allocator_inputs_and_clocks(
    monkeypatch, carried, block
):
    reference = run(monkeypatch, False, carried, block=block)
    candidate = run(monkeypatch, True, carried, block=block)
    assert candidate == reference
    assert candidate["events"][:2] == ["verify_done", "evict"]
    assert candidate["seq"] == [62, 130, 5]
    assert candidate["planning"] == [n + block for n in [62, 130, 5]]
    assert candidate["reserved"] == [
        max(c, n + 2 * block) for c, n in zip(carried, [62, 130, 5])
    ]
    assert [r[0] for r in candidate["reqs"]] == [62, 130, 5]
    assert all(r[2] == 6 for r in candidate["reqs"])
    assert candidate["topk"] == (7, None)
    if carried == [300, 300, 300]:
        assert not candidate["calls"]
