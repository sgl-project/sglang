"""Phase 2: P0-9 Runtime participant-set collective proof.

8-rank CPU/Gloo test modeling TP=4, PP=2 production topology.
"""

from __future__ import annotations

import datetime
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

WORLD_SIZE = 8
TP_SIZE = 4
PP_SIZE = 2
DRAFT_STAGE = 1
draft_ranks_list = list(range(DRAFT_STAGE * TP_SIZE, (DRAFT_STAGE + 1) * TP_SIZE))


def compute_rank_mapping(world_size, tp_size, pp_size):
    assert world_size == tp_size * pp_size
    mapping = {}
    for gr in range(world_size):
        pp_rank = gr // tp_size
        tp_rank = gr % tp_size
        tp_group_ranks = list(range(pp_rank * tp_size, (pp_rank + 1) * tp_size))
        pp_lane = tp_rank
        pp_peer_next = gr + tp_size if pp_rank < pp_size - 1 else None
        pp_peer_prev = gr - tp_size if pp_rank > 0 else None
        is_draft = (pp_rank == DRAFT_STAGE)
        mapping[gr] = {
            "global_rank": gr, "pp_rank": pp_rank, "tp_rank": tp_rank,
            "tp_group_ranks": tp_group_ranks, "pp_lane": pp_lane,
            "pp_peer_next": pp_peer_next, "pp_peer_prev": pp_peer_prev,
            "is_draft_participant": is_draft, "draft_tp_rank": tp_rank if is_draft else None,
        }
    return mapping


@dataclass
class TraceEvent:
    global_rank: int
    pp_rank: int
    tp_rank: int
    operation: str
    group_members: List[int]
    round: int
    sequence_number: int
    timestamp: float = field(default_factory=time.time)


def worker_fn(rank, world_size, backend, num_rounds, port, output_dir, result_queue):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=300))
        mapping = compute_rank_mapping(world_size, TP_SIZE, PP_SIZE)
        m = mapping[rank]
        traces: List[Dict] = []
        seq = 0

        tp_group = dist.new_group(ranks=m["tp_group_ranks"])
        draft_group = dist.new_group(ranks=draft_ranks_list)

        # Initial TP all-reduce
        t = torch.tensor([float(rank)], dtype=torch.float64)
        dist.all_reduce(t, group=tp_group)
        assert abs(t.item() - sum(m["tp_group_ranks"])) < 1e-6
        traces.append(TraceEvent(rank, m["pp_rank"], m["tp_rank"], "target_tp_all_reduce", m["tp_group_ranks"], -1, seq).__dict__)
        seq += 1

        for rnd in range(num_rounds):
            # 1. Target TP all-reduce
            t = torch.tensor([float(rank * 100 + rnd)], dtype=torch.float64)
            dist.all_reduce(t, group=tp_group)
            traces.append(TraceEvent(rank, m["pp_rank"], m["tp_rank"], "target_tp_all_reduce", m["tp_group_ranks"], rnd, seq).__dict__)
            seq += 1

            # 2. PP0→PP1 send/recv — post both isend and irecv, then wait
            # PP0 sends to PP1, PP1 receives from PP0
            # Use non-blocking ops to avoid deadlock
            if m["pp_rank"] == 0:
                send_buf = torch.tensor([float(rank * 1000 + rnd)], dtype=torch.float64)
                w = dist.isend(send_buf, dst=m["pp_peer_next"])
                w.wait()
                traces.append(TraceEvent(rank, m["pp_rank"], m["tp_rank"], "pp_send", [rank, m["pp_peer_next"]], rnd, seq).__dict__)
                seq += 1
            else:
                recv_buf = torch.tensor([0.0], dtype=torch.float64)
                w = dist.irecv(recv_buf, src=m["pp_peer_prev"])
                w.wait()
                assert abs(recv_buf.item() - m["pp_peer_prev"] * 1000 - rnd) < 1e-6
                traces.append(TraceEvent(rank, m["pp_rank"], m["tp_rank"], "pp_recv", [m["pp_peer_prev"], rank], rnd, seq).__dict__)
                seq += 1

            # 3. Draft collective (PP1 only) — must be AFTER PP send/recv completes
            # so PP0 ranks don't block
            if m["is_draft_participant"]:
                d = torch.tensor([float(rank * 10 + rnd)], dtype=torch.float64)
                dist.all_reduce(d, group=draft_group)
                expected = sum(draft_ranks_list) * 10 + rnd * len(draft_ranks_list)
                assert abs(d.item() - expected) < 1e-6
                traces.append(TraceEvent(rank, m["pp_rank"], m["tp_rank"], "draft_tp_all_reduce", draft_ranks_list, rnd, seq).__dict__)
                seq += 1
            else:
                traces.append(TraceEvent(rank, m["pp_rank"], m["tp_rank"], "draft_skipped_pp0", [], rnd, seq).__dict__)
                seq += 1

            # 4. PP1→PP0 relay (reverse direction)
            if m["pp_rank"] == 1:
                relay_buf = torch.tensor([float(rank * 2000 + rnd)], dtype=torch.float64)
                w = dist.isend(relay_buf, dst=m["pp_peer_prev"])
                w.wait()
                traces.append(TraceEvent(rank, m["pp_rank"], m["tp_rank"], "pp_relay_send", [rank, m["pp_peer_prev"]], rnd, seq).__dict__)
                seq += 1
            else:
                relay_recv = torch.tensor([0.0], dtype=torch.float64)
                w = dist.irecv(relay_recv, src=m["pp_peer_next"])
                w.wait()
                expected = m["pp_peer_next"] * 2000 + rnd
                assert abs(relay_recv.item() - expected) < 1e-6
                traces.append(TraceEvent(rank, m["pp_rank"], m["tp_rank"], "pp_relay_recv", [m["pp_peer_next"], rank], rnd, seq).__dict__)
                seq += 1

        assert m["pp_rank"] == rank // TP_SIZE
        assert m["tp_rank"] == rank % TP_SIZE
        assert m["pp_lane"] == m["tp_rank"]

        dist.barrier()
        trace_file = os.path.join(output_dir, f"trace_rank{rank}.json")
        with open(trace_file, "w") as f:
            json.dump(traces, f, indent=2)
        result_queue.put({"rank": rank, "status": "pass", "num_traces": len(traces)})
        dist.destroy_process_group()
    except Exception as e:
        try:
            result_queue.put({"rank": rank, "status": "fail", "error": str(e), "traceback": traceback.format_exc()})
        except:
            pass
        try:
            dist.destroy_process_group()
        except:
            pass


def verify_traces(output_dir, num_rounds):
    all_traces = []
    for rank in range(WORLD_SIZE):
        f = os.path.join(output_dir, f"trace_rank{rank}.json")
        if os.path.exists(f):
            with open(f) as fh:
                all_traces.extend(json.load(fh))
    errors = []
    for op in [t for t in all_traces if t["operation"] == "target_tp_all_reduce"]:
        r = op["global_rank"]
        exp = list(range((r // TP_SIZE) * TP_SIZE, ((r // TP_SIZE) + 1) * TP_SIZE))
        if sorted(op["group_members"]) != sorted(exp):
            errors.append(f"Rank {r}: target_tp wrong group")
    for op in [t for t in all_traces if t["operation"] == "draft_tp_all_reduce"]:
        if op["pp_rank"] != DRAFT_STAGE:
            errors.append(f"Rank {op['global_rank']}: draft on PP{op['pp_rank']}")
        if sorted(op["group_members"]) != sorted(draft_ranks_list):
            errors.append(f"Rank {op['global_rank']}: draft group mismatch")
    for op in [t for t in all_traces if t["operation"] in ("pp_send", "pp_relay_send")]:
        s, d = op["group_members"]
        if s % TP_SIZE != d % TP_SIZE:
            errors.append(f"PP lane mismatch: {s}->{d}")
    pp0_draft = [t for t in all_traces if t["operation"] == "draft_tp_all_reduce" and t["pp_rank"] == 0]
    if pp0_draft:
        errors.append(f"PP0 entered draft: {len(pp0_draft)} events")
    return (len(errors) == 0, "\n".join(errors[:20]) if errors else f"All {len(all_traces)} traces verified")


def run_test(num_rounds, label):
    output_dir = f"/tmp/glm52-eagle3-pp-validation-test-{label}"
    os.makedirs(output_dir, exist_ok=True)
    port = 29583 + abs(hash(label)) % 1000
    result_queue = mp.Queue()
    procs = []
    for rank in range(WORLD_SIZE):
        p = mp.Process(target=worker_fn, args=(rank, WORLD_SIZE, "gloo", num_rounds, port, output_dir, result_queue))
        p.start()
        procs.append(p)
    results = []
    for _ in range(WORLD_SIZE):
        try:
            results.append(result_queue.get(timeout=600))
        except:
            results.append({"rank": -1, "status": "fail", "error": "timeout"})
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    passed = all(r["status"] == "pass" for r in results)
    if not passed:
        failures = [r for r in results if r["status"] != "pass"]
        for f in failures:
            print(f"  Rank {f.get('rank')}: {f.get('error', 'unknown')[:200]}")
        return False, f"{len(failures)} ranks failed"
    ok, msg = verify_traces(output_dir, num_rounds)
    return ok, msg


if __name__ == "__main__":
    print("=== Phase 2: P0-9 Participant-Set Collective Proof (8-rank Gloo) ===")

    mapping = compute_rank_mapping(8, 4, 2)
    for r in range(4):
        assert not mapping[r]["is_draft_participant"]
    for r in range(4, 8):
        assert mapping[r]["is_draft_participant"]
    for lane in range(4):
        assert mapping[lane]["pp_peer_next"] == lane + 4
        assert mapping[lane + 4]["pp_peer_prev"] == lane
    print("  Rank mapping validation PASSED")

    for rounds, label in [(1, "1round"), (2, "2rounds"), (100, "100rounds"), (1000, "1000rounds")]:
        print(f"\nRunning {label} test...")
        ok, msg = run_test(rounds, label)
        if ok:
            print(f"  {label} test PASSED: {msg}")
        else:
            print(f"  {label} test FAILED: {msg}")
            sys.exit(1)

    print("\n=== All Phase 2 tests PASSED ===")
