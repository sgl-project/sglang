"""GPU checks and isolated timings for the native SGLang FlyDSL adapter.

This checks the kernel adapter, not model accuracy or serving throughput.
Run with the pinned AITER/FlyDSL stack on gfx950.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("m3_flydsl_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def caches(batch, max_length, page, heads=1, dim=128):
    pages = (max_length + page - 1) // page
    count = batch * pages + 1
    # Reserve page 0 and shuffle the remaining physical pages.
    table = (torch.randperm(count - 1, device="cuda") + 1).view(batch, pages)
    table = table.to(torch.int32)
    positions = torch.arange(pages * page, device="cuda")
    slots = table[:, positions // page].long() * page + positions % page
    k = (torch.randn(count, page, heads, dim, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn(count, page, heads, dim, device="cuda") * 0.3).to(k.dtype)
    shuffled_k = (
        k.view(count, page, heads, dim // 16, 16).permute(0, 2, 3, 1, 4).contiguous()
    )
    shuffled_v = (
        v.view(count, page // 16, 16, heads, dim).permute(0, 3, 1, 4, 2).contiguous()
    )
    return k, v, shuffled_k, shuffled_v, table, slots


def reference(q, k, v, slots, requests, lengths, ks, vs, topk=None, block=128):
    heads = k.shape[2]
    group = q.shape[1] // heads
    dim = q.shape[-1]
    result = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    k = k.flatten(0, 1).float() * ks
    v = v.flatten(0, 1).float() * vs
    for row, (req, length) in enumerate(zip(requests.tolist(), lengths.tolist())):
        for head in range(heads):
            if topk is None:
                positions = list(range(length))
            else:
                positions = [
                    p
                    for selected in topk[head, row].tolist()
                    if selected >= 0
                    for p in range(
                        selected * block, min((selected + 1) * block, length)
                    )
                ]
            if not positions:
                continue
            ids = slots[req, positions]
            query = q[row, head * group : (head + 1) * group].float()
            probs = (query @ k[ids, head].T * dim**-0.5).softmax(-1)
            result[row, head * group : (head + 1) * group] = probs @ v[ids, head]
    return result


def check(name, actual, expected, results, *, fail=False):
    actual = actual.float()
    delta = actual - expected
    record = {
        "case": name,
        "max_abs_error": delta.abs().max().item(),
        "relative_l2_error": (delta.norm() / expected.norm().clamp_min(1e-8)).item(),
        "finite": bool(torch.isfinite(actual).all()),
        "atol": 0.01,
        "rtol": 0.08,
    }
    record["passed"] = bool(torch.isclose(actual, expected, atol=0.01, rtol=0.08).all())
    results.append(record)
    print(json.dumps(record), flush=True)
    if fail:
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.08)


def independent_sparse_call(adapter, q, sk, sv, topk, slots, requests, lengths, ks, vs):
    """CPU-built page tables independently check the GPU metadata adapter."""
    batch, q_heads, dim = q.shape
    heads, page = sk.shape[1], sk.shape[3]
    width = topk.shape[-1] * (128 // page)
    tables = torch.zeros((batch * heads, width), dtype=torch.int32)
    selected_lengths = torch.zeros(batch * heads, dtype=torch.int32)
    host_slots, host_topk = slots.cpu(), topk.cpu()
    for b, (req, length) in enumerate(zip(requests.tolist(), lengths.tolist())):
        for h in range(heads):
            full, partial = [], []
            count = 0
            for block in host_topk[h, b].tolist():
                if block < 0:
                    continue
                for pos in range(block * 128, min((block + 1) * 128, length), page):
                    entry = int(host_slots[req, pos]) // page * heads + h
                    (full if pos + page <= length else partial).append(entry)
                    count += min(page, length - pos)
            entries = full + partial
            if entries:
                tables[b * heads + h, : len(entries)] = torch.tensor(entries)
            selected_lengths[b * heads + h] = count
    output = torch.empty_like(q)
    adapter.decode(
        output.view(batch * heads, q_heads // heads, dim),
        q.reshape(batch * heads, q_heads // heads, dim),
        sk.view(-1, 1, dim // 16, page, 16),
        sv.view(-1, 1, page // 16, dim, 16),
        selected_lengths.to(q.device),
        tables.to(q.device),
        dim**-0.5,
        ks,
        vs,
    )
    return output


def correctness(adapter, results):
    torch.manual_seed(20260927)
    for page in (16, 64, 128):
        for heads in (1, 2):
            lengths = torch.tensor(
                [0, 1, 17, 257, 513, 1025], device="cuda", dtype=torch.int64
            )
            batch = len(lengths)
            k, v, sk, sv, table, slots = caches(batch, 1152, page, heads)
            q = torch.randn(batch, heads * 16, 128, device="cuda", dtype=torch.bfloat16)
            out = torch.empty_like(q)
            ks = torch.tensor([0.7], dtype=torch.float32, device="cuda")
            vs = torch.tensor([1.3], dtype=torch.float32, device="cuda")
            requests = torch.arange(batch, device="cuda", dtype=torch.int64)
            expected = reference(q, k, v, slots, requests, lengths, ks, vs)
            label = f"page{page}-heads{heads}"
            adapter.decode(out, q, sk, sv, lengths, table, 128**-0.5, ks, vs)
            check(f"dense-static-{label}", out, expected, results)

            planner = adapter.DenseDecodePlanner(heads)
            fb = SimpleNamespace(
                seq_lens=lengths,
                batch_size=batch,
                forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True),
            )
            planner.prepare(fb, in_capture=True)
            plan = planner.active_plan

            def planned():
                planner.refresh(fb)
                adapter.decode(
                    out,
                    q,
                    sk,
                    sv,
                    planner.active_lengths,
                    table,
                    128**-0.5,
                    ks,
                    vs,
                    plan,
                )

            for _ in range(3):
                planned()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                planned()
            graph.replay()
            check(f"dense-planned-graph-{label}", out, expected, results)
            lengths.copy_(
                torch.tensor(
                    [1025, 513, 257, 17, 1, 0], device="cuda", dtype=torch.int32
                )
            )
            planner.prepare(fb)
            assert planner.active_plan is plan
            graph.replay()
            expected = reference(q, k, v, slots, requests, lengths, ks, vs)
            check(f"dense-graph-changed-lengths-{label}", out, expected, results)

            static_state = adapter.DenseDecodePlanner(heads, enable_plan=False)
            static_state.prepare(fb, in_capture=True)
            assert static_state.active_plan is None

            def static_graph_call():
                static_state.refresh(fb)
                adapter.decode(
                    out,
                    q,
                    sk,
                    sv,
                    static_state.active_lengths,
                    table,
                    128**-0.5,
                    ks,
                    vs,
                )

            static_graph_call()
            static_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(static_graph):
                static_graph_call()
            lengths.copy_(torch.tensor([0, 1, 17, 257, 513, 1025], device="cuda"))
            static_state.prepare(fb)
            static_graph.replay()
            expected = reference(q, k, v, slots, requests, lengths, ks, vs)
            check(f"dense-static-graph-int64-lengths-{label}", out, expected, results)

            # Noncontiguous top-k rows, different selections per KV head, and
            # the partial final block listed before full blocks.
            topk = torch.full((heads, batch, 8), -1, dtype=torch.int32, device="cuda")[
                :, :, ::2
            ]
            for h in range(heads):
                for b, length in enumerate(lengths.tolist()):
                    blocks = (length + 127) // 128
                    chosen = list(dict.fromkeys([blocks - 1, h, 0])) if blocks else []
                    chosen = [x for x in chosen if 0 <= x < blocks]
                    if chosen:
                        topk[h, b, : len(chosen)] = torch.tensor(
                            chosen, device="cuda", dtype=torch.int32
                        )
            sparse = adapter.sparse_decode(
                q, sk, sv, topk, slots, requests, lengths, 128, None, ks, vs
            )
            independent = independent_sparse_call(
                adapter, q, sk, sv, topk, slots, requests, lengths, ks, vs
            )
            torch.testing.assert_close(sparse, independent, atol=0, rtol=0)
            results.append({"case": f"sparse-metadata-exact-{label}", "passed": True})
            expected = reference(q, k, v, slots, requests, lengths, ks, vs, topk)
            check(f"sparse-{label}", sparse, expected, results)

            prefix = torch.tensor([0, 17, 257], device="cuda", dtype=torch.int32)
            cu_q = torch.tensor([0, 2, 5, 10], device="cuda", dtype=torch.int32)
            req = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int64)
            pq = torch.randn(10, heads * 16, 128, device="cuda", dtype=q.dtype)
            pt = torch.full((heads, 10, 4), -1, dtype=torch.int32, device="cuda")
            pt[:, :, 0] = 0
            pt[:, 5:, 1] = 2
            row_requests = req.repeat_interleave(torch.tensor([2, 3, 5], device="cuda"))
            row_lengths = torch.tensor(
                [1, 2, 18, 19, 20, 258, 259, 260, 261, 262],
                device="cuda",
                dtype=torch.int32,
            )
            po = adapter.sparse_prefill(
                pq, sk, sv, pt, slots, req, cu_q, prefix, 5, 128, None, ks, vs
            )
            independent = independent_sparse_call(
                adapter, pq, sk, sv, pt, slots, row_requests, row_lengths, ks, vs
            )
            torch.testing.assert_close(po, independent, atol=0, rtol=0)
            results.append({"case": f"prefill-metadata-exact-{label}", "passed": True})
            expected = reference(pq, k, v, slots, row_requests, row_lengths, ks, vs, pt)
            check(f"sparse-prefill-prefix-{label}", po, expected, results, fail=False)
    return results


def graph_latency(fn, warmup=5, iterations=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    samples = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {"median_us": statistics.median(samples), "rounds_us": samples}


def benchmark(adapter):
    from aiter.ops.triton.gluon.pa_decode_gluon import (
        get_recommended_splits,
        pa_decode_gluon,
    )

    records = []
    # These synthetic kernel workloads are not end-to-end serving traces.
    for batch in (1, 2, 10, 15, 20):
        for distribution in ("uniform", "mixed"):
            context = (
                [32768] * batch
                if distribution == "uniform"
                else [32768] + [1024] * (batch - 1)
            )
            k, v, sk, sv, table, slots = caches(batch, max(context), 16)
            del k, v, slots
            lengths = torch.tensor(context, device="cuda", dtype=torch.int32)
            q = torch.randn(batch, 16, 128, device="cuda", dtype=torch.bfloat16)
            out = torch.empty_like(q)
            scale = torch.ones(1, device="cuda", dtype=torch.float32)
            _, plan_fn, _ = adapter.load_flydsl()
            plan = plan_fn(lengths, 1)
            partitions = get_recommended_splits(batch, 1)
            es = torch.empty(
                (batch, 1, partitions, 16), device="cuda", dtype=torch.float32
            )
            ml = torch.empty_like(es)
            tmp = torch.empty((*es.shape, 128), device="cuda", dtype=q.dtype)

            def gluon():
                pa_decode_gluon(
                    output=out,
                    query=q,
                    key_cache=sk,
                    value_cache=sv,
                    context_lengths=lengths,
                    block_tables=table,
                    softmax_scale=128**-0.5,
                    query_length=1,
                    max_context_partition_num=partitions,
                    context_partition_size=256,
                    compute_type=sk.dtype,
                    key_scale=scale,
                    value_scale=scale,
                    exp_sums=es,
                    max_logits=ml,
                    temporary_output=tmp,
                    ps=True,
                )

            def static():
                adapter.decode(out, q, sk, sv, lengths, table, 128**-0.5, scale, scale)

            def planned():
                plan_fn(lengths, 1, plan=plan)
                adapter.decode(
                    out, q, sk, sv, lengths, table, 128**-0.5, scale, scale, plan
                )

            modes = {"gluon": gluon, "flydsl_static": static, "flydsl_planned": planned}
            timing = {name: graph_latency(fn) for name, fn in modes.items()}
            record = {
                "batch": batch,
                "distribution": distribution,
                "lengths": context,
                "timings": timing,
            }
            base = timing["gluon"]["median_us"]
            record["planned_latency_reduction_pct"] = 100 * (
                1 - timing["flydsl_planned"]["median_us"] / base
            )
            record["scope"] = (
                "isolated dense attention kernel; planner cost included; not serving throughput"
            )
            print(json.dumps(record), flush=True)
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if (
        not torch.cuda.is_available()
        or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    ):
        raise RuntimeError("This validation requires a gfx950 GPU")
    adapter = load_adapter(args.adapter)
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "checks": [],
    }
    try:
        correctness(adapter, report["checks"])
        if args.benchmark and all(x["passed"] for x in report["checks"]):
            report["kernel_benchmarks"] = benchmark(adapter)
        report["passed"] = all(x["passed"] for x in report["checks"])
    except Exception as exc:
        report["passed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
