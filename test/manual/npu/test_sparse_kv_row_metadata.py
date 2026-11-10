"""Exact row-plan parity, replay and paired latency checks for Ascend Triton.

Run with --device npu during the reserved device window. --device cpu with
TRITON_INTERPRET=1 checks kernel arithmetic without opening an NPU device.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
PKG = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload"
SOURCE = ROOT / "python" / Path(*PKG.split("."))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)


def mode(name):
    return SimpleNamespace(
        is_target_verify=lambda: name == "verify",
        is_draft_extend_v2=lambda: name == "draft",
        is_decode=lambda: name == "decode",
    )


def mtp_reference(batch, q, request_rows, context):
    b = batch.req_pool_indices.numel()
    slots = batch.req_pool_indices.long()[:, None].expand(b, q)
    pos = batch.positions[: b * q].long().view(b, q)
    valid = (
        (slots > 0)
        & (slots < request_rows)
        & (batch.seq_lens[:, None] > 0)
        & (pos >= 0)
        & (pos < context)
    )
    if batch.forward_mode.is_draft_extend_v2():
        valid &= (
            torch.arange(q, device=pos.device)[None]
            < batch.spec_info.num_accept_tokens[:, None]
        )
    if batch.out_cache_loc is not None:
        valid &= batch.out_cache_loc[: b * q].view(b, q) >= 0
    return slots, pos, valid


def copy_reference(sources, destinations, source_rows, destination_rows, valid):
    safe_s = torch.where(valid, source_rows, 0).long()
    safe_d = torch.where(valid, destination_rows, 0).long()
    src, dst, lengths = [], [], []
    for s, d in zip(sources, destinations):
        width = s.shape[-1] * s.element_size()
        src.append(safe_s * width + s.data_ptr())
        dst.append(safe_d * width + d.data_ptr())
        lengths.append(valid.int() * width)
    return (
        torch.cat(src),
        torch.cat(dst),
        torch.cat(lengths),
        torch.full(
            (1,), len(sources) * valid.numel(), device=valid.device, dtype=torch.int32
        ),
    )


def mapping_reference(
    plan, batch, valid, capacity, slots, req, lens, prefixes, extends
):
    dst, eligible, offset = [], [], 0
    for r, length, p, e, width in zip(req, lens, prefixes, extends, slots):
        for col in range(width):
            live = r > 0 and length > 0 and col < e
            dst.append(offset + p + col if live else 0)
            eligible.append(live)
        if r >= 0 and length > 0:
            offset += length
    device = plan.source.device
    dst = torch.tensor(dst, device=device)
    loc = batch.out_cache_loc.long()
    mask = torch.tensor(eligible, device=device) & valid & (loc >= 0) & (loc < capacity)
    safe = loc.clamp(0, capacity - 1)
    counts = torch.zeros(capacity, device=device, dtype=torch.int32)
    counts.scatter_add_(0, safe, mask.int())
    mask &= (counts[safe] == 1) & (plan.source[dst] == loc)
    occupied = torch.zeros(plan.total_tokens, device=device, dtype=torch.int32)
    occupied.scatter_add_(0, dst, mask.int())
    return torch.arange(loc.numel(), device=device), dst, mask, occupied == 0


def measure(reference, candidate, device, repeats):
    def sync():
        if device == "npu":
            torch.npu.synchronize()

    for _ in range(5):
        reference()
        candidate()
    sync()
    results = {"reference_us": [], "candidate_us": []}
    for r in range(5):
        pairs = [("reference_us", reference), ("candidate_us", candidate)]
        if r % 2:
            pairs.reverse()
        for key, fn in pairs:
            sync()
            start = time.perf_counter_ns()
            for _ in range(repeats):
                fn()
            sync()
            results[key].append((time.perf_counter_ns() - start) / repeats / 1000)
    results["speedup"] = statistics.median(results["reference_us"]) / statistics.median(
        results["candidate_us"]
    )
    if device == "npu":
        old_graph, old_outputs = replay(reference, reference)
        new_graph, new_outputs = replay(candidate, reference)
        graph_results = {"reference_graph_us": [], "candidate_graph_us": []}
        for r in range(5):
            pairs = [
                ("reference_graph_us", old_graph),
                ("candidate_graph_us", new_graph),
            ]
            if r % 2:
                pairs.reverse()
            for key, graph in pairs:
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    graph.replay()
                end.record()
                end.synchronize()
                graph_results[key].append(start.elapsed_time(end) * 1000 / repeats)
        exact(old_outputs, new_outputs)
        results.update(graph_results)
        results["graph_speedup"] = statistics.median(
            results["reference_graph_us"]
        ) / statistics.median(results["candidate_graph_us"])
    return results


def replay(fn, reference):
    # Change tensor contents after capture; replay must read the new values.
    warmup = torch.npu.Stream()
    warmup.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(warmup):
        for _ in range(3):
            fn()
    torch.npu.current_stream().wait_stream(warmup)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = fn()
    if outputs is None:
        outputs = ()
    elif hasattr(outputs, "source"):
        outputs = (outputs.source, outputs.destination, outputs.valid)
    graph.replay()
    torch.npu.synchronize()
    expected = reference()
    if expected is None:
        expected = ()
    elif hasattr(expected, "source"):
        expected = (expected.source, expected.destination, expected.valid)
    exact(outputs, expected)
    return graph, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--mf", action="store_true")
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
    kernels = load(PKG + ".row_metadata", SOURCE / "row_metadata.py")
    results = {
        "device": args.device,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__ if args.device == "npu" else None,
        "triton": kernels.triton.__version__,
        "row_metadata_source": str(SOURCE / "row_metadata.py"),
        "row_metadata_sha256": hashlib.sha256(
            (SOURCE / "row_metadata.py").read_bytes()
        ).hexdigest(),
        "reference_source": str(args.reference_source),
        "cases": [],
        "benchmarks": {},
    }
    torch.manual_seed(1703)
    d = args.device
    # Signed, int32/int64, invalid requests/positions, accept counts, suffix TP padding.
    for dtype in (torch.int32, torch.int64):
        for b, q, padding in ((1, 1, 0), (2, 3, 2), (16, 4, 0), (17, 14, 2)):
            for kind in ("verify", "draft"):
                req = torch.arange(b, device=d, dtype=dtype)
                if b == 1:
                    req[0] = 1
                else:
                    req[-1] = 40
                positions = torch.arange(b * q + padding, device=d, dtype=dtype) - 2
                seq = torch.full((b,), 64, device=d, dtype=dtype)
                loc = torch.arange(positions.numel(), device=d, dtype=dtype)
                loc[::3] = -1
                batch = SimpleNamespace(
                    req_pool_indices=req,
                    positions=positions,
                    seq_lens=seq,
                    out_cache_loc=loc,
                    forward_mode=mode(kind),
                    spec_info=SimpleNamespace(
                        num_tokens_per_req=q,
                        num_accept_tokens=torch.arange(b, device=d, dtype=dtype)
                        % (q + 1),
                    ),
                )
                ref = lambda: mtp_reference(batch, q, 32, 128)
                fused = lambda: kernels.mtp_rows(batch, q, 32, 128)
                exact(fused(), ref())
                if d == "npu" and b == 16 and dtype == torch.int32:
                    graph, output = replay(fused, ref)
                    batch.positions.add_(1)
                    graph.replay()
                    torch.npu.synchronize()
                    exact(output, ref())
                results["cases"].append(f"mtp/{dtype}/{b}/{q}/{kind}")
                if args.benchmark and b == 16 and dtype == torch.int32:
                    results["benchmarks"][f"mtp/{kind}/B{b}Q{q}"] = measure(
                        ref, fused, d, 100
                    )
    # Non-unit input strides and optional location tensors have the same ABI.
    if args.benchmark and args.reference_source:
        old_mtp = load(
            "reference_mtp", args.reference_source / Path(*PKG.split(".")) / "mtp.py"
        )
        new_mtp = load("candidate_mtp", SOURCE / "mtp.py")
        for kind in ("verify", "draft"):
            batch = SimpleNamespace(
                req_pool_indices=torch.arange(1, 17, device=d, dtype=torch.int32),
                positions=(torch.arange(64, device=d) % 4 + 64000),
                seq_lens=torch.full((16,), 64000, device=d, dtype=torch.int32),
                out_cache_loc=torch.arange(64, device=d),
                forward_mode=mode(kind),
                spec_info=SimpleNamespace(
                    num_tokens_per_req=4,
                    num_accept_tokens=torch.arange(16, device=d) % 5,
                ),
            )
            ref = lambda: old_mtp.prepare_mtp_rows(batch, 64, 17, 67584)
            fused = lambda: new_mtp.prepare_mtp_rows(batch, 64, 17, 67584)
            exact(fused(), ref())
            results["benchmarks"][f"prepare_mtp_rows/{kind}/B16Q4/context67584"] = (
                measure(ref, fused, d, 100)
            )
    for kind in ("decode", "verify", "draft"):
        q = 1 if kind == "decode" else 4
        batch = SimpleNamespace(
            req_pool_indices=torch.arange(32, device=d, dtype=torch.int32)[::2],
            seq_lens=torch.full((48,), 64, device=d, dtype=torch.int32)[::3],
            positions=torch.arange(16 * q * 3, device=d, dtype=torch.int64)[::3],
            out_cache_loc=None,
            forward_mode=mode(kind),
            spec_info=SimpleNamespace(
                num_accept_tokens=torch.arange(64, device=d)[::4] % (q + 1)
            ),
        )
        exact(kernels.mtp_rows(batch, q, 32, 128), mtp_reference(batch, q, 32, 128))
        results["cases"].append(f"mtp/strided/{kind}/no-locations")
    for n in (1, 65, 1024, 8192):
        sources = [
            torch.empty((32, 1, w), dtype=torch.bfloat16, device=d) for w in (512, 64)
        ]
        destinations = [torch.empty_like(s) for s in sources]
        src = torch.arange(n, device=d, dtype=torch.int32) % 32
        dst = src.flip(0).long()
        valid = torch.arange(n, device=d) % 3 != 0
        src[::3], dst[::3] = -1, -1
        ref = lambda: copy_reference(sources, destinations, src, dst, valid)
        fused = lambda: kernels.copy_rows_metadata(
            sources, destinations, src, dst, valid
        )
        exact(fused(), ref())
        if d == "npu" and n == 65:
            graph, output = replay(fused, ref)
            src.clamp_(min=0)
            dst.clamp_(min=0)
            valid.logical_not_()
            # All addresses, including invalid negative rows, follow reference semantics.
            graph.replay()
            torch.npu.synchronize()
            exact(output, ref())
        results["cases"].append(f"copy_rows/{n}")
        if args.benchmark:
            results["benchmarks"][f"copy_rows/{n}"] = measure(ref, fused, d, 100)
    src = torch.arange(66, device=d, dtype=torch.int32)[::2]
    dst = torch.arange(99, device=d, dtype=torch.int64)[::3]
    valid = torch.arange(132, device=d)[::4] % 3 != 0
    exact(
        kernels.copy_rows_metadata(sources, destinations, src, dst, valid),
        copy_reference(sources, destinations, src, dst, valid),
    )
    results["cases"].append("copy_rows/strided")
    for req, lens, prefixes, extends, slots in (
        ([1, 2], [4, 4], [2, 2], [2, 2], [2, 2]),
        ([1, 2], [3, 4], [2, 2], [1, 2], [2, 2]),
        ([1, -1, 2, 0], [3, 2, 4, 1], [2, 2, 2, 1], [1, 0, 2, 0], [2] * 4),
        ([1, 2], [0, 4], [0, 0], [0, 4], [0, 4]),
    ):
        table = (torch.arange(32 * 16, device=d).reshape(32, 16) * 7) % 127
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor(req, device=d, dtype=torch.int32),
            seq_lens=torch.tensor(lens, device=d, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(lens),
        )
        manager = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            actual_request_rows=32,
            logical_max_context_len=16,
        )
        total = sum(length for r, length in zip(req, lens) if r >= 0 and length > 0)
        source, dest, valid = kernels.prefill_plan(batch, manager, total)
        expected = torch.cat(
            [table[r, :length] for r, length in zip(req, lens) if r >= 0 and length > 0]
        )
        exact(
            (source, dest, valid),
            (
                expected,
                torch.arange(total, device=d),
                torch.ones(total, device=d, dtype=torch.bool),
            ),
        )
        plan = SimpleNamespace(source=source, total_tokens=total)
        loc = []
        for r, p, e, width in zip(req, prefixes, extends, slots):
            loc.extend(
                [
                    int(table[max(r, 0), p + col].cpu()) if col < e else -1
                    for col in range(width)
                ]
            )
        batch.out_cache_loc = torch.tensor(loc, device=d)
        current_valid = torch.ones(len(loc), device=d, dtype=torch.bool)
        for variation in ("normal", "duplicate", "invalid"):
            if variation == "duplicate" and len(loc) > 1:
                batch.out_cache_loc[1] = batch.out_cache_loc[0]
            if variation == "invalid":
                batch.out_cache_loc[0] = 999
                current_valid[-1] = False
            ref = lambda: mapping_reference(
                plan, batch, current_valid, 128, slots, req, lens, prefixes, extends
            )
            fused = lambda: kernels.prefill_current_mapping(
                plan, batch, current_valid, 128, slots, req, lens, prefixes, extends
            )
            exact(fused(), ref())
            results["cases"].append(f"mapping/{req}/{variation}")
    if d == "npu":
        large_prefill_cases(args, kernels, results)
    if args.mf:
        mf_case(args, results)
    print(json.dumps(results, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


def large_prefill_cases(args, kernels, results):
    """Check realistic full-history gathers and dynamic current-row masks."""
    d = args.device
    reference_module = None
    if args.reference_source:
        reference_module = load(
            "reference_block_tables",
            args.reference_source / Path(*PKG.split(".")) / "block_tables.py",
        )
    for b, length, width in ((1, 64000, 1024), (16, 64000, 4)):
        lens = [length - r for r in range(b)]
        req = list(range(1, b + 1))
        table = torch.randperm((b + 1) * length, device=d).reshape(b + 1, length).int()
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor(req, device=d, dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor(req),
            seq_lens=torch.tensor(lens, device=d, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(lens),
            extend_prefix_lens_cpu=[n - width for n in lens],
            extend_seq_lens_cpu=[width] * b,
        )
        manager = SimpleNamespace(
            device=torch.device(d),
            size=b + 1,
            max_context_len=length,
            logical_max_context_len=length,
            actual_request_rows=b + 1,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            host_source_indices=lambda rows, pos: table[
                rows.clamp(0, b), pos.clamp(0, length - 1)
            ].long(),
        )
        total = sum(lens)
        actual = kernels.prefill_plan(batch, manager, total)
        exact(
            actual,
            (
                torch.cat([table[r, :n] for r, n in zip(req, lens)]).long(),
                torch.arange(total, device=d),
                torch.ones(total, device=d, dtype=torch.bool),
            ),
        )
        plan = SimpleNamespace(source=actual[0], total_tokens=total)
        batch.out_cache_loc = torch.cat(
            [table[r, n - width : n] for r, n in zip(req, lens)]
        ).long()
        current_valid = torch.ones(b * width, device=d, dtype=torch.bool)
        ref = lambda: mapping_reference(
            plan,
            batch,
            current_valid,
            table.numel(),
            [width] * b,
            req,
            lens,
            batch.extend_prefix_lens_cpu,
            batch.extend_seq_lens_cpu,
        )
        fused = lambda: kernels.prefill_current_mapping(
            plan,
            batch,
            current_valid,
            table.numel(),
            [width] * b,
            req,
            lens,
            batch.extend_prefix_lens_cpu,
            batch.extend_seq_lens_cpu,
        )
        exact(fused(), ref())
        graph, output = replay(fused, ref)
        current_valid[::2] = False
        graph.replay()
        torch.npu.synchronize()
        exact(output, ref())
        results["cases"].append(f"prefill-large/B{b}/L{length}/Q{width}")
        if args.benchmark:
            results["benchmarks"][f"mapping/B{b}L{length}Q{width}"] = measure(
                ref, fused, d, 30
            )
            if reference_module is not None:
                old = lambda: reference_module.make_fused_prefill_plan(
                    batch, manager, None
                )
                new_module = load("candidate_block_tables", SOURCE / "block_tables.py")
                new = lambda: new_module.make_fused_prefill_plan(batch, manager, None)
                old_mapping = lambda: reference_module.fused_prefill_current_mapping(
                    plan, batch, current_valid, table.numel()
                )
                new_mapping = lambda: new_module.fused_prefill_current_mapping(
                    plan, batch, current_valid, table.numel()
                )
                exact(new_mapping(), old_mapping())
                results["benchmarks"][
                    f"fused_prefill_current_mapping/B{b}L{length}Q{width}"
                ] = measure(old_mapping, new_mapping, d, 30)
                results["benchmarks"][f"make_prefill_plan/B{b}L{length}"] = measure(
                    old, new, d, 30
                )

                def forward_method(path, module):
                    tree = ast.parse(path.read_text())
                    cls = next(
                        node
                        for node in tree.body
                        if isinstance(node, ast.ClassDef)
                        and node.name == "FusedSparseKVCacheManager"
                    )
                    method = next(
                        node
                        for node in cls.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "prepare_forward_kv_plan"
                    )
                    namespace = {
                        "torch": torch,
                        "prepare_prefill_gather_rows": module.prepare_prefill_gather_rows,
                        "make_fused_prefill_plan": module.make_fused_prefill_plan,
                    }
                    exec(
                        compile(
                            ast.Module(body=[method], type_ignores=[]),
                            str(path),
                            "exec",
                        ),
                        namespace,
                    )
                    return namespace[method.name]

                old_forward = forward_method(
                    args.reference_source / Path(*PKG.split(".")) / "fused_manager.py",
                    reference_module,
                )
                new_forward = forward_method(SOURCE / "fused_manager.py", new_module)
                manager.budget = SimpleNamespace(prefill_workspace_tokens=total)
                manager._prefill_workspace_stream = None
                results["benchmarks"][f"prepare_forward_kv_plan/B{b}L{length}"] = (
                    measure(
                        lambda: old_forward(manager, batch, None),
                        lambda: new_forward(manager, batch, None),
                        d,
                        30,
                    )
                )


def mf_case(args, results):
    """Validate fused descriptors through real MF D2H/H2D, eager and replay."""
    d = args.device
    assert d == "npu", "MF transport validation requires an NPU"
    pool_module = load("candidate_memfabric_pool", SOURCE / "memfabric_pool.py")
    group = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        broadcast_object=lambda x: x,
        barrier=lambda: None,
        broadcast=lambda tensor, src: None,
    )
    pool = pool_module.MemFabricKVPool(
        tokens=128,
        layers=1,
        widths=(512, 64),
        dtype=torch.bfloat16,
        device=d,
        group=group,
    )
    payload = [
        torch.randn((64, 1, w), dtype=torch.bfloat16, device=d) for w in (512, 64)
    ]
    host = tuple(component[0] for component in pool.buffers)
    src = torch.arange(64, device=d)
    dst = (src * 3) % 128
    valid = src % 5 != 0
    src = torch.where(valid, src, -1)
    dst = torch.where(valid, dst, -1)
    for component in host:
        component.zero_()
    output = [torch.full_like(component, -17) for component in payload]

    def transport():
        pool.copy_rows(payload, host, src, dst, valid)
        pool.copy_rows(host, output, dst, torch.arange(64, device=d), valid)
        return output

    def transport_reference():
        return [
            torch.where(
                valid[:, None, None], component, torch.full_like(component, -17)
            )
            for component in payload
        ]

    exact(transport(), transport_reference())
    graph, outputs = replay(transport, transport_reference)
    for component in payload:
        component.add_(1)
    graph.replay()
    torch.npu.synchronize()
    exact(outputs, transport_reference())
    results["cases"].append("MF/D2H-H2D/eager-dynamic-graph/exact")
    if args.benchmark and args.reference_source:
        old_pool = load(
            "reference_memfabric_pool",
            args.reference_source / Path(*PKG.split(".")) / "memfabric_pool.py",
        )
        ref = lambda: old_pool.MemFabricKVPool.copy_rows(
            pool, payload, host, src, dst, valid
        )
        fused = lambda: pool.copy_rows(payload, host, src, dst, valid)
        results["benchmarks"]["MF/copy_rows/transport/B16Q4"] = measure(
            ref, fused, d, 100
        )


if __name__ == "__main__":
    main()
