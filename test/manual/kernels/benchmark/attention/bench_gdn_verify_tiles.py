"""CUDA-graph timing and byte comparisons for circular GDN verify tiles."""

import argparse
import hashlib
import inspect
import json
import statistics

import torch

from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)


def make_case(tp, batch, length, history, seed=123):
    torch.manual_seed(seed)
    h, hv, k, v, slots, ring = 16 // tp, 64 // tp, 128, 128, 4, 16
    dev, dtype = "cuda", torch.bfloat16

    def rand(*shape):
        return torch.randn(*shape, device=dev, dtype=dtype) * 0.1

    return dict(
        q=rand(batch * length, h, k),
        k=rand(batch * length, h, k),
        v=rand(batch * length, hv, v),
        a=rand(batch * length, hv),
        b=rand(batch * length, hv),
        A_log=torch.zeros(hv, device=dev),
        dt_bias=torch.zeros(hv, device=dev),
        checkpoint_state=rand(slots, hv, v, k),
        d_cache=rand(slots, hv, ring, v),
        k_cache=rand(slots, h, ring, k),
        g_cache=-torch.rand(slots, hv, ring, device=dev) * 0.5,
        rawv_cache=rand(slots, hv, ring, v),
        rawk_cache=rand(slots, h, ring, k),
        beta_cache=None,
        out=torch.empty(batch * length, hv, v, device=dev, dtype=dtype),
        query_start_loc=torch.arange(
            0, (batch + 1) * length, length, device=dev, dtype=torch.int32
        ),
        ssm_state_indices=torch.arange(batch, device=dev, dtype=torch.int32),
        replay_indices=torch.arange(batch, device=dev, dtype=torch.int32),
        write_pos=torch.full((slots,), history, device=dev, dtype=torch.int32),
        cache_base=torch.full((slots,), 11, device=dev, dtype=torch.int32),
        is_flush=torch.zeros(slots, device=dev, dtype=torch.int8),
        max_cache_len=ring,
        max_spec_len=length,
        null_block_id=-1,
        launch_mode="verify",
    )


def snapshot(case):
    return {
        key: case[key].clone()
        for key in ("out", "d_cache", "k_cache", "g_cache", "rawv_cache", "rawk_cache")
    }


def benchmark(fn, repeats=7, calls=200):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(3):
        graph.replay()
    times = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return dict(
        p50_us=statistics.median(times),
        min_us=min(times),
        max_us=max(times),
        samples_us=times,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--tp", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--history", type=int, default=7)
    args = parser.parse_args()
    source = inspect.getsourcefile(gdn_replayssm_spec_decode)
    print(
        json.dumps(
            dict(
                source_md5=hashlib.md5(open(source, "rb").read()).hexdigest(),
                torch=torch.__version__,
                device=torch.cuda.get_device_name(),
                precision="bf16 compact checkpoint",
                mode="static microbenchmark",
                accuracy="PENDING",
            )
        ),
        flush=True,
    )
    configs = [(64, 1, 2, 2)]
    if args.sweep:
        configs += [
            (bv, nw, ns, nk)
            for bv in (16, 32, 64, 128)
            for nw in (2, 4, 8)
            for ns in (1, 2)
            for nk in (1, 2)
            if (bv, nw, ns, nk) not in configs
        ]
    for tp in args.tp:
        case = make_case(tp, args.batch, 7, args.history)
        gdn_replayssm_spec_decode(**case)
        torch.cuda.synchronize()
        expected = snapshot(case)
        for bv, nw, ns, nk in configs:
            opts = dict(block_v=bv, num_warps=nw, num_stages=ns, nk=nk)
            if args.direct:
                opts["direct_solve"] = True
            row = dict(tp=tp, batch=args.batch, tokens=7, history=args.history, **opts)
            try:

                def fn():
                    return gdn_replayssm_spec_decode(**case, **opts)

                fn()
                torch.cuda.synchronize()
                row["bitwise"] = {
                    key: bool(
                        torch.equal(case[key].view(torch.uint8), ref.view(torch.uint8))
                    )
                    for key, ref in expected.items()
                }
                row["max_abs_vs_stock"] = {
                    key: (case[key].float() - ref.float()).abs().max().item()
                    for key, ref in expected.items()
                }
                row.update(benchmark(fn))
            except Exception as exc:
                row["error"] = str(exc)[:1000]
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
