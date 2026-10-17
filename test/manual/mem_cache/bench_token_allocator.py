"""Fixed request-turnover microbenchmark of the production token allocator."""

import argparse
import json
import time

import torch

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator


def run(size, slots, rounds, device, need_sort):
    allocator = TokenToKVPoolAllocator(size, torch.float16, device, None, need_sort)
    for _ in range(rounds):
        indices = allocator.alloc(slots)
        assert indices is not None
        allocator.free(indices)
    assert allocator.available_size() == size
    recovered = allocator.alloc(size)
    assert recovered is not None
    assert torch.equal(torch.sort(recovered).values.cpu(), torch.arange(1, size + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = []
    for size in (8192, 131072, 524288):
        for need_sort in (False, True):
            slots, rounds = 128, 4096
            run(size, slots, 128, args.device, need_sort)
            samples = []
            for _ in range(5):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                run(size, slots, rounds, args.device, need_sort)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
            ) as prof:
                run(size, slots, rounds, args.device, need_sort)
            cats = [x for x in prof.key_averages() if x.key == "aten::cat"]
            cases.append(
                dict(
                    size=size,
                    slots=slots,
                    rounds=rounds,
                    need_sort=need_sort,
                    wall_ms=samples,
                    cat_calls=sum(x.count for x in cats),
                    cat_cpu_bytes=sum(x.self_cpu_memory_usage for x in cats),
                )
            )
    result = dict(device=args.device, torch=torch.__version__, cases=cases)
    if args.device == "cuda":
        result["gpu"] = torch.cuda.get_device_name()
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
