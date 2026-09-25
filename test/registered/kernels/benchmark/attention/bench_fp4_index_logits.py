import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    fp4_index_logits_decode,
    fp4_index_logits_paged,
    store_fp4_index_k_cache,
)
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def index_topk(q, weights, req, req_table, lens, table, capacity, paged):
    if paged:
        scores = fp4_index_logits_paged(
            q, weights, req, req_table, lens, table, 64, capacity, 1
        )
        indices = torch.empty((q.shape[0], 512), dtype=torch.int32, device=q.device)
        topk_transform_paged_v2(scores, lens, None, indices, 1, plan_topk_v2(lens))
        return indices
    positions = torch.arange(capacity, device=q.device)
    slots = req_table[req[:, None], positions[None, :]].long()
    slots.masked_fill_(positions[None, :] >= lens[:, None], 0)
    scores = fp4_index_logits_decode(q, weights, slots, lens, table, 64)
    return scores.topk(512, dim=-1, sorted=False).indices


@marker.parametrize("batch", [1, 4, 64], [4])
@marker.parametrize(
    "length,capacity",
    [(8192, 8192), (8192, 1048580), (32768, 1048580)],
    [(8192, 1048580)],
)
@marker.benchmark("impl", ["slots", "paged"])
def benchmark(batch, length, capacity, impl):
    if torch.cuda.get_device_capability()[0] != 9:
        marker.skip("Hopper decode indexer")
    q = torch.randn(batch, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch, 32, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(batch * length, 128, device="cuda", dtype=torch.bfloat16)
    locations = torch.arange(batch * length, device="cuda", dtype=torch.int64)
    table = torch.empty(batch * length // 64, 64 * 68, device="cuda", dtype=torch.uint8)
    store_fp4_index_k_cache(keys, table, locations, page_size=64, rne=True)
    req = torch.arange(batch, device="cuda", dtype=torch.int32)
    req_table = torch.full((batch, capacity), -1, device="cuda", dtype=torch.int32)
    req_table[:, :length] = locations.view(batch, length).to(torch.int32)
    lens = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    return marker.do_bench(
        index_topk,
        input_args=(q, weights, req, req_table, lens, table, capacity, impl == "paged"),
        # Rotate layer-specific KV and query operands. The request mapping is
        # shared between layers, as in serving; cloning its unused 1M capacity
        # would incorrectly reduce the rotation count for the visible KV.
        graph_clone_args=(0, 1, 2, 4, 5),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
