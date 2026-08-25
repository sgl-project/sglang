import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.elementwise.fast_topk import fast_topk as jit_fast_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


def torch_topk(score, lengths, topk):
    return torch.topk(score, topk, dim=1).indices


FN_MAP = {
    "jit": jit_fast_topk,
    "torch_topk": torch_topk,
}


@marker.parametrize(
    "batch,length",
    [(1, 2048), (128, 2048), (1024, 2048), (8000, 2048), (1024, 32768)],
    [(1024, 2048)],
)
@marker.benchmark("impl", ["jit", "torch_topk"])
def benchmark(batch: int, length: int, impl: str):
    topk = 512
    score = create_random(batch, length, dtype=torch.float32)
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(score, lengths, topk),
        # score is read -> clone it per iter to avoid L2 reuse.
        graph_clone_args=(0,),
        # indices output is derived; count only the score read.
        memory_args=(score,),
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark.run()
