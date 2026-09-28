import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.minicpm_sala import get_block_table
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

_HEAD_GROUP = 2
_SPARSE_BLOCK_SIZE = 64
_TOPK = 96


def _make_valid_inputs(token_num: int, topk: int, device: str = "cuda"):
    """Well-formed inputs shared by both expansion strategies.

    ``seqlen_q_max`` is tied to ``token_num`` so the per-token causal position
    (``token_pos_in_bs``) never indexes past ``block_table``.
    """
    seqlen_q_max = token_num
    num_blocks = max(1, seqlen_q_max // _SPARSE_BLOCK_SIZE)
    torch.manual_seed(0)
    topk_idx = torch.randint(
        0, num_blocks, (_HEAD_GROUP, token_num, topk), dtype=torch.int32, device=device
    )
    block_table = torch.arange(
        1, seqlen_q_max + 1, dtype=torch.int32, device=device
    ).reshape(1, seqlen_q_max)
    token_to_bs = torch.zeros((token_num,), dtype=torch.int32, device=device)
    token_pos_in_bs = torch.arange(1, token_num + 1, dtype=torch.int32, device=device)
    seqlen_q = torch.tensor([seqlen_q_max], dtype=torch.int32, device=device)
    return topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q


@marker.parametrize("token_num", [2**n for n in range(9, 15)], [512, 4096])
@marker.benchmark("provider", ["blockwise", "elementwise"])
def benchmark(token_num: int, provider: str):
    inputs = _make_valid_inputs(token_num, _TOPK)

    def fn(*args):
        return get_block_table(*args, elementwise=provider == "elementwise")

    return marker.do_bench(fn, input_args=inputs)


if __name__ == "__main__":
    benchmark.run()
