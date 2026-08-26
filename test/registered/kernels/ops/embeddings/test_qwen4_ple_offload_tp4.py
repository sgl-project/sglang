import os

import pytest
import torch

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    update_environment_variables,
)
from sglang.srt.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="4-gpu-b200")

_TP_SIZE = 4


def _run_tp4_parity(local_rank: int, world_size: int, master_port: int) -> None:
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
        }
    )
    torch.cuda.set_device(local_rank)
    torch.set_default_device(f"cuda:{local_rank}")
    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path="dummy",
            tp_size=world_size,
            disable_custom_all_reduce=True,
            enable_symm_mem=True,
        )
    )
    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        backend="nccl",
        enable_symm_mem=True,
    )

    try:
        for embedding_dim in (7, 257):
            baseline = VocabParallelEmbedding(
                17,
                embedding_dim,
                params_dtype=torch.bfloat16,
            )
            source = VocabParallelEmbedding(
                17,
                embedding_dim,
                params_dtype=torch.bfloat16,
            )
            offloaded = Qwen4ExpPinnedHostEmbedding(source)

            full_weight = (
                torch.arange(17 * embedding_dim, dtype=torch.int64, device="cpu")
                .remainder(127)
                .reshape(17, embedding_dim)
                .to(torch.bfloat16)
            )
            baseline.weight_loader(baseline.weight, full_weight)
            offloaded.weight_loader(offloaded.weight, full_weight)

            ids = torch.tensor(
                [[0, 1, 4, 5, 8], [11, 12, 15, 16, 7]],
                dtype=torch.int64,
                device=f"cuda:{local_rank}",
            )
            expected = baseline(ids)
            actual = offloaded(ids)

            assert offloaded.weight.is_pinned()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.cuda.synchronize()
    finally:
        destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_qwen4_ple_pinned_embedding_tp4_bitwise_parity():
    if not torch.cuda.is_available() or torch.cuda.device_count() < _TP_SIZE:
        pytest.skip("This test requires four CUDA devices.")

    torch.multiprocessing.spawn(
        _run_tp4_parity,
        args=(_TP_SIZE, get_open_port()),
        nprocs=_TP_SIZE,
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
