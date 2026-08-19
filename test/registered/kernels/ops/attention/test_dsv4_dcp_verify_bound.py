import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=1, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=1, stage="jit-kernel-unit", runner_config="amd")


def test_dspark_target_verify_expands_16k_page_table_bound() -> None:
    backend = object.__new__(DeepseekV4HipRadixBackend)
    backend.target_verify_num_draft_tokens = 6
    backend._move_to_device = lambda values: torch.tensor(values, dtype=torch.int32)

    captured = {}
    backend.init_forward_metadata_prefill = lambda **kwargs: captured.update(kwargs)
    backend.init_forward_metadata_target_verify_old(
        max_seq_len=16384,
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([16384], dtype=torch.int32),
        seq_lens_cpu=[16384],
        out_cache_loc=torch.zeros(6, dtype=torch.int64),
    )

    assert captured["max_seq_len"] == 16390
