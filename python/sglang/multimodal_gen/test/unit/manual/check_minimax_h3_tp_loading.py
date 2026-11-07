# SPDX-License-Identifier: Apache-2.0
"""Exact real-checkpoint parity against ordinary H3 TP loading (two H200s).

Run from the repository root, passing the native FL2VA/transformer directory:
    PYTHONPATH=python torchrun --standalone --nproc-per-node=2 \
        python/sglang/multimodal_gen/test/unit/manual/check_minimax_h3_tp_loading.py \
        --transformer /path/to/FL2VA/transformer

Both models remain resident for torch.equal on every parameter and buffer.
This is a correctness check, not a controlled startup benchmark. The ordinary
reference bypasses only the rank-local fast path; both use the same production
model construction, weight loaders, mixed precision and post-load processing.
"""

import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTConfig
from sglang.multimodal_gen.runtime.distributed import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.loader import rank_local_checkpoint
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=16)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    maybe_init_distributed_environment_and_model_parallel(tp_size=world_size, sp_size=1)
    publish(
        ServerArgs(model_path="dummy", tp_size=world_size), role="diffusion_gpu_worker"
    )
    hf_config = json.loads((args.transformer / "config.json").read_text())
    config = MiniMaxH3DiTConfig()
    config.update_model_arch(hf_config)
    kwargs = dict(
        model_cls=MiniMaxH3DiTModel,
        init_params={"config": config, "hf_config": hf_config},
        weight_dir_list=sorted(str(p) for p in args.transformer.glob("*.safetensors")),
        device=device,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    times = {}
    dist.barrier()
    start = time.perf_counter()
    with patch.object(
        rank_local_checkpoint, "try_load_rank_local_tp_state_dict", return_value=None
    ):
        reference = maybe_load_fsdp_model(**kwargs)
    torch.cuda.synchronize()
    times["ordinary_seconds"] = time.perf_counter() - start
    print(f"REFERENCE_LOADED rank={rank} {times}", flush=True)

    original = rank_local_checkpoint.try_load_rank_local_tp_state_dict
    local_bytes = 0

    def verified_local_load(*args, **kwargs):
        nonlocal local_bytes
        result = original(*args, **kwargs)
        assert result is not None, (
            "Native H3 unexpectedly fell back to ordinary loading"
        )
        local_bytes = sum(
            value.tensor.numel() * value.tensor.element_size()
            for value in result[0].values()
        )
        return result

    dist.barrier()
    start = time.perf_counter()
    with patch.object(
        rank_local_checkpoint, "try_load_rank_local_tp_state_dict", verified_local_load
    ):
        actual = maybe_load_fsdp_model(**kwargs)
    torch.cuda.synchronize()
    times["rank_local_seconds"] = time.perf_counter() - start
    expected_state, actual_state = reference.state_dict(), actual.state_dict()
    assert expected_state.keys() == actual_state.keys()
    assert (
        dict(reference.named_parameters()).keys()
        == dict(actual.named_parameters()).keys()
    )
    assert dict(reference.named_buffers()).keys() == dict(actual.named_buffers()).keys()
    for name, expected in expected_state.items():
        loaded = actual_state[name]
        assert (expected.shape, expected.dtype, expected.device) == (
            loaded.shape,
            loaded.dtype,
            loaded.device,
        ), name
        assert torch.equal(expected, loaded), name
    dist.barrier()
    print(
        "PARITY_OK "
        + json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "tensors": len(expected_state),
                "local_source_bytes": local_bytes,
                "state_bytes": sum(
                    t.numel() * t.element_size() for t in actual_state.values()
                ),
                "dtypes": sorted({str(t.dtype) for t in actual_state.values()}),
                **times,
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
