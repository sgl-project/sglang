"""Two-rank LingBot model-level sequence parallelism matches dense execution."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase, find_available_port

_WORLD = 2
_REL_FRO_LIMIT = 1e-2


def _build_tiny_dit():
    from sglang.multimodal_gen.configs.models.dits.lingbot_video_moe import (
        LingBotVideoMoEConfig,
    )
    from sglang.multimodal_gen.runtime.models.dits.lingbot_video_moe import (
        LingBotVideoTransformer3DModel,
    )

    config = LingBotVideoMoEConfig()
    config.update_model_arch(
        dict(
            hidden_size=128,
            num_attention_heads=2,
            axes_dims=(16, 24, 24),
            depth=2,
            intermediate_size=256,
            text_dim=32,
            freq_dim=32,
            num_experts=16,
            moe_intermediate_size=32,
        )
    )
    return LingBotVideoTransformer3DModel(config, {}).eval()


def _worker_impl() -> int:
    import torch.distributed as dist

    from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
        LingBotVideoMoEPipelineConfig,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.attention.selector import (
        global_force_attn_backend,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    from sglang.multimodal_gen.runtime.utils.precision import (
        set_mixed_precision_policy,
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    if get_context()._server_args is None:
        get_context().set_server_args(SrtServerArgs(model_path="dummy"))
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=_WORLD,
        ulysses_degree=_WORLD,
    )
    set_mixed_precision_policy(torch.bfloat16, torch.float32)
    global_force_attn_backend(AttentionBackendEnum.FA)
    set_global_server_args(
        SimpleNamespace(
            attention_backend="fa",
            attention_backend_config=None,
            kv_gather_degree=1,
            sp_split_auto=False,
            pipeline_config=LingBotVideoMoEPipelineConfig(),
        )
    )

    dense = _build_tiny_dit()
    generator = torch.Generator(device="cpu").manual_seed(31)
    with torch.no_grad():
        for parameter in dense.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    sequence_parallel = _build_tiny_dit()
    sequence_parallel.load_state_dict(dense.state_dict())

    def tensor(*shape):
        return torch.randn(*shape, generator=generator).cuda().to(torch.bfloat16)

    inputs = dict(
        hidden_states=tensor(1, 16, 3, 4, 4),
        timestep=torch.tensor([500.0], device="cuda"),
        encoder_hidden_states=tensor(1, 7, 32),
    )
    outputs = []
    for model, enable_sequence_shard in (
        (dense, False),
        (sequence_parallel, True),
    ):
        model = model.to(device="cuda", dtype=torch.bfloat16)
        with (
            torch.no_grad(),
            set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=SimpleNamespace(
                    enable_sequence_shard=enable_sequence_shard
                ),
            ),
        ):
            outputs.append(model(**inputs).float())

    dense_output, sp_output = outputs
    relative_error = (
        torch.linalg.vector_norm(sp_output - dense_output)
        / torch.linalg.vector_norm(dense_output)
    ).item()
    failed = (
        not torch.isfinite(sp_output).all().item()
        or not math.isfinite(relative_error)
        or relative_error > _REL_FRO_LIMIT
    )
    if failed:
        print(
            f"FAILURE rank{rank}: dense/SP relative Frobenius error "
            f"{relative_error:.3e} exceeds {_REL_FRO_LIMIT:.1e}",
            flush=True,
        )
    dist.barrier()
    if rank == 0 and not failed:
        print("LINGBOT_SP_PARITY PASS", flush=True)
    return 1 if failed else 0


def _worker() -> int:
    from sglang.multimodal_gen.runtime.distributed import cleanup_dist_env_and_memory

    try:
        return _worker_impl()
    finally:
        cleanup_dist_env_and_memory()


class TestLingBotVideoMoeSequenceParallel(CustomTestCase):
    def test_two_rank_parity(self):
        if not torch.cuda.is_available():
            self.skipTest("requires a CUDA or ROCm GPU")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        port = str(find_available_port(29500))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={_WORLD}",
                "--master-addr=127.0.0.1",
                f"--master-port={port}",
                __file__,
                "--worker",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        output = result.stdout + result.stderr
        if result.returncode:
            self.fail(f"torchrun worker failed ({result.returncode}):\n{output}")
        self.assertIn("LINGBOT_SP_PARITY PASS", output)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.exit(_worker())
    unittest.main()
