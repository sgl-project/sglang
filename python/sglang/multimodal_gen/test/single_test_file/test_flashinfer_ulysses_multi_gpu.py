# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=2 (or 4/8) on CUDA.

Validates the real model-parallel groups, Wan's USPAttention and H3's packed
varlen attention core, both layouts, strided/batched QKV, and group teardown.
"""

import os

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.device_communicators import (
    flashinfer_ulysses as fi,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_sp_group,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers import usp
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _minimax_h3_attention_core_impl,
)
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args


@torch.inference_mode()
def main():
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    os.environ.update(
        SGLANG_DIFFUSION_IPC_A2A="0",
        SGLANG_DIFFUSION_IPC_A2A_MULTI="0",
        SGLANG_DIFFUSION_FLASHINFER_ULYSSES_BACKEND="nvlink",
        # Leave the pipeline enabled to check that FlashInfer takes precedence.
        SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS="-1",
    )
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world, ulysses_degree=world
    )
    set_global_server_args(_make_unit_server_args())
    try:
        for dtype in (torch.float16, torch.bfloat16):
            for batch in (1, 2):
                torch.manual_seed(1000)
                full = torch.randn(
                    batch, world * 16, world * 2, 64, device="cuda", dtype=dtype
                )
                shard = full[:, rank * 16 : (rank + 1) * 16]
                envs.SGLANG_DIFFUSION_FLASHINFER_ULYSSES = True
                for head_dim in (1, 2):
                    x = shard if head_dim == 2 else shard.transpose(1, 2)
                    got = usp._usp_input_all_to_all(x, head_dim=head_dim)
                    spec = full[:, :, rank * 2 : (rank + 1) * 2]
                    spec = spec if head_dim == 2 else spec.transpose(1, 2)
                    torch.testing.assert_close(got, spec, rtol=0, atol=0)
                    # Same-shape exchange cannot overwrite the previous result.
                    retained = got.clone()
                    usp._usp_input_all_to_all(x + 1, head_dim=head_dim)
                    torch.testing.assert_close(got, retained, rtol=0, atol=0)
                    back = usp._usp_output_all_to_all(got, head_dim=head_dim)
                    torch.testing.assert_close(back, x, rtol=0, atol=0)
                # Legacy fixed-layout path used by Wan VSA.
                got = get_sp_group().all_to_all_4D(shard.contiguous(), 2, 1)
                torch.testing.assert_close(
                    got, full[:, :, rank * 2 : (rank + 1) * 2], rtol=0, atol=0
                )
                lengths = [16] * world
                got = usp._usp_input_all_to_all_varlen(shard, lengths, head_dim=2)
                back = usp._usp_output_all_to_all_varlen(got, lengths, head_dim=2)
                torch.testing.assert_close(back, shard, rtol=0, atol=0)
                fi.close_all()

        torch.manual_seed(1000 + rank)
        qkv = torch.randn(
            1, 32, world * 2, 3, 64, device="cuda", dtype=torch.bfloat16
        ).unbind(-2)
        layer = USPAttention(
            num_heads=world * 2,
            head_size=64,
            causal=False,
            required_attention_backend=AttentionBackendEnum.TORCH_SDPA,
            prefix="test.wan",
        )

        class H3Attention:
            # Use the real packed-varlen implementation already owned by layer.
            _attention_impl = layer.attn_impl
            _attention_backend_enum = AttentionBackendEnum.TORCH_SDPA

        outputs = []
        for enabled in (False, True):
            envs.SGLANG_DIFFUSION_FLASHINFER_ULYSSES = enabled
            with set_forward_context(0, None, None):
                wan = layer(*qkv)
                h3 = _minimax_h3_attention_core_impl(
                    H3Attention(),
                    *(t.squeeze(0) for t in qkv),
                    cu_seqlens=torch.tensor(
                        [0, 32 * world], dtype=torch.int32, device="cuda"
                    ),
                    cu_seqlens_host=(0, 32 * world),
                    max_seqlen=32 * world,
                    ulysses_active=True,
                )
            outputs.append((wan.clone(), h3.clone()))
        for baseline, candidate in zip(*outputs):
            torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
        assert fi._CONTEXTS, "model attention bypassed FlashInfer"
        assert all(c.comm.backend == "nvlink" for c in fi._CONTEXTS.values())
        if rank == 0:
            print(
                f"PASS: {world} GPUs, FP16/BF16, layouts, lifetimes, batched/strided QKV, Wan and H3 attention",
                flush=True,
            )
    finally:
        destroy_model_parallel()
        assert not fi._CONTEXTS
        destroy_distributed_environment()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
