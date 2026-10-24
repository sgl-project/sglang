# SPDX-License-Identifier: Apache-2.0
"""A stage without CFG shards a long enough sequence over the CFG ranks as one Ulysses group."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.distributed import group_coordinator
from sglang.multimodal_gen.runtime.distributed import parallel_state as ps
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.executors.parallel_executor import (
    ParallelExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av import (
    CFG_SP_MIN_TOKENS_PER_RANK,
    LTX2RefinementStage,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform


class _Sdpa:
    def forward(self, q, k, v, _ctx):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)


def _ulysses_attention() -> USPAttention:
    attn = USPAttention.__new__(USPAttention)
    attn.attn_impl = _Sdpa()
    attn.backend = AttentionBackendEnum.TORCH_SDPA
    attn.skip_sequence_parallel = False
    attn.enable_packed_qkv_input_a2a = False
    attn.sp_attention_mode = "ulysses"
    attn.sp_attention_mode_is_auto = False
    return attn


def _check_folded_topology(rank, rendezvous):
    with patch.object(group_coordinator, "current_platform", CpuPlatform()):
        ps.init_distributed_environment(
            world_size=2,
            rank=rank,
            distributed_init_method=rendezvous,
            local_rank=rank,
            backend="gloo",
            timeout=60,
        )
        ps.initialize_model_parallel(
            classifier_free_guidance_degree=2,
            sequence_parallel_degree=1,
            backend="gloo",
        )
    try:
        folded = ps.get_cfg_sequence_parallel_group()
        cfg = ps.get_cfg_group()
        # a view over the CFG communicators: no new process group per rank
        assert folded.device_group is cfg.device_group
        assert folded.ulysses_group is cfg.device_group
        assert (folded.ranks, folded.rank_in_group) == (cfg.ranks, cfg.rank_in_group)
        assert (folded.ulysses_world_size, folded.ulysses_rank) == (2, rank)
        assert folded.ring_world_size == 1
        assert ps.get_sp_world_size() == 1

        stage = object.__new__(LTX2RefinementStage)
        stage.server_args = SimpleNamespace(enable_cfg_parallel=True)
        stage.vae = None
        min_tokens = 2 * CFG_SP_MIN_TOKENS_PER_RANK
        short, long = (
            Req(sampling_params=SamplingParams(), latents=torch.empty(1, n, 1))
            for n in (min_tokens - 2, min_tokens)
        )
        server_args = SimpleNamespace(
            enable_cfg_parallel=True, enable_layerwise_nvtx_marker=False
        )
        with (
            patch.object(
                ParallelExecutor,
                "_component_residency_request",
                lambda *args: nullcontext(),
            ),
            patch.object(
                ParallelExecutor,
                "_run_stage_with_executor_hooks",
                lambda self, stage, index, batch, args, run_stage, nvtx: run_stage(
                    stage, batch
                ),
            ),
        ):
            executor = ParallelExecutor.__new__(ParallelExecutor)
            seen_short, seen_long = (
                executor._execute_stages(
                    [stage],
                    batch,
                    server_args,
                    lambda *_: (ps.get_sp_world_size(), ps.get_sp_parallel_rank()),
                )
                for batch in (short, long)
            )
        # a short stage 2 runs on the main rank alone; the other keeps its batch
        if rank == 0:
            assert seen_short == (1, 0)
        else:
            assert seen_short is short
        # a long one runs on every CFG rank inside the folded scope
        assert seen_long == (2, rank)
        assert ps.get_sp_world_size() == 1

        torch.manual_seed(0)
        config = LTX2PipelineConfig()
        tokens_per_frame = 6
        video = torch.randn(1, 16 * tokens_per_frame, 8)
        audio = torch.randn(1, 7, 8)
        q, k, v = (torch.randn(1, 8, 4, 16) for _ in range(3))
        expected = _Sdpa().forward(q, k, v, None)
        local = slice(4 * rank, 4 * rank + 4)
        attn = _ulysses_attention()
        with (
            ps.use_sequence_parallel_group(folded),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            batch = Req(
                height=config.vae_scale_factor * config.patch_size * 2,
                width=config.vae_scale_factor * config.patch_size * 3,
            )
            video_shard, _ = config.shard_latents_for_sp(batch, video)
            audio_shard, _ = config.shard_audio_latents_for_sp(batch, audio)
            assert video_shard.shape[1] == 8 * tokens_per_frame
            assert batch.sp_video_start_frame == 8 * rank
            assert audio_shard.shape[1] == 4 and batch.sp_audio_has_padding
            video_gathered = config.gather_latents_for_sp(video_shard, batch)
            audio_gathered = config.gather_audio_latents_for_sp(audio_shard, batch)

            out = attn.forward(q[:, local], k[:, local], v[:, local])
            # LTX-2 keeps audio whole under SP: every key appears once per rank,
            # so the gathered duplicates leave each softmax unchanged
            replicated = attn.forward(q, k, v)
        torch.testing.assert_close(video_gathered, video, rtol=0, atol=0)
        torch.testing.assert_close(audio_gathered, audio, rtol=0, atol=0)
        torch.testing.assert_close(out, expected[:, local])
        torch.testing.assert_close(replicated, expected)
        assert ps.get_sp_world_size() == 1
    finally:
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
    assert ps.get_cfg_sequence_parallel_group() is None


@pytest.mark.skipif(
    torch.version.hip is not None,
    reason="ROCm builds query GPU 0 while importing sglang, which fails with the "
    "GPUs hidden for these CPU workers",
)
def test_stage_without_cfg_shards_over_cfg_ranks(tmp_path, monkeypatch):
    # gloo ranks on CPU: hide the GPUs so no communicator binds rank 1 to a
    # device a 1-GPU runner lacks, and keep the CUDA-IPC all-to-all off
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "cpu")
    monkeypatch.setenv("SGLANG_DIFFUSION_IPC_A2A", "0")
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_check_folded_topology, args=(rendezvous,), nprocs=2)
