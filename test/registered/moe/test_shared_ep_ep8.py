import os
import unittest
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import triton.language as tl

from sglang.kernels.ops.attention.dsv4 import silu_and_mul_contig_post_quant
from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.shared_ep import (
    SharedEpQuantCapability,
    SharedEpQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.layers.moe.shared_ep.backend import (
    SharedEpDispatchOutput,
    run_shared_ep,
)
from sglang.srt.layers.moe.shared_ep.kernels import quantize_pack_input
from sglang.srt.layers.moe.shared_ep.layout import (
    SharedEpInputFormat,
    SharedEpLayout,
)
from sglang.srt.layers.moe.shared_ep.profiles import (
    DSV4_PRO_MXFP4_GFX950,
    GLM52_GFX950,
    SharedEpProfile,
    SharedEpQuantization,
    make_pull_cache_prefill_profile,
)
from sglang.srt.layers.moe.shared_ep.pull_cache_prefill import (
    allocate_pull_cache,
    make_pull_cache_prefill_plan,
)
from sglang.srt.layers.moe.shared_ep.state import create_shared_ep_state
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=40, stage="extra-b", runner_config="8-gpu-h200")
register_amd_ci(est_time=120, suite="stage-c-test-large-8-gpu-amd-mi35x")

_KERNEL_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 3,
}


@unittest.skipUnless(
    torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) == 8,
    "run with torchrun --nproc-per-node=8 on CUDA",
)
class TestSharedEpEp8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rank = int(os.environ["RANK"])
        cls.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(cls.local_rank)
        dist.init_process_group("gloo")
        cls.profile = SharedEpProfile(
            name="ep8_test",
            capability=(9, 5) if torch.version.hip else (9, 0),
            hidden_size=128,
            intermediate_size=256,
            top_k=6,
            num_experts=8,
            num_local_experts=1,
            ep_size=8,
            max_tokens_per_rank=32,
            block_shape=(128, 128),
            default_kernel_config=_KERNEL_CONFIG,
            small_kernel_config=None,
            small_kernel_max_tokens=0,
            route_kernel_config={"num_threads": 1024},
            quantization=SharedEpQuantization.BLOCK_FP8,
            platform="rocm" if torch.version.hip else "cuda",
        )
        cls.runner_config = MoeRunnerConfig(
            num_experts=8,
            num_local_experts=1,
            hidden_size=128,
            intermediate_size_per_partition=256,
            top_k=6,
        )

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def test_direct_read_return_reuse_and_graph_match_materialized_ep8(self):
        state = create_shared_ep_state(
            layout=SharedEpLayout.build(
                hidden_size=128,
                top_k=6,
                max_tokens_per_rank=32,
            ),
            cpu_group=dist.group.WORLD,
            device=torch.device("cuda", self.local_rank),
        )
        try:
            self.assertEqual(state.layout.output_row_bytes, 16 * 1024)
            pull_plan = make_pull_cache_prefill_plan(
                owners=self.profile.ep_size,
                source_tokens_per_owner=self.profile.max_tokens_per_rank,
                hidden_size=self.profile.hidden_size,
                top_k=self.profile.top_k,
                num_local_experts=self.profile.num_local_experts,
                expert_alignment=self.profile.block_size_m,
            )
            pull_cache = allocate_pull_cache(
                pull_plan,
                active_rows=pull_plan.cache_rows,
                device=torch.device("cuda", self.local_rank),
            )
            tokens = self.rank + 1
            topk_ids = torch.stack(
                [
                    (
                        torch.arange(6, device="cuda", dtype=torch.int32)
                        + self.rank
                        + token
                    )
                    % 8
                    for token in range(tokens)
                ]
            )
            topk_weights = (
                torch.arange(1, 7, device="cuda", dtype=torch.float32)
                .div_(21)
                .expand(tokens, -1)
                .contiguous()
            )
            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=None,
            )

            full_w13 = torch.stack(
                [
                    torch.full(
                        (512, 128),
                        (expert + 1) / 512,
                        dtype=torch.float8_e4m3fn,
                        device="cuda",
                    )
                    for expert in range(8)
                ]
            )
            full_w2 = torch.stack(
                [
                    torch.full(
                        (128, 256),
                        (expert + 1) / 512,
                        dtype=torch.float8_e4m3fn,
                        device="cuda",
                    )
                    for expert in range(8)
                ]
            )
            full_w13_scale = torch.ones(
                (8, 4, 1),
                dtype=torch.float32,
                device="cuda",
            )
            full_w2_scale = torch.ones(
                (8, 1, 2),
                dtype=torch.float32,
                device="cuda",
            )
            local_w13 = full_w13[self.rank : self.rank + 1]
            local_w2 = full_w2[self.rank : self.rank + 1]
            local_w13_scale = full_w13_scale[self.rank : self.rank + 1]
            local_w2_scale = full_w2_scale[self.rank : self.rank + 1]
            fallback_quant = DeepGemmMoeQuantInfo(
                w13_weight=local_w13,
                w2_weight=local_w2,
                use_fp8=True,
                w13_scale=local_w13_scale,
                w2_scale=local_w2_scale,
                block_shape=[128, 128],
            )
            local_quant = SharedEpQuantInfo(
                w13_weight=local_w13,
                w2_weight=local_w2,
                w13_scale=local_w13_scale,
                w2_scale=local_w2_scale,
                block_shape=(128, 128),
                fallback_quant_info=fallback_quant,
                fallback_backend=MoeRunnerBackend.DEEP_GEMM,
            )

            for generation in (1, 2):
                hidden = self._hidden(generation, tokens)
                hidden_fp8, hidden_scale = per_token_group_quant_fp8(
                    hidden,
                    128,
                )
                expected = self._run_materialized(
                    hidden_fp8,
                    hidden_scale,
                    topk_output,
                    full_w13,
                    full_w2,
                    full_w13_scale,
                    full_w2_scale,
                    tokens,
                )
                actual = self._run_shared(
                    state,
                    hidden,
                    topk_output,
                    local_quant,
                    skew_input_writer=generation == 2,
                    skew_output_consumer=generation == 1,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                pull_actual = self._run_shared(
                    state,
                    hidden,
                    topk_output,
                    local_quant,
                    phase="prefill",
                    pull_cache=pull_cache,
                )
                torch.testing.assert_close(pull_actual, expected, rtol=0, atol=0)

            graph_hidden = self._hidden(3, tokens)
            graph_q, graph_scale = per_token_group_quant_fp8(
                graph_hidden,
                128,
            )
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph):
                graph_output = self._run_shared(
                    state,
                    graph_hidden,
                    topk_output,
                    local_quant,
                )
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            expected = self._run_materialized(
                graph_q,
                graph_scale,
                topk_output,
                full_w13,
                full_w2,
                full_w13_scale,
                full_w2_scale,
                tokens,
            )
            torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
        finally:
            dist.barrier()
            state.close()

    @unittest.skipUnless(torch.version.hip is not None, "requires ROCm")
    def test_glm52_pull_cache_matches_direct_ep8(self):
        arch = torch.cuda.get_device_properties(self.local_rank).gcnArchName.split(
            ":",
            1,
        )[0]
        if arch != "gfx950":
            self.skipTest(f"requires gfx950, got {arch}")

        decode_profile = GLM52_GFX950
        prefill_profile = make_pull_cache_prefill_profile(decode_profile, 64)
        self.assertIsNotNone(prefill_profile)
        runner_config = MoeRunnerConfig(
            num_experts=decode_profile.num_experts,
            num_local_experts=decode_profile.num_local_experts,
            hidden_size=decode_profile.hidden_size,
            intermediate_size_per_partition=decode_profile.intermediate_size,
            top_k=decode_profile.top_k,
        )

        def create_state(profile):
            return create_shared_ep_state(
                layout=SharedEpLayout.build(
                    hidden_size=profile.hidden_size,
                    top_k=profile.top_k,
                    max_tokens_per_rank=profile.max_tokens_per_rank,
                ),
                cpu_group=dist.group.WORLD,
                device=torch.device("cuda", self.local_rank),
            )

        decode_state = create_state(decode_profile)
        prefill_state = create_state(prefill_profile)
        pull_plan = make_pull_cache_prefill_plan(
            owners=prefill_profile.ep_size,
            source_tokens_per_owner=prefill_profile.max_tokens_per_rank,
            hidden_size=prefill_profile.hidden_size,
            top_k=prefill_profile.top_k,
            num_local_experts=prefill_profile.num_local_experts,
            expert_alignment=prefill_profile.block_size_m,
        )
        pull_cache = allocate_pull_cache(
            pull_plan,
            active_rows=pull_plan.cache_rows,
            device=torch.device("cuda", self.local_rank),
        )
        w13 = torch.empty(
            (
                decode_profile.num_local_experts,
                2 * decode_profile.intermediate_size,
                decode_profile.hidden_size,
            ),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        w2 = torch.empty(
            (
                decode_profile.num_local_experts,
                decode_profile.hidden_size,
                decode_profile.intermediate_size,
            ),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        local_start = self.rank * decode_profile.num_local_experts
        for local_expert in range(decode_profile.num_local_experts):
            value = (local_start + local_expert + 1) / 4096
            w13[local_expert].fill_(value)
            w2[local_expert].fill_(value)
        w13_scale = torch.ones(
            (
                decode_profile.num_local_experts,
                2 * decode_profile.intermediate_size // 128,
                decode_profile.hidden_size // 128,
            ),
            dtype=torch.float32,
            device="cuda",
        )
        w2_scale = torch.ones(
            (
                decode_profile.num_local_experts,
                decode_profile.hidden_size // 128,
                decode_profile.intermediate_size // 128,
            ),
            dtype=torch.float32,
            device="cuda",
        )
        quant_info = SharedEpQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=(128, 128),
            fallback_quant_info=Mock(),
            fallback_backend=MoeRunnerBackend.AITER,
        )
        hidden = (
            torch.arange(
                decode_profile.hidden_size,
                dtype=torch.float32,
                device="cuda",
            )
            .remainder(97)
            .sub(48)
            .div_(4096)
            .to(torch.bfloat16)
            .view(1, -1)
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.full(
                (1, decode_profile.top_k),
                1 / decode_profile.top_k,
                dtype=torch.float32,
                device="cuda",
            ),
            topk_ids=torch.tensor(
                [
                    [
                        ((self.rank + slot) % decode_profile.ep_size)
                        * decode_profile.num_local_experts
                        + slot
                        for slot in range(decode_profile.top_k)
                    ]
                ],
                dtype=torch.int32,
                device="cuda",
            ),
            router_logits=None,
        )

        def run(profile, state, phase, cache=None):
            quantize_pack_input(
                state.local_input,
                source=hidden,
                source_ids=topk_output.topk_ids,
                source_weights=topk_output.topk_weights,
                group_size=128,
            )
            state.local_output[:1].zero_()
            state.input_epoch.publish()
            return run_shared_ep(
                SharedEpDispatchOutput(
                    hidden_states=state.global_input.activations,
                    hidden_states_scale=state.global_input.scales,
                    topk_output=topk_output,
                    state=state,
                    profile=profile,
                    num_tokens=1,
                    local_expert_start=local_start,
                    phase=phase,
                    pull_cache=cache,
                ),
                quant_info,
                runner_config,
            ).hidden_states

        try:
            direct = run(decode_profile, decode_state, "decode")
            pulled = run(
                prefill_profile,
                prefill_state,
                "prefill",
                pull_cache,
            )
            torch.testing.assert_close(pulled, direct, rtol=5e-2, atol=2e-2)
        finally:
            dist.barrier()
            prefill_state.close()
            decode_state.close()

    @unittest.skipUnless(torch.version.hip is not None, "requires ROCm")
    def test_mxfp4_direct_return_reuses_ep8_epochs(self):
        arch = torch.cuda.get_device_properties(self.local_rank).gcnArchName.split(
            ":",
            1,
        )[0]
        if arch != "gfx950":
            self.skipTest(f"requires gfx950, got {arch}")

        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        }
        profile = SharedEpProfile(
            name="ep8_mxfp4_test",
            capability=(9, 5),
            hidden_size=128,
            intermediate_size=128,
            top_k=2,
            num_experts=8,
            num_local_experts=1,
            ep_size=8,
            max_tokens_per_rank=128,
            quantization=SharedEpQuantization.MXFP4,
            block_shape=(1, 32),
            default_kernel_config=config,
            small_kernel_config=None,
            small_kernel_max_tokens=128,
            route_kernel_config={"num_threads": 256},
            default_w13_kernel_config=config,
            default_w2_kernel_config=config,
            platform="rocm",
        )
        runner_config = MoeRunnerConfig(
            num_experts=8,
            num_local_experts=1,
            hidden_size=128,
            intermediate_size_per_partition=128,
            top_k=2,
        )
        state = create_shared_ep_state(
            layout=SharedEpLayout.build(
                hidden_size=profile.hidden_size,
                top_k=profile.top_k,
                max_tokens_per_rank=profile.max_tokens_per_rank,
                input_format=SharedEpInputFormat.BF16,
                direct_output=True,
            ),
            cpu_group=dist.group.WORLD,
            device=torch.device("cuda", self.local_rank),
        )
        w13 = torch.full(
            (1, 2 * profile.intermediate_size, profile.hidden_size // 2),
            0x22,
            dtype=torch.uint8,
            device="cuda",
        )
        w2 = torch.full(
            (1, profile.hidden_size, profile.intermediate_size // 2),
            0x22,
            dtype=torch.uint8,
            device="cuda",
        )
        w13_scale = torch.full(
            (1, 2 * profile.intermediate_size, profile.hidden_size // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        )
        w2_scale = torch.full(
            (1, profile.hidden_size, profile.intermediate_size // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        )
        quant_info = SharedEpQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=(1, 32),
            fallback_quant_info=Mock(),
            fallback_backend=MoeRunnerBackend.AITER,
            quantization=SharedEpQuantization.MXFP4,
            weight_group_size=32,
            scale_format="e8m0",
            capabilities=frozenset({SharedEpQuantCapability.CANONICAL_MXFP4}),
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.tensor(
                [[0.6, 0.4]],
                dtype=torch.float32,
                device="cuda",
            ),
            topk_ids=torch.tensor(
                [[self.rank, (self.rank + 1) % 8]],
                dtype=torch.int32,
                device="cuda",
            ),
            router_logits=None,
        )

        previous = None
        try:
            from sglang.srt.layers.moe.shared_ep.fp4 import (
                publish_bf16_owner_input,
            )

            for generation in (1, 4):
                hidden = torch.full(
                    (1, profile.hidden_size),
                    (self.rank + 1) * generation / 1024,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                publish_bf16_owner_input(
                    state.local_input,
                    source=hidden,
                    source_ids=topk_output.topk_ids,
                    source_weights=topk_output.topk_weights,
                )
                state.local_output[:1].zero_()
                state.input_epoch.publish()
                actual = run_shared_ep(
                    SharedEpDispatchOutput(
                        hidden_states=state.global_input.activations,
                        hidden_states_scale=None,
                        topk_output=topk_output,
                        state=state,
                        profile=profile,
                        num_tokens=1,
                        local_expert_start=self.rank,
                    ),
                    quant_info,
                    runner_config,
                ).hidden_states
                self.assertEqual(tuple(actual.shape), (1, profile.hidden_size))
                self.assertTrue(torch.isfinite(actual).all().item())
                self.assertGreater(actual.abs().max().item(), 0)
                if previous is not None:
                    self.assertFalse(torch.equal(actual, previous))
                previous = actual.clone()
        finally:
            dist.barrier()
            state.close()

    @unittest.skipUnless(torch.version.hip is not None, "requires ROCm")
    def test_dsv4_pro_mxfp4_production_shape_ep8(self):
        arch = torch.cuda.get_device_properties(self.local_rank).gcnArchName.split(
            ":",
            1,
        )[0]
        if arch != "gfx950":
            self.skipTest(f"requires gfx950, got {arch}")

        profile = DSV4_PRO_MXFP4_GFX950
        runner_config = MoeRunnerConfig(
            num_experts=profile.num_experts,
            num_local_experts=profile.num_local_experts,
            hidden_size=profile.hidden_size,
            intermediate_size_per_partition=profile.intermediate_size,
            top_k=profile.top_k,
        )
        state = create_shared_ep_state(
            layout=SharedEpLayout.build(
                hidden_size=profile.hidden_size,
                top_k=profile.top_k,
                max_tokens_per_rank=profile.max_tokens_per_rank,
                input_format=SharedEpInputFormat.BF16,
                direct_output=True,
            ),
            cpu_group=dist.group.WORLD,
            device=torch.device("cuda", self.local_rank),
        )
        w13 = torch.full(
            (
                profile.num_local_experts,
                2 * profile.intermediate_size,
                profile.hidden_size // 2,
            ),
            0x22,
            dtype=torch.uint8,
            device="cuda",
        )
        w2 = torch.full(
            (
                profile.num_local_experts,
                profile.hidden_size,
                profile.intermediate_size // 2,
            ),
            0x22,
            dtype=torch.uint8,
            device="cuda",
        )
        w13_scale = torch.full(
            (
                profile.num_local_experts,
                2 * profile.intermediate_size,
                profile.hidden_size // 32,
            ),
            127,
            dtype=torch.uint8,
            device="cuda",
        )
        w2_scale = torch.full(
            (
                profile.num_local_experts,
                profile.hidden_size,
                profile.intermediate_size // 32,
            ),
            127,
            dtype=torch.uint8,
            device="cuda",
        )
        quant_info = SharedEpQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=(1, 32),
            fallback_quant_info=Mock(),
            fallback_backend=MoeRunnerBackend.AITER,
            quantization=SharedEpQuantization.MXFP4,
            weight_group_size=32,
            scale_format="e8m0",
            capabilities=frozenset({SharedEpQuantCapability.CANONICAL_MXFP4}),
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.full(
                (1, profile.top_k),
                1 / profile.top_k,
                dtype=torch.float32,
                device="cuda",
            ),
            topk_ids=torch.tensor(
                [
                    [
                        ((self.rank + slot) % profile.ep_size)
                        * profile.num_local_experts
                        for slot in range(profile.top_k)
                    ]
                ],
                dtype=torch.int32,
                device="cuda",
            ),
            router_logits=None,
        )

        try:
            from sglang.srt.layers.moe.shared_ep.fp4 import (
                publish_bf16_owner_input,
            )

            for generation in range(1, 81):
                hidden = torch.full(
                    (1, profile.hidden_size),
                    (self.rank + 1) * ((generation - 1) % 4 + 1) / 4096,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                publish_bf16_owner_input(
                    state.local_input,
                    source=hidden,
                    source_ids=topk_output.topk_ids,
                    source_weights=topk_output.topk_weights,
                )
                state.local_output[:1].zero_()
                state.input_epoch.publish()
                actual = run_shared_ep(
                    SharedEpDispatchOutput(
                        hidden_states=state.global_input.activations,
                        hidden_states_scale=None,
                        topk_output=topk_output,
                        state=state,
                        profile=profile,
                        num_tokens=1,
                        local_expert_start=self.rank * profile.num_local_experts,
                    ),
                    quant_info,
                    runner_config,
                ).hidden_states
                self.assertEqual(tuple(actual.shape), (1, profile.hidden_size))
                self.assertTrue(torch.isfinite(actual).all().item())
                self.assertGreater(actual.abs().max().item(), 0)

            idle_tokens = 1 if self.rank == 0 else 0
            idle_hidden = torch.full(
                (idle_tokens, profile.hidden_size),
                1 / 1024,
                dtype=torch.bfloat16,
                device="cuda",
            )
            idle_ids = topk_output.topk_ids[:idle_tokens]
            idle_weights = topk_output.topk_weights[:idle_tokens]
            idle_topk = StandardTopKOutput(
                topk_weights=idle_weights,
                topk_ids=idle_ids,
                router_logits=None,
            )
            publish_bf16_owner_input(
                state.local_input,
                source=idle_hidden,
                source_ids=idle_ids,
                source_weights=idle_weights,
            )
            state.local_output[:idle_tokens].zero_()
            state.input_epoch.publish()
            idle_output = run_shared_ep(
                SharedEpDispatchOutput(
                    hidden_states=state.global_input.activations,
                    hidden_states_scale=None,
                    topk_output=idle_topk,
                    state=state,
                    profile=profile,
                    num_tokens=idle_tokens,
                    local_expert_start=self.rank * profile.num_local_experts,
                ),
                quant_info,
                runner_config,
            ).hidden_states
            self.assertEqual(
                tuple(idle_output.shape),
                (idle_tokens, profile.hidden_size),
            )
            self.assertTrue(torch.isfinite(idle_output).all().item())
        finally:
            del quant_info, w13, w2, w13_scale, w2_scale
            dist.barrier()
            state.close()

    def _hidden(self, generation: int, tokens: int) -> torch.Tensor:
        values = torch.arange(
            tokens * 128,
            dtype=torch.float32,
            device="cuda",
        ).view(tokens, 128)
        values = (values + self.rank * 17 + generation * 29) % 97
        return ((values - 48) / 4096).to(torch.bfloat16)

    def _run_shared(
        self,
        state,
        hidden,
        topk_output,
        quant_info,
        *,
        skew_input_writer=False,
        skew_output_consumer=False,
        phase="decode",
        pull_cache=None,
    ) -> torch.Tensor:
        if skew_input_writer and self.rank == 0:
            torch.cuda._sleep(250_000_000)
        quantize_pack_input(
            state.local_input,
            source=hidden,
            source_ids=topk_output.topk_ids,
            source_weights=topk_output.topk_weights,
            group_size=128,
        )
        state.input_epoch.publish()
        dispatch_output = SharedEpDispatchOutput(
            hidden_states=state.global_input.activations,
            hidden_states_scale=state.global_input.scales,
            topk_output=topk_output,
            state=state,
            profile=self.profile,
            num_tokens=hidden.shape[0],
            local_expert_start=self.rank,
            phase=phase,
            pull_cache=pull_cache,
        )
        if not skew_output_consumer:
            return run_shared_ep(
                dispatch_output,
                quant_info,
                self.runner_config,
            ).hidden_states

        output_epoch_type = type(state.output_epoch)
        original_wait_all = output_epoch_type.wait_all

        def wait_all_with_rank_zero_skew(epoch):
            original_wait_all(epoch)
            if epoch is state.output_epoch and self.rank == 0:
                torch.cuda._sleep(250_000_000)

        with patch.object(output_epoch_type, "wait_all", wait_all_with_rank_zero_skew):
            return run_shared_ep(
                dispatch_output,
                quant_info,
                self.runner_config,
            ).hidden_states

    def _run_materialized(
        self,
        hidden_fp8,
        hidden_scale,
        topk_output,
        w13,
        w2,
        w13_scale,
        w2_scale,
        num_tokens,
    ) -> torch.Tensor:
        sorted_ids, expert_ids, padded = moe_align_block_size(
            topk_output.topk_ids,
            _KERNEL_CONFIG["BLOCK_SIZE_M"],
            8,
        )
        route_count = topk_output.topk_ids.numel()
        gate_up = torch.empty(
            (route_count, 512),
            dtype=torch.bfloat16,
            device="cuda",
        )
        invoke_fused_moe_kernel(
            A=hidden_fp8,
            B=w13,
            bias=None,
            C=gate_up,
            A_scale=hidden_scale,
            B_scale=w13_scale,
            B_zp=None,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=padded,
            mul_routed_weight=False,
            top_k=6,
            config=_KERNEL_CONFIG,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=[128, 128],
            filter_expert=False,
            a_is_prequantized=True,
        )
        down_fp8 = torch.empty(
            (route_count, 256),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        down_scale = torch.empty(
            (route_count, 2),
            dtype=torch.float32,
            device="cuda",
        )
        silu_and_mul_contig_post_quant(
            input=gate_up,
            output=down_fp8,
            output_scale=down_scale,
            quant_group_size=128,
        )
        contributions = torch.empty(
            (route_count, 128),
            dtype=torch.bfloat16,
            device="cuda",
        )
        invoke_fused_moe_kernel(
            A=down_fp8,
            B=w2,
            bias=None,
            C=contributions,
            A_scale=down_scale,
            B_scale=w2_scale,
            B_zp=None,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=padded,
            mul_routed_weight=True,
            top_k=1,
            config=_KERNEL_CONFIG,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=[128, 128],
            filter_expert=False,
            a_is_prequantized=True,
        )
        return contributions.view(num_tokens, 6, 128).sum(dim=1)


if __name__ == "__main__":
    if "WORLD_SIZE" in os.environ:
        unittest.main()
    else:
        multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
