import os
from unittest.mock import patch

import msgspec
import pytest
import torch
import torch.distributed as dist
import triton.language as tl

from sglang.kernels.ops.attention.dsv4 import silu_and_mul_contig_post_quant
from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.layers.moe.shared_ep.backend import (
    SharedEpDispatchOutput,
    run_shared_ep,
)
from sglang.srt.layers.moe.shared_ep.kernels import quantize_pack_input
from sglang.srt.layers.moe.shared_ep.layout import SharedEpLayout
from sglang.srt.layers.moe.shared_ep.profiles import (
    DSV4_FLASH,
    SharedEpProfile,
    make_pull_cache_prefill_profile,
)
from sglang.srt.layers.moe.shared_ep.pull_cache_prefill import (
    allocate_pull_cache,
    make_pull_cache_prefill_plan,
)
from sglang.srt.layers.moe.shared_ep.state import create_shared_ep_state
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=40, stage="extra-b", runner_config="8-gpu-h200")

_KERNEL_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 3,
}


@pytest.mark.skipif(
    not (torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) == 8),
    reason="run with torchrun --nproc-per-node=8 on CUDA",
)
class TestSharedEpEp8:
    @classmethod
    def setup_class(cls):
        cls.rank = int(os.environ["RANK"])
        cls.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(cls.local_rank)
        dist.init_process_group("gloo")
        cls.profile = SharedEpProfile(
            name="ep8_test",
            capability=(9, 0),
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
        )
        cls.runner_config = MoeRunnerConfig(
            num_experts=8,
            num_local_experts=1,
            hidden_size=128,
            intermediate_size_per_partition=256,
            top_k=6,
        )

    @classmethod
    def teardown_class(cls):
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
            pull_plan = make_pull_cache_prefill_plan(
                owners=8,
                source_tokens_per_owner=32,
                hidden_size=128,
                top_k=6,
                num_local_experts=1,
                expert_alignment=16,
            )
            pull_cache = allocate_pull_cache(
                pull_plan,
                active_rows=pull_plan.cache_rows,
                device=torch.device("cuda", self.local_rank),
            )
            assert state.layout.output_row_bytes == 32 * 1024
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
            local_quant = TritonMoeQuantInfo(
                w13_weight=full_w13[self.rank : self.rank + 1],
                w2_weight=full_w2[self.rank : self.rank + 1],
                use_fp8_w8a8=True,
                w13_scale=full_w13_scale[self.rank : self.rank + 1],
                w2_scale=full_w2_scale[self.rank : self.rank + 1],
                block_shape=[128, 128],
            )

            for generation in (1, 2):
                hidden = self._hidden(generation, tokens)
                hidden_fp8, hidden_scale = sglang_per_token_group_quant_fp8(
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
            graph_q, graph_scale = sglang_per_token_group_quant_fp8(
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

    def test_dsv4_profile_m64_prefill_matches_materialized_ep8(self):
        profile = make_pull_cache_prefill_profile(
            DSV4_FLASH,
            max_tokens_per_rank=1024,
        )
        assert profile is not None
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
            ),
            cpu_group=dist.group.WORLD,
            device=torch.device("cuda", self.local_rank),
        )
        try:
            pull_plan = make_pull_cache_prefill_plan(
                owners=profile.ep_size,
                source_tokens_per_owner=profile.max_tokens_per_rank,
                hidden_size=profile.hidden_size,
                top_k=profile.top_k,
                num_local_experts=profile.num_local_experts,
                expert_alignment=profile.block_size_m,
            )
            pull_cache = allocate_pull_cache(
                pull_plan,
                active_rows=pull_plan.cache_rows,
                device=torch.device("cuda", self.local_rank),
            )
            tokens = 1
            topk_ids = torch.stack(
                [
                    (
                        (
                            torch.arange(
                                profile.top_k,
                                device="cuda",
                                dtype=torch.int32,
                            )
                            + self.rank
                            + token
                        )
                        % profile.ep_size
                    )
                    * profile.num_local_experts
                    + (
                        (
                            torch.arange(
                                profile.top_k,
                                device="cuda",
                                dtype=torch.int32,
                            )
                            * 5
                        )
                        + self.rank * 7
                        + token * 3
                    )
                    % profile.num_local_experts
                    for token in range(tokens)
                ]
            )
            topk_weights = (
                torch.arange(
                    1,
                    profile.top_k + 1,
                    device="cuda",
                    dtype=torch.float32,
                )
                .div_(sum(range(1, profile.top_k + 1)))
                .expand(tokens, -1)
                .contiguous()
            )
            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=None,
            )
            reference_topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=torch.arange(
                    profile.top_k,
                    device="cuda",
                    dtype=torch.int32,
                ).view(1, -1),
                router_logits=None,
            )
            reference_profile = msgspec.structs.replace(
                DSV4_FLASH,
                num_experts=profile.top_k,
                num_local_experts=profile.top_k,
            )

            local_w13 = torch.empty(
                (
                    profile.num_local_experts,
                    2 * profile.intermediate_size,
                    profile.hidden_size,
                ),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            local_w2 = torch.empty(
                (
                    profile.num_local_experts,
                    profile.hidden_size,
                    profile.intermediate_size,
                ),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            local_expert_start = self.rank * profile.num_local_experts
            for local_expert in range(profile.num_local_experts):
                value = (local_expert_start + local_expert + 1) / 4096
                local_w13[local_expert].fill_(value)
                local_w2[local_expert].fill_(value)
            local_w13_scale = torch.ones(
                (
                    profile.num_local_experts,
                    2 * profile.intermediate_size // 128,
                    profile.hidden_size // 128,
                ),
                dtype=torch.float32,
                device="cuda",
            )
            local_w2_scale = torch.ones(
                (
                    profile.num_local_experts,
                    profile.hidden_size // 128,
                    profile.intermediate_size // 128,
                ),
                dtype=torch.float32,
                device="cuda",
            )
            reference_w13 = torch.empty(
                (
                    profile.top_k,
                    2 * profile.intermediate_size,
                    profile.hidden_size,
                ),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            reference_w2 = torch.empty(
                (
                    profile.top_k,
                    profile.hidden_size,
                    profile.intermediate_size,
                ),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            for reference_expert, global_expert in enumerate(topk_ids[0].tolist()):
                value = (global_expert + 1) / 4096
                reference_w13[reference_expert].fill_(value)
                reference_w2[reference_expert].fill_(value)
            reference_w13_scale = torch.ones(
                (
                    profile.top_k,
                    2 * profile.intermediate_size // 128,
                    profile.hidden_size // 128,
                ),
                dtype=torch.float32,
                device="cuda",
            )
            reference_w2_scale = torch.ones(
                (
                    profile.top_k,
                    profile.hidden_size // 128,
                    profile.intermediate_size // 128,
                ),
                dtype=torch.float32,
                device="cuda",
            )
            local_quant = TritonMoeQuantInfo(
                w13_weight=local_w13,
                w2_weight=local_w2,
                use_fp8_w8a8=True,
                w13_scale=local_w13_scale,
                w2_scale=local_w2_scale,
                block_shape=[128, 128],
            )

            generation = 1
            with self.subTest(generation=generation):
                hidden = self._hidden(
                    generation,
                    tokens,
                    hidden_size=profile.hidden_size,
                )
                hidden_fp8, hidden_scale = sglang_per_token_group_quant_fp8(
                    hidden,
                    128,
                )
                expected = self._run_materialized(
                    hidden_fp8,
                    hidden_scale,
                    reference_topk_output,
                    reference_w13,
                    reference_w2,
                    reference_w13_scale,
                    reference_w2_scale,
                    tokens,
                    profile=reference_profile,
                )
                actual = self._run_shared(
                    state,
                    hidden,
                    topk_output,
                    local_quant,
                    phase="prefill",
                    pull_cache=pull_cache,
                    profile=profile,
                    runner_config=runner_config,
                )
                # Expert-local execution changes the FP8/BF16 reduction order
                # relative to the materialized all-expert reference.
                actual_fp32 = actual.float()
                expected_fp32 = expected.float()
                difference = (actual_fp32 - expected_fp32).abs()
                tolerance = 1e-7 + 0.05 * expected_fp32.abs()
                max_excess = (difference - tolerance).amax().item()
                max_absolute = difference.amax().item()
                assert max_excess <= 0, f"{max_absolute=} {max_excess=}"
        finally:
            dist.barrier()
            state.close()

    def _hidden(
        self,
        generation: int,
        tokens: int,
        *,
        hidden_size: int = 128,
    ) -> torch.Tensor:
        values = torch.arange(
            tokens * hidden_size,
            dtype=torch.float32,
            device="cuda",
        ).view(tokens, hidden_size)
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
        profile=None,
        runner_config=None,
    ) -> torch.Tensor:
        profile = self.profile if profile is None else profile
        runner_config = self.runner_config if runner_config is None else runner_config
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
            profile=profile,
            num_tokens=hidden.shape[0],
            local_expert_start=self.rank * profile.num_local_experts,
            phase=phase,
            pull_cache=pull_cache,
        )
        if not skew_output_consumer:
            return run_shared_ep(
                dispatch_output,
                quant_info,
                runner_config,
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
                runner_config,
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
        *,
        profile=None,
    ) -> torch.Tensor:
        profile = self.profile if profile is None else profile
        kernel_config = profile.kernel_config(num_tokens)
        sorted_ids, expert_ids, padded = moe_align_block_size(
            topk_output.topk_ids,
            kernel_config["BLOCK_SIZE_M"],
            profile.num_experts,
        )
        route_count = topk_output.topk_ids.numel()
        gate_up = torch.empty(
            (route_count, 2 * profile.intermediate_size),
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
            top_k=profile.top_k,
            config=profile.w13_kernel_config(num_tokens),
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=list(profile.block_shape),
            filter_expert=False,
            a_is_prequantized=True,
        )
        down_fp8 = torch.empty(
            (route_count, profile.intermediate_size),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        down_scale = torch.empty(
            (route_count, profile.intermediate_size // 128),
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
            (route_count, profile.hidden_size),
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
            config=profile.w2_kernel_config(num_tokens),
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=list(profile.block_shape),
            filter_expert=False,
            a_is_prequantized=True,
        )
        return contributions.view(
            num_tokens,
            profile.top_k,
            profile.hidden_size,
        ).sum(dim=1)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
