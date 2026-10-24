"""Layer-owned quantized latent shards for the merged Kimi front projection."""

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.all_reduce import Communicator, PullPlane
from sglang.kernels.ops.moe.moe_route_quant_fused import (
    QuantGather,
    available,
    route_quant_fused,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    _SEMAPHORE_BYTES,
    _allocate_symmetric_memory,
)
from sglang.srt.runtime_context import get_buffer

# Serving wins measured for single-token decode and an eight-token verify block.
# The gather kernel itself takes a runtime row count.
FRONT_SHARD_ROWS = (1, 8)


def sharded_front_available(group, hidden_size: int) -> bool:
    def check():
        # All TP ranks must select the same collective path.
        ready = torch.tensor(
            [available(group.world_size, hidden_size)], dtype=torch.int32, device="cpu"
        )
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group.cpu_group)
        return bool(ready.item())

    return get_buffer(
        f"k3_front_gather_available_{group.unique_name}_{hidden_size}", check
    )


class KimiK3FrontWorkspace:
    def __init__(self, group, num_layers: int, hidden_size: int, max_tokens: int):
        from flashinfer.fused_moe import core

        device = torch.device("cuda", torch.cuda.current_device())
        self.rank = group.rank_in_group
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        local_groups = hidden_size // group.world_size // 32
        scale_stride = (local_groups + 15) // 16 * 16
        self.scale_row_bytes = group.world_size * scale_stride
        scale_start = num_layers * max_tokens * hidden_size
        scale_end = scale_start + num_layers * max_tokens * self.scale_row_bytes
        sem_start = (
            (scale_end + _SEMAPHORE_BYTES - 1) // _SEMAPHORE_BYTES * _SEMAPHORE_BYTES
        )
        num_blocks = 2 * max_tokens
        total = sem_start + num_blocks * _SEMAPHORE_BYTES
        with torch.inference_mode(False):
            self.region, self.handle = _allocate_symmetric_memory(
                total, device, group.cpu_group
            )
        self.region[sem_start:].zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group.cpu_group)
        peers = [
            self.handle.get_buffer(i, [total], torch.uint8)[sem_start:].view(
                num_blocks, _SEMAPHORE_BYTES
            )
            for i in range(group.world_size)
        ]
        mc_base = int(self.handle.multicast_ptr)
        # Separate from the shared expert's overlapping AllReduce semaphore plane.
        self.comm = Communicator(
            pull=PullPlane(
                self.rank,
                group.world_size,
                semaphores=peers,
                mc_semaphore=mc_base + sem_start,
            )
        )
        self.quantized = (
            self.region[:scale_start]
            .view(torch.float8_e4m3fn)
            .view(num_layers, max_tokens, hidden_size)
        )
        self.scales = (
            self.region[scale_start:scale_end]
            .view(torch.int32)
            .view(num_layers, max_tokens, group.world_size, scale_stride // 4)
        )
        self.q_mc = mc_base
        self.scale_mc = mc_base + scale_start

        self._routing_slot_type = getattr(core, "TrtllmMoERoutingMetadataSlot", None)
        self.routing = None

    def route(self, layer_id, scores, bias, local_x, config):
        rows = scores.shape[0]
        assert rows <= self.max_tokens
        if rows == 1 and self.routing is None and self._routing_slot_type is not None:
            self.routing = SingleTokenRouting(
                scores.device, scores.shape[1], config.top_k, self._routing_slot_type
            )
        routing = self.routing if rows == 1 else None
        # Finalize/AllReduce fences the consumer before this layer reuses its slot.
        gather = QuantGather(
            self.comm,
            self.quantized[layer_id, :rows],
            self.scales[layer_id, :rows],
            self.q_mc + layer_id * self.max_tokens * self.hidden_size,
            self.scale_mc + layer_id * self.max_tokens * self.scale_row_bytes,
            routing.slot.expert_weights if routing else None,
            routing.batch if routing else None,
        )
        result = route_quant_fused(
            scores,
            bias,
            local_x,
            config.top_k,
            routed_scaling_factor=(
                config.routed_scaling_factor
                if config.routed_scaling_factor is not None
                else 1.0
            ),
            renormalize=config.renormalize,
            apply_scale=config.apply_routed_scaling_factor_on_output,
            gather=gather,
        )
        return result, routing


class SingleTokenRouting:
    """Reuse native MoE metadata when one row selects distinct, sorted experts.

    Only tile-N=8 uses the prepared layout; other autotuner choices retain the
    native preparation. Capacities follow the native one-row routing layout.
    """

    def __init__(self, device, num_experts, top_k, slot_type):
        def ints(n):
            return torch.empty(n, dtype=torch.int32, device=device)

        tile_n = 8
        padded = top_k * tile_n
        expanded = torch.arange(top_k, dtype=torch.int32, device=device) * tile_n
        permuted = torch.full((padded + 1,), -1, dtype=torch.int32, device=device)
        permuted[:padded:tile_n] = 0
        # FlashInfer's multi-tile allocator only accepts top_k <= 8. Its body
        # also accepts this one-row layout, with one tile per distinct expert.
        self.slot = slot_type(
            tile_n=tile_n,
            total_num_padded_tokens=ints(1).fill_(padded),
            expanded_idx_to_permuted_idx=expanded,
            permuted_idx_to_token_idx=permuted,
            expert_weights=torch.empty((1, top_k), dtype=torch.bfloat16, device=device),
            # Native routing ABI reserves at least 512 histogram entries.
            expert_count_histogram=ints(max(2 * num_experts, 512)),
            num_tokens_per_expert=ints(num_experts),
            cta_idx_xy_to_batch_idx=ints(top_k),
            cta_idx_xy_to_mn_limit=expanded + 1,
            num_non_exiting_ctas=ints(1).fill_(top_k),
        )
        self.batch = self.slot.cta_idx_xy_to_batch_idx
        self.workspaces = {}

    def __call__(self, **kwargs):
        from flashinfer.autotuner import AutoTuner
        from flashinfer.fused_moe import core

        x = kwargs["hidden_states"]
        scale = kwargs["hidden_states_scale"]
        native = core.get_trtllm_moe_sm100_module()
        runner = native.MoERunner(
            native.moe_op,
            top_k=kwargs["top_k"],
            num_local_experts=kwargs["local_num_experts"],
            dtype_act=core.deduce_trtllm_gen_tensor_dtype(x, scale),
            dtype_weights=core.deduce_trtllm_gen_tensor_dtype(
                kwargs["gemm1_weights"], kwargs["gemm1_weights_scale"]
            ),
            fp8_quantization_type=core.Fp8QuantizationType.NoneFp8,
            hidden_size=x.shape[1],
            intermediate_size=kwargs["intermediate_size"],
            activation_type=kwargs["activation_type"],
            use_shuffled_weight=True,
            num_experts=kwargs["num_experts"],
        )
        inputs = core.MoeRunnerInputs(
            output=kwargs["output"],
            routing_logits=None,
            topk_ids=kwargs["topk_ids"],
            expert_weights=self.slot.expert_weights,
            hidden_states=x,
            hidden_states_scale=scale,
            gemm1_lora_delta=None,
            per_token_scale=None,
        )
        tuning = runner._make_tuning_config(
            inputs,
            tune_max_num_tokens=1,
            use_cold_l2_cache=True,
            use_cuda_graph=True,
        )
        options = dict(kwargs)
        for name in (
            "hidden_states",
            "hidden_states_scale",
            "output",
            "topk_ids",
            "top_k",
            "local_num_experts",
            "intermediate_size",
            "tune_max_num_tokens",
        ):
            options.pop(name)
        options["routing_input_mode"] = core.RoutingInputMode.PackedPrecomputed
        _, tactic = AutoTuner.get().choose_one(
            "flashinfer::trtllm_fp4_block_scale_moe",
            [runner],
            tuning,
            inputs.to_list(),
            **options,
        )
        if tactic == -1 or tactic[0] != self.slot.tile_n:
            return core.trtllm_fp4_block_scale_routed_moe(**kwargs)
        key = (tuple(tactic), runner.get_cache_key_extras(inputs.to_list()))
        workspace = self.workspaces.get(key)
        if workspace is None:
            assert not torch.cuda.is_current_stream_capturing()
            workspace = tuple(
                core._torch_view_of_ffi_tensor(t)
                for t in runner.forward(
                    inputs.to_list(),
                    tactic=tactic,
                    do_preparation=True,
                    da_routing_metadata=self.slot.tensors(),
                    **options,
                )
            )
            self.workspaces[key] = workspace
        result = runner.forward(
            inputs.to_list(),
            tactic=tactic,
            da_routing_metadata=self.slot.tensors(),
            da_body_workspace=workspace,
            **options,
        )
        return [core._torch_view_of_ffi_tensor(t) for t in result]
