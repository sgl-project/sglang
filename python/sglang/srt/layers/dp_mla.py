from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.dp_attention import (
    dp_gather_replicate,
    dp_scatter,
    get_attention_dp_rank,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class DpMlaWrapper:
    def __init__(self):
        self.enable_dp_mla = global_server_args_dict["enable_dp_mla"]
        if not self.enable_dp_mla:
            return
        self.group = get_tp_group()
        self.rank_in_node = get_tensor_model_parallel_rank()
        self.world_size_in_node = get_tensor_model_parallel_world_size()
        self.dp_tp_size = get_attention_tp_size()
        self.dp_tp_rank = get_attention_tp_rank()
        self.dp_size_in_node = get_attention_dp_size()
        self.dp_rank_in_node = get_attention_dp_rank()
        self.mla_mask = None
        self.gathered_buffer_tp2dp = None
        self.gathered_buffer_dp2tp = None

    def get_tp_to_dp_plan_meta(
        self,
        forward_batch: "ForwardBatch",
        output,
        input,
        chunk_size,
    ):
        if forward_batch.dp_mla_tp2dp_plan_meta is None:
            dp_size = self.dp_size_in_node  # 8//4=2
            world_size = self.world_size_in_node
            dp_tp_rank = self.dp_tp_rank

            all_lens = forward_batch.global_num_tokens_gpu.to(torch.int64)
            new_seq = all_lens[self.dp_rank_in_node]
            # dp0~3[h0,h1,h2,h3,h4,h5,h6,h7] -> [dp0[h0~3],dp0[h4~7],dp1[h0~3],dp1[h4~7],...]
            # recv from every rank
            output_split_sizes = torch.zeros(
                world_size, dtype=torch.int64, device=all_lens.device
            )
            seq_with_pad = forward_batch.gathered_buffer_tp2dp.shape[0]
            output_split_sizes[dp_tp_rank * dp_size : (dp_tp_rank + 1) * dp_size] = (
                new_seq
            )
            output_split_offsets = torch.zeros(
                world_size, dtype=torch.int64, device=all_lens.device
            )
            if seq_with_pad > 0:
                output_split_offsets[
                    dp_tp_rank * dp_size : (dp_tp_rank + 1) * dp_size
                ] = torch.arange(
                    0,
                    seq_with_pad * dp_size,
                    seq_with_pad,
                    dtype=torch.int64,
                    device=all_lens.device,
                )

            # send to every rank
            input_split_sizes = all_lens.unsqueeze(-1) * self.mla_mask
            input_split_sizes = input_split_sizes.view(-1)

            forward_batch.dp_mla_tp2dp_plan_meta = self.group.custom_all_to_all_plan(
                output,
                input,
                output_split_sizes,
                input_split_sizes,
                chunk_size=chunk_size,
                output_split_offsets=output_split_offsets,
            )

        return forward_batch.dp_mla_tp2dp_plan_meta

    @torch.compiler.disable
    def tp_to_dp(
        self,
        input_tensor: torch.Tensor,
        forward_batch: "ForwardBatch",
    ):
        world_size = self.world_size_in_node
        dp_size = self.dp_size_in_node

        # input_tensor: [seq0+seq1+seq2+..., head/8, head_dim] -> [seq0, head/2, head_dim], [seq1, head/2, head_dim]
        if world_size == 1:
            return input_tensor
        # seq, 16, (128+64) or 576
        total_seq, local_head, seq_head_dim = input_tensor.shape
        new_head = local_head * dp_size

        output_tensor = forward_batch.gathered_buffer_tp2dp

        chunk_size = local_head * seq_head_dim
        input_tensor = input_tensor.view(-1, 1, chunk_size)
        output_tensor = output_tensor.view(-1, dp_size, chunk_size)
        # to dp_size, seq+seq_pad, chunk_size
        output_tensor_t = output_tensor.transpose(0, 1)
        plan_meta = self.get_tp_to_dp_plan_meta(
            forward_batch,
            output_tensor_t,
            input_tensor,
            chunk_size,
        )
        self.group.custom_all_to_all(output_tensor_t, input_tensor, plan_meta, "tp2dp")
        return output_tensor.view(-1, new_head, seq_head_dim)

    def get_dp_to_tp_plan_meta(
        self,
        forward_batch: "ForwardBatch",
        output,
        input,
        chunk_size,
    ):
        if forward_batch.dp_mla_dp2tp_plan_meta is None:
            world_size = self.world_size_in_node
            dp_size = self.dp_size_in_node  # 8//2==4
            dp_tp_rank = self.dp_tp_rank

            all_lens = forward_batch.global_num_tokens_gpu.to(torch.int64)
            cur_seq = all_lens[self.dp_rank_in_node]

            # [dp0[h0~3],dp0[h4~7],dp1[h0~3],dp1[h4~7],...] -> dp0~3[h0,h1,h2,h3,h4,h5,h6,h7]
            # recv from every rank
            output_split_sizes = all_lens.unsqueeze(-1) * self.mla_mask
            output_split_sizes = output_split_sizes.view(-1)
            # send to every rank
            input_split_sizes = torch.zeros(
                world_size, dtype=torch.int64, device=all_lens.device
            )
            input_split_sizes[dp_tp_rank * dp_size : (dp_tp_rank + 1) * dp_size] = (
                cur_seq
            )
            seq_with_pad = forward_batch.gathered_buffer_tp2dp.shape[0]
            input_split_offsets = torch.zeros(
                world_size, dtype=torch.int64, device=all_lens.device
            )
            if seq_with_pad > 0:
                input_split_offsets[
                    dp_tp_rank * dp_size : (dp_tp_rank + 1) * dp_size
                ] = torch.arange(
                    0,
                    seq_with_pad * dp_size,
                    seq_with_pad,
                    dtype=torch.int64,
                    device=all_lens.device,
                )
            forward_batch.dp_mla_dp2tp_plan_meta = self.group.custom_all_to_all_plan(
                output,
                input,
                output_split_sizes,
                input_split_sizes,
                chunk_size=chunk_size,
                input_split_offsets=input_split_offsets,
            )
        return forward_batch.dp_mla_dp2tp_plan_meta

    @torch.compiler.disable
    def dp_to_tp(
        self,
        input_tensor: torch.Tensor,
        forward_batch: "ForwardBatch",
    ):
        world_size = self.world_size_in_node
        dp_tp_size = self.dp_tp_size

        if world_size == 1:
            return input_tensor
        # input_tensor: [seq_r0, head/2, 512], [seq_r1, head/2, 512], ... -> [seq_r0+seq_r1+seq_r2+...,head/8, 512]
        _, local_head, seq_head_dim = input_tensor.shape

        dp_size = world_size // dp_tp_size
        new_head = local_head // dp_size

        output_tensor = forward_batch.gathered_buffer_dp2tp

        chunk_size = new_head * seq_head_dim
        output_tensor = output_tensor.view(-1, 1, chunk_size)
        input_tensor = input_tensor.view(-1, dp_size, chunk_size)
        # to [dp_size, seq+seq_pad, new_head, seq_head_dim)
        input_tensor_t = input_tensor.transpose(0, 1)
        plan_meta = self.get_dp_to_tp_plan_meta(
            forward_batch,
            output_tensor,
            input_tensor_t,
            chunk_size,
        )
        self.group.custom_all_to_all(output_tensor, input_tensor_t, plan_meta, "dp2tp")
        return output_tensor.view(-1, new_head, seq_head_dim)

    def mla_inputs_tp_to_dp(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.enable_dp_mla:
            if q_pe is not None:
                last_dims = [q_nope.shape[-1], q_pe.shape[-1]]
                q_nope = torch.cat([q_nope, q_pe], dim=-1)
            q_nope = self.tp_to_dp(
                q_nope.contiguous(),
                forward_batch,
            )
            if q_pe is not None:
                q_nope, q_pe = q_nope.split(last_dims, dim=-1)

            hidden_states = self._scatter_hidden_states(
                hidden_states.contiguous(),
                forward_batch,
            )
        return q_nope, q_pe, hidden_states

    def _scatter_hidden_states(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if forward_batch.can_run_dp_cuda_graph:
            first_dim = forward_batch.input_ids.shape[0]
            last_dim = hidden_states.shape[-1]
            hidden_states, global_hidden_states = (
                forward_batch.gathered_buffer.view(-1)[: first_dim * last_dim].view(
                    first_dim, last_dim
                ),
                hidden_states,
            )
            dp_scatter(
                hidden_states,
                global_hidden_states,
                forward_batch,
            )
        else:
            all_lens = forward_batch.global_num_tokens_cpu
            start_pos = sum(all_lens[: self.dp_rank_in_node])
            end_pos = start_pos + all_lens[self.dp_rank_in_node]
            hidden_states = hidden_states[start_pos:end_pos]
        return hidden_states

    def mla_outputs_dp_to_tp(
        self,
        attn_output: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.enable_dp_mla:
            attn_output = self.dp_to_tp(
                attn_output,
                forward_batch,
            )
        return attn_output

    def model_inputs_dp_to_tp(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.enable_dp_mla and forward_batch.gathered_buffer.shape[0] != 0:
            input_ids, local_input_ids = (
                torch.empty(
                    (forward_batch.gathered_buffer.shape[0],),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
                input_ids,
            )
            dp_gather_replicate(input_ids, local_input_ids, forward_batch)
        return input_ids

    def model_outputs_tp_to_dp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.enable_dp_mla:
            hidden_states = self._scatter_hidden_states(
                hidden_states,
                forward_batch,
            )
        return hidden_states

    def init_gathered_buffer(self, runner: "ModelRunner"):
        if self.enable_dp_mla:
            self.get_gathered_buffer_dp2tp(runner)
            self.get_gathered_buffer_dp2tp(runner)

    def get_gathered_buffer_tp2dp(self, runner: "ModelRunner"):
        if self.gathered_buffer_tp2dp is None:
            config = runner.model_config
            server_args = runner.server_args
            dp_mla_tp_size = server_args.tp_size // server_args.dp_size
            num_dp_key_value_heads = config.num_key_value_heads // dp_mla_tp_size
            q_dim = config.kv_lora_rank + config.qk_rope_head_dim
            self.gathered_buffer_tp2dp = torch.empty(
                (
                    server_args.chunked_prefill_size,
                    num_dp_key_value_heads,
                    q_dim,
                ),
                dtype=config.dtype,
                device=runner.device,
            )
            self.group.register_output_buffer(self.gathered_buffer_tp2dp, "tp2dp")

            mla_mask = torch.zeros(
                self.dp_tp_size, dtype=torch.int64, device=runner.device
            )
            # to dp0~n tp dp_tp_rank
            mla_mask[self.rank_in_node // self.dp_size_in_node] = 1
            self.mla_mask = mla_mask
        return self.gathered_buffer_tp2dp

    def get_gathered_buffer_dp2tp(self, runner: "ModelRunner"):
        if self.gathered_buffer_dp2tp is None:
            config = runner.model_config
            server_args = runner.server_args
            num_tp_key_value_heads = (
                config.num_key_value_heads // self.world_size_in_node
            )
            kv_lora_rank = config.kv_lora_rank

            sum_len = server_args.chunked_prefill_size * self.dp_size_in_node
            self.gathered_buffer_dp2tp = torch.empty(
                (
                    sum_len,
                    num_tp_key_value_heads,
                    kv_lora_rank,
                ),
                dtype=config.dtype,
                device=runner.device,
            )
            self.group.register_output_buffer(self.gathered_buffer_dp2tp, "dp2tp")
        return self.gathered_buffer_dp2tp


_DP_MLA_WRAPPER_INSTANCE: Optional[DpMlaWrapper] = None


def get_dp_mla_wrapper() -> DpMlaWrapper:
    global _DP_MLA_WRAPPER_INSTANCE
    if _DP_MLA_WRAPPER_INSTANCE is None:
        _DP_MLA_WRAPPER_INSTANCE = DpMlaWrapper()
    return _DP_MLA_WRAPPER_INSTANCE
