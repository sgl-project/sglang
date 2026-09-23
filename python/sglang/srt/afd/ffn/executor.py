"""FFN-side transport execution for eager AFD steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.afd.afd_sync import AFDSyncMsg
from sglang.srt.afd.comm import AFDCommBackend
from sglang.srt.afd.ffn.compute import AFDFFNCompute
from sglang.srt.layers.dp_attention import set_is_extend_in_batch
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class AFDFFNExecutor:
    def __init__(
        self,
        scheduler: Scheduler,
        comm: AFDCommBackend,
        compute: AFDFFNCompute,
    ):
        self._comm = comm
        self._compute = compute
        self._device = torch.device(comm.config.device)
        self._dtype = scheduler.tp_worker.model_runner.dtype
        self._layers = compute.layers
        (attn_rank,) = comm.config.attn_ranks_for_ffn(comm.config.instance_rank)
        self._attn_global_rank = comm.config.attn_global_rank(attn_rank)
        if compute.device != self._device:
            raise RuntimeError(
                f"AFD ffn compute device {compute.device} does not match "
                f"communication device {self._device}"
            )
        tp_group = scheduler.tp_group
        if (
            tp_group.world_size != comm.config.num_ffn_ranks
            or tp_group.rank_in_group != comm.config.instance_rank
        ):
            raise RuntimeError(
                "AFD ffn rank topology does not match the local TP group"
            )

    @torch.inference_mode()
    def run_step(self, msg: AFDSyncMsg) -> None:
        if len(msg.tokens_per_rank) != self._comm.config.num_attn_ranks:
            raise RuntimeError(
                f"AFD sync tokens_per_rank has {len(msg.tokens_per_rank)} entries, "
                f"but the attn instance has {self._comm.config.num_attn_ranks} ranks"
            )
        num_tokens = sum(msg.tokens_per_rank)
        if msg.wire_capacity != num_tokens:
            raise RuntimeError(
                "AFD eager execution requires exact-size transport: "
                f"tokens={num_tokens}, wire={msg.wire_capacity}"
            )
        forward_mode = ForwardMode(msg.forward_mode)
        if forward_mode.is_idle() and num_tokens != 0:
            raise RuntimeError("AFD idle step must have zero capacity")
        if num_tokens == 0:
            return
        set_is_extend_in_batch(forward_mode.is_extend())

        for layer_index, layer in enumerate(self._layers):
            hidden_states = torch.empty(
                (num_tokens, layer.hidden_size), dtype=self._dtype, device=self._device
            )
            topk_ids = torch.empty(
                (num_tokens, layer.top_k), dtype=torch.int32, device=self._device
            )
            topk_weights = torch.empty(
                (num_tokens, layer.top_k), dtype=torch.float32, device=self._device
            )
            # STANDARD routing is already materialized by A; runners use the
            # logits shape for metadata without reading its values.
            router_logits = torch.empty(
                (num_tokens, layer.num_experts),
                dtype=torch.float32,
                device=self._device,
            )
            self._comm.recv_from(
                [hidden_states, topk_ids, topk_weights], self._attn_global_rank
            )
            output = self._compute.run(
                layer_index, hidden_states, topk_ids, topk_weights, router_logits
            )
            self._comm.send_to([output.contiguous()], self._attn_global_rank)
