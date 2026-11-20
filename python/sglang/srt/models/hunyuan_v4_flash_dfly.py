import dataclasses
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.hy_v4_flash import HYV4FlashDFlyConfig, HYV4FlashTextConfig
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.layers.layer_boundary.adapters.attention import AttentionInputs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP
from sglang.srt.models.hunyuan_v4_flash import HYV4FlashAttention
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dfly_utils import DFlyRequestState, build_draft_topk
from sglang.srt.utils import BumpAllocator

CHECKPOINT_PREFIX = "model.dfly_layers."


class DFlyExternalIndexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.rows: Optional[torch.Tensor] = None

    def forward(
        self, x, q_lora, positions, forward_batch, layer_id, return_indices=True
    ):
        return self.rows[: q_lora.shape[0]]


class HYV4FlashDFlyLayer(nn.Module):
    def __init__(
        self,
        config: HYV4FlashTextConfig,
        index: int,
        target_layer_id: int,
        quant_config,
    ):
        super().__init__()
        self.prefix = f"{CHECKPOINT_PREFIX}{index}"
        self.quant_config = quant_config
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act
        self.indexer = DFlyExternalIndexer()
        # The target layer id makes RadixAttention address that layer's KV pages.
        self.self_attn = HYV4FlashAttention(
            config,
            target_layer_id,
            quant_config,
            f"{self.prefix}.self_attn",
            alt_stream=None,
            external_indexer=self.indexer,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp: Optional[DeepseekV2MLP] = None

    def materialize_mlp(self, width: int, device, dtype) -> None:
        if self.mlp is not None:
            if self.mlp.gate_up_proj.output_sizes[0] != width:
                raise ValueError("Inconsistent DFly FFN widths in the checkpoint")
            return
        if width % get_parallel().tp_size:
            raise ValueError("DFly FFN width must be divisible by TP")
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                self.mlp = DeepseekV2MLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=width,
                    hidden_act=self.hidden_act,
                    quant_config=self.quant_config,
                    reduce_results=True,
                    prefix=f"{self.prefix}.mlp",
                )
        finally:
            torch.set_default_dtype(previous)

    def forward(self, positions, hidden, residual, forward_batch, zero_allocator):
        if residual is not None:
            hidden = hidden + residual
        residual = hidden
        x = self.input_layernorm(hidden)
        context = get_attn_tp_context()
        context.set_attn_inputs(
            AttentionInputs(x, forward_batch, self.self_attn.prepare_qkv_latent)
        )
        try:
            x = self.self_attn(positions, x, forward_batch, zero_allocator)
        finally:
            context.clear_attn_inputs()
        hidden = x + residual
        residual = hidden
        return self.mlp(self.post_attention_layernorm(hidden)), residual


class HYV4FlashDFlyFusion(nn.Module):
    def __init__(self, config: HYV4FlashTextConfig):
        super().__init__()
        d = config.hidden_size
        self.enorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * d,
            d,
            bias=False,
            quant_config=None,
            prefix=f"{CHECKPOINT_PREFIX}mtp_fusion.eh_proj",
        )
        self.gate = nn.Parameter(torch.empty(d), requires_grad=False)

    def forward(self, anchors, target_hidden):
        fused, _ = self.eh_proj(
            torch.cat((self.enorm(anchors), self.hnorm(target_hidden)), -1)
        )
        return anchors + self.gate * fused


class HYV4FlashDFlyHiddenCorrection(nn.Module):
    def __init__(self, config: HYV4FlashTextConfig):
        super().__init__()
        d = config.hidden_size
        width = getattr(config, "dfly_hidden_correction_intermediate_size", None) or d
        prefix = f"{CHECKPOINT_PREFIX}hidden_correction"
        self.hidden_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.embed_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            2 * d,
            [width, width],
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            width,
            d,
            bias=False,
            quant_config=None,
            reduce_results=True,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden, previous_embed):
        combined = torch.cat(
            (self.hidden_norm(hidden), self.embed_norm(previous_embed)), -1
        )
        gate_up, _ = self.gate_up_proj(combined)
        delta, _ = self.down_proj(self.act(gate_up))
        return hidden + delta


class HYV4FlashDFly(nn.Module):
    def __init__(
        self,
        config: HYV4FlashTextConfig,
        dfly_config: HYV4FlashDFlyConfig,
        quant_config=None,
    ):
        super().__init__()
        self.dfly_config = dfly_config
        self.layers = nn.ModuleList(
            HYV4FlashDFlyLayer(config, i, target_layer, quant_config)
            for i, target_layer in enumerate(dfly_config.target_kv_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_fusion = HYV4FlashDFlyFusion(config)
        self.hidden_correction = HYV4FlashDFlyHiddenCorrection(config)
        self.embed_tokens = None
        self.lm_head = None
        self.state: Optional[DFlyRequestState] = None
        self.draft_rows: Optional[torch.Tensor] = None
        self.draft_counts: Optional[torch.Tensor] = None
        self.candidates_out: Optional[torch.Tensor] = None

    def bind_target(self, embed_tokens, lm_head) -> None:
        self.__dict__["embed_tokens"] = embed_tokens
        self.__dict__["lm_head"] = lm_head

    def bind_buffers(
        self,
        *,
        state: DFlyRequestState,
        draft_rows: torch.Tensor,
        draft_counts: torch.Tensor,
        candidates_out: torch.Tensor,
    ) -> None:
        self.state = state
        self.draft_rows = draft_rows
        self.draft_counts = draft_counts
        self.candidates_out = candidates_out
        for i, layer in enumerate(self.layers):
            layer.indexer.rows = draft_rows[i]

    def draft_into_verify(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        num_rows = input_ids.shape[0]
        width = self.dfly_config.block_size
        bs = num_rows // width
        req_pool_indices = forward_batch.req_pool_indices[:bs]
        build_draft_topk(
            state_topk=self.state.topk,
            req_pool_indices=req_pool_indices,
            verify_slots=forward_batch.out_cache_loc[:num_rows].view(bs, width),
            out_rows=self.draft_rows,
            out_counts=self.draft_counts,
        )
        backend = get_attn_backend()
        metadata = backend.forward_metadata
        backend.forward_metadata = dataclasses.replace(
            metadata, dsa_cache_seqlens_int32=self.draft_counts[:num_rows]
        )
        try:
            block_ids = input_ids.view(bs, width)
            drafts = self.propose(
                block_ids=block_ids,
                positions=positions,
                target_hidden=self.state.hidden[req_pool_indices],
                forward_batch=forward_batch,
            )
        finally:
            backend.forward_metadata = metadata
        block_ids[:, 1:] = drafts
        self.candidates_out[:num_rows].copy_(input_ids)

    @torch.no_grad()
    def propose(
        self,
        block_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        bs, width = block_ids.shape
        embeds = self.embed_tokens(block_ids.reshape(-1)).view(bs, width, -1)
        embeds[:, 0] = self.mtp_fusion(embeds[:, 0], target_hidden)
        hidden = embeds.reshape(bs * width, -1)
        residual = None
        zero_allocator = BumpAllocator(
            buffer_size=2 * len(self.layers),
            dtype=torch.float32,
            device=hidden.device,
        )
        for layer in self.layers:
            hidden, residual = layer(
                positions, hidden, residual, forward_batch, zero_allocator
            )
        hidden = self.norm(hidden + residual).view(bs, width, -1)
        previous = block_ids[:, 0]
        drafts = []
        for step in range(width - 1):
            corrected = self.hidden_correction(
                hidden[:, step + 1], self.embed_tokens(previous)
            )
            previous = self._greedy(corrected)
            drafts.append(previous)
        return torch.stack(drafts, dim=1)

    def _greedy(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self.lm_head
        shard = head.shard_indices
        weight = head.weight[: shard.num_org_elements]
        logits = torch.mm(hidden, weight.T, out_dtype=torch.float32)
        local_max, local_arg = logits.max(dim=-1)
        local_ids = local_arg + shard.org_vocab_start_index
        tp_group = get_parallel().tp_group
        if tp_group.world_size == 1:
            return local_ids
        rows = hidden.shape[0]
        maxes = local_max.new_empty((tp_group.world_size * rows,))
        ids = local_ids.new_empty((tp_group.world_size * rows,))
        tp_group.all_gather_into_tensor(maxes, local_max.contiguous())
        tp_group.all_gather_into_tensor(ids, local_ids.contiguous())
        best = maxes.view(-1, rows).argmax(dim=0)
        return ids.view(-1, rows).gather(0, best[None]).squeeze(0)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        pending_mlp: list[Tuple[str, torch.Tensor]] = []
        qkv_a: dict[str, dict[str, torch.Tensor]] = {}
        loaded: set[str] = set()
        device = self.norm.weight.device
        dtype = self.norm.weight.dtype

        def load(name, value, shard_id=None):
            if name not in params:
                raise ValueError(f"Unexpected DFly weight {name!r}")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                loader(param, value)
            else:
                loader(param, value, shard_id)
            loaded.add(name)

        for name, value in weights:
            if name.startswith("confidence_head."):
                continue
            if name.endswith(".weight_scale"):
                name += "_inv"
            if name.startswith("final_norm."):
                name = name.replace("final_norm.", "norm.", 1)
            elif name.split(".", 1)[0].isdigit():
                name = f"layers.{name}"
            if ".mlp." in name and name.startswith("layers."):
                layer = self.layers[int(name.split(".")[1])]
                if name.endswith((".gate_proj.weight", ".up_proj.weight")):
                    layer.materialize_mlp(value.shape[0], device, dtype)
                elif name.endswith(".down_proj.weight"):
                    layer.materialize_mlp(value.shape[1], device, dtype)
                if layer.mlp is None:
                    pending_mlp.append((name, value))
                    continue
                params = dict(self.named_parameters())
                for pending_name, pending_value in pending_mlp:
                    self._load_mapped(load, pending_name, pending_value)
                pending_mlp.clear()
            if ".q_a_proj." in name or ".kv_a_proj_with_mqa." in name:
                kind = "q" if ".q_a_proj." in name else "kv"
                fused = name.replace(".q_a_proj.", ".fused_qkv_a_proj_with_mqa.")
                fused = fused.replace(
                    ".kv_a_proj_with_mqa.", ".fused_qkv_a_proj_with_mqa."
                )
                parts = qkv_a.setdefault(fused, {})
                parts[kind] = value
                if len(parts) == 2:
                    load(fused, torch.cat([parts["q"], parts["kv"]], dim=0))
                    del qkv_a[fused]
                continue
            self._load_mapped(load, name, value)
        if pending_mlp or qkv_a:
            raise ValueError("Incomplete DFly checkpoint: missing paired weights")
        missing = {
            name
            for name in dict(self.named_parameters())
            if not name.endswith((".k_scale", ".v_scale"))
        } - loaded
        if missing:
            raise ValueError(f"Missing DFly weights: {sorted(missing)}")
        for layer in self.layers:
            _split_kv_b_proj(layer.self_attn)

    @staticmethod
    def _load_mapped(load, name, value):
        for shard, shard_id in ((".gate_proj.", 0), (".up_proj.", 1)):
            if shard in name:
                load(name.replace(shard, ".gate_up_proj."), value, shard_id)
                return
        load(name, value)


def _split_kv_b_proj(attn) -> None:
    w = attn.kv_b_proj.weight
    if w.dtype != torch.bfloat16:
        raise ValueError("The DFly kv_b_proj must be bf16")
    w_kc, w_vc = w.unflatten(0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)).split(
        [attn.qk_nope_head_dim, attn.v_head_dim], dim=1
    )
    attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
    attn.w_vc = w_vc.contiguous().transpose(1, 2)
