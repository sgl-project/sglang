# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    MultimemAllGatherer,
    recommended_max_tokens,
)
from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.factories import layer_stack
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.models.minimax_m3 import MiniMaxM3Attention, MiniMaxM3DecoderLayer
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    get_dflash_attention_sliding_window_size,
)


class DSparkMiniMaxAttention(MiniMaxM3Attention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # All queries inside a diffusion draft block attend to one another.
        self.attn.attn_type = AttentionType.ENCODER_ONLY
        self.attn_type = AttentionType.ENCODER_ONLY
        self.sliding_window_size = (
            get_dflash_attention_sliding_window_size(kwargs["config"]) or -1
        )
        self.attn.sliding_window_size = self.sliding_window_size

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        Used by the DFlash worker to materialize ctx tokens into the draft KV
        cache: only K/V are needed for cached tokens; Q is never consumed.
        """
        can_slice_qkv_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice_qkv_weight:
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            return k, v

        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Use a minimal dummy query (1 head) to avoid doing full-Q work.
        dummy_q = k.new_empty((k.shape[0], self.head_dim))
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DSparkMiniMaxDecoder(MiniMaxM3DecoderLayer):
    attention_cls = DSparkMiniMaxAttention


class DSparkMiniMaxLayer(nn.Module):
    attention_conv = mlp_conv = None

    def __init__(self, decoder_layer: DSparkMiniMaxDecoder):
        super().__init__()
        self.decoder_layer = decoder_layer

    @property
    def self_attn(self):
        return self.decoder_layer.self_attn

    def forward(self, positions, hidden_states, forward_batch):
        return self.decoder_layer(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )


class MiniMaxDraftBackbone(DFlashDraftModel):
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.config = config
        self.is_nemotron_35_draft = False
        self.embed_tokens = None
        self.candidate_selector = None
        self.block_size = config.block_size
        self.num_context_features = len(config.dflash_config["target_layer_ids"])
        norm = GemmaRMSNorm if config.use_gemma_norm else RMSNorm
        with layer_stack():
            self.layers = nn.ModuleList(
                [
                    DSparkMiniMaxLayer(
                        DSparkMiniMaxDecoder(
                            config=config,
                            layer_id=i,
                            quant_config=quant_config,
                            prefix=f"language_model.model.dspark.layers.{i}.decoder_layer",
                        )
                    )
                    for i in range(config.num_hidden_layers)
                ]
            )
        self.norm = norm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = norm(config.hidden_size, eps=config.rms_norm_eps)
        # Each rank projects its hidden columns; the gather rebuilds the row.
        self.fc = ColumnParallelLinear(
            config.hidden_size * self.num_context_features,
            config.hidden_size,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix="language_model.model.dspark.fc",
        )
        # Rows are the injected target tokens, up to a prefill chunk.
        self._fc_gatherer = MultimemAllGatherer(
            max_tokens=recommended_max_tokens(include_prefill=True, floor=512)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        # DFlashDraftModel.forward threads (hidden, residual) through its layers;
        # the target's decoder layers keep that pair on the forward batch's stream.
        hidden_states = (
            self.forward_embed(input_ids) if input_embeds is None else input_embeds
        )
        residual_batch.start(forward_batch)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)
        hidden_states = residual_batch.final_norm(
            hidden_states, forward_batch, self.norm, skip_empty=True
        )
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    def project_target_hidden(self, target_hidden):
        if target_hidden.shape[-1] != self.fc.input_size:
            raise ValueError(
                "DSpark target hidden features do not match checkpoint projection"
            )
        return self.hidden_norm(self._fc_gatherer(self.fc(target_hidden)[0]))


class DSparkMiniMaxDraftModel(DSparkDraftMixin, MiniMaxDraftBackbone):
    supports_vocab_parallel_draft_logits = True

    def prune_to_ctx_kv_injection(self) -> None:
        self.markov_head = None
        self.confidence_head = None
        for layer in self.layers:
            layer.decoder_layer.mlp = None
            layer.self_attn.o_proj = None
        torch.cuda.empty_cache()

    def _fused_k_norm_weights(self) -> list[torch.Tensor]:
        # GemmaRMSNorm scales by 1 + w, formed in fp32 like its reference.
        return [
            1.0 + n.weight.float() if isinstance(n, GemmaRMSNorm) else n.weight.data
            for n in (layer.self_attn.k_norm for layer in self.layers)
        ]

    def _build_fused_kv_write_bundle(self, pool):
        stacked = self._stacked_ctx_kv_params()
        if stacked is None:
            return super()._build_fused_kv_write_bundle(pool)
        method, projection = stacked["method"], stacked["projection"]
        return self._fused_kv_write_layout(
            pool, project=lambda x: method.apply(projection, x)
        )

    def _stacked_ctx_kv_params(self):
        if not envs.SGLANG_DSPARK_STACKED_CTX_KV.get():
            return None
        cached = getattr(self, "_mxfp8_ctx_kv_cache", False)
        if cached is not False:
            return cached
        self._mxfp8_ctx_kv_cache = None
        weights, scales, widths = [], [], []
        method0 = None
        for layer in self.layers:
            attn = layer.self_attn
            proj = attn.qkv_proj
            method = proj.quant_method
            if not (
                isinstance(method, Fp8LinearMethod)
                and method.use_mxfp8
                and not method.use_marlin
                and method.mxfp8_dense_backend is not None
                and (
                    method.mxfp8_dense_backend.is_flashinfer_cutlass()
                    or method.mxfp8_dense_backend.is_flashinfer_cutedsl()
                )
                and proj.bias is None
                and attn.q_size % 128 == 0
                and (2 * attn.kv_size) % 128 == 0
                and proj.weight.dtype == torch.float8_e4m3fn
                and proj.weight.is_cuda
            ):
                return None
            if method0 is None:
                method0 = method
            elif (
                method.mxfp8_dense_backend != method0.mxfp8_dense_backend
                or proj.weight.shape[1] != weights[0].shape[1]
                or proj.weight.device != weights[0].device
            ):
                return None
            kv_slice = slice(attn.q_size, attn.q_size + 2 * attn.kv_size)
            sf_cols = ((proj.weight.shape[1] + 127) // 128) * 4
            weights.append(proj.weight[kv_slice])
            scales.append(
                proj.weight_scale_inv_swizzled.view(-1, sf_cols)[kv_slice].reshape(-1)
            )
            widths.append(2 * attn.kv_size)
        if not weights:
            return None
        self._mxfp8_ctx_kv_cache = {
            "method": method0,
            "projection": SimpleNamespace(
                weight=torch.cat(weights, dim=0),
                weight_scale_inv_swizzled=torch.cat(scales),
            ),
            "widths": widths,
        }
        return self._mxfp8_ctx_kv_cache

    def _project_ctx_kv_stacked(self, *, ctx_hidden, positions, stacked):
        kv_all = stacked["method"].apply(stacked["projection"], ctx_hidden)
        ks, vs = [], []
        for layer, kv in zip(self.layers, kv_all.split(stacked["widths"], dim=-1)):
            attn = layer.self_attn
            k, v = kv.split(attn.kv_size, dim=-1)
            k = attn.apply_k_norm(k)
            k = attn.apply_k_rope(positions, k)
            ks.append(k.view(-1, attn.num_kv_heads, attn.head_dim))
            vs.append(v.view(-1, attn.num_kv_heads, attn.head_dim))
        return ks, vs

    def load_weights(self, weights):
        self._mxfp8_ctx_kv_cache = False
        # The fused write bundle snapshots stacked weights and k-norm multipliers.
        self._fused_kv_write_cache = None
        params = dict(self.named_parameters())
        loaded = set()
        loaded_shards = {}
        prefixes = ("language_model.model.dspark.", "model.dspark.", "dspark.")
        stacked = (
            (".q_proj.", ".qkv_proj.", "q"),
            (".k_proj.", ".qkv_proj.", "k"),
            (".v_proj.", ".qkv_proj.", "v"),
            (".gate_proj.", ".gate_up_proj.", 0),
            (".up_proj.", ".gate_up_proj.", 1),
        )
        for raw_name, weight in weights:
            name = next(
                (raw_name[len(p) :] for p in prefixes if raw_name.startswith(p)), None
            )
            if name is None or "rotary_emb." in name:
                continue
            if name.startswith("confidence_head.") and self.confidence_head is None:
                continue
            name = name.replace("block_sparse_moe", "mlp").replace("final_norm", "norm")
            shard = None
            for source, target, shard_id in stacked:
                if source in name:
                    name = name.replace(source, target)
                    shard = shard_id
                    break
            if name not in params:
                raise ValueError(f"Unexpected DSpark checkpoint tensor: {raw_name}")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                loader(param, weight)
            else:
                loader(param, weight, shard)
                loaded_shards.setdefault(name, set()).add(shard)
            loaded.add(name)
        for name, shards in loaded_shards.items():
            expected = {"q", "k", "v"} if ".qkv_proj." in name else {0, 1}
            if shards != expected:
                raise ValueError(
                    f"Missing DSpark checkpoint shards for {name}: {expected - shards}"
                )
        missing = {
            name
            for name, param in params.items()
            if name not in loaded and not getattr(param, "_skip_weight_check", False)
        }
        if missing:
            raise ValueError(f"Missing DSpark checkpoint tensors: {sorted(missing)}")
        return loaded


EntryClass = DSparkMiniMaxDraftModel
