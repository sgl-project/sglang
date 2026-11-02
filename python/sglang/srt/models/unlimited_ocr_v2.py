"""Unlimited-OCR v2: v1 vision stack with a DASH language tower (block-sparse
attention, entmax routing, centered log-prior bias, page2d RoPE).
"""

import logging
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.kernels.ops.attention.unlimited_ocr_dash.page2d_rope import (
    apply_page2d_rope_pair,
)
from sglang.srt.configs.unlimited_ocr import UnlimitedVLConfig
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek import DeepseekMLP, DeepseekMoE
from sglang.srt.models.deepseek_ocr import clear_sam_rel_pos_cache
from sglang.srt.models.unlimited_ocr import UnlimitedOCRForCausalLM
from sglang.srt.models.unlimited_ocr_v2_support.crop_transfer import (
    UnlimitedOcrCropTransfer,
)
from sglang.srt.models.unlimited_ocr_v2_support.document_prefetch import EarlyEmbeddings
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class UnlimitedOCRV2Attention(nn.Module):
    """Plain MHA over the DASH sparse backend, with page2d RoPE on both branches."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        sparse_cfg = config.sparse_attention_config
        tp_size = get_parallel().tp_size
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5
        self.idx_head_dim = int(sparse_cfg.get("sparse_index_dim", 128))
        # sparse_num_index_heads is forced to 1 for the DASH router: the stride-0
        # expand the attend relies on only holds for a single-head router.
        assert int(sparse_cfg.get("sparse_num_index_heads", 1)) == 1, (
            "the DASH router is single-head; sparse_num_index_heads must be 1"
        )

        # main attention heads are sharded independently of the single router head
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # Router branch (checkpoint: self_attn.dash_router.*). Gemma-style RMSNorm
        # (x * (1 + w)): trained weights are ~0, so plain x*w would zero the scores.
        eps = config.rms_norm_eps
        self.index_q_proj = ReplicatedLinear(
            self.hidden_size,
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("index_q_proj", prefix),
        )
        self.index_k_proj = ReplicatedLinear(
            self.hidden_size,
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("index_k_proj", prefix),
        )
        self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=eps)
        # Learned stage-0 summary query. Zero-init == mean pooling, which is what
        # training starts from; the checkpoint always carries it in head_cls mode.
        self.head_cls = nn.Parameter(torch.zeros(self.idx_head_dim))

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Project q/k/v and index branches, apply page2d RoPE, then attend."""
        backend = get_attn_backend()
        num_tokens = hidden_states.shape[0]
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        idx_q, _ = self.index_q_proj(hidden_states)
        idx_k, _ = self.index_k_proj(hidden_states)
        idx_q = self.index_q_norm(idx_q.view(-1, 1, self.idx_head_dim)).view(
            num_tokens, self.idx_head_dim
        )
        idx_k = self.index_k_norm(idx_k.view(-1, 1, self.idx_head_dim)).view(
            num_tokens, self.idx_head_dim
        )

        # page2d: one cos/sin table per branch per step, shared by all layers.
        # No 1D-rope fallback; a step with no coordinates is the dummy run.
        cs_main = backend.cos_sin("main", q.dtype)
        if cs_main is not None:
            q, k = apply_page2d_rope_pair(q, k, cs_main[0], cs_main[1], self.head_dim)
            cs_idx = backend.cos_sin("index", idx_q.dtype)
            idx_q, idx_k = apply_page2d_rope_pair(
                idx_q, idx_k, cs_idx[0], cs_idx[1], self.idx_head_dim
            )

        attn_out = self.attn(
            q,
            k,
            v,
            forward_batch,
            idx_q=idx_q,
            idx_k=idx_k.view(num_tokens, 1, self.idx_head_dim),
            head_cls=self.head_cls,
        )
        out, _ = self.o_proj(attn_out)
        return out


class UnlimitedOCRV2DecoderLayer(nn.Module):
    """One decoder layer: DASH sparse self-attention + Deepseek MLP/MoE."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = UnlimitedOCRV2Attention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        if (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        ):
            self.mlp = DeepseekMoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = DeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ):
        """Pre-norm self-attention + MLP block, returning (hidden, residual)."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class UnlimitedOCRV2Model(nn.Module):
    """Token embedding + the stack of DASH decoder layers + final norm."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                UnlimitedOCRV2DecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """Embed (or accept precomputed embeds) and run all decoder layers."""
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class UnlimitedOCRV2LanguageModel(nn.Module):
    """`model` + `lm_head`, matching the v1 nesting so the vision remap is shared."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.model = UnlimitedOCRV2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding module."""
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run the language model and project the last hidden states to logits."""
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )


class UnlimitedOCRV2ForCausalLM(UnlimitedOCRForCausalLM):
    """Use the v1 multimodal stack with the v2 DASH language tower."""

    def __init__(
        self,
        *,
        config: UnlimitedVLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.crop_transfer = UnlimitedOcrCropTransfer(
            model_dtype=self.vision_model.dtype
        )
        # The image encoder; split-document prefetch fills it ahead of prefill.
        self.early_embeddings = EarlyEmbeddings(encode=self.get_image_feature)

    def _build_language_model(self, config, quant_config, prefix):
        # construct the DASH tower directly, without allocating a discarded v1 tower
        text_config = config.text_config
        text_config.sparse_attention_config = config.sparse_attention_config
        return UnlimitedOCRV2LanguageModel(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("language", prefix),
        )

    def _stack_crop_inputs(
        self,
        mm_items: List[MultimodalDataItem],
        *,
        has_local_crops: Optional[List[bool]],
        target_dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crops through the pinned crop transfer; crop shapes stay on the CPU."""
        if has_local_crops is not None and not any(has_local_crops):
            shape = (len(mm_items),) + tuple(mm_items[0].images_crop.shape)
            images_crop = torch.zeros(shape, dtype=target_dtype, device=device)
        else:
            images_crop = self.crop_transfer.stack(
                tensors=[item.images_crop for item in mm_items],
                dtype=target_dtype,
                device=device,
            )
        # Crop-shape metadata stays on CPU to avoid GPU scalar synchronization.
        images_spatial_crop = torch.cat(
            [item.images_spatial_crop for item in mm_items], dim=0
        ).type(torch.long)
        return images_crop, images_spatial_crop

    def _encode_ocr1_features(self, images: torch.Tensor) -> torch.Tensor:
        """Exclude cuDNN SDPA because nondeterminism can flip OCR logits."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            return super()._encode_ocr1_features(images)

    def _process_image_input(self, mm_items: List[MultimodalDataItem]) -> torch.Tensor:
        """Batch matching crop geometries; encode mixed geometries separately."""
        if (
            len(mm_items) == 1
            or len({tuple(item.images_crop.shape) for item in mm_items}) == 1
        ):
            return super()._process_image_input(mm_items)
        encode_one = super()._process_image_input
        features = [encode_one([item]) for item in mm_items]
        return torch.cat(features, dim=0)

    # SAM's relative-position bias is about 0.4 GiB per 1024px image; cap
    # vision batches at eight to bound memory.
    _VIT_MAX_BATCH = 8

    def _encode_ocr1_features_grouped(
        self, tiles: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Order follows ``tiles``, split by shape at ``_VIT_MAX_BATCH``; batched
        bf16 results may differ from per-stack encoding.
        """
        n = len(tiles)
        counts = [t.shape[0] for t in tiles]
        out: List[Optional[torch.Tensor]] = [None] * n
        groups: dict = {}
        for idx, t in enumerate(tiles):
            groups.setdefault(tuple(t.shape[1:]), []).append(idx)
        for _shape, idxs in groups.items():
            batch = torch.cat([tiles[i] for i in idxs], dim=0)
            feat_chunks = [
                self._encode_ocr1_features(batch[s : s + self._VIT_MAX_BATCH])
                for s in range(0, batch.shape[0], self._VIT_MAX_BATCH)
            ]
            feats = (
                feat_chunks[0]
                if len(feat_chunks) == 1
                else torch.cat(feat_chunks, dim=0)
            )
            pos = 0
            for i in idxs:
                out[i] = feats[pos : pos + counts[i]]
                pos += counts[i]
        return out  # type: ignore[return-value]

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
        has_local_crops: Optional[List[bool]] = None,
    ):
        """Trim crop padding, encode pages in batches, then assemble each page."""
        n = images_spatial_crop.size(0)

        local_tiles = {}
        for j in range(n):
            patches = images_crop[j][0].to(torch.bfloat16)
            crop_shape = images_spatial_crop[j][0]
            n_tiles = int(crop_shape[0]) * int(crop_shape[1])
            if 0 < n_tiles < patches.shape[0]:
                patches = patches[:n_tiles]
            use_local = (
                has_local_crops[j]
                if has_local_crops is not None
                else torch.sum(patches).item() != 0
            )
            if use_local:
                local_tiles[j] = patches

        with torch.no_grad():
            # group global and local views separately, preserving page and tile order
            global_feats = self._encode_ocr1_features_grouped(
                [pixel_values[j] for j in range(n)]
            )
            local_feats = {}
            if local_tiles:
                encoded = self._encode_ocr1_features_grouped(list(local_tiles.values()))
                local_feats = dict(zip(local_tiles, encoded))
            pages = []
            for j in range(n):
                parts = [
                    self._format_ocr1_global_features(global_feats[j]),
                    self.view_seperator[None, :],
                ]
                if j in local_feats:
                    local = self._format_ocr1_local_features(
                        local_feats[j], images_spatial_crop[j][0]
                    )
                    parts.insert(0, local)
                pages.append(torch.cat(parts, dim=0))
        return pages

    def get_attention_sliding_window_size(self) -> Optional[int]:
        """Use a full KV pool; DASH applies its local window in the kernel."""
        return None

    def is_prefill_aware_swa(self) -> bool:
        """No prefill-aware SWA: see ``get_attention_sliding_window_size``."""
        return False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ):
        """Route through the shared multimodal embedding + language model path."""
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            data_embedding_funcs={Modality.IMAGE: self.early_embeddings},
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Remap checkpoint keys (see ``_rewrite_v2_names``) and load all params."""
        clear_sam_rel_pos_cache()
        # Only the MLP is fused; attention keeps separate q/k/v (see __init__).
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        shadowed = 0
        dash_layers: Set[str] = set()

        for name, loaded_weight in _rewrite_v2_names(weights, dash_layers):
            if name == "__shadowed__":
                shadowed += 1
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        unloaded = params_dict.keys() - loaded_params
        if unloaded:
            raise RuntimeError(
                f"Some weights are not initialized from checkpoints: {unloaded}"
            )
        logger.info(
            "[Unlimited-OCR v2] loaded %d DASH router layers; dropped %d indexer "
            "tensors shadowed by dash_router weights",
            len(dash_layers),
            shadowed,
        )
        self.post_load_weights()

    def post_load_weights(self):
        """CPU-AMX repacking for the MoE experts; no-op on GPU."""
        from sglang.srt.utils import cpu_has_amx_support, is_cpu

        if not (is_cpu() and cpu_has_amx_support()):
            return
        from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading

        layer_ids = int(self.config.num_hidden_layers)
        first_k_dense_replace = getattr(self.config, "first_k_dense_replace", -1)
        moe_layer_freq = getattr(self.config, "moe_layer_freq", 1)
        for layer_id in range(layer_ids):
            if layer_id >= first_k_dense_replace and layer_id % moe_layer_freq == 0:
                moe = self.model.model.layers[layer_id].mlp
                if hasattr(moe, "w1") and hasattr(moe, "w2"):
                    _amx_process_weight_after_loading(moe, ["w1", "w2"])


def _rewrite_v2_names(
    weights: Iterable[Tuple[str, torch.Tensor]], dash_layers: Set[str]
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Map v2 checkpoint keys onto the runtime module tree.

    Drops zeroed ``indexer`` keys that would clobber trained ``dash_router``
    weights and optional ``mtp`` tensors; vision keys stay top-level,
    language-model keys gain one ``model.`` prefix.
    """
    for name, weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue
        if name.startswith("mtp."):
            continue
        # The CLIP tower saved `position_ids` as a persistent buffer; newer
        # transformers registers it non-persistently, so it has no parameter.
        if name.endswith("embeddings.position_ids"):
            continue

        router, _, tail = name.partition(".self_attn.dash_router.")
        if tail:
            dash_layers.add(router)
            if tail == "head_cls":
                yield _lm_prefix(f"{router}.self_attn.head_cls"), weight
            else:
                yield _lm_prefix(f"{router}.self_attn.index_{tail}"), weight
            continue

        idx, _, tail = name.partition(".self_attn.indexer.")
        if tail:
            if idx in dash_layers:
                yield "__shadowed__", weight
                continue
            yield _lm_prefix(f"{idx}.self_attn.index_{tail}"), weight
            continue

        if name == "lm_head.weight":
            yield "model.lm_head.weight", weight
            continue
        yield _lm_prefix(name), weight


_VISION_KEYS = (
    "image_newline",
    ".projector",
    "vision_model",
    "sam_model",
    "view_seperator",
)


def _lm_prefix(name: str) -> str:
    """Add the extra `model.` level the LM tower needs; vision keys lose theirs."""
    if not name.startswith("model."):
        return name
    if any(key in name for key in _VISION_KEYS):
        return name[len("model.") :]
    return name.replace("model.", "model.model.", 1)


EntryClass = [UnlimitedOCRV2ForCausalLM]
