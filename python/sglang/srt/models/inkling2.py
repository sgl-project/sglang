from array import array
from collections.abc import Iterable

import torch
from torch import nn

from sglang.kernels.ops.activation.softcap import (
    softcap_inplace_logits as fused_softcap,
)
from sglang.srt.configs.inkling2 import Inkling2MMConfig, Inkling2ModelConfig
from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.inkling_common.inkling2_attn import (
    AttentionSources,
    Inkling2Attention,
)
from sglang.srt.models.inkling_common.inkling2_attn_res import (
    Inkling2AttnRes,
    Inkling2AttnResState,
)
from sglang.srt.models.inkling_common.inkling2_moe import (
    Inkling2Experts,
    Inkling2MoE,
    RoutedOutput,
)
from sglang.srt.models.inkling_common.inkling2_norm import RMSNorm
from sglang.srt.models.inkling_common.inkling2_sconv import ShortConvolution
from sglang.srt.models.inkling_common.sconv import SconvType
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_mm,
    get_parallel,
    get_spec,
)

VERIFY_ROUTED_EPILOGUE_MAX_TOKENS = 32


class Inkling2DecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.layer_id = layer_id
        self.is_local = layer_id in config.local_layer_ids
        self.attn = Inkling2Attention(
            config, layer_id, self.is_local, prefix=add_prefix("attn", prefix)
        )
        self.mlp = Inkling2MoE(
            config, layer_id, add_prefix("mlp", prefix), quant_config
        )
        norm_names = (
            ("attn_norm",)
            if self.is_local
            else ("q_input_norm", "k_input_norm", "v_input_norm")
        )
        for name in (*norm_names, "mlp_norm"):
            setattr(
                self,
                name,
                RMSNorm(
                    config.hidden_size,
                    config.rms_norm_eps,
                    pre_affine=config.pre_norm_pre_affine,
                    output_dtype=torch.bfloat16,
                ),
            )
        self.attn_sconv = ShortConvolution(
            config.hidden_size,
            config.sconv_kernel_size,
            SconvType.ATTN,
            layer_id=layer_id,
            tp_rank=0,
            tp_size=1,
        )
        self.mlp_sconv = ShortConvolution(
            config.hidden_size,
            config.sconv_kernel_size,
            SconvType.MLP,
            layer_id=layer_id,
            tp_rank=0,
            tp_size=1,
        )

    def forward(self, state, positions, forward_batch):
        streams = state.attention_input(self.layer_id)
        if self.is_local:
            attn_input = self.attn_norm(streams[0])
        else:
            stacked = streams[0].new_empty(
                (3, *streams[0].shape),
                dtype=self.q_input_norm.output_dtype or streams[0].dtype,
            )
            attn_input = AttentionSources(
                self.q_input_norm(streams[0], out=stacked[0]),
                self.k_input_norm(streams[1], out=stacked[1]),
                self.v_input_norm(streams[2], out=stacked[2]),
            )
        attention = self.attn(attn_input, positions, forward_batch)
        attention = self.attn_sconv(attention, positions, forward_batch)
        mlp_input = self.mlp_norm(state.mlp_input(self.layer_id, attention))
        routed = forward_batch.forward_mode.is_decode() or (
            forward_batch.forward_mode.is_target_verify()
            and attention.shape[0] <= VERIFY_ROUTED_EPILOGUE_MAX_TOKENS
        )
        mlp = self.mlp(mlp_input, forward_batch, combine=not routed)
        if isinstance(mlp, torch.Tensor) and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        ):
            mlp = RoutedOutput(mlp, None, 0)
        if isinstance(mlp, RoutedOutput):
            state.finalize_layer_with(
                self.layer_id,
                lambda partial: self.mlp_sconv.routed_decode_epilogue(
                    mlp, attention, partial, forward_batch
                ),
            )
            return
        mlp = self.mlp_sconv(mlp, positions, forward_batch)
        state.finalize_layer(self.layer_id, attention, mlp)


class _Fp32AsBf16Gather:
    """Last-dim all-gather of FP32 rows as bfloat16 pairs (pure data movement)."""

    def __init__(self, gatherer):
        self.gatherer = gatherer

    def __call__(self, x):
        if x.dtype != torch.float32 or x.dim() != 2:
            return self.gatherer(x)
        return self.gatherer(x.view(torch.bfloat16)).view(torch.float32)


class Inkling2LogitsProcessor(LogitsProcessor):
    def __init__(self, config, output_dtype=torch.bfloat16):
        super().__init__(config)
        self.vocab_size = config.padded_vocab_size
        self.input_logprob_processor.vocab_size = self.vocab_size
        self.use_fp32_lm_head = output_dtype == torch.float32
        # FP32 logits: softcap each rank's vocab shard before the gather, and
        # gather them as bfloat16 pairs through the multimem all-gather.
        self.shard_softcap = None
        if self.use_fp32_lm_head:
            self.shard_softcap = self.final_logit_softcapping
            self.final_logit_softcapping = None
            self._logits_gatherer = _Fp32AsBf16Gather(self._logits_gatherer)

    def warm_logits_gather(self, shard_width, device):
        """Build the multimem logits all-gather before CUDA graph capture
        (collective)."""
        if self.use_fp32_lm_head and self.do_tensor_parallel_all_gather:
            self._logits_gatherer(
                torch.zeros((1, shard_width), dtype=torch.float32, device=device)
            )

    def _compute_lm_head(
        self, hidden_states, lm_head, embedding_bias=None, *, out=None
    ):
        logits = super()._compute_lm_head(
            hidden_states, lm_head, embedding_bias, out=out
        )
        if self.shard_softcap:
            if logits.is_cuda:
                fused_softcap(logits, self.shard_softcap)
            else:
                logits = self.shard_softcap * torch.tanh(logits / self.shard_softcap)
        return logits

    def _get_logits(
        self,
        hidden_states,
        lm_head,
        logits_metadata,
        embedding_bias=None,
        use_logits_buffer=True,
    ):
        logits = super()._get_logits(
            hidden_states, lm_head, logits_metadata, embedding_bias, use_logits_buffer
        )
        logits[..., self.config.vocab_size :] = float("-inf")
        return logits


class Inkling2CausalLLM(nn.Module):
    def __init__(self, config: Inkling2ModelConfig, quant_config, prefix):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.start_layer, self.end_layer = 0, config.num_hidden_layers
        self.layers_to_capture: list[int] = []
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.embed_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                Inkling2DecoderLayer(
                    config, i, quant_config, add_prefix(f"layers.{i}", prefix)
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.attn_res = Inkling2AttnRes(
            config.num_hidden_layers,
            config.hidden_size,
            config.n_layers_per_attn_res_block,
            config.local_layer_ids,
            config.attn_res_normalize_values,
        )
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            pre_affine=config.final_norm_pre_affine,
            output_dtype=torch.bfloat16,
        )
        self.lm_head = ParallelLMHead(
            config.padded_vocab_size,
            config.hidden_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = Inkling2LogitsProcessor(
            config,
            quant_config.unembed_output_dtype
            if quant_config is not None
            else torch.bfloat16,
        )

    def get_input_embeddings(self):
        def embed(input_ids):
            return self.embed_norm(self.embed_tokens(input_ids))

        embed.weight = self.embed_tokens.weight
        embed.num_embeddings = self.embed_tokens.num_embeddings
        return embed

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden = (
            self.get_input_embeddings()(input_ids)
            if input_embeds is None
            else input_embeds
        )
        captures = (
            AuxHiddenStatePacker.for_batch(forward_batch, len(self.layers_to_capture))
            if self.layers_to_capture
            else None
        )
        if hidden.shape[0] == 0:
            if captures is not None:
                for _ in self.layers_to_capture:
                    captures.append(hidden.to(self.lm_head.weight.dtype))
                return hidden, captures.finalize()
            return hidden
        state = Inkling2AttnResState(self.attn_res, hidden)
        for i, layer in enumerate(self.layers):
            if captures is not None and i in self.layers_to_capture:
                captures.append(state.partial.to(self.lm_head.weight.dtype))
            layer(state, positions, forward_batch)
        hidden = self.norm(state.compute_output())
        if captures is not None:
            captures.append(hidden.to(self.lm_head.weight.dtype))
            return hidden, captures.finalize()
        return hidden


class Inkling2ForConditionalGeneration(nn.Module):
    fall_back_to_pt_during_load = False
    allow_patterns_overrides = ["model*.safetensors"]

    def __init__(self, config: Inkling2MMConfig, quant_config=None, prefix: str = ""):
        super().__init__()
        if quant_config is not None and not isinstance(
            quant_config, Inkling2QuantConfig
        ):
            raise ValueError("Inkling2 requires its checkpoint precision metadata")
        if get_parallel().pp_group.world_size != 1:
            raise ValueError(
                "Inkling2 attention residuals do not support pipeline parallelism"
            )
        # EPD: encoders build only the vision tower, language servers only the LLM.
        self.encoder_only = get_disagg().encoder_only
        self.language_only = get_disagg().language_only
        # Encoder data parallelism replicates the vision tower; the LLM rejects DP.
        if get_parallel().dp_size != 1 and not self.encoder_only:
            raise ValueError("Inkling2 does not support data parallel inference")
        if get_exec().comm.enable_scattered_sconv:
            raise ValueError("Inkling2 requires full-width convolution states")
        self.config, self.text_config, self.quant_config = (
            config,
            config.text_config,
            quant_config,
        )
        self.llm = (
            None
            if self.encoder_only
            else Inkling2CausalLLM(
                config.text_config, quant_config, add_prefix("llm", prefix)
            )
        )
        self.visual = None
        if self.encoder_only or (get_mm().enable_multimodal and not self.language_only):
            if config.vision_config is None:
                raise ValueError("Inkling2 image inputs require vision_config")
            from sglang.srt.models.inkling2_vision import Inkling2VisionEncoder

            self.visual = Inkling2VisionEncoder(
                config.vision_config, config.text_config.hidden_size
            )
        self.mm_pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    @property
    def model(self):
        return self.llm

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        if any(item.modality != Modality.IMAGE for item in mm_inputs.mm_items):
            raise ValueError("Inkling2 only supports image multimodal inputs")
        return self.mm_pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if self.visual is None and self.language_only:
            # The receiver falls back to local encoding when no encoder is reachable.
            raise ValueError(
                "Inkling2 language-only servers have no vision tower; "
                "image inputs need a reachable --encoder-urls encoder"
            )
        if self.visual is None:
            raise ValueError("Inkling2 image inputs require --enable-multimodal")
        grids = torch.cat(
            [item.model_specific_data["vision_grid_thw"] for item in items]
        )
        num_patches = int(grids.prod(dim=1).sum())
        group = get_parallel().attn_tp_group
        param = next(self.visual.parameters())
        if group.rank_in_group == 0:
            patches = torch.cat([item.feature for item in items]).to(
                device=param.device, dtype=torch.bfloat16
            )
            output = self.visual(patches, grids)
            if output.shape != (num_patches, self.text_config.hidden_size):
                raise ValueError(
                    "Vision encoder output does not match image token count"
                )
        else:
            output = torch.empty(
                (num_patches, self.text_config.hidden_size),
                device=param.device,
                dtype=torch.bfloat16,
            )
        group.broadcast(output, src=0)
        return output

    @property
    def lm_head(self):
        return self.llm.lm_head

    @property
    def capture_aux_hidden_states(self) -> bool:
        return self.llm is not None and bool(self.llm.layers_to_capture)

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        num_layers = len(self.llm.layers)
        if (
            not layer_ids
            or layer_ids != sorted(set(layer_ids))
            or layer_ids[0] < 0
            or layer_ids[-1] != num_layers
        ):
            raise ValueError(
                "Inkling2 draft feature layers must be sorted, unique, and end "
                f"with {num_layers}, the post-final-norm hidden state"
            )
        self.llm.layers_to_capture = list(layer_ids)

    def get_attention_sliding_window_size(self):
        window = self.text_config.sliding_window_size
        return window if get_spec().speculative_algorithm == "DSPARK" else window - 1

    def get_draft_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def preprocess_mm_for_encoder(self, mm_data, modality, config):
        from sglang.srt.multimodal.inkling2.processor import build_encoder_inputs

        return build_encoder_inputs(mm_data, modality)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        **kwargs,
    ):
        if forward_batch.contains_mm_inputs():
            # Language-only servers receive precomputed embeddings from encoders.
            if self.visual is None and not self.language_only:
                raise ValueError("Inkling2 image inputs require --enable-multimodal")
            if input_embeds is not None:
                raise ValueError("Image inputs cannot be combined with input_embeds")
            hidden = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.llm,
                data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
                positions=positions,
            )
        else:
            if input_embeds is not None:
                input_embeds = self.llm.embed_norm(input_embeds)
            hidden = self.llm(input_ids, positions, forward_batch, input_embeds)
        auxiliary = None
        if self.capture_aux_hidden_states:
            hidden, auxiliary = hidden
        return self.llm.logits_processor(
            input_ids,
            hidden,
            self.llm.lm_head,
            forward_batch,
            aux_hidden_states=auxiliary,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        projections = (
            "wq_du",
            "wk_dv",
            "wv_dv",
            "wr_du",
            "gate_attn",
            "attn_sink_proj",
        )
        packed_shards: dict[str, set[int]] = {}
        for checkpoint_name, weight in weights:
            name = checkpoint_name.removeprefix("model.")
            if name.startswith(("audio.", "mtp.")):
                continue
            if self.visual is None and name.startswith(("visual.", "vision.")):
                continue
            if self.llm is None and not name.startswith("visual."):
                continue
            if self.quant_config is not None:
                expected_dtype = self.quant_config.tensor_dtypes[checkpoint_name]
                if weight.dtype != expected_dtype:
                    raise ValueError(
                        f"{checkpoint_name}: expected {expected_dtype}, got {weight.dtype}"
                    )
            name = name.replace(".embed.weight", ".embed_tokens.weight")
            name = name.replace(".unembed.weight", ".lm_head.weight")
            name = name.replace(".embed_tokens.embed_norm.", ".embed_norm.")
            name = name.replace(".embed.embed_norm.", ".embed_norm.")
            shard = None
            for i, projection in enumerate(projections):
                if f".attn.{projection}." in name:
                    name = name.replace(f".{projection}.", ".qkvr.")
                    shard = i
                    break
            for leaf in ("w13_weight", "w2_weight"):
                name = name.replace(f".mlp.{leaf}", f".mlp.experts.{leaf}")
            for suffix in ("scale", "amax", "outer_scale_denominator"):
                name = name.replace(f"_weight_{suffix}", f"_{suffix}")
            if name not in params:
                raise ValueError(f"Unrecognized checkpoint tensor: {checkpoint_name}")
            param = params[name]
            if param.dtype != weight.dtype:
                raise ValueError(
                    f"{checkpoint_name}: parameter dtype {param.dtype} would change checkpoint precision {weight.dtype}"
                )
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is not None:
                if name in loaded and name not in packed_shards:
                    raise ValueError(
                        f"Checkpoint mixes packed and unpacked tensors: {checkpoint_name}"
                    )
                if shard in packed_shards.get(name, set()):
                    raise ValueError(f"Repeated checkpoint tensor: {checkpoint_name}")
                loader(param, weight, shard)
                packed_shards.setdefault(name, set()).add(shard)
            else:
                if name in loaded:
                    raise ValueError(f"Repeated checkpoint tensor: {checkpoint_name}")
                loader(param, weight)
            loaded.add(name)
        missing = set(params) - loaded
        for name, shards in packed_shards.items():
            expected = set(range(len(projections))) if name.endswith(".weight") else {0}
            if not expected.issubset(shards):
                missing.add(f"{name} shards {sorted(expected - shards)}")
        if missing:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        for module in self.modules():
            if isinstance(module, Inkling2Experts):
                module.swizzle_nvfp4_scales()
                module.prepare_mxfp8_cutlass_scales()
                module.prepare_mega_moe()
                module.prepare_routed_exchange()
            elif isinstance(module, Inkling2LogitsProcessor):
                module.warm_logits_gather(
                    self.llm.lm_head.weight.shape[0], self.llm.lm_head.weight.device
                )
            elif isinstance(module, Inkling2AttnRes):
                module.scale_queries()
        return loaded


EntryClass = Inkling2ForConditionalGeneration
