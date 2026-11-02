"""Muse 1.2 Omni model with torch-native vision and audio encoders."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

import sglang.srt.managers.mm_utils as mm_utils
from sglang.srt.layers.linear import (
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.muse_spark.muse_spark_audio import AudioEncoder
from sglang.srt.layers.muse_spark.muse_spark_layers import MuseSpark1xRMSNorm
from sglang.srt.layers.muse_spark.muse_spark_navit import (
    NaViT,
    NaViTAdapter,
    get_navit_patch_count,
    get_navit_video_patch_count,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.muse_spark import MuseSparkTextForCausalLM
from sglang.srt.multimodal.encoder_preprocessing import EncoderMediaProcessorConfig
from sglang.srt.multimodal.mm_utils import (
    _pad_mrope_vision_embeddings_for_tp_gather,
    get_dp_encoder_lb_assignment,
)
from sglang.srt.runtime_context import get_disagg, get_mm, get_parallel
from sglang.srt.utils import add_prefix, is_cpu


def _run_dp_sharded_encode(
    features: list[torch.Tensor],
    encode_cost_per_item: list[int],
    tokens_per_item: list[int],
    encode_fn,
    out_hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Mirrors run_dp_sharded_mrope_vision_model in sglang.srt.multimodal.mm_utils.
    parallel = get_parallel()
    tp_size = parallel.attn_tp_size
    tp_rank = parallel.attn_tp_rank
    if tp_size == 1:
        return encode_fn(features)

    image_to_tp_rank, gpu_sample_counts, _ = get_dp_encoder_lb_assignment(
        encode_cost_per_item, tp_size
    )
    cum_counts = [0, *itertools.accumulate(gpu_sample_counts)]
    local_idxs = image_to_tp_rank[cum_counts[tp_rank] : cum_counts[tp_rank + 1]]

    local_embeds = (
        encode_fn([features[i] for i in local_idxs])
        if local_idxs
        else torch.empty((0, out_hidden_size), device=device, dtype=dtype)
    )

    # Single-item fast path: bit-identical to the all-gather below.
    if len(features) == 1:
        if tp_rank != 0:
            local_embeds = torch.empty(
                (tokens_per_item[0], out_hidden_size), device=device, dtype=dtype
            )
        get_parallel().attn_tp_group.broadcast(local_embeds, src=0)
        return local_embeds

    grouped_out_lengths = []
    offset = 0
    for count in gpu_sample_counts:
        grouped_out_lengths.append(
            sum(tokens_per_item[i] for i in image_to_tp_rank[offset : offset + count])
        )
        offset += count
    max_len_per_rank = max(grouped_out_lengths)

    padded = _pad_mrope_vision_embeddings_for_tp_gather(local_embeds, max_len_per_rank)
    gathered = parallel.attn_tp_group.all_gather(padded, dim=0)

    per_rank = []
    for rank in range(tp_size):
        start = rank * max_len_per_rank
        per_rank.append(gathered[start : start + grouped_out_lengths[rank]])

    ordered = [None] * len(features)
    offset = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        rank_images = image_to_tp_rank[offset : offset + count]
        rank_embed = per_rank[rank]
        embed_start = 0
        for item_idx in rank_images:
            n_tok = tokens_per_item[item_idx]
            ordered[item_idx] = rank_embed[embed_start : embed_start + n_tok]
            embed_start += n_tok
        offset += count
    return torch.cat(ordered, dim=0)


class _RMSNormWithoutScale(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(value, (value.shape[-1],), None, self.eps)


class _NormalizedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Module, norm: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_embedding", embedding)
        object.__setattr__(self, "_norm", norm)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._norm(self._embedding(input_ids))

    @property
    def weight(self):
        return self._embedding.weight

    @property
    def num_embeddings(self):
        return self._embedding.num_embeddings


class MuseSparkForConditionalGeneration(MuseSparkTextForCausalLM):
    """Muse 1.2 decoder with NaViT image/video and waveform audio inputs."""

    # Features are single tensors, eagerly reconstructed by the scheduler on
    # every consumer rank before entering the encoder (no deferred DP lease).
    supports_cuda_vmm_feature_transport = True
    encoder_media_processor_config = EncoderMediaProcessorConfig(
        preprocess_off_event_loop=True, audio_sample_rate=24_000
    )

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.encoder_only = get_disagg().encoder_only
        self.language_only = get_disagg().language_only
        if self.encoder_only:
            nn.Module.__init__(self)
            self.config = config.text_config
            self.quant_config = quant_config
            self.model = nn.Module()
            # Audio uses the text embedding norm, which has no learned weight.
            # Preserve the exact norm without constructing the decoder or vocab.
            self.model.language_model = nn.Module()
            self.model.language_model.embed_norm = MuseSpark1xRMSNorm(
                config.text_config.hidden_size,
                eps=config.text_config.rms_norm_eps,
                has_weight=False,
            )
        else:
            super().__init__(config.text_config, quant_config, prefix)
        self._normalized_embedding = None
        if not self.encoder_only and self.model.language_model.embed_norm is not None:
            self._normalized_embedding = _NormalizedEmbedding(
                self.model.language_model.embed_tokens,
                self.model.language_model.embed_norm,
            )
        if self.language_only:
            return
        self.use_data_parallel_encoder = get_mm().mm_enable_dp_encoder
        self.model.vision_tower = NaViT(
            config.vision_config,
            prefix=add_prefix("model.vision_tower", prefix),
            use_data_parallel=self.use_data_parallel_encoder,
        )
        if config.out_hidden_size != self.model.vision_tower.output_size:
            raise ValueError("out_hidden_size must match the merged vision hidden size")
        if config.projector_hidden_act != "gelu":
            raise ValueError("Muse vision projector supports gelu only")
        if self.use_data_parallel_encoder or (
            # CPU cannot pad the checkpoint-fixed rowwise input dimension.
            is_cpu() and config.projector_hidden_size % get_parallel().tp_size
        ):
            projection_class = ReplicatedLinear
            projection_kwargs = {}
        else:
            projection_class = RowParallelLinear
            projection_kwargs = {"input_is_parallel": False}
        self.model.vision_adapter = NaViTAdapter(
            config.out_hidden_size,
            config.projector_hidden_size,
            add_prefix("model.vision_adapter", prefix),
            use_data_parallel=self.use_data_parallel_encoder
            or (
                is_cpu() and config.projector_hidden_size % get_parallel().tp_size != 0
            ),
        )
        self.model.vision_projection = projection_class(
            input_size=config.projector_hidden_size,
            output_size=config.text_config.hidden_size,
            bias=False,
            prefix=add_prefix("model.vision_projection", prefix),
            **projection_kwargs,
        )
        self.perception_emb_norm = _RMSNormWithoutScale()
        self.model.audio_tower = AudioEncoder(config.audio_config)
        if config.audio_config.out_hidden_size != config.text_config.hidden_size:
            raise ValueError("audio out_hidden_size must match text hidden_size")

    def preprocess_mm_for_encoder(self, mm_data, modality, config):
        from sglang.srt.multimodal.processors.muse_spark_omni import (
            build_encoder_inputs,
            make_preprocess_config,
        )

        return build_encoder_inputs(
            mm_data,
            modality,
            make_preprocess_config(self.model.vision_tower.config),
            self.model.audio_tower.patch_size,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        if input_ids is None:
            if input_embeds is None:
                raise ValueError("input_embeds is required for multimodal forwarding")
            return super().forward(
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )

        if (
            not forward_batch.forward_mode.is_decode()
            and forward_batch.contains_mm_inputs()
        ):
            return mm_utils.general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self,
                data_embedding_funcs={
                    Modality.IMAGE: self.get_image_feature,
                    Modality.VIDEO: self.get_video_feature,
                    Modality.AUDIO: self.get_audio_feature,
                },
                positions=positions,
            )

        if input_embeds is None:
            # Breakable prefill graphs capture the multimodal model's inner
            # transformer against a stable input_embeds buffer.  The eager
            # outer tail must therefore materialize text embeddings too; if it
            # forwards input_ids alone, the replay buffer retains its
            # capture-time zeros and every prefill logit is invalid.  Use the
            # same normalized embedding path as multimodal requests.
            input_embeds = self.get_input_embeddings()(input_ids)
        return super().forward(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )

    def get_input_embeddings(self) -> nn.Module:
        return self._normalized_embedding or super().get_input_embeddings()

    def _encode_vision(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if not inputs:
            raise ValueError("vision encoder received an empty input list")
        parameter = next(self.model.vision_tower.parameters())
        inputs = [
            value.to(device=parameter.device, dtype=parameter.dtype) for value in inputs
        ]
        output, _ = self.model.vision_projection(
            self.model.vision_adapter(self.model.vision_tower(inputs))
        )
        if self.perception_emb_norm is not None:
            output = self.perception_emb_norm(output)
        return output

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        images = []
        for item in items:
            value = torch.as_tensor(item.feature)
            if value.ndim not in (3, 4):
                raise ValueError(f"unexpected image shape {tuple(value.shape)}")
            images.append(value)
        if self.use_data_parallel_encoder:
            navit = self.model.vision_tower
            costs, tokens = [], []
            for v in images:
                frames = 1 if v.ndim == 3 else v.shape[0] // navit.patch_temporal
                grid = (v.shape[-2] // navit.patch_height) * (
                    v.shape[-1] // navit.patch_width
                )
                costs.append(frames * grid)
                tokens.append(
                    frames
                    * get_navit_patch_count(
                        v.shape[-2],
                        v.shape[-1],
                        navit.patch_height,
                        navit.patch_width,
                        navit.downsample_factor,
                    )
                )
            parameter = next(self.model.vision_tower.parameters())
            return _run_dp_sharded_encode(
                images,
                costs,
                tokens,
                self._encode_vision,
                self.config.hidden_size,
                parameter.device,
                parameter.dtype,
            )
        return self._encode_vision(images)

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        videos = []
        for item in items:
            value = torch.as_tensor(item.feature)
            if value.ndim != 4:
                raise ValueError(f"unexpected video shape {tuple(value.shape)}")
            videos.append(value)
        if self.use_data_parallel_encoder:
            navit = self.model.vision_tower
            costs = [
                (v.shape[0] // navit.patch_temporal)
                * (v.shape[-2] // navit.patch_height)
                * (v.shape[-1] // navit.patch_width)
                for v in videos
            ]
            tokens = [
                get_navit_video_patch_count(
                    v.shape[0],
                    v.shape[-2],
                    v.shape[-1],
                    navit.patch_temporal,
                    navit.patch_height,
                    navit.patch_width,
                    navit.downsample_factor,
                )
                for v in videos
            ]
            parameter = next(self.model.vision_tower.parameters())
            return _run_dp_sharded_encode(
                videos,
                costs,
                tokens,
                self._encode_vision,
                self.config.hidden_size,
                parameter.device,
                parameter.dtype,
            )
        return self._encode_vision(videos)

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if self.model.audio_tower is None:
            raise RuntimeError(
                "this checkpoint does not configure the waveform encoder"
            )
        parameter = next(self.model.audio_tower.parameters())
        waveforms = [
            torch.as_tensor(item.feature).to(
                device=parameter.device, dtype=parameter.dtype
            )
            for item in items
        ]

        def _encode_audio(local_waveforms):
            output = torch.cat(self.model.audio_tower(local_waveforms), dim=0)
            if self.model.language_model.embed_norm is not None:
                output = self.model.language_model.embed_norm(output)
            return output

        if self.use_data_parallel_encoder:
            patch = self.model.audio_tower.patch_size
            costs = [w.numel() for w in waveforms]
            tokens = [-(-w.numel() // patch) for w in waveforms]
            return _run_dp_sharded_encode(
                waveforms,
                costs,
                tokens,
                _encode_audio,
                self.config.hidden_size,
                parameter.device,
                parameter.dtype,
            )
        return _encode_audio(waveforms)

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        pattern = mm_utils.MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def post_load_weights(self) -> None:
        if not self.encoder_only:
            return super().post_load_weights()
        for module in self.modules():
            if isinstance(module, MuseSpark1xRMSNorm):
                module.precompute_effective_weight()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        parameters = dict(self.named_parameters())

        def text_weights():
            for name, weight in weights:
                if name.startswith("model.language_model.") or name == "lm_head.weight":
                    if not self.encoder_only:
                        yield name, weight
                    continue
                if self.language_only and name.startswith(
                    (
                        "model.vision_tower.",
                        "model.vision_adapter.",
                        "model.vision_projection.",
                        "model.audio_tower.",
                    )
                ):
                    continue
                if ".attn." in name:
                    for projection in ("q", "k", "v"):
                        marker = f".attn.{projection}_proj."
                        if marker not in name:
                            continue
                        if projection in ("q", "k"):
                            heads = self.model.vision_tower.config.num_attention_heads
                            shape = weight.shape
                            weight = (
                                weight.reshape(
                                    heads, 2, shape[0] // heads // 2, *shape[1:]
                                )
                                .transpose(1, 2)
                                .reshape(shape)
                                .contiguous()
                            )
                        parameter = parameters[name.replace(marker, ".attn.qkv_proj.")]
                        parameter.weight_loader(parameter, weight, projection)
                        break
                    else:
                        parameter = parameters[name]
                        getattr(parameter, "weight_loader", default_weight_loader)(
                            parameter, weight
                        )
                else:
                    parameter = parameters[name]
                    getattr(parameter, "weight_loader", default_weight_loader)(
                        parameter, weight
                    )

        if self.encoder_only:
            # Iteration loads encoder tensors while filtering every decoder tensor.
            for _ in text_weights():
                pass
        else:
            super().load_weights(text_weights())


EntryClass = [MuseSparkForConditionalGeneration]

__all__ = ["MuseSparkForConditionalGeneration"]
