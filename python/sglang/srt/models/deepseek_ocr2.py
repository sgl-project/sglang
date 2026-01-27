# Adapted from
# https://github.com/deepseek-ai/DeepSeek-OCR-2/blob/main/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/deepseek_ocr2.py
"""Inference-only Deepseek-OCR model compatible with HuggingFace weights."""

from typing import Iterable, List, Optional, Set, Tuple, TypeAlias, Union

import torch
from torch import Tensor, nn

from sglang.srt.configs.deepseek_ocr import DeepseekVLV2Config
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek import DeepseekForCausalLM
from sglang.srt.models.deepseek_ocr import (
    DeepseekOCRForCausalLM,
    merge_multimodal_embeddings,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.models.utils import WeightsMapper

MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]

from sglang.srt.configs.deepseek_ocr import DeepseekVLV2Config
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepencoder2 import (
    build_qwen2_decoder_as_encoder,
    build_sam_vit_b,
)
from sglang.srt.models.deepseek_ocr import MlpProjector

NestedTensors: TypeAlias = Union[
    list["NestedTensors"],
    list["torch.Tensor"],
    "torch.Tensor",
    tuple["torch.Tensor", ...],
]


class NoRepeatNGramLogitsProcessor:
    def __init__(
        self,
        ngram_size: int,
        window_size: int,
        whitelist_token_ids: set[int] | None = None,
    ):
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if len(output_ids) < self.ngram_size:
            return logits

        current_prefix = tuple(output_ids[-(self.ngram_size - 1) :])

        search_start = max(0, len(output_ids) - self.window_size)
        search_end = len(output_ids) - self.ngram_size + 1

        banned_tokens = set()
        for i in range(search_start, search_end):
            ngram = tuple(output_ids[i : i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned_tokens.add(ngram[-1])

        banned_tokens = banned_tokens - self.whitelist_token_ids

        if banned_tokens:
            logits[list(banned_tokens)] = -float("inf")

        return logits


class DeepseekOCR2ForCausalLM(DeepseekOCRForCausalLM):
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            # map prefix for language backbone
            "model.embed_tokens.": "language_model.model.embed_tokens.",
            "model.layers.": "language_model.model.layers.",
            "model.norm.": "language_model.model.norm.",
            "lm_head.": "language_model.lm_head.",
            "model.qwen2_model.": "vision_model.",
            # remove "model." prefix for other components
            "model.": "",
        }
    )

    def __init__(
        self,
        *,
        config: DeepseekVLV2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.pp_group = get_pp_group()
        self.config = config

        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config

        n_embed = 1280

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # special token for image token sequence format
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

        if self.text_config.topk_method == "noaux_tc":
            self.model = DeepseekV3ForCausalLM(
                config=config.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )
        elif not self.text_config.use_mla:
            self.model = DeepseekForCausalLM(
                config=config.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )
        else:
            self.model = DeepseekV2ForCausalLM(
                config=config.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_qwen2_decoder_as_encoder()

        vm = self.vision_model
        if not hasattr(vm, "dtype"):
            cls = type(vm)
            if not hasattr(cls, "dtype"):
                cls.dtype = property(lambda self: next(self.parameters()).dtype)

        n_embed = 1280
        self.projector = MlpProjector(
            projector_type="linear",
            input_dim=896,
            n_embed=n_embed,
        )

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # special token for image token sequence format
        n_embed = self.projector_config.n_embed
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # This is a typo in original implementation
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> NestedTensors:

        # Pixel_values (global view): [n_image, batch_size, 3, height, width]
        # images_spatial_crop: [n_image, batch_size, [num_tiles_w, num_tiles_h]]
        # images_crop (local view): [n_image, batch_size, num_pathes, 3, h, w]
        # split the pixel and image_crop, all batch_size = 1

        images_in_this_batch = []

        with torch.no_grad():
            for jdx in range(images_spatial_crop.size(0)):
                # with torch.set_grad_enabled(False):
                patches = images_crop[jdx][0].to(torch.bfloat16)  # batch_size = 1
                # patches = images_crop[jdx][0]
                image_ori = pixel_values[jdx]
                crop_shape = images_spatial_crop[jdx][0]

                if torch.sum(patches).item() != 0:  # if all values = 0, no crop
                    # P, C, H, W = patches.shape
                    # crop_flag = 1
                    local_features_1 = self.sam_model(patches)
                    # TODO del patches
                    # torch.compiler.cudagraph_mark_step_begin()
                    local_features_2 = self.qwen2_model(local_features_1)

                    # local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                    local_features = self.projector(local_features_2)

                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.qwen2_model(global_features_1)
                    # global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                    global_features = self.projector(global_features_2)

                    _, hw, n_dim = global_features.shape
                    # h = w = int(hw ** 0.5)

                    _2, hw2, n_dim2 = local_features.shape
                    # h2 = w2 = int(hw2 ** 0.5)

                    # width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                    # global_features = global_features.view(h, w, n_dim)

                    # global_features = torch.cat(
                    #     [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                    # )

                    global_features = global_features.view(-1, n_dim)

                    # local_features = local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2).permute(0, 2, 1, 3, 4).reshape(height_crop_num*h2, width_crop_num*w2, n_dim2)
                    # local_features = torch.cat(
                    #     [local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2)], dim=1
                    # )
                    local_features = local_features.view(-1, n_dim2)

                    global_local_features = torch.cat(
                        [local_features, global_features, self.view_seperator[None, :]],
                        dim=0,
                    )

                else:
                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.qwen2_model(global_features_1)
                    # global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
                    global_features = self.projector(global_features_2)

                    _, hw, n_dim = global_features.shape
                    # h = w = int(hw ** 0.5)

                    # global_features = global_features.view(h, w, n_dim)

                    # global_features = torch.cat(
                    #     [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
                    # )

                    global_features = global_features.view(-1, n_dim)

                    global_local_features = torch.cat(
                        [global_features, self.view_seperator[None, :]], dim=0
                    )

                images_in_this_batch.append(global_local_features)

        return images_in_this_batch

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        vision_embeddings = self._process_image_input(items)
        return vision_embeddings

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def _parse_and_validate_image_input(self, **kwargs: object):

        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)

        if pixel_values is None or torch.sum(pixel_values).item() == 0:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
                )

            if not isinstance(images_spatial_crop, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image sizes. "
                    f"Got type: {type(images_spatial_crop)}"
                )

            if not isinstance(images_crop, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image crop. " f"Got type: {type(images_crop)}"
                )

            return [pixel_values, images_crop, images_spatial_crop]

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(self, image_input) -> torch.Tensor:

        # image_input: [pixel_values, images_crop, images_spatial_crop]

        pixel_values = image_input[0].to(torch.bfloat16)

        # images_crop = image_input[1].to(torch.bfloat16)
        images_crop = image_input[1]
        # images_crop = image_input[1]
        images_spatial_crop = image_input[2].to(dtype=torch.long)

        # local_start = time.time()
        vision_features = self._pixel_values_to_embedding(
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
        )

        return vision_features

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        # input_ids.to(torch.bfloat16)
        # self.image_token_id.to(torch.bfloat16)

        if multimodal_embeddings is not None:
            # multimodal_embeddings = multimodal_embeddings.to(torch.bfloat16)
            # multimodal_embeddings = [emb.to(torch.bfloat16) for emb in multimodal_embeddings]
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, self.image_token_id
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ):

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility
        vision_embeddings = self.get_multimodal_embeddings(**kwargs)
        inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
        input_ids = None
        hidden_states = self.model(
            input_ids=None,
            forward_batch=forward_batch,
            input_embeds=inputs_embeds,
            positions=positions,
            **kwargs,
        )

        return hidden_states

    @staticmethod
    def _is_mm_key(raw_name: str) -> bool:
        return any(
            k in raw_name
            for k in (
                "image_newline",
                "view_seperator",
                "projector",
                "vision_model",
                "sam_model",
                "qwen2_model",
            )
        )

    def _remap_name(
        self, raw_name: str, params_dict: dict[str, torch.nn.Parameter]
    ) -> str:
        """
        HF checkpoint -> wrapper named_parameters mapping.

        Key cases:
        - lm_head.weight -> model.lm_head.weight
        - model.projector.* -> projector.*
        - model.image_newline / model.view_seperator -> image_newline / view_seperator
        - model.qwen2_model.* -> vision_model.* (with candidates)
        - language model: model.* -> model.model.* (prefix-only, once)
        """
        if raw_name == "lm_head.weight":
            return "model.lm_head.weight"

        if not raw_name.startswith("model."):
            return raw_name

        # projector and special params may be under model.*
        if raw_name.startswith("model.projector."):
            return raw_name[len("model.") :]  # projector.*
        if raw_name.startswith("model.image_newline"):
            return raw_name[len("model.") :]  # image_newline
        if raw_name.startswith("model.view_seperator"):
            return raw_name[len("model.") :]  # view_seperator

        # vision encoder path in HF
        if raw_name.startswith("model.qwen2_model."):
            # common possibilities (pick the first that exists)
            cand_a = (
                "vision_model." + raw_name[len("model.qwen2_model.") :]
            )  # drop qwen2_model
            cand_b = "vision_model." + raw_name[len("model.") :]  # keep qwen2_model
            cand_c = "vision_model.qwen2_model." + raw_name[len("model.qwen2_model.") :]
            for c in (cand_a, cand_b, cand_c):
                if c in params_dict:
                    return c
            # return a reasonable default so error is informative
            return cand_a

        # If HF already has explicit vision_model/sam_model etc under model., strip that model. prefix
        if any(
            k in raw_name
            for k in (
                "vision_model",
                "sam_model",
                "projector",
                "image_newline",
                "view_seperator",
            )
        ):
            return raw_name[len("model.") :]

        # language model weights: add one "model." prefix ONLY ONCE
        # raw: model.layers.0... -> model.model.layers.0...
        if raw_name.startswith("model.model."):
            return raw_name
        return "model." + raw_name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # stacked mapping: (fused_name, shard_name, shard_id)
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded: Set[str] = set()

        for raw_name, w in weights:
            if "rotary_emb.inv_freq" in raw_name:
                continue

            # PP weight tying: sometimes only embed_tokens is present but lm_head is needed on last stage
            if (
                getattr(self.pp_group, "is_last_rank", False)
                and raw_name == "model.embed_tokens.weight"
            ):
                lm_key = "model.lm_head.weight"
                if lm_key in params_dict and lm_key not in loaded:
                    param = params_dict[lm_key]
                    wl = getattr(param, "weight_loader", default_weight_loader)
                    wl(param, w)
                    loaded.add(lm_key)

            name = self._remap_name(raw_name, params_dict)

            # Try stacked/fused mapping only if the target exists
            stacked_done = False
            for fused_suffix, shard_suffix, shard_id in stacked_params_mapping:
                if shard_suffix not in name:
                    continue
                candidate = name.replace(shard_suffix, fused_suffix)
                if candidate not in params_dict:
                    continue
                param = params_dict[candidate]
                wl = getattr(param, "weight_loader", None)
                if wl is None:
                    # fused param should provide a sharded loader; if not, skip to normal path
                    continue
                wl(param, w, shard_id)
                loaded.add(candidate)
                stacked_done = True
                break

            if stacked_done:
                continue

            # Normal load
            if name not in params_dict:
                raise KeyError(
                    f"Unmapped weight name: raw='{raw_name}' -> mapped='{name}'"
                )

            param = params_dict[name]
            wl = getattr(param, "weight_loader", default_weight_loader)
            wl(param, w)
            loaded.add(name)

        # unloaded = params_dict.keys() - loaded
        # if unloaded:
        #     raise RuntimeError(f"Unloaded params count={len(unloaded)}; examples={list(sorted(unloaded))[:50]}")


EntryClass = [DeepseekOCR2ForCausalLM]
