# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only LLaVa model compatible with HuggingFace weights."""

import math
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    LlavaConfig,
    MistralConfig,
    Qwen2Config,
    SiglipVisionModel,
)
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector

# leave till last and symbol only in case circular import
import sglang.srt.models as sgl_models
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.models.mistral import MistralForCausalLM
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.multimodal.mm_utils import (
    get_anyres_image_grid_shape,
    unpad_image,
    unpad_image_shape,
)
from sglang.srt.utils import add_prefix, flatten_nested_list, logger


class LlavaBaseForCausalLM(nn.Module):
    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        image_sizes = flatten_nested_list(
            [item.image_sizes for item in image_inputs.mm_items]
        )

        pad_values = [item.pad_value for item in image_inputs.mm_items]

        # hardcode for spatial_unpad + anyres
        if any(
            item.modality == Modality.MULTI_IMAGES or item.modality == Modality.VIDEO
            for item in image_inputs.mm_items
        ):
            image_aspect_ratio = "pad"
        else:
            image_aspect_ratio = "anyres"
        offset_list = []
        image_inputs.image_pad_len = []
        for image_idx, image_s in enumerate(image_sizes):
            if len(image_sizes) > 16:
                # 2x2 pooling with stride 2
                new_image_feature_len = (
                    math.ceil(self.image_size / self.patch_size / 2) ** 2
                )
            else:
                new_image_feature_len = self.image_feature_len  # multi-image

            height = width = self.num_patches_per_side
            if "anyres" in image_aspect_ratio:
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_s,
                    self.image_grid_pinpoints,
                    self.vision_tower.config.image_size,
                )
                h = num_patch_height * height
                w = num_patch_width * width
                new_h, new_w = unpad_image_shape(h, w, image_s)

                if "anyres_max" in self.config.image_aspect_ratio:
                    matched_anyres_max_num_patches = re.match(
                        r"anyres_max_(\d+)", self.config.image_aspect_ratio
                    )
                    if matched_anyres_max_num_patches:
                        max_num_patches = int(matched_anyres_max_num_patches.group(1))
                    # times = math.sqrt(h * w / (max_num_patches * unit**2))
                    times = math.sqrt(
                        new_h * new_w / (max_num_patches * self.image_feature_len)
                    )
                    if times > 1.1:
                        new_h = int(new_h // times)
                        new_w = int(new_w // times)
                new_image_feature_len += new_h * (new_w + 1)

            try:
                offset = input_ids.index(self.config.image_token_index)
            except ValueError:
                offset = 0
            # old_len + pad_len - 1, because we need to remove image_token_id
            input_ids = (
                input_ids[:offset]
                + [pad_values[image_idx % len(pad_values)]] * new_image_feature_len
                + input_ids[offset + 1 :]
            )
            offset_list.append(offset)
            image_inputs.image_pad_len.append(new_image_feature_len)

        image_inputs.image_offsets = offset_list
        return input_ids

    def encode_images(
        self, pixel_values: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        encode images by vision tower and multimodal projector
        Args:
            pixel_values: torch.Tensor or List[torch.Tensor]: each tensor for an input image
        Returns:
            torch.Tensor: encoded image features from the input image; if multiple, flattened by seq_len axis
        """
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        # NOTE: This is not memory efficient. (output_hidden_states=True) will save all the hidden stated.
        selected_image_feature = image_outputs.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy in ["default", "patch"]:
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch,
    ) -> torch.Tensor:
        image_inputs = forward_batch.mm_inputs

        if forward_batch.forward_mode.is_extend():
            # 图片占位的哈希 token 需要 clamp；真正的视觉特征稍后会覆盖
            input_ids.clamp_(min=0, max=self.config.vocab_size - 1)

            # 文本嵌入；稍后把视觉特征 scatter 回到这里
            input_embeds = self.language_model.model.embed_tokens(input_ids)

            # 计算哪些样本在本 step 需要填入视觉特征
            max_image_offset = []
            for im in image_inputs:
                if im and im.image_offsets:
                    max_image_offset.append(
                        np.max(np.array(im.image_offsets) + np.array(im.image_pad_len))
                    )
                else:
                    max_image_offset.append(-1)

            start_positions = positions[forward_batch.extend_start_loc].cpu().numpy()
            need_vision = start_positions <= np.array(max_image_offset)
            need_indices = [
                i for i in range(forward_batch.batch_size) if need_vision[i]
            ]

            print(
                f"[llava.forward] extend step: bs={forward_batch.batch_size} "
                f"start_pos={start_positions.tolist()} "
                f"max_image_offset={max_image_offset} "
                f"need_indices={need_indices}"
            )

            if need_vision.any():
                bs = forward_batch.batch_size

                # 每个需要视觉的请求包含的图片数（用于 split 分组）
                per_req_img_counts = [
                    len(image_inputs[i].mm_items) for i in range(bs) if need_vision[i]
                ]
                print(
                    f"[llava.forward] per-req image counts (need only)={per_req_img_counts}"
                )

                # 扁平化像素与尺寸；encode 后再按请求还原二维分组结构 List[req][img]
                pixel_values = flatten_nested_list(
                    [
                        [item.feature for item in image_inputs[i].mm_items]
                        for i in need_indices
                    ]
                )
                image_sizes_grouped = [
                    flatten_nested_list(
                        [item.image_sizes for item in image_inputs[i].mm_items]
                    )
                    for i in need_indices
                ]

                if len(pixel_values) == 0:
                    print("[llava.forward] no pixel_values -> fall back to LM only")
                    return self.language_model(
                        input_ids, positions, forward_batch, input_embeds=input_embeds
                    )

                # ---- 尺寸解析小工具：从 [(H,W)] / [[(H,W)]] 等结构里剥到 (H,W)
                def _resolve_img_size(s):
                    cur = s
                    for _ in range(4):
                        if isinstance(cur, (list, tuple)):
                            if len(cur) == 0:
                                break
                            if isinstance(cur[0], (int, float)) and len(cur) >= 2:
                                return int(cur[0]), int(cur[1])
                            cur = cur[0]
                        else:
                            break
                    return None

                # ---- 统一二维化，避免 cat 时报维度不一致
                def _to_2d(t: torch.Tensor) -> torch.Tensor:
                    if t is None:
                        return t
                    if t.dim() == 2:
                        return t
                    if t.dim() == 3:
                        # (#tiles_or_frames, H*W, hidden) -> (tiles*H*W, hidden)
                        return t.flatten(0, 1)
                    if t.dim() == 4:
                        # 比如 (N, C, H, W) 等，将前 3 维合并为 tokens，最后一维作为 hidden
                        return t.flatten(0, 2)
                    # 其它情况：保留最后一维为 hidden，前面合并
                    return t.view(-1, t.shape[-1])

                ########## 1) 编码图片：保持 List[req][img] 结构 ##########
                if pixel_values[0] is None:
                    print(
                        "[llava.forward] pixel_values[0] is None -> fall back to LM only"
                    )
                    return self.language_model(
                        input_ids, positions, forward_batch, input_embeds=input_embeds
                    )

                if getattr(pixel_values[0], "ndim", None) == 4:
                    # HD：每张图为 (num_patch, C=3, H, W)
                    per_img_patch_counts = [img.shape[0] for img in pixel_values]
                    print(
                        f"[llava.forward] HD path: per_img_patch_counts={per_img_patch_counts}"
                    )
                    concat_images = torch.tensor(
                        np.concatenate(pixel_values, axis=0),
                        device=self.vision_tower.device,
                    )
                    flat_feats = self.encode_images(
                        concat_images
                    )  # (sum_patches, hidden)
                    # 先按每张图的 patch 数拆，再按请求聚合
                    per_img_feats = list(
                        torch.split(flat_feats, per_img_patch_counts, dim=0)
                    )
                    image_features = []
                    cursor = 0
                    for cnt in per_req_img_counts:
                        image_features.append(per_img_feats[cursor : cursor + cnt])
                        cursor += cnt
                    # 记录每张图编码后 tokens
                    enc_tokens = [f.shape[0] for f in per_img_feats]
                    print(f"[llava.forward] HD enc per-image tokens={enc_tokens}")
                else:
                    # normal：每张图 (3, H, W)，encode 输出 (sum_imgs, T, hidden)（通常 T=576）
                    print(
                        f"[llava.forward] normal path: num_images={len(pixel_values)} "
                        f"sample_feature_shape={getattr(pixel_values[0], 'shape', None)}"
                    )
                    pixel_values_tensor = torch.tensor(
                        np.array(pixel_values), device=self.vision_tower.device
                    )
                    flat_feats = self.encode_images(
                        pixel_values_tensor
                    )  # (sum_imgs, T, hidden)
                    per_req_feats = list(
                        torch.split(flat_feats, per_req_img_counts, dim=0)
                    )
                    image_features = [
                        list(req_feat) for req_feat in per_req_feats
                    ]  # List[req][img]
                    # 记录每张图编码后 tokens
                    enc_tokens = (
                        [f.shape[1] for f in flat_feats]
                        if flat_feats.dim() == 3
                        else []
                    )
                    if not enc_tokens:
                        # 逐图统计
                        tmp = []
                        for req_feat in per_req_feats:
                            for f in req_feat:
                                tmp.append(
                                    f.shape[0]
                                    if f.dim() == 2
                                    else (f.shape[1] if f.dim() == 3 else -1)
                                )
                        enc_tokens = tmp
                    print(f"[llava.forward] normal enc per-image tokens={enc_tokens}")

                ########## 2) 空间合并 / anyres / video 等后处理（严格使用 req_idx,image_idx 的尺寸） ##########
                if self.mm_patch_merge_type.startswith("spatial"):
                    height = width = self.num_patches_per_side
                    grouped_merged = []
                    for req_idx, per_req in enumerate(image_features):
                        merged_per_req = []
                        for image_idx, image_feat in enumerate(per_req):
                            # 当前图片的 modality 与尺寸（逐图取，不要用 0 号）
                            modality = (
                                image_inputs[need_indices[req_idx]]
                                .mm_items[image_idx]
                                .modality
                            )
                            raw_size = None
                            if req_idx < len(image_sizes_grouped) and image_idx < len(
                                image_sizes_grouped[req_idx]
                            ):
                                raw_size = _resolve_img_size(
                                    image_sizes_grouped[req_idx][image_idx]
                                )

                            # multi-images / video 统一按 pad；单图走 config
                            if modality == Modality.IMAGE:
                                image_aspect_ratio = self.config.image_aspect_ratio
                            elif modality in (Modality.MULTI_IMAGES, Modality.VIDEO):
                                image_aspect_ratio = "pad"
                            else:
                                image_aspect_ratio = self.config.image_aspect_ratio

                            print(
                                f"[llava.forward] postproc req={need_indices[req_idx]} img={image_idx} "
                                f"modality={modality} aspect={image_aspect_ratio} raw_size={raw_size} "
                                f"feat_shape={tuple(image_feat.shape)}"
                            )

                            # ---- anyres 分支（仅单图 IMAGE 才会走）
                            if (
                                image_feat.shape[0] > 1
                                and "anyres" in image_aspect_ratio
                                and modality == Modality.IMAGE
                            ):
                                base_image_feature = _to_2d(
                                    image_feat[0]
                                )  # (H*W, hidden)
                                tiles_feature = image_feat[
                                    1:
                                ]  # (tiles, H*W, hidden) 3D

                                matched_anyres_max_num_patches = None
                                if "anyres_max" in image_aspect_ratio:
                                    matched_anyres_max_num_patches = re.match(
                                        r"anyres_max_(\d+)", image_aspect_ratio
                                    )
                                    if matched_anyres_max_num_patches:
                                        max_num_patches = int(
                                            matched_anyres_max_num_patches.group(1)
                                        )

                                if (
                                    image_aspect_ratio == "anyres"
                                    or "anyres_max" in image_aspect_ratio
                                ):
                                    vision_tower_image_size = self.image_size
                                    try:
                                        if raw_size is None:
                                            num_patch_width, num_patch_height = 2, 2
                                        else:
                                            num_patch_width, num_patch_height = (
                                                get_anyres_image_grid_shape(
                                                    raw_size,
                                                    self.config.image_grid_pinpoints,
                                                    vision_tower_image_size,
                                                )
                                            )
                                    except Exception as e:
                                        print(
                                            f"[llava.forward][anyres] grid shape error: {e}"
                                        )
                                        num_patch_width, num_patch_height = 2, 2
                                    # (tiles, H*W, hidden) -> (H_tiles, W_tiles, H, W, hidden)
                                    tiles_feature = tiles_feature.view(
                                        num_patch_height,
                                        num_patch_width,
                                        height,
                                        width,
                                        -1,
                                    )
                                else:
                                    tiles_feature = tiles_feature.view(
                                        2, 2, height, width, -1
                                    )

                                if "unpad" in self.mm_patch_merge_type:
                                    unit = tiles_feature.shape[2]
                                    tiles_feature = tiles_feature.permute(
                                        4, 0, 2, 1, 3
                                    ).contiguous()
                                    tiles_feature = tiles_feature.flatten(1, 2).flatten(
                                        2, 3
                                    )
                                    if raw_size is not None:
                                        tiles_feature = unpad_image(
                                            tiles_feature, raw_size
                                        )
                                    if (
                                        "anyres_max" in image_aspect_ratio
                                        and matched_anyres_max_num_patches
                                    ):
                                        c, h, w = tiles_feature.shape
                                        times = math.sqrt(
                                            h * w / (max_num_patches * unit * unit)
                                        )
                                        if times > 1.1:
                                            tiles_feature = tiles_feature[None]
                                            tiles_feature = nn.functional.interpolate(
                                                tiles_feature,
                                                [int(h // times), int(w // times)],
                                                mode="bilinear",
                                            )[0]
                                    tiles_feature = torch.cat(
                                        (
                                            tiles_feature,
                                            self.language_model.model.image_newline[
                                                :, None, None
                                            ].expand(*tiles_feature.shape[:-1], 1),
                                        ),
                                        dim=-1,
                                    )
                                    tiles_feature = tiles_feature.flatten(
                                        1, 2
                                    ).transpose(
                                        0, 1
                                    )  # -> (tokens, hidden)
                                else:
                                    # non-unpad：拍平到 2D
                                    tiles_feature = tiles_feature.permute(
                                        0, 2, 1, 3, 4
                                    ).contiguous()
                                    tiles_feature = tiles_feature.flatten(
                                        0, 3
                                    )  # -> (tiles*H*W, hidden)

                                # ✅ 保证 2D 再 cat
                                tiles_feature = _to_2d(tiles_feature)
                                base_image_feature = _to_2d(base_image_feature)
                                image_feat = torch.cat(
                                    (base_image_feature, tiles_feature), dim=0
                                )  # (tokens, hidden)

                            else:
                                # ---- 非 anyres：保持 (tokens, hidden)；video 做 2x2 下采样后转 2D
                                if modality == Modality.VIDEO:
                                    num_of_frames = image_feat.shape[0]
                                    image_feat = image_feat.view(
                                        num_of_frames, height, width, -1
                                    )
                                    image_feat = image_feat.permute(
                                        0, 3, 1, 2
                                    ).contiguous()  # N,C,H,W
                                    hh, ww = image_feat.shape[2:]
                                    scaled = [math.ceil(hh / 2), math.ceil(ww / 2)]
                                    image_feat = nn.functional.interpolate(
                                        image_feat, size=scaled, mode="bilinear"
                                    )
                                    image_feat = (
                                        image_feat.flatten(2)
                                        .transpose(1, 2)
                                        .contiguous()
                                    )  # N, H*W, C

                                image_feat = _to_2d(image_feat)  # ✅ 强制 2D

                            # 统一在这里（按需）追加 newline（spatial-unpad 分支已经追加过）
                            if "unpad" in self.mm_patch_merge_type:
                                image_feat = torch.cat(
                                    (
                                        image_feat,
                                        self.language_model.model.image_newline[
                                            None, :
                                        ].expand(1, image_feat.shape[-1]),
                                    ),
                                    dim=0,
                                )

                            print(
                                f"[llava.forward] postproc out req={need_indices[req_idx]} img={image_idx} "
                                f"tokens={image_feat.shape[0]}"
                            )
                            merged_per_req.append(image_feat)  # (tokens, hidden)
                        grouped_merged.append(merged_per_req)
                    image_features = (
                        grouped_merged  # List[req][img], each (tokens, hidden)
                    )

                ########## 3) 把每张图的 tokens scatter 回本 step 的 input_embeds ##########
                extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
                extend_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()
                prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

                print(
                    f"[llava.forward] scatter window: start={extend_start_loc_cpu.tolist()} "
                    f"seqlen={extend_seq_lens.tolist()} prefix={prefix_lens_cpu}"
                )

                pt = 0  # 指向 image_features 中“第几个需要视觉的请求”
                for i in range(bs):
                    if not need_vision[i]:
                        continue

                    start_idx = extend_start_loc_cpu[i]
                    seq_len = extend_seq_lens[i]
                    prefix_len = prefix_lens_cpu[i]

                    if pt >= len(image_features):
                        print(
                            f"[llava.forward] pt({pt}) >= len(image_features)({len(image_features)}), break"
                        )
                        break  # 安全保护

                    # === 基于 model_specific_data.original_index 构建取数映射 ===
                    mm_items_i = getattr(image_inputs[i], "mm_items", []) or []
                    index_by_orig = {}
                    for j, it in enumerate(mm_items_i):
                        msd = getattr(it, "model_specific_data", None) or {}
                        orig = msd.get("original_index", j)
                        index_by_orig[orig] = j

                    # 逐图 scatter
                    for image_idx, image_offset in enumerate(
                        image_inputs[i].image_offsets
                    ):
                        fetch_idx = index_by_orig.get(image_idx, image_idx)

                        if fetch_idx >= len(image_features[pt]):
                            print(
                                f"[llava.forward][skip] req={i} img={image_idx} fetch_idx={fetch_idx} "
                                f">= len(image_features[{pt}])={len(image_features[pt])}"
                            )
                            continue  # 安全保护

                        placeholder_len = image_inputs[i].image_pad_len[image_idx]

                        # 完全在 prefix 左侧或右侧则跳过
                        if (image_offset + placeholder_len) <= prefix_len:
                            print(
                                f"[llava.forward][skip-left] req={i} img={image_idx} "
                                f"offset={image_offset} placeholder={placeholder_len} prefix={prefix_len}"
                            )
                            continue
                        if image_offset >= prefix_len + seq_len:
                            print(
                                f"[llava.forward][skip-right] req={i} img={image_idx} "
                                f"offset={image_offset} seq_end={prefix_len + seq_len}"
                            )
                            break

                        tmp_image_feature = image_features[pt][
                            fetch_idx
                        ]  # (tokens, hidden)
                        if tmp_image_feature.dim() != 2:
                            tmp_image_feature = _to_2d(tmp_image_feature)
                            if tmp_image_feature.dim() != 2:
                                print(
                                    f"[llava.forward][skip-shape] req={i} img={image_idx} "
                                    f"feat_dim={tmp_image_feature.dim()}"
                                )
                                continue

                        enc_len = tmp_image_feature.shape[0]
                        input_offset = image_offset - prefix_len
                        left_idx = start_idx + input_offset
                        right_idx = left_idx + enc_len

                        print(
                            f"[llava.forward] scatter req={i} img={image_idx} "
                            f"fetch_idx={fetch_idx} offset={image_offset} "
                            f"pad_len(placeholder)={placeholder_len} pad_len(enc)={enc_len} "
                            f"window=[{start_idx},{start_idx+seq_len}) "
                            f"write=[{left_idx},{right_idx})"
                        )

                        # 左裁剪
                        if left_idx < start_idx:
                            cut = start_idx - left_idx
                            if cut < enc_len:
                                tmp_image_feature = tmp_image_feature[cut:]
                                left_idx = start_idx
                            else:
                                print(
                                    f"[llava.forward][skip-cut-left] req={i} img={image_idx} cut={cut} >= enc_len={enc_len}"
                                )
                                continue

                        # 右裁剪
                        if right_idx > start_idx + seq_len:
                            cut = right_idx - (start_idx + seq_len)
                            if cut < tmp_image_feature.shape[0]:
                                tmp_image_feature = tmp_image_feature[:-cut]
                                right_idx = start_idx + seq_len
                            else:
                                print(
                                    f"[llava.forward][skip-cut-right] req={i} img={image_idx} cut={cut} >= cur_len={tmp_image_feature.shape[0]}"
                                )
                                continue

                        if right_idx > left_idx and tmp_image_feature.numel() > 0:
                            try:
                                input_embeds[left_idx:right_idx] = tmp_image_feature
                                print(
                                    f"[llava.forward] scatter OK req={i} img={image_idx} "
                                    f"final_write=[{left_idx},{right_idx}) tokens={tmp_image_feature.shape[0]}"
                                )
                            except RuntimeError as e:
                                print(f"[llava.forward][scatter-failed] {e}")
                                print(
                                    f"{input_embeds.shape=}, {tmp_image_feature.shape=}"
                                )
                                print(
                                    f"{start_idx=}, {image_offset=}, {prefix_len=}, {enc_len=}"
                                )
                        else:
                            print(
                                f"[llava.forward][skip-empty] req={i} img={image_idx} "
                                f"write=[{left_idx},{right_idx})"
                            )

                    pt += 1

            # 进入语言模型
            return self.language_model(
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )

        elif forward_batch.forward_mode.is_decode():
            return self.language_model(input_ids, positions, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Load clip vision model by cfg['mm_vision_tower']:
        # huggingface_name or path_of_clip_relative_to_llava_model_dir
        # We put the initialization here instead of __init__ to allow it being reused by other subclasses.
        vision_path = self.config.mm_vision_tower
        if "clip" in vision_path:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                vision_path, torch_dtype=torch.float16
            ).cuda()
        elif "siglip" in vision_path:
            self.vision_tower = SiglipVisionModel.from_pretrained(
                vision_path, torch_dtype=torch.float16
            ).cuda()
            # Siglip needs all feature tokens
            self.config.mm_vision_select_feature = "full"
        self.vision_tower.eval()

        self.vision_feature_layer = self.config.mm_vision_select_layer
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size

        self.mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        self.image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        self.image_grid_pinpoints = getattr(self.config, "image_grid_pinpoints", None)

        self.image_feature_len = int((self.image_size // self.patch_size) ** 2)
        if (
            self.vision_feature_select_strategy == "patch"
            or self.vision_feature_select_strategy == "full"
        ):
            pass
        elif self.vision_feature_select_strategy == "cls_patch":
            self.image_feature_len += 1
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        # load mm_projector
        projector_weights = {
            "model.mm_projector.0": "multi_modal_projector.linear_1",
            "model.mm_projector.2": "multi_modal_projector.linear_2",
            "model.vision_tower.vision_tower": "vision_tower",
            # Update the vision tower weights if we find them in the checkpoint (it may be finetuned).
            "model.image_newline": "language_model.model.image_newline",
        }
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "projector" in name or "vision_tower" in name or "image_newline" in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                self.language_model.load_weights([(name, loaded_weight)])

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size


class LlavaLlamaForCausalLM(LlavaBaseForCausalLM):
    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.vision_tower = None
        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = LlamaForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )


class LlavaQwenForCausalLM(LlavaBaseForCausalLM):
    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.vision_tower = None

        if getattr(self.config, "vision_config", None) is None:
            self.config.vision_config = CLIPVisionConfig(self.config.mm_vision_tower)
        if getattr(self.config, "text_config", None) is None:
            self.config.text_config = Qwen2Config(self.config._name_or_path)

        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size

        if getattr(self.config, "projector_hidden_act", None) is None:
            self.config.projector_hidden_act = "gelu"
        if getattr(self.config, "image_token_index", None) is None:
            self.config.image_token_index = 151646

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = Qwen2ForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )


class LlavaMistralForCausalLM(LlavaBaseForCausalLM):
    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.vision_tower = None

        if getattr(self.config, "vision_config", None) is None:
            self.config.vision_config = CLIPVisionConfig(self.config.mm_vision_tower)
        if getattr(self.config, "text_config", None) is None:
            self.config.text_config = MistralConfig(self.config._name_or_path)

        self.config.vision_config.hidden_size = config.mm_hidden_size
        self.config.text_config.hidden_size = config.hidden_size

        if getattr(self.config, "projector_hidden_act", None) is None:
            self.config.projector_hidden_act = "gelu"
        if getattr(self.config, "image_token_index", None) is None:
            self.config.image_token_index = 32000

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = MistralForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(config.text_config.hidden_size, dtype=torch.float16)
            )


class LlavaForConditionalGeneration(LlavaBaseForCausalLM):
    """
    An adaptor class to enable support for multiple mmlm such as mistral-community/pixtral-12b
    It follows the structure of (vision_tower, multi_modal_projector, language_model)

    Once a model config is loaded, text_config and vision_config will be extracted, and
    LlavaForConditionalGeneration will load the language_model and vision_tower models
    according to config.
    """

    MULTIMODAL_PROJECTOR_TYPE = LlavaMultiModalProjector

    @property
    def dtype(self):
        return self.torch_dtype

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        if hasattr(self.vision_tower, "pad_input_ids"):
            return self.vision_tower.pad_input_ids(input_ids, image_inputs)
        else:
            return super().pad_input_ids(input_ids, image_inputs)

    def _get_sgl_model_cls(self, config, auto_model_type: Type[AutoModel] = AutoModel):
        """
        Get the SGLang model implementation class according to config.

        Args:
            config: The config object of the model.
            auto_model_type: The type of the auto model.

        Returns:
            The SGLang model implementation class.
        """
        config_cls_name = config.__class__.__name__
        arch_name_mapping = self._config_cls_name_to_arch_name_mapping(auto_model_type)
        if arch := arch_name_mapping.get(config_cls_name):
            if isinstance(arch, tuple):
                arch = arch[0]
                logger.warning(
                    f"Multiple {auto_model_type.__name__} models found for submodule config `{config_cls_name}`, defaulting to [0]: {arch.__name__}"
                )
            try:
                return sgl_models.registry.ModelRegistry.resolve_model_cls(arch)[0]
            except Exception as e:
                raise ValueError(
                    f"{auto_model_type.__name__} found a corresponding model `{arch}` for config class `{config_cls_name}`, but failed to load it from SGLang ModelRegistry. \n{e}"
                )
        else:
            raise ValueError(
                f"{auto_model_type.__name__} cannot find a corresponding model for config class `{config_cls_name}`"
            )

    @lru_cache
    def _config_cls_name_to_arch_name_mapping(
        self, auto_model_type: Type[AutoModel]
    ) -> Dict[str, str]:
        mapping = {}
        for config_cls in auto_model_type._model_mapping.keys():
            archs = auto_model_type._model_mapping.get(config_cls, None)
            if archs is not None:
                if isinstance(archs, tuple):
                    mapping[config_cls.__name__] = tuple(
                        arch.__name__ for arch in archs
                    )
                else:
                    mapping[config_cls.__name__] = archs.__name__
        return mapping

    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert hasattr(config, "text_config")
        assert hasattr(config, "vision_config")
        self.config = config
        self.text_config = self.config.text_config
        self.vision_config = self.config.vision_config
        self.torch_dtype = getattr(self.config, "torch_dtype")

        if not getattr(self.text_config, "torch_dtype"):
            self.text_config.torch_dtype = self.torch_dtype
        if not getattr(self.vision_config, "torch_dtype"):
            self.vision_config.torch_dtype = self.torch_dtype

        if not hasattr(self.config, "vocab_size"):
            self.config.vocab_size = self.text_config.vocab_size
        if not hasattr(self.config, "image_aspect_ratio"):
            self.config.image_aspect_ratio = "anyres"
        if not hasattr(self.config, "image_grid_pinpoints"):
            # from transformers.models.llava_onevision.configuration_llava_onevision import LlavaOnevisionConfig
            # self.config.image_grid_pinpoints = LlavaOnevisionConfig().image_grid_pinpoints
            self.config.image_grid_pinpoints = [
                [96, 96],
                [224, 224],
                [384, 384],
                [512, 512],
                [768, 768],
                [1024, 1024],
            ]
        if not hasattr(self.config, "mm_patch_merge_type"):
            self.config.mm_patch_merge_type = "flat"
        if not hasattr(self.config, "image_token_index"):
            self.config.image_token_index = 10
        if not hasattr(self.config, "projector_hidden_act"):
            self.config.projector_hidden_act = "gelu"

        self.vision_feature_layer = getattr(self.config, "vision_feature_layer", -1)
        self.vision_feature_select_strategy = getattr(
            self.config, "vision_feature_select_strategy", "full"
        )
        self.image_size = self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size

        self.mm_patch_merge_type = self.config.mm_patch_merge_type
        self.image_aspect_ratio = self.config.image_aspect_ratio
        self.image_grid_pinpoints = self.config.image_grid_pinpoints

        self.image_feature_len = int((self.image_size // self.patch_size) ** 2)

        self.multi_modal_projector = self.MULTIMODAL_PROJECTOR_TYPE(config)

        language_model_cls = self._get_sgl_model_cls(
            self.text_config, AutoModelForCausalLM
        )
        vision_model_cls = self._get_sgl_model_cls(self.vision_config, AutoModel)
        self.language_model = language_model_cls(
            self.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.vision_tower = vision_model_cls(
            self.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )

        if "unpad" in getattr(self.config, "mm_patch_merge_type", ""):
            self.language_model.model.image_newline = nn.Parameter(
                torch.empty(self.text_config.hidden_size, dtype=self.torch_dtype)
            )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract features from image inputs.

        Args:
            items: List of MultimodalDataItem objects containing image data
                Note that an item can be either "image" or "multi-images"

        Returns:
            torch.Tensor: features from image inputs, concatenated
        """
        features = []
        for item in items:
            # in each item, we assume pixel_values is always batched
            pixel_values, image_sizes = item.feature, item.image_sizes
            image_outputs = self.vision_tower(
                pixel_values, image_sizes, output_hidden_states=True
            )
            selected_image_feature = image_outputs.hidden_states[
                self.vision_feature_layer
            ]

            if self.vision_feature_select_strategy in ["default", "patch"]:
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    f"Unexpected select feature: {self.vision_feature_select_strategy}"
                )
            features.append(
                self.multi_modal_projector(selected_image_feature.squeeze(0))
            )
        ret = torch.cat(features, dim=0)
        return ret

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            get_embedding=get_embedding,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            placeholder_tokens=None,  # using mm_item.pad_value
            positions=positions,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for LlavaForConditionalGeneration.

        Unlike the base class implementation, this one doesn't need to handle
        weight name remapping as the weights are already properly structured with
        'language_model' and 'vision_tower' prefixes in the safetensors files.
        """
        if (
            self.vision_feature_select_strategy == "patch"
            or self.vision_feature_select_strategy == "full"
        ):
            pass
        elif self.vision_feature_select_strategy == "cls_patch":
            self.image_feature_len += 1
        else:
            raise ValueError(
                f"Unexpected select feature: {self.vision_feature_select_strategy}"
            )

        # Create dictionaries for direct parameter loading
        params_dict = dict(self.named_parameters())

        # Load weights directly without remapping
        for name, loaded_weight in weights:
            for part in ("language_model", "vision_tower"):
                if name.startswith(part):
                    name = name[len(part + ".") :]
                    getattr(self, part).load_weights([(name, loaded_weight)])
                    break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [
    LlavaLlamaForCausalLM,
    LlavaQwenForCausalLM,
    LlavaMistralForCausalLM,
    LlavaForConditionalGeneration,
]
