# Reference: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sglang.srt.models.paddleocr_vl import PaddleOCRVLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor


class PaddleOCRVLImageProcessor(QwenVLImageProcessor):
    models = [PaddleOCRVLForConditionalGeneration]

    # A document page is a far heavier preprocessing unit than a chat image:
    # resize + normalize + patchify of a full-resolution scan costs tens of
    # milliseconds, so a single worker caps request throughput at
    # 1 / preprocess_time regardless of how much GPU is left idle. Overlap it
    # across workers; the work itself is unchanged.
    #
    # Two, not more: measured on an H200 at 32-way concurrency, two workers beat
    # both one and four on every shape tried (1080p pages 6.72 -> 9.55 req/s at
    # two, 8.92 at four; 360p pages with 512-token outputs 22.38 -> 25.20 at
    # two, 25.06 at four). Past two, spreading request arrivals fragments the
    # GPU prefill batches faster than the extra overlap pays for itself.
    auto_mm_processor_worker_num = 2
    auto_mm_io_worker_num = 16
    supports_mm_processor_concurrency = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>",
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
        ).build(_processor)
