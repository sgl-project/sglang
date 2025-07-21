# Copyright 2024 SGLang Team
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

import logging
import re
from typing import Any, List, Mapping, Optional, Sequence, Union

import torch
from transformers.processing_utils import ProcessorMixin

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.phi3v import Phi3VForCausalLM, _IMAGE_TOKEN_ID
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)


class Phi3VProcessorAdapter(ProcessorMixin):
    """Adapter to make HF Phi3V processor work with SGLang."""
    
    def __init__(self, _processor) -> None:
        self._processor = _processor
        # Store image tokens from the processor
        self.img_tokens = getattr(_processor, "img_tokens", ["<|image|>"])
    
    def __getattr__(self, name):
        return getattr(self._processor, name)
    
    def __call__(self, **kwargs):
        # Call the underlying processor
        result = self._processor(**kwargs)
        
        # Ensure input_ids is a tensor
        if "input_ids" in result and isinstance(result["input_ids"], list):
            result["input_ids"] = torch.tensor(result["input_ids"])
        
        # Replace placeholder tokens with _IMAGE_TOKEN_ID
        if "input_ids" in result:
            input_ids = result["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                # Phi3v processor has inserted -1, -2 etc as placeholder in prompt_ids,
                # which will cause OverflowError when decoding the prompt_ids.
                # Therefore, we need to do an early replacement here
                input_ids.masked_fill_(input_ids < 0, _IMAGE_TOKEN_ID)
                result["input_ids"] = input_ids
        
        return result
    
    def calc_num_image_tokens_from_image_size(self, width: int, height: int) -> int:
        """Calculate the number of image tokens based on image size."""
        # Default implementation based on Phi3V's HD transform
        # Each 336x336 crop generates 12x12 = 144 tokens after 2x2 pooling
        # Plus additional tokens for newlines and separators
        h_crop = height // 336
        w_crop = width // 336
        
        # Global features: 12*12 + 12 (newlines) + 1 (separator) = 157
        global_tokens = 12 * 12 + 12 + 1
        
        # Sub features: (h_crop * 12) * (w_crop * 12 + 1) tokens
        sub_tokens = (h_crop * 12) * (w_crop * 12 + 1)
        
        return sub_tokens + global_tokens


class Phi3VMultimodalProcessor(BaseMultimodalProcessor):
    models = [Phi3VForCausalLM]

    def __init__(self, hf_config, server_args, _processor):
        # Wrap the processor with our adapter
        self.processor = Phi3VProcessorAdapter(_processor)
        super().__init__(hf_config, server_args, self.processor)

        # Get image tokens from processor
        self.image_tokens = getattr(self.processor, "img_tokens", ["<|image|>"])
        
        # Build multimodal tokens
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_tokens[0] if self.image_tokens else "<|image|>",
            image_token_id=_IMAGE_TOKEN_ID,
        ).build(self.processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        # Load multimodal data
        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        # Process and combine multimodal data
        mm_items, input_ids, prompt = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        # Apply Phi3V-specific prompt formatting if needed
        if len(mm_items) > 0:
            # Get tokenizer for special token handling
            tokenizer = self.processor.tokenizer
            
            # Convert input_ids to text for pattern matching
            if len(input_ids) and input_ids[0] == tokenizer.bos_token_id:
                # Remove initial BOS token
                input_ids_for_decode = input_ids[1:]
            else:
                input_ids_for_decode = input_ids
                
            text = tokenizer.decode(input_ids_for_decode)
            
            # Remove spaces after special tokens
            for special_tokens in tokenizer.special_tokens_map.values():
                if isinstance(special_tokens, str):
                    text = text.replace(f"{special_tokens} ", special_tokens)
                elif isinstance(special_tokens, list):
                    for special_token in special_tokens:
                        text = text.replace(f"{special_token} ", special_token)
            
            # Apply HF behavior for image token formatting
            pattern = r"<\|image_\d+\|>"
            prompt_chunks = [
                tokenizer(chunk).input_ids
                for chunk in re.split(pattern, text)
            ]
            image_tags = [
                tokenizer(chunk, add_special_tokens=False).input_ids
                for chunk in re.findall(pattern, text)
            ]
            if len(prompt_chunks) > len(image_tags):
                image_tags.append([])
            
            # Rebuild input_ids with proper formatting
            new_input_ids = []
            for chunk, tag in zip(prompt_chunks, image_tags):
                new_input_ids.extend(chunk)
                new_input_ids.extend(tag)
            
            # Add remaining chunks if any
            if len(prompt_chunks) > len(image_tags):
                for chunk in prompt_chunks[len(image_tags):]:
                    new_input_ids.extend(chunk)
            
            input_ids = torch.tensor(new_input_ids)

        # Return the processed data
        return {
            "input_ids": input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids,
            "mm_items": mm_items,
            "image_token_id": self.mm_tokens.image_token_id,
        }
    
    def _get_num_image_tokens(self, image_width: int, image_height: int) -> int:
        """Calculate the number of tokens an image will produce."""
        return self.processor.calc_num_image_tokens_from_image_size(
            width=image_width,
            height=image_height,
        )