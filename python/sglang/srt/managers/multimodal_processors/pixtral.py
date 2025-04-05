import asyncio
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers.processing_utils import ProcessorMixin
from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.llava import LlavaForConditionalGeneration
from sglang.srt.utils import load_image
from sglang.utils import get_exception_traceback

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(pathname)s:%(lineno)d - %(message)s')

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


class PixtralImageProcessor(BaseMultimodalProcessor, ProcessorMixin):
    """
    Processor for the Pixtral vision-language model.
    
    This processor handles image processing for Pixtral models and is designed
    to work with the LlavaForConditionalGeneration class (which includes Pixtral
    vision model support).

    """
    
    # Register for LlavaForConditionalGeneration which has Pixtral vision model support
    models = [LlavaForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.NUM_TOKEN_PER_FRAME = 330  # Default estimate for Pixtral
        logger.debug(f"PixtralImageProcessor initialized: hf_config={type(hf_config)}, NUM_TOKEN_PER_FRAME={self.NUM_TOKEN_PER_FRAME}")
        if hasattr(self._processor, 'image_processor'):
            logger.debug(f"Image processor type: {type(self._processor.image_processor)}")
            if hasattr(self._processor.image_processor, 'crop_size'):
                logger.debug(f"Image processor crop_size: {self._processor.image_processor.crop_size}")
            if hasattr(self._processor.image_processor, 'size'):
                logger.debug(f"Image processor size: {self._processor.image_processor.size}")

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes],
        processor=None,
    ):
        """
        Process a single image for the Pixtral model.
        
        Args:
            image_data: Image data as a path, bytes, or PIL Image
            processor: The image processor to use
            
        Returns:
            Tuple of (processed image, image hash, image size)
        """
        try:
            # Load the image
            logger.debug(f"Processing image data type: {type(image_data)}")
            image, image_size = load_image(image_data)
            image_hash = hash(image_data)
            logger.debug(f"Loaded image size: {image_size}, hash: {image_hash}, mode: {image.mode}")
            
            # Ensure the image is in RGB format
            if image.mode != "RGB":
                logger.debug(f"Converting image from {image.mode} to RGB")
                image = image.convert("RGB")
            
            # Process the image using the provided processor
            # For Pixtral, we want the raw pixel values
            logger.debug(f"Processing image with processor type: {type(processor)}")
            pixel_values = processor.image_processor(image)["pixel_values"][0]
            logger.debug(f"Processed pixel_values type: {type(pixel_values)}, shape: {pixel_values.shape}")
            
            # Convert numpy arrays to PyTorch tensors which are hashable
            if isinstance(pixel_values, np.ndarray):
                logger.debug(f"Converting numpy array to torch tensor, dtype before: {pixel_values.dtype}")
                pixel_values = torch.tensor(pixel_values, dtype=torch.float16)
                logger.debug(f"Converted to tensor, shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
            elif isinstance(pixel_values, torch.Tensor):
                logger.debug(f"Ensuring tensor is float16, dtype before: {pixel_values.dtype}")
                pixel_values = pixel_values.to(dtype=torch.float16)
                logger.debug(f"Tensor converted, shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
            
            logger.debug(f"Image processing complete: tensor shape={pixel_values.shape}, dtype={pixel_values.dtype}")
            
            # Check for NaN or inf values
            if torch.isnan(pixel_values).any():
                logger.debug("WARNING: NaN values detected in the image tensor")
            if torch.isinf(pixel_values).any():
                logger.debug("WARNING: Inf values detected in the image tensor")
                
            # Log tensor statistics for debugging
            logger.debug(f"Tensor stats: min={pixel_values.min().item():.4f}, max={pixel_values.max().item():.4f}, mean={pixel_values.mean().item():.4f}")
            
            return pixel_values, image_hash, image.size
            
        except Exception:
            logger.error("Exception in PixtralImageProcessor:\n" + get_exception_traceback())
            raise

    async def _process_single_image(self, image_data: Union[bytes, str]):
        """
        Asynchronously process a single image using the CPU executor.
        
        Args:
            image_data: Image data to process
            
        Returns:
            Processed image data
        """
        logger.debug(f"_process_single_image called with data type: {type(image_data)}")
        if self.cpu_executor is not None:
            logger.debug("Using CPU executor for image processing")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.cpu_executor,
                PixtralImageProcessor._process_single_image_task,
                image_data,
                self._processor,
            )
        else:
            logger.debug("No CPU executor available, processing directly")
            return self._process_single_image_task(
                image_data,
                self._processor,
            )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        """
        Process multimodal data asynchronously.
        
        Args:
            image_data: List of image data (paths, bytes, PIL Images)
            input_text: The text input associated with the images
            request_obj: The request object containing additional parameters
            
        Returns:
            Dictionary with processed multimodal items
        """
        logger.debug(f"process_mm_data_async called with {len(image_data) if isinstance(image_data, list) else image_data} images, input_text: {input_text[:50]}...")
        
        if not image_data:
            logger.debug("No image data provided, returning None")
            return None
            
        # Normalize input to a list if it's a single item
        if isinstance(image_data, str) or not isinstance(image_data, list):
            logger.debug(f"Converting single image data to list, type: {type(image_data)}")
            image_data = [image_data]
            
        # Process each image
        if len(image_data) > 0:
            pixel_values, data_hashes, image_sizes = [], [], []
            
            logger.debug(f"Processing {len(image_data)} images concurrently")
            # Process all images concurrently
            processing_tasks = [
                self._process_single_image(img_data) 
                for img_data in image_data
            ]
            results = await asyncio.gather(*processing_tasks)
            logger.debug(f"Gathered results for {len(results)} images")
            
            # Collect results
            for pixel_v, image_h, image_s in results:
                pixel_values.append(pixel_v)
                data_hashes.append(image_h)
                image_sizes.append(image_s)
                logger.debug(f"Collected result: tensor shape={pixel_v.shape}, dtype={pixel_v.dtype}, image_size={image_s}")
                
            # Determine the modality based on the request
            # Default to IMAGE for single images
            modality = Modality.IMAGE
            if hasattr(request_obj, 'modalities') and request_obj.modalities:
                logger.debug(f"Request modalities: {request_obj.modalities}")
                if request_obj.modalities[0] == "multi-images":
                    logger.debug("Setting modality to MULTI_IMAGES")
                    modality = Modality.MULTI_IMAGES
                elif request_obj.modalities[0] == "video":
                    logger.debug("Setting modality to VIDEO")
                    modality = Modality.VIDEO
            logger.debug(f"Final modality: {modality}")
            
            # Create a multimodal item for this batch of images
            pad_value = 1 if hasattr(self._processor, 'image_token_id') else 1
            logger.debug(f"Using pad_value: {pad_value}")
            
            mm_item = MultimodalDataItem(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                modality=modality,
                # For Pixtral, use a standard pad value for image token placeholders
                pad_value=pad_value
            )
            logger.debug(f"Created MultimodalDataItem: modality={mm_item.modality}, images={len(mm_item.pixel_values)}")
            
            return {
                "mm_items": [mm_item],
            }
        else:
            raise ValueError(f"Invalid image data: {image_data}")
