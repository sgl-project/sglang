import torch
from torch import nn
import torchvision
import numpy as np
from PIL import Image
from typing import List, Any, Dict, Optional, Iterable, Tuple

from transformers.image_transforms import (
    convert_to_rgb,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    make_list_of_images
)
from transformers import CLIPVisionModel

from sglang.srt.models.llama import Phi3ForCausalLM
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.configs import Phi3VConfig, Phi3VCLIPVisionConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader


# adapted from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/c45209e90a4c4f7d16b2e9d48503c7f3e83623ed/image_processing_phi3_v.py
class ImageProcessor(nn.Module):
    def __init__(self, num_crops=1, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD):
        super().__init__()
        self.num_crops = num_crops
        self.image_mean = image_mean
        self.image_std = image_std
        self.img_processor = self._create_image_processor()

    def _create_image_processor(self):
        """Create the torchvision transform pipeline."""
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.image_mean, self.image_std),
        ])

    def _pad_image_336(self, image):
        """Pad image to multiple of 336 pixels."""
        width, height = image.size
        tar = int(np.ceil(height / 336) * 336)
        top_padding = int((tar - height)/2)
        bottom_padding = tar - height - top_padding
        left_padding = 0
        right_padding = 0
        padded_image = torchvision.transforms.functional.pad(
            image, 
            [left_padding, top_padding, right_padding, bottom_padding],
            fill=[255, 255, 255]
        )
        return padded_image

    def _calc_padded_size(self, width, height, padding_unit=336):
        """Calculate padded dimensions."""
        target_height = int(np.ceil(height / padding_unit) * padding_unit)
        top_padding = int((target_height - height) / 2)
        bottom_padding = target_height - height - top_padding
        left_padding = 0
        right_padding = 0
        padded_width = width + left_padding + right_padding
        padded_height = height + top_padding + bottom_padding
        return padded_width, padded_height

    def _calc_hd_transform_size(self, width, height, hd_num=16):
        """Calculate dimensions after HD transform."""
        transposed = False
        if width < height:
            width, height = height, width
            transposed = True

        ratio = width / height
        scale = 1
        while scale * np.ceil(scale / ratio) <= hd_num:
            scale += 1
        scale -= 1

        new_width = int(scale * 336)
        new_height = int(new_width / ratio)

        padded_width, padded_height = self._calc_padded_size(new_width, new_height)

        if transposed:
            padded_width, padded_height = padded_height, padded_width

        return padded_width, padded_height

    def _apply_hd_transform(self, img, hd_num=16):
        """Apply HD transform to a single image."""
        width, height = img.size
        trans = False
        if width < height:
            img = img.transpose(Image.TRANSPOSE)
            trans = True
            width, height = img.size
            
        ratio = (width / height)
        scale = 1
        while scale * np.ceil(scale/ratio) <= hd_num:
            scale += 1
        scale -= 1
        
        new_w = int(scale * 336)
        new_h = int(new_w / ratio)

        img = torchvision.transforms.functional.resize(img, [new_h, new_w])
        img = self._pad_image_336(img)
        
        if trans:
            img = img.transpose(Image.TRANSPOSE)

        return img

    def _pad_to_max_num_crops_tensor(self, images, max_crops=5):
        """Pad tensor with zeros to reach max_crops."""
        B, _, H, W = images.shape
        if B < max_crops:
            pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
            images = torch.cat([images, pad], dim=0)
        return images

    def _preprocess_images(self, pixel_values):
        """Convert and standardize input images to RGB format."""
        images = make_list_of_images(pixel_values)
        images = [convert_to_rgb(image) for image in images]
        return [image.convert('RGB') for image in images]

    def _convert_to_tensors(self, transformed_pil):
        """Convert PIL images to normalized tensors."""
        return [self.img_processor(img) for img in transformed_pil]

    def _create_global_crops(self, hd_images):
        """Create global crops by interpolating to fixed size."""
        return [
            torch.nn.functional.interpolate(
                im.unsqueeze(0).float(),
                size=(336, 336),
                mode='bicubic'
            ).to(im.dtype)
            for im in hd_images
        ]

    def _calculate_shapes_and_tokens(self, hd_images):
        """Calculate image shapes and token counts."""
        shapes = [[im.size(1), im.size(2)] for im in hd_images]
        num_img_tokens = [
            int(((h//336)*(w//336)+1)*144 + 1 + (h//336+1)*12)
            for h, w in shapes
        ]
        return shapes, num_img_tokens

    def _reshape_hd_images(self, hd_images, shapes):
        """Reshape HD images to the required format."""
        return [
            im.reshape(1, 3, h//336, 336, w//336, 336)
              .permute(0,2,4,1,3,5)
              .reshape(-1, 3, 336, 336)
              .contiguous()
            for im, (h, w) in zip(hd_images, shapes)
        ]

    def _combine_global_and_local(self, global_image, hd_images_reshape):
        """Combine global and local image crops."""
        return [
            torch.cat([_global_image] + [_im], dim=0)
            for _global_image, _im in zip(global_image, hd_images_reshape)
        ]

    def _finalize_output(self, image_transformed, shapes, num_img_tokens):
        """Create the final output dictionary."""
        image_sizes = [torch.LongTensor(_shapes) for _shapes in shapes]
        return {
            "pixel_values": image_transformed,
            "image_sizes": image_sizes,
            "num_img_tokens": num_img_tokens
        }

    def forward(self, pixel_values: List[Any]) -> Dict[str, Any]:
        """
        Process raw images into a batched tensor suitable for a "Phi3V-like" pipeline.
        
        Args:
            pixel_values (List[Any]): List of raw images (PIL.Image.Image, NumPy array, etc.)
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - "pixel_values": torch.FloatTensor of shape (batch_size, num_crops+1, 3, 336, 336)
                - "image_sizes": list of (height, width) after HD_transform
                - "num_img_tokens": list of ints indicating token counts
        """
        # Step 1: Preprocess images
        images = self._preprocess_images(pixel_values)
        
        # Step 2: Apply HD transform
        transformed_pil = [self._apply_hd_transform(im, hd_num=self.num_crops) for im in images]
        
        # Step 3: Convert to tensors
        hd_images = self._convert_to_tensors(transformed_pil)
        
        # Step 4: Create global crops
        global_image = self._create_global_crops(hd_images)
        
        # Step 5: Calculate shapes and tokens
        shapes, num_img_tokens = self._calculate_shapes_and_tokens(hd_images)
        
        # Step 6: Reshape HD images
        hd_images_reshape = self._reshape_hd_images(hd_images, shapes)
        
        # Step 7: Combine global and local images
        combined_images = self._combine_global_and_local(global_image, hd_images_reshape)
        
        # Step 8: Pad and stack
        image_transformed = [
            self._pad_to_max_num_crops_tensor(im, self.num_crops+1)
            for im in combined_images
        ]
        image_transformed = torch.stack(image_transformed, dim=0)
        
        # Step 9: Create final output
        return self._finalize_output(image_transformed, shapes, num_img_tokens)

# adapted from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_embedding_phi3_v.py
class Phi3VMultiModalProjector(nn.Module):
    def __init__(self, config: Phi3VConfig):
        super().__init__()

        self.image_dim_out = config.img_processor['image_dim_out']
        self.hidden_size = config.get("n_embd", config.get("hidden_size"))
        self.dim_projection = self.hidden_size
        self.depth = 2

        self.glb_GN = nn.Parameter(torch.zeros([1, 1, self.image_dim_out * 4]))
        self.sub_GN = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out * 4]))

        # default projection_cls mlp is considered here
        self.layer1 = nn.Linear(self.image_dim_out * 4, self.dim_projection)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(self.dim_projection, self.dim_projection)

    def forward(self, image_features, image_sizes, hidden_states, image_positions):

        image_features_proj = self.hd_feature_transform(image_features, image_sizes)
        hidden_states = hidden_states.index_put(
                image_positions, image_features_proj, accumulate=False
            ) # position changed to image_position from original code

        return hidden_states

    def hd_feature_transform(self, image_features, image_sizes):
        """
        image_features: (num_images, num_crops+1, 24*24, 1024)
        """

        global_image_features = image_features[:, 0]  # (num_images, 24*24, 1024)
        # global feature can be viewed as a special HD case with num_crops 1x1
        global_image_features_hd = self.reshape_hd_patches_2x2merge(global_image_features, 1, 1)
        global_image_features_hd_newline = self.add_image_newline(global_image_features_hd)

        all_image_embeddings = []
        # need a for loop to process each image because of different image sizes
        # (patch arrangement is different for each image)
        for i, img_size in enumerate(image_sizes):
            h, w = img_size
            h_crop = h // 336
            w_crop = w // 336
            num_crops = h_crop * w_crop

            # NOTE: real num_crops is padded
            # (num_crops, 24*24, 1024)
            sub_image_features = image_features[i, 1 : 1 + num_crops]
            sub_image_features_hd = self.reshape_hd_patches_2x2merge(
                sub_image_features, h_crop, w_crop
            )
            sub_image_features_hd_newline = self.add_image_newline(sub_image_features_hd)

            # [sub features, separator, global features]
            all_image_embeddings.extend(
                [
                    sub_image_features_hd_newline.squeeze(0),  # (h_crop*12*(w_crop*12+1), 4096)
                    self.glb_GN.squeeze(0),
                    global_image_features_hd_newline[i],
                ]
            )

        image_features_proj = self.image_projection(all_image_embeddings)

        return image_features_proj

    def image_projection(self, all_image_embeddings):

        target_device = self.layer1.bias.device
        target_dtype = self.layer1.bias.dtype

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0).to(target_device, target_dtype)

        image_features_proj = self.layer1(all_image_embeddings)
        image_features_proj = self.gelu(image_features_proj)
        image_features_proj = self.layer2(image_features_proj)

        return image_features_proj

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape
        # add the newline token to the HD image feature patches
        newline_embeddings = self.sub_GN.expand(num_images, h, -1, -1)  # (n_img, h, 1, hid_dim)
        image_features_hd_newline = torch.cat(
            [image_features_hd, newline_embeddings], dim=2
        ).reshape(num_images, -1, hid_dim)
        return image_features_hd_newline

    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        """
        image_features: (num_images*num_crops, 24*24, 1024)
        output: (num_images, h_crop*12, w_crop*12, 4096), h_crop*w_crop == num_crops
        """
        N, L, C = image_features.shape
        assert L == 24 * 24 and C == 1024 and N % (h_crop * w_crop) == 0
        num_images = N // (h_crop * w_crop)
        H = int(L**0.5)
        image_features_hd = (
            image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
            .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
            .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
            .reshape(N, -1, 4 * C)  # N, 144, 4096
            .reshape(
                num_images, h_crop, w_crop, H // 2, H // 2, -1
            )  # n_img, h_crop, w_crop, 12, 12, 4096
            .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
            .reshape(
                num_images, h_crop * H // 2, w_crop * H // 2, 4 * C
            )  # n_img, h_crop*12, w_crop*12, 4096
        )

        return image_features_hd

class Phi3VForCausalLM(nn.Module):
    def __init__(
        self,
        config: Phi3VConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = None
        self.language_model = Phi3ForCausalLM(config, quant_config=quant_config)
        self._process_images = ImageProcessor()

        self.layer_idx = -2 # fixed to default value of clip embedding feature extraction

    # adapted from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_embedding_phi3_v.py
    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Encode images using the vision tower, returning patch embeddings.

        Args:
            pixel_values (torch.FloatTensor): A float tensor of shape 
                (num_images, num_crops, 3, 336, 336).
        
        Returns:
            torch.FloatTensor: A tensor of shape 
                (num_images, num_crops, seq_len-1, self.image_dim_out)
                containing patch embeddings for each crop.
        """

        # 1. Flatten to (num_images * num_crops, 3, 336, 336)
        num_images, num_crops, c, h, w = pixel_values.shape
        assert c == 3 and h == w == 336, "Expected images of shape (N, C, 3, 336, 336)."
        pixel_values_flat = pixel_values.flatten(0, 1)

        # 2. Forward pass through vision model
        vision_outputs = self.vision_tower(pixel_values_flat, output_hidden_states=True)
        
        # 3. Get the desired hidden state and remove [CLS] token
        img_feature = vision_outputs.hidden_states[self.layer_idx]  # shape: (N*C, seq_len, hidden_size)
        patch_feature = img_feature[:, 1:]                     # remove the CLS token -> (N*C, seq_len-1, hidden_size)

        # 4. Reshape to (num_images, num_crops, seq_len-1, image_dim_out)
        image_dim_out = self.config['img_processor'].get('image_dim_out', 1024)
        image_features = patch_feature.reshape(
            num_images, num_crops, -1, image_dim_out
        )

        return image_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        if forward_batch.forward_mode.is_extend():
            # Clamp input ids. This is because the input_ids for the image tokens are
            # filled with the hash values of the image for the prefix matching in the radix attention.
            # There values are useless because their embeddings will be replaced by vision embeddings anyway.
            input_ids.clamp_(min=0, max=self.config.vocab_size - 1)

            # Embed text inputs
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)

            # 1. Embed each image in image_inputs
            # 2. Concatenate text and image embeddings
            extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
            prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

            for i, image in enumerate(forward_batch.image_inputs):
                if image is None:
                    continue
                start_idx = extend_start_loc_cpu[i]
                prefix_len = prefix_lens_cpu[i]

                pixel_values = torch.tensor(image.pixel_values, device="cuda")
                processed_image = self._process_images(pixel_values)
                num_image_tokens = processed_image['num_image_tokens'] # or self.config['img_processor']['num_image_tokens'] ?

                # Embed image
                image_embeds = self.encode_images(processed_image['pixel_values'])

                image_offsets = image.image_offsets

                image_embeds_offset = 0
                for idx, image_offset in enumerate(image_offsets):
                    if image_offset < prefix_len:
                        continue

                    left_idx = start_idx + (image_offset - prefix_len)
                    right_idx = (
                        start_idx + (image_offset - prefix_len) + num_image_tokens
                    )
                    inputs_embeds[left_idx:right_idx] = image_embeds[
                        image_embeds_offset : image_embeds_offset + num_image_tokens
                    ]
                    image_embeds_offset += num_image_tokens

            return self.language_model(
                input_ids, positions, forward_batch, input_embeds=inputs_embeds
            )

        elif forward_batch.forward_mode.is_decode():
            return self.language_model(
                input_ids, positions, forward_batch
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        vision_path = self.config['img_processor']['model_name']
        clip_config = Phi3VCLIPVisionConfig().to_transformers_config()
        self.vision_tower = CLIPVisionModel.from_pretrained(
                    vision_path, 
                    config=clip_config,
                    torch_dtype=torch.float16
                ).cuda()

        self.vision_tower.eval()

        # load mm_projector
        projector_weights = {
            "model.vision_embed_tokens.img_projection.0": "multi_modal_projector.linear_1", # Weights for the first linear layer of the image projection.
            "model.vision_embed_tokens.img_projection.2": "multi_modal_projector.linear_2", # Weights for the second linear layer of the image projection.
            "model.vision_embed_tokens.glb_GN": "multi_modal_projector.glb_GN",
            "model.vision_embed_tokens.sub_GN": "multi_modal_projector.sub_GN",
            "model.vision_embed_tokens.img_processor.vision_model": "vision_tower",  # Update the vision tower weights if we find them in the checkpoint (it may be finetuned).
        }

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "img_projection" in name or "vision_model" in name or "glb_GN" in name or "sub_GN" in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                self.language_model.load_weights([(name, loaded_weight)])

EntryClass = [Phi3VForCausalLM]