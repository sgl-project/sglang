import base64
import re
from io import BytesIO

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoProcessor, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12, upscale=False):
    image = image_file.convert("RGB")
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLChat:
    def __init__(
        self,
        model_path: str,
        max_new_tokens=2048,
        temperature=0.01,
        repetition_penalty=1.0,
    ):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=None,
            do_sample=False,
            temperature=temperature,
        )
        self.sampling_params = dict(
            max_new_tokens=max_new_tokens,
            stop_token_ids=[self.tokenizer.added_tokens_encoder["<|im_end|>"]],
            temperature=temperature,
        )

    def build_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        self.model.eval().cuda()
        torch.cuda.empty_cache()

    def build_prompt_hf(self, sample):
        prompt = sample["final_input_prompt"]
        count = len(sorted(set(int(m) for m in re.findall(r"<image ([1-7])>", prompt))))
        num_patches_list = []
        if count == 1:
            prompt = "<image>\n" + prompt
            pixel_values = (
                load_image(sample["image_1"], upscale=True).to(torch.bfloat16).cuda()
            )
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values_list = []
            for idx in range(1, count + 1):
                prompt = f"Image-{count+1-idx}: <image>\n" + prompt
                pixel_values = load_image(sample[f"image_{idx}"], upscale=True).to(
                    torch.bfloat16
                )
                num_patches_list.append(pixel_values.size(0))
                pixel_values_list.append(pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0).cuda()
        print(f"\033[31m{prompt}\033[0m")
        return {
            "prompt": prompt,
            "pixel_values": pixel_values,
            "num_patches_list": num_patches_list,
        }

    def build_prompt_sglang(self, sample):
        prompt = sample["final_input_prompt"]
        count = len(sorted(set(int(m) for m in re.findall(r"<image ([1-7])>", prompt))))
        request_dict = {
            "model": "",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        }
        for idx in range(1, count + 1):
            image = sample[f"image_{idx}"]
            bytes_io = BytesIO()
            image.save(bytes_io, format="PNG")
            base64_str = base64.b64encode(bytes_io.getvalue()).decode("utf-8")
            request_dict["messages"][0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"},
                }
            )
        return request_dict

    def chat(self, sample):
        inputs = self.build_prompt_hf(sample)
        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=inputs["pixel_values"],
            num_patches_list=inputs["num_patches_list"],
            question=inputs["prompt"],
            generation_config=self.generate_kwargs,
            verbose=True,
        )
        # print(f'\033[32m{response}\033[0m')
        return response
