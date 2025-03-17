import base64
import re
from io import BytesIO

import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor


class Qwen2VLChat:
    def __init__(
        self,
        model_path: str,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
    ):
        self.model_path = model_path
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.sampling_params = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            stop_token_ids=[
                self.processor.tokenizer.added_tokens_encoder["<|im_end|>"]
            ],
            top_k=top_k,
            temperature=temperature,
        )

    def build_model(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="cpu",
            attn_implementation="flash_attention_2",
        )
        self.model.cuda().eval()
        torch.cuda.empty_cache()

    def build_prompt_hf(self, sample):
        prompt = sample["final_input_prompt"]
        image_count = len(
            sorted(set(int(m) for m in re.findall(r"<image ([1-7])>", prompt)))
        )
        image = sample["image_1"]
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Question: " + prompt},
                    ],
                }
            ]
            for i in range(1, image_count + 1):
                messages[0]["content"].append(
                    {
                        "type": "image",
                        "image": sample["image_1"],
                        "min_pixels": 1003520,
                        "max_pixels": 12845056,
                    }
                )
            print(f"\033[31m{messages}\033[0m")
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images, videos = process_vision_info([messages])
            inputs = self.processor(
                text=[text],
                images=[images],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
        return inputs

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
                            "text": "Question: " + prompt,
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
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        print(f"\033[32m{response}\033[0m")
        return response
