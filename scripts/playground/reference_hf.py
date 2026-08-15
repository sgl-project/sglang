"""
Usage: python3 scripts/playground/reference_hf.py --model-path MODEL_PATH --model-type {text,vlm} [--max-new-tokens NUM] [--dtype DTYPE]
  --model-path MODEL_PATH: Path to model (default: TinyLlama/TinyLlama-1.1B-Chat-v0.4)
  --model-type {text,vlm}: Model type, text or vlm (default: text)
  --max-new-tokens NUM: Max new tokens to generate (default: 16)
  --dtype DTYPE: Data type for computation (default: float16)
Note: '--model' is deprecated; use '--model-path'. Runs normal_text() for text, vlm_text_with_image() for vlm.

Reference output:
========== Prompt 0 ==========
prefill logits (final) tensor([-8.3125, -7.1172,  3.3398,  ..., -4.9531, -4.1328, -3.4141],
       device='cuda:0')
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.

========== Prompt 1 ==========
prefill logits (final) tensor([-8.9062, -9.0156,  4.1484,  ..., -4.9922, -4.4961, -4.0742],
       device='cuda:0')
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of

========== Prompt 2 ==========
prefill logits (final) tensor([-9.6328, -9.0547,  4.0234,  ..., -5.3047, -4.7148, -4.4609],
       device='cuda:0')
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the
"""

import argparse

import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
)

from sglang.srt.utils.hf_transformers_utils import get_tokenizer


@torch.no_grad()
def vlm_text_with_image(args):
    # Load the processor and model for ImageTextToText tasks
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=args.dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )

    torch.cuda.set_device(0)

    # List of image URLs to process
    image_urls = [
        "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    ]

    # Conversation template for the processor
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    max_new_tokens = args.max_new_tokens

    for i, url in enumerate(image_urls):
        # Load the image from the URL
        image = Image.open(requests.get(url, stream=True).raw)

        # Apply the chat template to the text prompt
        # Notice that not all processors support chat templates.
        # LLaVA and QWen are two processors that support chat templates.
        if not hasattr(processor, "apply_chat_template"):
            raise ValueError("The processor does not support chat templates.")
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # Prepare inputs for the model
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(
            "cuda:0"
        )

        # Generate output from the model
        output_ids = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens
        )
        output_str = processor.decode(output_ids[0])

        # Get the logits from the model's forward pass
        outputs = model.forward(**inputs)
        logits = outputs.logits[0, -1, :]

        print(f"\n========== Image {i} ==========")
        print("prefill logits (final)", logits)
        # TODO(gaocegege): The output contains numerous <|image_pad|> tokens,
        # making it cluttered and difficult to read.
        # These tokens should be removed or cleaned up for better readability.
        print(output_str)


@torch.no_grad()
def normal_text(args):
    t = get_tokenizer(args.model_path, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=args.dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )

    prompts = [
        "The capital of France is",
        "The capital of the United Kindom is",
        "Today is a sunny day and I like",
    ]
    max_new_tokens = args.max_new_tokens

    torch.cuda.set_device(0)

    for i, p in enumerate(prompts):
        if isinstance(p, str):
            input_ids = t.encode(p, return_tensors="pt").to("cuda:0")
        else:
            input_ids = torch.tensor([p], device="cuda:0")

        output_ids = m.generate(
            input_ids, do_sample=False, max_new_tokens=max_new_tokens
        )
        output_str = t.decode(output_ids[0])

        prefill_logits = m.forward(input_ids).logits[0][-1]

        print(f"\n========== Prompt {i} ==========")
        print("prefill logits (final)", prefill_logits)
        print(output_str)


@torch.no_grad()
def synthetic_tokens(args):
    m = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    m.cuda()
    print(m)

    input_len = 256
    output_len = 8
    prompts = [list(range(5, 5 + input_len))]

    for p in prompts:
        input_ids = p
        for i in range(output_len + 1):
            prefill_logits = m.forward(torch.tensor([input_ids], device="cuda")).logits[
                0
            ][-1]

            if i == 0:
                print("prefill logits", prefill_logits)
            else:
                print("decode", i - 1, prefill_logits)

            input_ids.append(torch.argmax(prefill_logits).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v0.4",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)

    parser.add_argument("--dtype", type=str, default="float16")

    parser.add_argument("--model-type", type=str, default="text")

    args = parser.parse_args()

    if args.model_type == "vlm":
        vlm_text_with_image(args)
    else:
        normal_text(args)
