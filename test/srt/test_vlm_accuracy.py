"""
"""

import unittest
from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.conversation import generate_chat_conv
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.openai_api.protocol import ChatCompletionRequest
from sglang.srt.server_args import ServerArgs


def vlm_text_with_image(args):
    max_new_tokens = 100
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


# Test the logits output between HF and SGLang
class VisionLLMLogitsBase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_url = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model_path = ""
        cls.chat_template = ""
        cls.processor = ""
        response = requests.get(cls.image_url)
        cls.main_image = Image.open(BytesIO(response.content))

    def compare_outputs(self, sglang_output: torch.Tensor, hf_output: torch.Tensor):
        # Convert to float32 for numerical stability if needed
        hf = hf_output.float()
        sg = sglang_output.float()

        # Basic shape and dtype comparison
        print("\n=== Basic Properties ===")
        print(f"Shapes match: {hf.shape == sg.shape}")
        print(f"HF shape: {hf.shape}, SGLang shape: {sg.shape}")
        print(f"HF dtype: {hf.dtype}, SGLang dtype: {sg.dtype}")

        # Move tensors to CPU for numpy operations
        hf_np = hf.cpu().numpy()
        sg_np = sg.cpu().numpy()

        # Statistical metrics
        print("\n=== Statistical Metrics ===")
        print(f"Mean absolute difference: {torch.mean(torch.abs(hf - sg)).item():.6f}")
        print(f"Max absolute difference: {torch.max(torch.abs(hf - sg)).item():.6f}")
        print(f"Mean squared error: {torch.mean((hf - sg) ** 2).item():.6f}")
        print(
            f"Root mean squared error: {torch.sqrt(torch.mean((hf - sg) ** 2)).item():.6f}"
        )

        # Cosine similarity (across feature dimension)
        cos_sim = F.cosine_similarity(hf, sg)
        print(f"Mean cosine similarity: {torch.mean(cos_sim).item():.6f}")
        print(f"Min cosine similarity: {torch.min(cos_sim).item():.6f}")

        # Find largest absolute differences
        print("\n=== Largest Absolute Differences ===")
        diffs = torch.abs(hf - sg)
        flat_diffs = diffs.flatten()

        # Get indices of top 10 differences
        top_k = 10
        top_values, top_flat_indices = torch.topk(flat_diffs, top_k)

        # Convert flat indices to multidimensional indices
        top_indices = np.unravel_index(top_flat_indices.cpu().numpy(), diffs.shape)

        print(f"\nTop {top_k} largest absolute differences:")
        print(
            "Index".ljust(30)
            + "Difference".ljust(15)
            + "HF Value".ljust(15)
            + "SGLang Value"
        )
        print("-" * 75)

        for i in range(top_k):
            # Get the index tuple for this difference
            idx = tuple(dim[i] for dim in top_indices)
        diff_val = top_values[i].item()
        hf_val = hf[idx].item()
        sg_val = sg[idx].item()

        # Format the index tuple and values
        idx_str = str(idx)
        print(f"{idx_str:<30}{diff_val:<15.6f}{hf_val:<15.6f}{sg_val:.6f}")

        np.testing.assert_allclose(hf_np, sg_np)

    def get_processor_output(self):
        json_str = f"""
        {{
  "model": "{self.model_path}",
  "messages": [
    {{
      "role": "user",
      "content": [
        {{
          "type": "image_url",
          "image_url": {{
            "url": "{self.image_url}"
          }}
        }},
        {{
          "type": "text",
          "text": "Whats in this picture?"
        }}
      ]
    }}
  ]
}}
        """

        req = ChatCompletionRequest.model_validate_json(json_str)

        conv = generate_chat_conv(req, template_name=self.chat_template)

        text = conv.get_prompt()

        # Process inputs using processor
        # FIXME: the formal arguments may differ
        inputs = self.processor(
            text=[text],
            images=[self.main_image],
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def get_sglang_model(self):
        model_runner = ModelRunner(
            model_config=ModelConfig(self.model_path, model_override_args="{}"),
            mem_fraction_static=0.8,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            nccl_port=12435,
            server_args=ServerArgs(
                model_path=self.model_path,
                disable_cuda_graph=True,
            ),
        )
        return model_runner.model


class TestMiniCPMVLogits(VisionLLMLogitsBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = "openbmb/MiniCPM-V-2_6"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.chat_template = "minicpmv"

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = AutoModel.from_pretrained(
            cls.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).eval()
        cls.model.to(cls.device)

    async def test_encode_output(self):
        inputs = self.get_processor_output()

        with torch.no_grad():
            model_inputs = {
                "input_ids": inputs.input_ids,
                "image_bound": inputs.image_bound,
                "pixel_values": inputs.pixel_values,
                "tgt_sizes": inputs.tgt_sizes,
            }
            (hf_output, _) = self.model.get_vllm_embedding(
                model_inputs,
            )
            hf_output = hf_output.squeeze(0)

        with torch.no_grad():
            model = self.get_sglang_model()
            input_ids = inputs["input_ids"].to(self.device).flatten()
            image_inputs = model._parse_and_validate_inputs(
                input_ids=input_ids,
                **{
                    "pixel_values": [inputs["pixel_values"]],
                    "tgt_sizes": [inputs["tgt_sizes"]],
                    "im_start_id": [self.tokenizer.im_start_id],
                    "im_end_id": [self.tokenizer.im_end_id],
                    "slice_start_id": [self.tokenizer.slice_start_id],
                    "slice_end_id": [self.tokenizer.slice_end_id],
                },
            )
            (sglang_output, _) = model.get_embedding(
                input_ids=input_ids, image_inputs=image_inputs
            )

        self.compare_outputs(sglang_output, hf_output)


class TestQwen2_5VLLogits(VisionLLMLogitsBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = MiniCPMV
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.chat_template = "minicpmv"

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = AutoModel.from_pretrained(
            cls.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).eval()
        cls.model.to(cls.device)

    async def test_output_ids(self):
        max_new_tokens = 128
        inputs = self.get_processor_output()

        with torch.no_grad():
            model_inputs = {
                "input_ids": inputs.input_ids,
                "image_bound": inputs.image_bound,
                "pixel_values": inputs.pixel_values,
                "tgt_sizes": inputs.tgt_sizes,
            }
            hf_outputs = self.model.forward(**inputs)
            hf_logits = hf_outputs.logits[0, -1, :]

        # with torch.no_grad():
        #     model = self.get_sglang_model()
        #     input_ids = inputs["input_ids"].to(self.device).flatten()
        #     positions = compute_position_triton()
        #     forward_batch =
        #     image_inputs = model.forward(
        #         input_ids,
        #         positions,
        #         for
        #     )
        #     (sglang_output, _) = model.get_embedding(
        #         input_ids=input_ids, image_inputs=image_inputs
        #     )
        #
        # self.compare_outputs(sglang_output, hf_output)


if __name__ == "__main__":
    unittest.main()
