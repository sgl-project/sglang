"""
"""

import unittest
from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
import dataclasses
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
import sglang as sgl
from sglang.srt.conversation import chat_templates

MiniCPMV = "openbmb/MiniCPM-V-2_6"
QWEN25VL = "Qwen/Qwen2.5-VL-7B-Instruct"


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
                    "im_start_id": self.tokenizer.im_start_id,
                    "im_end_id": self.tokenizer.im_end_id,
                    "slice_start_id": self.tokenizer.slice_start_id,
                    "slice_end_id": self.tokenizer.slice_end_id,
                },
            )
            (sglang_output, _) = model.get_embedding(
                input_ids=input_ids, image_inputs=image_inputs
            )

        self.compare_outputs(sglang_output, hf_output)


class TestQWEN25VLLogits(VisionLLMLogitsBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = QWEN25VL
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.chat_template = "qwen2-vl"

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = AutoModelForImageTextToText.from_pretrained(
            cls.model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map=cls.device,
            trust_remote_code=True,
        ).eval()
        cls.model.to(cls.device)
        cls.max_new_tokens = 30
        cls.temperature = 0.4
        cls.top_k = 0
        cls.top_p = 1.0
        cls.debug_tensor_dump_output_folder = "logits"

    async def test_decode_logits(self):
        # Initialize SGLang Engine with logits dumping enabled
        server_args = ServerArgs(
            model_path=self.model_path,
            chat_template=self.chat_template,
            debug_tensor_dump_output_folder=self.debug_tensor_dump_output_folder,
            disable_cuda_graph=True
        )
        
        vlm = sgl.Engine(**dataclasses.asdict(server_args))
        
        # Get conversation template and image token
        conv = chat_templates[self.chat_template].copy()
        image_token = conv.image_token
        
        # Prepare prompt with image
        prompt = f"What's in this image?\n{image_token}"
        
        # Generate with same parameters as HF
        sampling_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p
        }
        
        # Generate output from SGLang
        sgl_output = vlm.generate(
            prompt=prompt,
            image_data=self.image_url,
            sampling_params=sampling_params,
        )
        
        # Get HF output with scores
        inputs = self.get_processor_output()
        with torch.no_grad():
            hf_outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            hf_logits = hf_outputs.scores
        
        # Load SGLang logits
        data = np.load(f"self.debug_tensor_dump_output_folder/pytorch_dump_{self.debug_tensor_dump_output_folder}.npz")
        decode_logits = []
        
        # Extract only decode step logits
        for key in data.files:
            tensor = data[key]
            if tensor.shape[0] == 1:  # decode step logits
                decode_logits.append(tensor)
        
        sgl_logits = torch.tensor(np.array(decode_logits)).to(self.device)
        
        # Convert to numpy and compare
        hf_logits_np = hf_logits.cpu().numpy()
        sgl_logits_np = sgl_logits.cpu().numpy()
        
        np.testing.assert_allclose(hf_logits_np, sgl_logits_np)
        
        # Cleanup
        vlm.shutdown()

if __name__ == "__main__":
    unittest.main()