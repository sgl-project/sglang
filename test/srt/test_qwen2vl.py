import asyncio
import unittest
from unittest.mock import patch, AsyncMock
import os
import json
import tempfile
import math
import base64
import torch
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from pathlib import Path

import torch.nn.functional as F

QWEN2_VL_MODEL = "Qwen/Qwen2-VL-7B-Instruct"


class RawSGLangTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN2_VL_MODEL, trust_remote_code=True)
        self.image_token_id = self.tokenizer.encode("<|image_pad|>")[0]

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(QWEN2_VL_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Path to the test PDF
        self.test_pdf_path = Path(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "ambiguous.pdf"))
        self.maxDiff = None

    async def test_vision_encoder(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
  
        # Apply chat template to get the text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = requests.get(messages[0]["content"][0]["image"])
        main_image = Image.open(BytesIO(response.content))


        # Process inputs using processor
        inputs = self.processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="pt",
        )

        print(self.model.visual)

        with torch.no_grad():
            hf_output = self.model.visual(inputs["pixel_values"].to(self.device), grid_thw=inputs["image_grid_thw"].to(self.device))

        print("HF", hf_output, hf_output.shape)

        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.hf_transformers_utils import get_tokenizer
        from sglang.srt.server_args import ServerArgs, PortArgs

        model_config = ModelConfig(
            QWEN2_VL_MODEL,
            model_override_args="{}"
        )
        
        server_args = ServerArgs(model_path=QWEN2_VL_MODEL)
        # Initialize model runner
        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=0.8,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            nccl_port=12435,
            server_args=server_args,
        )

        print(model_runner.model.visual)
        with torch.no_grad():
            sglang_output = model_runner.model.visual(inputs["pixel_values"].to(self.device), grid_thw=inputs["image_grid_thw"].to(self.device))

        print("SGLANG", sglang_output, sglang_output.shape)

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
        print(f"Root mean squared error: {torch.sqrt(torch.mean((hf - sg) ** 2)).item():.6f}")
        
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
        print("Index".ljust(30) + "Difference".ljust(15) + "HF Value".ljust(15) + "SGLang Value")
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