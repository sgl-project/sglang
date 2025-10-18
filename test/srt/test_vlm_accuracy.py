"""
"""

import unittest
from io import BytesIO
from typing import List, Optional

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.managers.mm_utils import embed_mm_inputs, init_embedding_cache
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.server_args import ServerArgs

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)


# Test the logits output between HF and SGLang
class VisionLLMLogitsBase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_url = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
        cls.device = torch.device(device_type)
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

    def get_completion_request(self) -> ChatCompletionRequest:
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
          "text": "What's in this picture?"
        }}
      ]
    }}
  ]
}}
        """

        return ChatCompletionRequest.model_validate_json(json_str)

    def get_processor_output(self, req: Optional[ChatCompletionRequest] = None):
        if req is None:
            req = self.get_completion_request()
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
        self.model_runner = ModelRunner(
            model_config=ModelConfig(self.model_path, model_override_args="{}"),
            mem_fraction_static=0.8,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=12435,
            server_args=ServerArgs(
                model_path=self.model_path,
                disable_cuda_graph=True,
            ),
        )
        return self.model_runner.model


class TestMiniCPMV2_6Logits(VisionLLMLogitsBase):
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

        cls.device = torch.device(device_type)
        cls.hf_model = (
            AutoModel.from_pretrained(
                cls.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .eval()
            .to(cls.device)
        )
        init_embedding_cache()

    async def test_vlm_embedding_output(self):
        """
        Compares the embedding output of vlm
        """
        inputs = self.get_processor_output()

        with torch.no_grad():
            # hf
            model_inputs = {
                "input_ids": inputs.input_ids,
                "image_bound": inputs.image_bound,
                "pixel_values": inputs.pixel_values,
                "tgt_sizes": inputs.tgt_sizes,
            }
            (hf_output, _) = self.hf_model.get_vllm_embedding(
                model_inputs,
            )
            hf_output = hf_output.squeeze(0)

            # sglang
            model = self.get_sglang_model()
            input_ids = inputs["input_ids"].to(self.device).flatten()

            pixel_values = inputs["pixel_values"]
            tgt_sizes = inputs["tgt_sizes"]
            pixel_values_flat: List[torch.Tensor] = []
            tgt_sizes_flat: List[torch.Tensor] = []
            for pixel_b, tgt_b in zip(pixel_values, tgt_sizes):
                # per image
                if len(pixel_b) != len(tgt_b):
                    raise ValueError(
                        "Inconsistent N lengths, found: "
                        f"{len(pixel_b)} vs {len(tgt_b)}"
                    )
                for pixel_n, tgt_n in zip(pixel_b, tgt_b):
                    pixel_values_flat += [pixel_n]
                    tgt_sizes_flat += [tgt_n]

            im_start_id, im_end_id = (
                self.tokenizer.im_start_id,
                self.tokenizer.im_end_id,
            )
            slice_start_id, slice_end_id = (
                self.tokenizer.slice_start_id,
                self.tokenizer.slice_end_id,
            )

            image_offsets = BaseMultimodalProcessor.get_mm_items_offset_by_pair(
                input_ids=input_ids, mm_start_id=im_start_id, mm_end_id=im_end_id
            )
            slice_offsets = BaseMultimodalProcessor.get_mm_items_offset_by_pair(
                input_ids=input_ids, mm_start_id=slice_start_id, mm_end_id=slice_end_id
            )
            image_offsets.extend(slice_offsets)
            image_offsets = sorted(image_offsets)

            sglang_output = embed_mm_inputs(
                mm_inputs_list=[
                    MultimodalInputs(
                        mm_items=[
                            MultimodalDataItem(
                                feature=pixel_values_flat,
                                offsets=image_offsets,
                                tgt_size=tgt_sizes_flat,
                                modality=Modality.IMAGE,
                                pad_value=self.processor.tokenizer.unk_token_id,
                            )
                        ]
                    ),
                ],
                extend_prefix_lens=[0],
                extend_seq_lens=[input_ids.shape[0]],
                input_ids=input_ids,
                input_embedding=model.get_input_embeddings(),
                multimodal_model=model,
                placeholder_tokens={
                    Modality.IMAGE: self.processor.tokenizer.unk_token_id,
                },
            )

        self.compare_outputs(sglang_output, hf_output)


class TestMiniCPMV4Logits(VisionLLMLogitsBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = "openbmb/MiniCPM-V-4"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.chat_template = "minicpmv"

        cls.device = torch.device(device_type)
        cls.hf_model = (
            AutoModel.from_pretrained(
                cls.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .eval()
            .to(cls.device)
        )
        init_embedding_cache()

    async def test_vlm_embedding_output(self):
        """
        Compares the embedding output of vlm
        """
        inputs = self.get_processor_output()

        with torch.no_grad():
            # hf
            model_inputs = {
                "input_ids": inputs.input_ids,
                "image_bound": inputs.image_bound,
                "pixel_values": inputs.pixel_values,
                "tgt_sizes": inputs.tgt_sizes,
            }
            hf_output = self.hf_model.get_input_embeddings()(inputs.input_ids)

            # sglang
            model = self.get_model()
            sglang_output = self.vlm_func(
                model,
                input_ids=inputs.input_ids.to(self.device),
                pixel_values=inputs.pixel_values,
                image_bound=inputs.image_bound.to(self.device),
                tgt_sizes=inputs.tgt_sizes.to(self.device),
                input_embedding=model.get_input_embeddings(),
                multimodal_model=model,
                placeholder_tokens={
                    Modality.IMAGE: self.processor.tokenizer.unk_token_id,
                },
            )

        self.compare_outputs(sglang_output, hf_output)
