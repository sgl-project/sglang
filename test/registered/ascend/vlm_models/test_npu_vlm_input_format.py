import json
import os
import unittest
from typing import Optional

import torch

# Compatibility shim: Kimi-VL dynamic module expects PytorchGELUTanh which may
# be missing in transformers==4.57.1. Inject a lightweight implementation so
# the model can import successfully without downgrading transformers.
import transformers.activations as _hf_activations
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from sglang.test.ascend.test_ascend_utils import (
    GEMMA_3_4B_IT_WEIGHTS_PATH,
    IMAGE_MAN_IRONING_PATH,
    IMAGE_SGL_LOGO_PATH,
    KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH,
    QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

if not hasattr(_hf_activations, "PytorchGELUTanh"):

    class PytorchGELUTanh(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.gelu(x, approximate="tanh")

    _hf_activations.PytorchGELUTanh = PytorchGELUTanh
    _hf_activations.ACT2FN.setdefault(
        "pytorch_gelu_tanh",
        lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
    )

from sglang import Engine
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.utils.hf_transformers_utils import _fix_added_tokens_encoding

register_npu_ci(est_time=747, suite="full-2-npu-a3", nightly=True)

IMAGE_MAN_IRONING_URL = IMAGE_MAN_IRONING_PATH
IMAGE_SGL_LOGO_URL = IMAGE_SGL_LOGO_PATH


class VLMInputTestBase:
    model_path = None
    chat_template = None
    processor = None
    visual = None  # Should be a callable for precomputed embeddings

    @classmethod
    def setUpClass(cls):
        assert cls.model_path is not None, "Set model_path in subclass"
        assert cls.chat_template is not None, "Set chat_template in subclass"

        cls.image_urls = [IMAGE_MAN_IRONING_URL, IMAGE_SGL_LOGO_URL]
        cls.device = torch.device("npu")

        cls.main_image = []
        for image_url in cls.image_urls:
            if not os.path.exists(image_url):
                raise FileNotFoundError(f"Local image file not found: {image_url}")
            cls.main_image.append(Image.open(image_url))

        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True, use_fast=True
        )
        _fix_added_tokens_encoding(cls.processor.tokenizer)
        cls._init_visual()

    @classmethod
    def _init_visual(cls):
        """Override in subclass to set up cls.visual as a callable for precomputed embeddings."""
        raise NotImplementedError

    def setUp(self):
        self.engine = Engine(
            model_path=self.model_path,
            chat_template=self.chat_template,
            device=self.device.type,
            mem_fraction_static=0.8,
            enable_multimodal=True,
            disable_cuda_graph=True,
            trust_remote_code=True,
            attention_backend="ascend",
            tp_size=2,
            mm_attention_backend="ascend_attn",
        )

    def tearDown(self):
        self.engine.shutdown()

    def verify_response(self, output):
        # The goal is to check that the model roughly understands:
        #   - image 1: taxi / car scene
        #   - image 2: SGL logo / company
        # We intentionally keep the check keyword-based and loose to avoid
        # overfitting to a specific phrasing.
        if isinstance(output, list):
            result = output[0]
        else:
            result = output
        out_text = result["text"].lower()

        assert any(w in out_text for w in ("taxi", "cab", "car")), out_text

        has_sg_or_logo_side = any(
            kw in out_text
            for kw in (
                "sg ",
                "sgl",
                " sgl",
                "logo",
                "software guidance",
                "labs",
                "laborator",
                "company",
                " text",
            )
        )
        assert has_sg_or_logo_side, out_text

    def get_completion_request(self) -> ChatCompletionRequest:
        json_structure = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.image_urls[0]}},
                        {"type": "image_url", "image_url": {"url": self.image_urls[1]}},
                        {
                            "type": "text",
                            "text": "Describe both the first image and the second image in detail separately.",
                            # update prompt, ensure kimi-vl understands the images separately.
                        },
                    ],
                }
            ],
        }
        json_str = json.dumps(json_structure)
        return ChatCompletionRequest.model_validate_json(json_str)

    def get_processor_output(self, req: Optional[ChatCompletionRequest] = None):
        if req is None:
            req = self.get_completion_request()
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()

        # Process inputs using processor
        inputs = self.processor(
            text=[text],
            images=self.main_image,
            return_tensors="pt",
        ).to(self.device)

        return inputs, text

    async def test_accepts_image(self):
        req = self.get_completion_request()
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()
        output = await self.engine.async_generate(
            prompt=text,
            image_data=self.main_image,
            sampling_params=dict(temperature=0.0, max_new_tokens=512),
        )
        self.verify_response(output)

    async def test_accepts_precomputed_embeddings(self):
        req = self.get_completion_request()
        processor_output, _ = self.get_processor_output(req=req)

        with torch.inference_mode():
            precomputed_embeddings = self.__class__.visual(processor_output)

        output = await self.engine.async_generate(
            input_ids=processor_output["input_ids"][0].tolist(),
            image_data=[
                self._precomputed_image_data(processor_output, precomputed_embeddings)
            ],
            sampling_params=dict(temperature=0.0, max_new_tokens=512),
        )
        self.verify_response(output)

    async def test_accepts_processor_output(self):
        req = self.get_completion_request()
        processor_output, prompt = self.get_processor_output(req=req)
        output = await self.engine.async_generate(
            input_ids=processor_output["input_ids"][0].tolist(),
            image_data=[self._processor_output_image_data(processor_output)],
            sampling_params=dict(temperature=0.0, max_new_tokens=512),
        )
        self.verify_response(output)

    def _precomputed_image_data(self, processor_output, precomputed_embeddings):
        """This should not be overridden."""
        return dict(
            processor_output,
            format="precomputed_embedding",
            feature=precomputed_embeddings,
        )

    def _processor_output_image_data(self, processor_output):
        """Override in subclass to pass the correct set of arguments."""
        raise NotImplementedError


class TestQwenVLUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
    chat_template = "qwen2-vl"

    @classmethod
    def _init_visual(cls):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cls.model_path, torch_dtype=torch.bfloat16
        ).eval()
        # In transformers v5, .visual moved under .model
        visual = model.model.visual
        cls.visual_model = visual.to(cls.device)

        # In transformers v5, the visual encoder returns BaseModelOutputWithPooling;
        # pooler_output has the spatially-merged embeddings we need.
        def visual(processor_output):
            out = cls.visual_model(
                processor_output["pixel_values"], processor_output["image_grid_thw"]
            )
            return out.pooler_output if hasattr(out, "pooler_output") else out

        cls.visual = visual

    def _processor_output_image_data(self, processor_output):
        return dict(processor_output, format="processor_output")


class TestGemmaUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = GEMMA_3_4B_IT_WEIGHTS_PATH
    chat_template = "gemma-it"

    @classmethod
    def _init_visual(cls):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            cls.model_path, torch_dtype=torch.bfloat16
        )
        base_model = model.model

        cls.vision_tower = base_model.vision_tower.eval().to(cls.device)

        if hasattr(base_model, "multi_modal_projector"):
            cls.mm_projector = base_model.multi_modal_projector.eval().to(cls.device)
        else:
            cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)

        cls.visual = lambda processor_output: cls.mm_projector(
            cls.vision_tower(
                pixel_values=processor_output["pixel_values"]
            ).last_hidden_state
        )

    def _processor_output_image_data(self, processor_output):
        return dict(processor_output, format="processor_output")


# Updated Kimi-VL test to use the new input format.
class TestKimiVLImageUnderstandsImage(
    VLMInputTestBase, unittest.IsolatedAsyncioTestCase
):
    model_path = KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH
    chat_template = "kimi-vl"

    @classmethod
    def _init_visual(cls):
        import inspect

        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(cls.model_path, trust_remote_code=True)

        # Transformers v5 auto-populates rope_scaling with
        # {"rope_theta": ..., "rope_type": "default"} even when the original
        # config had rope_scaling: null. The remote KimiVL code branches on
        # `if self.config.rope_scaling is None` so we must reset it.
        tc = getattr(config, "text_config", None)
        if tc is not None:
            rs = getattr(tc, "rope_scaling", None)
            if isinstance(rs, dict) and rs.get("rope_type") == "default":
                tc.rope_scaling = None

        # Transformers v5 calls tie_weights(recompute_mapping=False) in
        # post_init, but KimiVL's tie_weights doesn't accept that kwarg.
        auto_map = getattr(config, "auto_map", {})
        model_ref = auto_map.get("AutoModel")
        if model_ref:
            model_cls = get_class_from_dynamic_module(model_ref, cls.model_path)
            orig_tie = model_cls.tie_weights
            if "recompute_mapping" not in inspect.signature(orig_tie).parameters:

                def _patched_tie(self, **kwargs):
                    return orig_tie(self)

                model_cls.tie_weights = _patched_tie

        model = AutoModel.from_pretrained(
            cls.model_path, config=config, trust_remote_code=True
        )
        cls.vision_tower = model.vision_tower.eval().to(cls.device)
        cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)
        _vt_dtype = next(cls.vision_tower.parameters()).dtype

        cls.visual = lambda tokenizer_output: cls.mm_projector(
            cls.vision_tower(
                pixel_values=tokenizer_output["pixel_values"].to(_vt_dtype),
                grid_hws=tokenizer_output["image_grid_hws"],
            )
        )

    def _processor_output_image_data(self, processor_output):
        return dict(processor_output, format="processor_output")


if __name__ == "__main__":
    unittest.main()
