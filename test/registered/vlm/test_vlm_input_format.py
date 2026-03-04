import json
import unittest
from io import BytesIO
from typing import Optional

import requests
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

from sglang.test.ci.ci_register import register_cuda_ci

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

register_cuda_ci(est_time=447, suite="stage-b-test-large-1-gpu")

IMAGE_MAN_IRONING_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"
IMAGE_SGL_LOGO_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/sgl_logo.png"


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
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.main_image = []
        for image_url in cls.image_urls:
            response = requests.get(image_url)
            cls.main_image.append(Image.open(BytesIO(response.content)))
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True, use_fast=True
        )
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
        )

    def tearDown(self):
        self.engine.shutdown()

    def verify_response(self, output):
        # The goal is to check that the model roughly understands:
        #   - image 1: taxi / car scene
        #   - image 2: SGL logo / company
        # We intentionally keep the check keyword-based and loose to avoid
        # overfitting to a specific phrasing.
        out_text = output["text"].lower()

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
                            "text": "Describe both the first image and the second image in detail separately.",  # update prompt, ensure kimi-vl understands the images separately.
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
            input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
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
            input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
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
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    chat_template = "qwen2-vl"

    @classmethod
    def _init_visual(cls):
        cls.visual_model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cls.model_path, torch_dtype=torch.bfloat16
            )
            .eval()
            .visual.to(cls.device)
        )
        cls.visual = lambda processor_output: cls.visual_model(
            processor_output["pixel_values"], processor_output["image_grid_thw"]
        )

    def _processor_output_image_data(self, processor_output):
        return dict(processor_output, format="processor_output")


class TestGemmaUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = "google/gemma-3-4b-it"
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
    model_path = "moonshotai/Kimi-VL-A3B-Instruct"
    chat_template = "kimi-vl"

    @classmethod
    def _init_visual(cls):
        model = AutoModel.from_pretrained(cls.model_path, trust_remote_code=True)
        cls.vision_tower = model.vision_tower.eval().to(cls.device)
        cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)

        cls.visual = lambda tokenizer_output: cls.mm_projector(
            cls.vision_tower(
                pixel_values=tokenizer_output["pixel_values"],
                grid_hws=tokenizer_output["image_grid_hws"],
            )
        )

    def _processor_output_image_data(self, processor_output):
        return dict(processor_output, format="processor_output")


# not for CI: too large
# class TestLlama4ImageUnderstandsImage(
#     VLMInputTestBase, unittest.IsolatedAsyncioTestCase
# ):
#     # Allow overriding via env for local/offline runs.
#     model_path = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
#     chat_template = "llama-4"

#     def setUp(self):
#         if torch.cuda.device_count() < 4:
#             self.skipTest("Skipping Llama-4 test: requires 4 GPUs for TP=4")
#         self.engine = Engine(
#             model_path=self.model_path,
#             trust_remote_code=True,
#             chat_template=self.chat_template,
#             enable_multimodal=True,
#             mem_fraction_static=0.8,
#             tp_size=4,
#             attention_backend="fa3",
#             context_length=65536,
#         )

#     @classmethod
#     def _init_visual(cls):
#         model = AutoModel.from_pretrained(
#             cls.model_path,
#             trust_remote_code=True,
#             torch_dtype="auto",
#             force_download=True,
#         )
#         cls.vision_tower = model.vision_model.eval().to(cls.device)
#         cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)

#         cls.visual = lambda tokenizer_output: cls.mm_projector(
#             cls.vision_tower(
#                 pixel_values=tokenizer_output["pixel_values"],
#             ).last_hidden_state.flatten(0, -2)
#         )

#     def _processor_output_image_data(self, processor_output):
#         # Llama-4 vision expects processor_output format with pixel_values
#         return dict(processor_output, format="processor_output")


# class TestLlavaUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
#     model_path = "llava-hf/llava-1.5-7b-hf"
#     chat_template = "vicuna_v1.1"

#     @classmethod
#     def _init_visual(cls):
#         from transformers import LlavaForConditionalGeneration

#         model = LlavaForConditionalGeneration.from_pretrained(
#             cls.model_path,
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#         )
#         cls.vision_tower = model.vision_tower.eval().to(cls.device)
#         cls.multi_modal_projector = model.multi_modal_projector.eval().to(cls.device)
#         cls.config = model.config

#         def visual_func(processor_output):
#             pixel_values = processor_output["pixel_values"].to(
#                 cls.device, dtype=torch.float16
#             )

#             vision_outputs = cls.vision_tower(pixel_values, output_hidden_states=True)
#             image_features = vision_outputs.hidden_states[-2]

#             if cls.config.vision_feature_select_strategy == "default":
#                 image_features = image_features[:, 1:]
#             elif cls.config.vision_feature_select_strategy == "full":
#                 image_features = image_features

#             image_features = cls.multi_modal_projector(image_features)
#             return image_features

#         cls.visual = visual_func

#     def _processor_output_image_data(self, processor_output):
#         return dict(processor_output, format="processor_output")


class TestInternVLUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = "OpenGVLab/InternVL2-2B"
    chat_template = "internvl-2-5"

    @classmethod
    def setUpClass(cls):
        assert cls.model_path is not None, "Set model_path in subclass"
        assert cls.chat_template is not None, "Set chat_template in subclass"
        cls.image_urls = [IMAGE_MAN_IRONING_URL, IMAGE_SGL_LOGO_URL]
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.main_image = []
        for image_url in cls.image_urls:
            response = requests.get(image_url)
            cls.main_image.append(Image.open(BytesIO(response.content)))

        # InternVL models (2, 3, 3.5) do not ship a standard HuggingFace
        # Processor; AutoProcessor.from_pretrained returns a bare tokenizer.
        # Use AutoTokenizer explicitly so the intent is clear.
        from transformers import AutoTokenizer

        cls.processor = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls._init_visual()

    @classmethod
    def _init_visual(cls):
        model = AutoModel.from_pretrained(
            cls.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        cls.vision_model = model.vision_model.eval().to(cls.device)
        cls.mlp1 = model.mlp1.eval().to(cls.device)

        config = model.config
        cls.internvl_config = config
        image_size = getattr(config, "force_image_size", None) or (
            config.vision_config.image_size
        )
        patch_size = config.vision_config.patch_size
        cls.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        cls.internvl_image_size = image_size
        cls.internvl_downsample_ratio = config.downsample_ratio
        cls.internvl_ps_version = config.ps_version
        cls.internvl_select_layer = config.select_layer

        del model

        def pixel_shuffle(x, scale_factor):
            n, w, h, c = x.size()
            x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(
                n,
                int(h * scale_factor),
                int(w * scale_factor),
                int(c / (scale_factor * scale_factor)),
            )
            if cls.internvl_ps_version != "v1":
                x = x.permute(0, 2, 1, 3).contiguous()
            return x

        def visual_func(processor_output):
            pixel_values = processor_output["pixel_values"].to(
                cls.device, dtype=torch.bfloat16
            )
            if cls.internvl_select_layer == -1:
                vit_embeds = cls.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
            else:
                vit_embeds = cls.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[cls.internvl_select_layer]
            vit_embeds = vit_embeds[:, 1:, :]

            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = pixel_shuffle(
                vit_embeds, scale_factor=cls.internvl_downsample_ratio
            )
            vit_embeds = vit_embeds.reshape(
                vit_embeds.shape[0], -1, vit_embeds.shape[-1]
            )
            vit_embeds = cls.mlp1(vit_embeds)
            return vit_embeds

        cls.visual = visual_func

    def get_processor_output(self, req=None):
        """Override to handle InternVL's custom preprocessing.

        Uses shared ``image_to_pixel_values`` from ``internvl_utils`` for
        image preprocessing (dynamic tiling + normalize) and expands
        ``<IMG_CONTEXT>`` placeholders into ``<img>`` + context tokens +
        ``</img>`` — mirroring the logic in
        ``InternVLProcessor.process_internlm2_mm_data_async``.
        """
        from sglang.srt.multimodal.internvl_utils import image_to_pixel_values
        from sglang.srt.multimodal.processors.internvl import InternVLProcessor

        if req is None:
            req = self.get_completion_request()
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()

        # Preprocess images using the shared utility (dynamic tiling +
        # bicubic resize + ImageNet normalize), same pipeline as the engine.
        all_pixel_values = []
        num_patches_list = []
        for img in self.main_image:
            pv = image_to_pixel_values(
                img,
                input_size=self.internvl_image_size,
                max_num_tiles=InternVLProcessor.IMAGE_MAX_NUM,
                use_thumbnail=True,
            )
            all_pixel_values.append(pv)
            num_patches_list.append(pv.shape[0])

        pixel_values = torch.cat(all_pixel_values, dim=0).to(self.device)

        # Expand each <IMG_CONTEXT> placeholder into <img> + <IMG_CONTEXT>*N + </img>.
        # This mirrors InternVLProcessor.process_internlm2_mm_data_async.
        ph = "<<<__IMG_PH__>>>"
        expanded_text = text.replace(InternVLProcessor.IMG_CONTEXT, ph)
        for num_patches in num_patches_list:
            image_tokens = (
                InternVLProcessor.IMG_START
                + InternVLProcessor.IMG_CONTEXT * (self.num_image_token * num_patches)
                + InternVLProcessor.IMG_END
            )
            expanded_text = expanded_text.replace(ph, image_tokens, 1)
        # Remove any remaining placeholders (more placeholders than images)
        expanded_text = expanded_text.replace(ph, "")

        # Tokenize the expanded text
        input_ids = self.processor(expanded_text, return_tensors="pt")["input_ids"]

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }, text

    def _processor_output_image_data(self, processor_output):
        return dict(processor_output, format="processor_output")


if __name__ == "__main__":
    unittest.main()
