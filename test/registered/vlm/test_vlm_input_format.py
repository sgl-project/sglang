import json
import unittest
from io import BytesIO
from typing import List, Optional, Tuple, Union

import requests
import torch

# Compatibility shim: Kimi-VL dynamic module expects PytorchGELUTanh which may
# be missing in transformers==4.57.1. Inject a lightweight implementation so
# the model can import successfully without downgrading transformers.
import transformers.activations as _hf_activations
from einops import rearrange
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
from sglang.srt.utils.common import is_cuda, is_xpu
from sglang.srt.utils.hf_transformers_utils import _fix_added_tokens_encoding

register_cuda_ci(est_time=747, stage="base-b", runner_config="1-gpu-large")

IMAGE_MAN_IRONING_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"
IMAGE_SGL_LOGO_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/sgl_logo.png"

_is_cuda = is_cuda()
_is_xpu = is_xpu()


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
        if _is_cuda:
            cls.device = torch.device("cuda")
        elif _is_xpu:
            cls.device = torch.device("xpu")
        else:
            cls.device = torch.device("cpu")

        cls.main_image = []
        for image_url in cls.image_urls:
            response = requests.get(image_url)
            cls.main_image.append(Image.open(BytesIO(response.content)))

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
        try:
            model = AutoModel.from_pretrained(
                cls.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            )
        except (RuntimeError, AttributeError) as e:
            if isinstance(e, RuntimeError) and "meta" not in str(e):
                raise
            # Transformers v5 always uses meta tensors for init, which breaks
            # models calling .item() in __init__ (e.g. InternVL's drop_path_rate).
            # Transformers v5.5.3 may also raise AttributeError for remote-code
            # models missing new internal attributes (e.g. all_tied_weights_keys).
            # Fall back to from_config + manual weight loading.
            import gc
            import glob
            import os

            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(cls.model_path, trust_remote_code=True)
            with torch.device("cpu"):
                model = AutoModel.from_config(
                    config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
            model_dir = snapshot_download(cls.model_path)
            for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
                shard = load_file(f)
                model.load_state_dict(shard, strict=False)
                del shard
            gc.collect()

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


@unittest.skip("temporarily disabled: NaN in next_token_logits")
class TestMiniCPMVUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = "openbmb/MiniCPM-V-4"
    chat_template = "minicpmv"

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
            cls.model_path, trust_remote_code=True
        )
        # In transformers v5.5.3, AutoTokenizer may return TokenizersBackend
        # which lacks model-specific attributes (e.g. im_start_id for MiniCPM-V).
        # Replace with sglang's tokenizer which handles this via declared-class
        # fallback, then fix added tokens encoding.
        from sglang.srt.utils.hf_transformers import get_tokenizer

        cls.processor.tokenizer = get_tokenizer(cls.model_path, trust_remote_code=True)
        _fix_added_tokens_encoding(cls.processor.tokenizer)
        cls._init_visual()

    @classmethod
    def _init_visual(cls):
        try:
            model = AutoModel.from_pretrained(
                cls.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
        except (AttributeError, RuntimeError) as e:
            err = str(e)
            if "all_tied_weights_keys" not in err and "meta" not in err:
                raise
            # Transformers v5: remote model code may lack all_tied_weights_keys
            # or meta-tensor init may break .item() calls.  Fall back to
            # from_config + manual weight loading.
            import gc
            import glob
            import os

            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(cls.model_path, trust_remote_code=True)
            with torch.device("cpu"):
                model = AutoModel.from_config(
                    config,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
            model_dir = snapshot_download(cls.model_path)
            for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
                shard = load_file(f)
                model.load_state_dict(shard, strict=False)
                del shard
            gc.collect()

        cls.vpm_model = model.vpm.eval().to(cls.device)
        cls.resampler_model = model.resampler.eval().to(cls.device)
        del model

        def visual_func(processor_output):
            pixel_values = processor_output["pixel_values"]
            tgt_sizes = processor_output["tgt_sizes"]

            pixel_values_flat = []
            tgt_sizes_flat = []
            for pixel_b, tgt_b in zip(pixel_values, tgt_sizes):
                if isinstance(pixel_b, (list, tuple)):
                    for pixel_n, tgt_n in zip(pixel_b, tgt_b):
                        pixel_values_flat.append(pixel_n)
                        tgt_sizes_flat.append(tgt_n)
                else:
                    pixel_values_flat.append(pixel_b)
                    tgt_sizes_flat.append(tgt_b)

            tgt_sizes_tensor = torch.stack(tgt_sizes_flat, dim=0)
            device = cls.vpm_model.embeddings.position_embedding.weight.device
            dtype = cls.vpm_model.embeddings.position_embedding.weight.dtype

            all_pixel_values_lst = [
                i.flatten(end_dim=1).permute(1, 0) for i in pixel_values_flat
            ]
            max_patches = int(
                (tgt_sizes_tensor[:, 0] * tgt_sizes_tensor[:, 1]).max().item()
            )
            all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                all_pixel_values_lst, batch_first=True, padding_value=0.0
            )
            B, L, _ = all_pixel_values.shape
            all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
            patch_attn_mask = torch.zeros(
                (B, 1, max_patches), dtype=torch.bool, device=device
            )
            tgt_sizes_dev = tgt_sizes_tensor.to(device)
            mask_shapes = tgt_sizes_dev[:, 0] * tgt_sizes_dev[:, 1]
            patch_attn_mask[:, 0, :] = torch.arange(
                max_patches, device=device
            ).unsqueeze(0) < mask_shapes.unsqueeze(1)

            vision_output = cls.vpm_model(
                all_pixel_values.type(dtype),
                patch_attention_mask=patch_attn_mask,
                tgt_sizes=tgt_sizes_tensor,
            )
            vision_embedding = vision_output.last_hidden_state
            return cls.resampler_model(vision_embedding, tgt_sizes_tensor)

        cls.visual = visual_func

    def _processor_output_image_data(self, processor_output):
        return dict(processor_output, format="processor_output")


class TestJanusProUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = "deepseek-ai/Janus-Pro-7B"
    chat_template = "janus-pro"

    @classmethod
    def _init_visual(cls):
        import glob
        import os

        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import AutoConfig

        from sglang.srt.model_loader.weight_utils import (
            download_weights_from_hf,
            pt_weights_iterator,
        )

        # Construct a custom VisionTower class here because there is no remote code
        # available on Huggingface Hub for (https://huggingface.co/deepseek-ai/Janus-Pro-7B/tree/main).
        # Hence, ``AutoModel.from_pretrained`` can't be used.
        #
        # Reuse some components from ``sglang.srt.models.deepseek_janus_pro`` as
        # some parts of the code are ported from the official repository.
        from sglang.srt.models.deepseek_janus_pro import (
            CLIPVisionTower,
            DropPath,
            LayerScale,
            Mlp,
            MlpProjector,
            SigLIP_MODEL_CONFIG,
            SigLIPVisionCfg,
            VisionTransformer,
        )

        # Rewrite the embeddings computation parts that involve attention to
        # make it not depend on QKVParallelLinear / RowParallelLinear /
        # tensor-parallel state which are unavailable outside the engine
        # runtime.
        #
        # The code is adapted from the official ``deepseek-ai/Janus`` repository
        # (https://github.com/deepseek-ai/Janus/tree/main).
        class Attention(nn.Module):
            fused_attn: bool

            def __init__(
                self,
                dim: int,
                num_heads: int = 8,
                qkv_bias: bool = False,
                qk_norm: bool = False,
                attn_drop: float = 0.0,
                proj_drop: float = 0.0,
                norm_layer: nn.Module = nn.LayerNorm,
            ) -> None:
                super().__init__()
                assert dim % num_heads == 0, "dim should be divisible by num_heads"
                self.num_heads = num_heads
                self.head_dim = dim // num_heads
                self.scale = self.head_dim**-0.5
                self.fused_attn = True

                self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
                self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
                self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
                self.attn_drop = nn.Dropout(attn_drop)
                self.proj = nn.Linear(dim, dim)
                self.proj_drop = (
                    nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, N, C = x.shape
                qkv = (
                    self.qkv(x)
                    .reshape(B, N, 3, self.num_heads, self.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv.unbind(0)
                q, k = self.q_norm(q), self.k_norm(k)

                if self.fused_attn:
                    x = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=self.attn_drop.p if self.training else 0.0,
                    )
                else:
                    q = q * self.scale
                    attn = q @ k.transpose(-2, -1)
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)
                    x = attn @ v

                x = x.transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x

        class AttentionBlock(nn.Module):
            def __init__(
                self,
                dim: int,
                num_heads: int,
                mlp_ratio: float = 4.0,
                qkv_bias: bool = False,
                qk_norm: bool = False,
                proj_drop: float = 0.0,
                attn_drop: float = 0.0,
                init_values: Optional[float] = None,
                drop_path: float = 0.0,
                act_layer: nn.Module = nn.GELU,
                norm_layer: nn.Module = nn.LayerNorm,
                mlp_layer: nn.Module = Mlp,
            ) -> None:
                super().__init__()
                self.norm1 = norm_layer(dim)
                self.attn = Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                )
                self.ls1 = (
                    LayerScale(dim, init_values=init_values)
                    if init_values
                    else nn.Identity()
                )
                self.drop_path1 = (
                    DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
                )

                self.norm2 = norm_layer(dim)
                self.mlp = mlp_layer(
                    in_features=dim,
                    hidden_features=int(dim * mlp_ratio),
                    act_layer=act_layer,
                    drop=proj_drop,
                )
                self.ls2 = (
                    LayerScale(dim, init_values=init_values)
                    if init_values
                    else nn.Identity()
                )
                self.drop_path2 = (
                    DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x

        class VisionTower(CLIPVisionTower):
            def __init__(
                self,
                model_name: str = "siglip_large_patch16_384",
                image_size: Union[Tuple[int, int], int] = 336,
                select_feature: str = "patch",
                select_layer: int = -2,
                select_layers: Optional[List] = None,
                ckpt_path: str = "",
                pixel_mean: Optional[List[float]] = None,
                pixel_std: Optional[List[float]] = None,
                **kwargs,
            ):
                super().__init__(
                    model_name=model_name,
                    image_size=image_size,
                    select_feature=select_feature,
                    select_layer=select_layer,
                    select_layers=select_layers,
                    ckpt_path=ckpt_path,
                    pixel_mean=pixel_mean,
                    pixel_std=pixel_std,
                    **kwargs,
                )

            def build_vision_tower(self, vision_tower_params):
                # We already know that the configs from https://huggingface.co/deepseek-ai/Janus-Pro-7B/blob/main/config.json#L49
                # SigLIP is used. For this test, to avoid unnecessary code,
                # we override the ``build_vision_tower`` method to construct
                # ``VisionTransformer``.
                self.select_feature = "same"
                model_name = vision_tower_params.get(
                    "model_name", "siglip_so400m_patch14_384"
                )
                vision_cfg = SigLIPVisionCfg(**SigLIP_MODEL_CONFIG[model_name])

                select_layer = vision_tower_params.get("select_layer", 1)
                if select_layer <= 0:
                    layers = min(
                        vision_cfg.layers, vision_cfg.layers + select_layer + 1
                    )
                else:
                    layers = min(vision_cfg.layers, select_layer)

                vision_tower = VisionTransformer(
                    img_size=vision_tower_params.get("image_size", -1),
                    patch_size=vision_cfg.patch_size,
                    embed_dim=vision_cfg.width,
                    depth=layers,
                    num_heads=vision_cfg.heads,
                    mlp_ratio=vision_cfg.mlp_ratio,
                    class_token=vision_cfg.class_token,
                    global_pool=vision_cfg.global_pool,
                    ignore_head=vision_tower_params.get("ignore_head", True),
                    weight_init=vision_tower_params.get("weight_init", "skip"),
                    num_classes=0,
                    # Force to use testing custom ``AttentionBlock``
                    block_fn=AttentionBlock,
                )

                forward_kwargs = dict()
                return vision_tower, forward_kwargs

        config = AutoConfig.from_pretrained(cls.model_path)
        vision_config = config.vision_config

        cls.vision_model = VisionTower(**vision_config.params).eval().to(cls.device)

        aligner_config = config.aligner_config
        cls.aligner = MlpProjector(aligner_config.params).eval().to(cls.device)

        # Load weights to vision tower and aligner

        hf_folder = download_weights_from_hf(
            cls.model_path,
            cache_dir=None,
            # Janus-Pro ships only *.bin on the hub
            allow_patterns=["*.bin"],
        )

        shard_files = sorted(glob.glob(os.path.join(hf_folder, "*.bin")))
        if not shard_files:
            raise RuntimeError(f"No .bin weight shards found in {hf_folder}")
        iterator = pt_weights_iterator(shard_files)

        vision_state, aligner_state = {}, {}
        for name, tensor in iterator:
            if name.startswith("vision_model."):
                vision_state[name[len("vision_model.") :]] = tensor
            elif name.startswith("aligner."):
                aligner_state[name[len("aligner.") :]] = tensor

        miss_v, _ = cls.vision_model.load_state_dict(vision_state, strict=False)
        miss_a, _ = cls.aligner.load_state_dict(aligner_state, strict=False)

        if miss_v:
            raise RuntimeError(f"vision_model missing weights: {miss_v[:5]}")
        if miss_a:
            raise RuntimeError(f"aligner missing weights: {miss_a[:5]}")

        def visual_func(processor_output):
            pixel_values = processor_output["pixel_values"]
            bs, n = pixel_values.shape[0:2]
            pixel_values = pixel_values.to(
                device=cls.vision_model.device, dtype=cls.vision_model.dtype
            )

            images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            images_embeds = cls.aligner(cls.vision_model(images))
            images_embeds = rearrange(
                images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n
            )

            return images_embeds

        cls.visual = visual_func

    def _processor_output_image_data(self, processor_output):
        return dict(processor_output, format="processor_output")

    def get_processor_output(self, req: Optional[ChatCompletionRequest] = None):
        if req is None:
            req = self.get_completion_request()
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()

        # Unlike other processors, ``VLChatProcessor`` takes the argument ``prompt``
        # instead of ``text``
        inputs = self.processor(
            prompt=text,
            images=self.main_image,
            return_tensors="pt",
        )

        return inputs, text


if __name__ == "__main__":
    unittest.main()
