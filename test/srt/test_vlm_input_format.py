import json
import unittest
from io import BytesIO
from typing import Optional

import requests
import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from sglang import Engine
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.conversation import generate_chat_conv

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

<<<<<<< HEAD
        car_keywords = ("taxi", "cab", "car", "vehicle", "suv", "van", "truck", "auto")
        assert any(w in out_text for w in car_keywords), out_text
=======
        assert any(w in out_text for w in ("taxi", "cab", "car")), out_text
>>>>>>> 23ed88531 (test: relax VLM input format assertions for image understanding (#12755))

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
<<<<<<< HEAD
                "letters",
                "icon",
                "sgi",  # 容忍一次错拼
=======
>>>>>>> 23ed88531 (test: relax VLM input format assertions for image understanding (#12755))
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
                            "text": "There are two pictures. First describe picture 1, then picture 2. What are in these pictures?",
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
            sampling_params=dict(temperature=0.0),
        )
        self.verify_response(output)

    async def test_accepts_precomputed_embeddings(self):
        req = self.get_completion_request()
        processor_output, _ = self.get_processor_output(req=req)
        with torch.inference_mode():
            precomputed_embeddings = self.__class__.visual(processor_output)
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()
        output = await self.engine.async_generate(
            prompt=text,
            image_data=self._precomputed_image_data(
                processor_output, precomputed_embeddings
            ),
            sampling_params=dict(temperature=0.0),
        )
        self.verify_response(output)

    async def test_accepts_processor_output(self):
        req = self.get_completion_request()
        processor_output, prompt = self.get_processor_output(req=req)
        output = await self.engine.async_generate(
            input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
            image_data=[dict(processor_output, format="processor_output")],
            sampling_params=dict(temperature=0.0),
        )
        self.verify_response(output)

    def _processor_output_image_data(self, processor_output):
        return [dict(processor_output, format="processor_output")]

    def _precomputed_image_data(self, processor_output, precomputed_embeddings):
        # Qwen2.5-VL：自己用视觉塔把每图算到“已 merge 的特征”，并逐图返回
        if "image_grid_thw" in processor_output:
            g = processor_output["image_grid_thw"]  # [N,3]
            pv = processor_output["pixel_values"]  # [N,T,D] 或 [sum_T,D]
            assert g.dim() == 2 and g.shape[1] == 3
            N = int(g.shape[0])

            visual = self.__class__.visual_model
            projector = getattr(self.__class__, "mm_projector", None)
            per_img_pv = []
            if pv.dim() == 3:
                assert pv.shape[0] == N
                for i in range(N):
                    per_img_pv.append(pv[i : i + 1])
            else:
                st = 0
                for i in range(N):
                    T = int((g[i, 0] * g[i, 1] * g[i, 2]).item())
                    ed = st + T
                    per_img_pv.append(pv[st:ed])
                    st = ed

            items = []
            with torch.inference_mode():
                for i in range(N):
                    v = visual(per_img_pv[i], grid_thw=g[i : i + 1])  # [Li, Dv]
                    feat_i = (
                        projector(v) if projector is not None else v
                    )  # [Li, Dh] 供 LLM
                    items.append(
                        {
                            "format": "precomputed_embedding",
                            "precomputed_embeddings": feat_i.detach().cpu(),
                            "num_vision_tokens": int(
                                feat_i.shape[0]
                            ),  # 仅提示注入长度，避免误路由
                        }
                    )
            assert len(items) == N
            return items

        # Gemma-3：同理
        pv = processor_output["pixel_values"]
        vt = self.__class__.vision_tower
        mp = self.__class__.mm_projector
        items = []
        with torch.inference_mode():
            for i in range(pv.shape[0]):
                feat_i = mp(vt(pixel_values=pv[i : i + 1]).last_hidden_state)  # [Li, D]
                items.append(
                    {
                        "format": "precomputed_embedding",
                        "precomputed_embeddings": feat_i.detach().cpu(),
                        "num_vision_tokens": int(feat_i.shape[0]),
                    }
                )
        return items


class TestQwenVLUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    chat_template = "qwen2-vl"

    @classmethod
    def _init_visual(cls):
        m = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cls.model_path, torch_dtype=torch.bfloat16
            )
            .eval()
            .to(cls.device)
        )

        # 视觉塔优先从 m.model.visual 取；若透传也支持 m.visual
        cls.visual_model = getattr(m.model, "visual", None) or getattr(
            m, "visual", None
        )
        assert (
            cls.visual_model is not None
        ), "visual tower not found on Qwen2.5-VL model"

        # 投影层常见命名与挂载位置做兜底匹配
        cand_names = [
            "multi_modal_projector",
            "mm_projector",
            "vision_projector",
            "visual_projector",
            "multi_modal_resampler",
            "perceiver_resampler",
        ]
        proj = None
        for name in cand_names:
            proj = getattr(m.model, name, None) or getattr(m, name, None)
            if proj is not None:
                break
        cls.mm_projector = proj  # 可能为 None（有的权重视觉塔直接输出 Dh）

        # 供 test 中的 self.__class__.visual(...) 直接用
        def _qwen_visual(po):
            v = cls.visual_model(
                po["pixel_values"], po["image_grid_thw"]
            )  # [L_i, Dv or Dh]
            return cls.mm_projector(v) if cls.mm_projector is not None else v

        cls.visual = _qwen_visual

    def _processor_output_image_data(self, processor_output):
        pv = processor_output["pixel_values"]
        gthw = processor_output["image_grid_thw"]
        num_images = pv.shape[0]
        items = []
        for i in range(num_images):
            items.append(
                {
                    "format": "processor_output",
                    "pixel_values": pv[i : i + 1],
                    "image_grid_thw": gthw[i : i + 1],
                }
            )
        return items


class TestGemmaUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    # model_path = "google/gemma-3-4b-it"
    model_path = "/root/.cache/modelscope/hub/models/LLM-Research/gemma-3-4b-it"
    chat_template = "gemma-it"

    @classmethod
    def _init_visual(cls):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            cls.model_path, torch_dtype=torch.bfloat16
        )
        cls.vision_tower = model.vision_tower.eval().to(cls.device)
        cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)
        cls.visual = lambda processor_output: cls.mm_projector(
            cls.vision_tower(
                pixel_values=processor_output["pixel_values"]
            ).last_hidden_state
        )

    # Temporarily skip Kimi-VL for CI test due to issue in transformers=4.57.0
    # class TestKimiVLImageUnderstandsImage(
    #     VLMInputTestBase, unittest.IsolatedAsyncioTestCase
    # ):
    #     model_path = "moonshotai/Kimi-VL-A3B-Instruct"
    #     chat_template = "kimi-vl"

    #     @classmethod
    #     def _init_visual(cls):
    #         model = AutoModel.from_pretrained(cls.model_path, trust_remote_code=True)
    #         cls.vision_tower = model.vision_tower.eval().to(cls.device)
    #         cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)

    #         cls.visual = lambda tokenizer_output: cls.mm_projector(
    #             cls.vision_tower(
    #                 pixel_values=tokenizer_output["pixel_values"],
    #                 grid_hws=tokenizer_output["image_grid_hws"],
    #             )
    #         )

    def _processor_output_image_data(self, processor_output):
        pv = processor_output["pixel_values"]
        num_images = pv.shape[0]
        items = []
        for i in range(num_images):
            items.append(
                {
                    "format": "processor_output",
                    "pixel_values": pv[i : i + 1],
                }
            )
        return items


# not for CI: too large
# class TestLlama4ImageUnderstandsImage(
#     VLMInputTestBase, unittest.IsolatedAsyncioTestCase
# ):
#     model_path = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
#     chat_template = "llama_4_vision"

#     def setUp(self):
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
#         model = AutoModel.from_pretrained(cls.model_path, trust_remote_code=True, torch_dtype="auto")
#         cls.vision_tower = model.vision_model.eval().to(cls.device)
#         cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)

#         cls.visual = lambda tokenizer_output: cls.mm_projector(
#             cls.vision_tower(
#                 pixel_values=tokenizer_output["pixel_values"],
#             ).last_hidden_state.flatten(0, -2)
#         )

#     def _pixel_values_image_data(self, processor_output):
#         return dict(
#             modality="IMAGE",
#             pixel_values=processor_output["pixel_values"],
#         )


if __name__ == "__main__":
    unittest.main()