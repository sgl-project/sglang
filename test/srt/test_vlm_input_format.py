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
from sglang.srt.conversation import generate_chat_conv
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

TEST_IMAGE_URL = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"


class VLMInputTestBase:
    model_path = None
    chat_template = None
    processor = None
    visual = None  # Should be a callable for precomputed features

    @classmethod
    def setUpClass(cls):
        assert cls.model_path is not None, "Set model_path in subclass"
        assert cls.chat_template is not None, "Set chat_template in subclass"
        cls.image_url = TEST_IMAGE_URL
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        response = requests.get(cls.image_url)
        cls.main_image = Image.open(BytesIO(response.content))
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True, use_fast=True
        )
        cls._init_visual()

    @classmethod
    def _init_visual(cls):
        """Override in subclass to set up cls.visual as a callable for precomputed features."""
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
        out_text = output["text"].lower()
        assert "taxi" in out_text or "cab" in out_text or "car" in out_text, out_text

    def get_completion_request(self) -> ChatCompletionRequest:
        json_structure = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.image_url}},
                        {"type": "text", "text": "What's in this picture?"},
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
            images=[self.main_image],
            return_tensors="pt",
        ).to(self.device)

        return inputs

    async def test_understands_image(self):
        req = self.get_completion_request()
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()
        output = await self.engine.async_generate(
            prompt=text,
            image_data=[self.main_image],
            sampling_params=dict(temperature=0.0),
        )
        self.verify_response(output)

    async def test_understands_precomputed_embeddings(self):
        req = self.get_completion_request()
        processor_output = self.get_processor_output(req=req)
        with torch.inference_mode():
            precomputed_embeddings = self.__class__.visual(processor_output)
        output = await self.engine.async_generate(
            input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
            image_data=[
                self._precomputed_image_data(processor_output, precomputed_embeddings)
            ],
            sampling_params=dict(temperature=0.0),
        )
        self.verify_response(output)

    async def test_understands_pixel_values(self):
        req = self.get_completion_request()
        processor_output = self.get_processor_output(req=req)
        output = await self.engine.async_generate(
            input_ids=processor_output["input_ids"][0].detach().cpu().tolist(),
            image_data=[self._pixel_values_image_data(processor_output)],
            sampling_params=dict(temperature=0.0),
        )
        self.verify_response(output)

    def _precomputed_image_data(self, processor_output, precomputed_embeddings):
        """This should not be overridden."""
        return dict(
            modality="IMAGE",
            precomputed_embeddings=precomputed_embeddings,
        )

    def _pixel_values_image_data(self, processor_output):
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

    def _pixel_values_image_data(self, processor_output):
        return dict(
            modality="IMAGE",
            image_grid_thw=processor_output["image_grid_thw"],
            pixel_values=processor_output["pixel_values"],
        )


class TestGemmaUnderstandsImage(VLMInputTestBase, unittest.IsolatedAsyncioTestCase):
    model_path = "google/gemma-3-4b-it"
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

    def _pixel_values_image_data(self, processor_output):
        return dict(
            modality="IMAGE",
            pixel_values=processor_output["pixel_values"][0],
        )


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

    def _pixel_values_image_data(self, processor_output):
        return dict(
            modality="IMAGE",
            pixel_values=processor_output["pixel_values"],
            image_grid_hws=processor_output["image_grid_hws"],
        )


class TestLlama4ImageUnderstandsImage(
    VLMInputTestBase, unittest.IsolatedAsyncioTestCase
):
    model_path = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    chat_template = "llama_4_vision"

    def setUp(self):
        self.engine = Engine(
            model_path=self.model_path,
            tokenizer_path=self.model_path,
            tokenizer_mode="auto",
            skip_tokenizer_init=False,
            load_format="auto",
            model_loader_extra_config="{}",
            trust_remote_code=True,
            dtype="auto",
            kv_cache_dtype="auto",
            quantization_param_path=None,
            context_length=20000,
            served_model_name=self.model_path,
            chat_template=self.chat_template,
            completion_template=None,
            is_embedding=False,
            enable_multimodal=True,
            revision=None,
            hybrid_kvcache_ratio=None,
            host="0.0.0.0",
            port=8000,
            mem_fraction_static=0.8,
            max_running_requests=None,
            max_total_tokens=None,
            chunked_prefill_size=32768,
            max_prefill_tokens=32768,
            schedule_policy="fcfs",
            schedule_conservativeness=1.0,
            cpu_offload_gb=0,
            page_size=1,
            tp_size=8,
            pp_size=1,
            max_micro_batch_size=None,
            stream_interval=1,
            stream_output=False,
            random_seed=941523765,
            constrained_json_whitespace_pattern=None,
            watchdog_timeout=300,
            dist_timeout=None,
            download_dir=None,
            base_gpu_id=0,
            gpu_id_step=1,
            sleep_on_idle=False,
            log_level="info",
            log_level_http=None,
            log_requests=False,
            log_requests_level=0,
            crash_dump_folder=None,
            show_time_cost=False,
            enable_metrics=False,
            bucket_time_to_first_token=None,
            bucket_e2e_request_latency=None,
            bucket_inter_token_latency=None,
            collect_tokens_histogram=False,
            decode_log_interval=40,
            enable_request_time_stats_logging=False,
            kv_events_config=None,
            api_key=None,
            file_storage_path="sglang_storage",
            enable_cache_report=False,
            reasoning_parser=None,
            tool_call_parser=None,
            dp_size=1,
            load_balance_method="round_robin",
            dist_init_addr=None,
            nnodes=1,
            node_rank=0,
            json_model_override_args="{}",
            preferred_sampling_params=None,
            lora_paths=None,
            max_loras_per_batch=8,
            lora_backend="triton",
            attention_backend=None,
            sampling_backend="flashinfer",
            grammar_backend="xgrammar",
            mm_attention_backend=None,
            speculative_algorithm=None,
            speculative_draft_model_path=None,
            speculative_num_steps=None,
            speculative_eagle_topk=None,
            speculative_num_draft_tokens=None,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
            speculative_token_map=None,
            ep_size=1,
            enable_ep_moe=False,
            enable_deepep_moe=False,
            enable_flashinfer_moe=False,
            deepep_mode="auto",
            ep_num_redundant_experts=0,
            ep_dispatch_algorithm="static",
            init_expert_location="trivial",
            enable_eplb=False,
            eplb_algorithm="auto",
            eplb_rebalance_num_iterations=1000,
            eplb_rebalance_layers_per_chunk=None,
            expert_distribution_recorder_mode=None,
            expert_distribution_recorder_buffer_size=1000,
            enable_expert_distribution_metrics=False,
            deepep_config=None,
            moe_dense_tp_size=None,
            enable_double_sparsity=False,
            ds_channel_config_path=None,
            ds_heavy_channel_num=32,
            ds_heavy_token_num=256,
            ds_heavy_channel_type="qk",
            ds_sparse_decode_threshold=4096,
            disable_radix_cache=True,
            cuda_graph_max_bs=2000,
            cuda_graph_bs=[2000],
            disable_cuda_graph=False,
        )

    @classmethod
    def _init_visual(cls):
        model = AutoModel.from_pretrained(cls.model_path, trust_remote_code=True, torch_dtype="auto")
        cls.vision_tower = model.vision_model.eval().to(cls.device)
        cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)

        cls.visual = lambda tokenizer_output: cls.mm_projector(
            cls.vision_tower(
                pixel_values=tokenizer_output["pixel_values"],
            ).last_hidden_state.flatten(0, -2)
        )

    def _pixel_values_image_data(self, processor_output):
        return dict(
            modality="IMAGE",
            pixel_values=processor_output["pixel_values"],
        )


if __name__ == "__main__":
    unittest.main()
