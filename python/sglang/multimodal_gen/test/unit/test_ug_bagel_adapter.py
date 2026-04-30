# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.pipelines.ug import _load_ug_bridge
from sglang.srt.session.session_controller import SessionController
from sglang.srt.ug.adapter import UGModelRunnerAdapter, UGModelSessionView
from sglang.srt.ug.bagel import (
    BAGELAdapterError,
    BAGELInterleaveContextBackend,
    BAGELNativeSRTDenoiseExecutor,
    BAGELNativeSRTPreparedDenoise,
    BAGELNativeSRTUForwardExecutor,
    BAGELUForwardBridge,
    BAGELUGModelAdapter,
    MockBAGELBackend,
    _build_native_srt_bagel_inferencer_shell,
    _ensure_bagel_transformers_compat,
    create_bagel_ug_model_adapter,
)
from sglang.srt.ug.bagel_cache import (
    BAGELPagedKVCacheBacking,
    BAGELSRTKVCacheFactory,
    InMemoryBAGELSRTKVCacheBacking,
)
from sglang.srt.ug.context import UGSessionHandle, UGSRTKVTokenBinding
from sglang.srt.ug.runtime import (
    UGInterleavedMessage,
    UGLatentDecodeRequest,
    UGLatentPrepareRequest,
    UGSegmentState,
    UGSessionRuntime,
    UGVelocityRequest,
)
from sglang.srt.ug.sampling import (
    build_bagel_denoise_schedule,
    get_bagel_effective_cfg_scales,
)
from sglang.srt.ug.srt_executor import UGSRTSchedulerExecutor


class TestBAGELUGModelAdapter(unittest.TestCase):
    def test_bagel_sampling_helpers_match_official_boundaries(self):
        schedule = build_bagel_denoise_schedule(
            num_inference_steps=4,
            timestep_shift=3.0,
        )
        self.assertTrue(
            torch.allclose(
                schedule.timesteps,
                torch.tensor([1.0, 6.0 / 7.0, 0.6]),
            )
        )
        self.assertTrue(
            torch.allclose(
                schedule.dts,
                torch.tensor([1.0 / 7.0, 9.0 / 35.0, 0.6]),
            )
        )

        cfg = SimpleNamespace(
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_interval=(0.5, 1.0),
        )
        self.assertEqual(
            get_bagel_effective_cfg_scales(cfg, torch.tensor([1.0])),
            (4.0, 1.5),
        )
        self.assertEqual(
            get_bagel_effective_cfg_scales(cfg, torch.tensor([0.4])),
            (1.0, 1.0),
        )

    def test_missing_checkpoint_path_reports_actionable_error(self):
        with self.assertRaisesRegex(
            BAGELAdapterError,
            "requires a local BAGEL checkpoint directory",
        ):
            BAGELUGModelAdapter("ByteDance-Seed/BAGEL-7B-MoT")

    def test_missing_checkpoint_files_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                BAGELAdapterError,
                "missing required files",
            ):
                BAGELUGModelAdapter(tmpdir)

    def test_real_loader_requires_native_srt_executor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_required_checkpoint_files(Path(tmpdir))
            with self.assertRaisesRegex(BAGELAdapterError, "native-SRT only"):
                BAGELUGModelAdapter(tmpdir)

    def test_real_loader_passes_native_srt_denoise_executor(self):
        inferencer = FakeBAGELInferencer()
        native_executor = BAGELNativeSRTDenoiseExecutor(FakeNativeSRTVelocityModel())
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            _write_required_checkpoint_files(checkpoint_dir)
            with patch(
                "sglang.srt.ug.bagel._build_native_srt_bagel_inferencer_shell",
                return_value=inferencer,
            ) as build_shell:
                adapter = BAGELUGModelAdapter(
                    tmpdir,
                    native_srt_denoise_executor=native_executor,
                )

        self.assertIs(adapter.backend.native_srt_denoise_executor, native_executor)
        self.assertIs(adapter.backend.inferencer, inferencer)
        build_shell.assert_called_once_with(checkpoint_dir)
        self.assertIsInstance(
            adapter.backend.u_forward_bridge.srt_u_forward_executor,
            BAGELNativeSRTUForwardExecutor,
        )

    def test_native_real_loader_does_not_probe_external_bagel_modules(self):
        inferencer = FakeBAGELInferencer()
        native_executor = BAGELNativeSRTDenoiseExecutor(FakeNativeSRTVelocityModel())

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            _write_required_checkpoint_files(checkpoint_dir)
            with patch(
                "sglang.srt.ug.bagel._build_native_srt_bagel_inferencer_shell",
                return_value=inferencer,
            ):
                adapter = BAGELUGModelAdapter(
                    tmpdir,
                    native_srt_denoise_executor=native_executor,
                )

        self.assertIs(adapter.backend.inferencer, inferencer)

    def test_load_bagel_bridge_wires_scheduler_executor_to_native_denoise(self):
        native_model = FakeNativeSRTVelocityModel()
        model_runner = FakeModelRunner(native_model)
        scheduler = FakeSchedulerWithModelRunner(model_runner)
        adapter = BAGELUGModelAdapter(
            "already-loaded-bagel",
            backend=MockBAGELBackend(),
        )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines.ug.create_bagel_ug_model_adapter",
            return_value=adapter,
        ) as create:
            bridge = _load_ug_bridge("local-bagel", scheduler=scheduler)

        create.assert_called_once()
        native_executor = create.call_args.kwargs["native_srt_denoise_executor"]
        self.assertTrue(create.call_args.kwargs["native_srt_u_context"])
        self.assertIs(native_executor.srt_model, native_model)
        self.assertIs(
            native_executor.forward_batch_provider.__self__,
            bridge.runtime.srt_request_executor,
        )
        self.assertIs(bridge.runtime.session_controller, scheduler.session_controller)
        self.assertEqual(bridge.runtime.srt_image_tokenization, "multimodal")

    def test_mock_bagel_adapter_factory_runs_interleave_loop(self):
        adapter = create_bagel_ug_model_adapter("sglang-internal/mock-bagel")
        self.assertIsInstance(adapter.backend, MockBAGELBackend)

        bridge = _load_ug_bridge("sglang-internal/mock-bagel")
        contexts = bridge.build_contexts(prompt="draw then explain", image=None)
        self.assertIsInstance(contexts.full.session, UGSessionHandle)

        latents = torch.zeros(1, 2, 4)
        for step in range(2):
            latents = bridge.predict_velocity(
                contexts=contexts,
                latent_tokens=latents,
                timestep=torch.tensor([1.0 - step * 0.5]),
                latent_position_ids=torch.arange(2),
                sampling_params=None,
            )

        bridge.append_generated_image(contexts=contexts, image=object())
        post_image = bridge.decode_next_segment(contexts=contexts)

        self.assertEqual(post_image.type, "text")
        self.assertEqual(post_image.text, "bagel_mock_text_after_image")
        self.assertEqual(contexts.full.token_count, 5)

        counters = bridge.runtime.get_debug_counters(contexts.full.session)
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 2)
        self.assertEqual(counters["append_image_count"], 1)
        self.assertEqual(counters["decode_count"], 2)
        self.assertEqual(counters["state"], "u_decode")

    def test_mock_bagel_velocity_depends_on_srt_session_view(self):
        adapter = create_bagel_ug_model_adapter("sglang-internal/mock-bagel")
        session = UGModelSessionView(
            handle=UGSessionHandle(
                session_id="bagel-view",
                anchor_request_id="bagel-view:u1",
                context_length=3,
                context_version=1,
            ),
            state=UGSegmentState.G_DENOISE,
            srt_request_count=3,
            srt_last_request_id="bagel-view:u1",
            srt_last_origin_input_len=3,
        )
        request = UGVelocityRequest(
            session=session.handle,
            latent_tokens=torch.zeros(1, 1, 2),
            timestep=torch.tensor([0.5]),
            latent_position_ids=torch.arange(1),
            sampling_params=None,
        )

        velocity = adapter.predict_velocity_from_session(
            session=session,
            request=request,
        )

        self.assertTrue(torch.allclose(velocity, torch.full_like(velocity, 1.15)))


class FakeContextBAGELModel:
    def __init__(self):
        self.forward_flow_calls = []
        self.language_model = SimpleNamespace(model=SimpleNamespace())
        self.prepare_vae_latent_calls = []
        self.prepare_vae_latent_cfg_calls = []
        self.hidden_size = 4
        self.latent_patch_size = 1
        self.latent_channel = 1
        self.language_model.model.embed_tokens = torch.nn.Embedding(8, self.hidden_size)
        self.vae2llm = torch.nn.Linear(1, self.hidden_size, bias=False)
        self.time_embedder = FakeTimeEmbedder(self.hidden_size)
        self.latent_pos_embed = torch.nn.Embedding(16, self.hidden_size)
        self.vit_model = FakeVITModel(self.hidden_size)
        self.connector = torch.nn.Identity()
        self.vit_pos_embed = torch.nn.Embedding(16, self.hidden_size)
        self.embed_grad_enabled = []

    def prepare_vae_latent(
        self,
        *,
        curr_kvlens,
        curr_rope,
        image_sizes,
        new_token_ids,
    ):
        self.prepare_vae_latent_calls.append(
            {
                "curr_kvlens": list(curr_kvlens),
                "curr_rope": list(curr_rope),
                "image_sizes": list(image_sizes),
                "new_token_ids": dict(new_token_ids),
            }
        )
        payload = dict(_fake_bagel_generation_input())
        payload["key_values_lens"] = torch.tensor(curr_kvlens, dtype=torch.int)
        return payload

    def prepare_vae_latent_cfg(self, *, curr_kvlens, curr_rope, image_sizes):
        self.prepare_vae_latent_cfg_calls.append(
            {
                "curr_kvlens": list(curr_kvlens),
                "curr_rope": list(curr_rope),
                "image_sizes": list(image_sizes),
            }
        )
        payload = dict(_fake_bagel_cfg_generation_input())
        payload["cfg_key_values_lens"] = torch.tensor(curr_kvlens, dtype=torch.int)
        return payload

    def prepare_vae_images(
        self,
        *,
        curr_kvlens,
        curr_rope,
        images,
        transforms,
        new_token_ids,
        timestep=0,
    ):
        del images, transforms
        self.prepare_vae_latent_calls.append(
            {
                "kind": "image",
                "curr_kvlens": list(curr_kvlens),
                "curr_rope": list(curr_rope),
            }
        )
        position = int(curr_rope[0])
        generation_input = {
            "padded_images": torch.ones(1, 1, 2, 2),
            "patchified_vae_latent_shapes": [(2, 2)],
            "packed_vae_position_ids": torch.arange(4, dtype=torch.long),
            "packed_timesteps": torch.tensor([timestep], dtype=torch.float32),
            "packed_vae_token_indexes": torch.tensor([1, 2, 3, 4]),
            "packed_text_ids": torch.tensor(
                [new_token_ids["start_of_image"], new_token_ids["end_of_image"]],
                dtype=torch.long,
            ),
            "packed_text_indexes": torch.tensor([0, 5], dtype=torch.long),
            "packed_position_ids": torch.full((6,), position, dtype=torch.long),
            "packed_seqlens": torch.tensor([6], dtype=torch.int32),
            "packed_indexes": torch.arange(
                int(curr_kvlens[0]),
                int(curr_kvlens[0]) + 6,
                dtype=torch.long,
            ),
            "packed_key_value_indexes": torch.arange(
                int(curr_kvlens[0]),
                dtype=torch.long,
            ),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int32),
        }
        return generation_input, [int(curr_kvlens[0]) + 6], [position + 1]

    def prepare_vit_images(
        self,
        *,
        curr_kvlens,
        curr_rope,
        images,
        transforms,
        new_token_ids,
    ):
        del images, transforms
        position = int(curr_rope[0])
        generation_input = {
            "packed_text_ids": torch.tensor(
                [new_token_ids["start_of_image"], new_token_ids["end_of_image"]],
                dtype=torch.long,
            ),
            "packed_text_indexes": torch.tensor([0, 3], dtype=torch.long),
            "vit_token_seqlens": torch.tensor([2], dtype=torch.int32),
            "packed_vit_tokens": torch.ones(2, self.hidden_size),
            "packed_vit_position_ids": torch.arange(2, dtype=torch.long),
            "packed_vit_token_indexes": torch.tensor([1, 2], dtype=torch.long),
            "packed_position_ids": torch.full((4,), position, dtype=torch.long),
            "packed_seqlens": torch.tensor([4], dtype=torch.int32),
            "packed_indexes": torch.arange(
                int(curr_kvlens[0]),
                int(curr_kvlens[0]) + 4,
                dtype=torch.long,
            ),
            "packed_key_value_indexes": torch.arange(
                int(curr_kvlens[0]),
                dtype=torch.long,
            ),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int32),
        }
        return generation_input, [int(curr_kvlens[0]) + 4], [position + 1]

    def embed_bagel_vae_image(self, generation_input):
        self.embed_grad_enabled.append(torch.is_grad_enabled())
        return self._fake_embeds_from_generation_input(generation_input)

    def embed_bagel_vit_image(self, generation_input):
        self.embed_grad_enabled.append(torch.is_grad_enabled())
        return self._fake_embeds_from_generation_input(generation_input)

    def decode_bagel_image(self, latent, image_shape):
        del latent
        return FakeImage(size=(image_shape[1], image_shape[0]))

    def _fake_embeds_from_generation_input(self, generation_input):
        seq_len = int(generation_input["packed_seqlens"].sum().item())
        values = torch.arange(seq_len * self.hidden_size, dtype=torch.float32)
        return values.reshape(seq_len, self.hidden_size)


class FakeVAEModel(torch.nn.Module):
    def encode(self, padded_images):
        return padded_images


class FakeVITModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        *,
        packed_pixel_values,
        packed_flattened_position_ids,
        cu_seqlens,
        max_seqlen,
    ):
        del packed_flattened_position_ids, cu_seqlens, max_seqlen
        return packed_pixel_values.to(torch.float32)


class FakeTimeEmbedder(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, timestep):
        return timestep.reshape(-1, 1).to(torch.float32) * self.weight.reshape(1, -1)


class FakeNativeSRTVelocityModel(FakeContextBAGELModel):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.grad_enabled = []
        self.latent_downsample = 16

    def predict_velocity_from_packed_gen(self, **kwargs):
        self.calls.append(kwargs)
        self.grad_enabled.append(torch.is_grad_enabled())
        return kwargs["latent_tokens"] + 9.0


class FakeBranchNativeSRTVelocityModel(FakeNativeSRTVelocityModel):
    def predict_velocity_from_packed_gen(self, **kwargs):
        self.calls.append(kwargs)
        self.grad_enabled.append(torch.is_grad_enabled())
        marker = float(kwargs["packed_position_ids"][0].item())
        return torch.full_like(kwargs["latent_tokens"], marker)


class FakeReqToTokenPool:
    def __init__(self, size=2, max_context_len=16):
        self.device = "cpu"
        self.max_context_len = max_context_len
        self.req_to_token = torch.full(
            (size, max_context_len),
            -1,
            dtype=torch.int32,
        )
        self.free_slots = list(range(size))
        self.freed_reqs = []

    def alloc(self, reqs):
        if len(reqs) > len(self.free_slots):
            return None
        allocated = self.free_slots[: len(reqs)]
        self.free_slots = self.free_slots[len(reqs) :]
        for req, idx in zip(reqs, allocated):
            req.req_pool_idx = idx
        return allocated

    def free(self, req):
        self.freed_reqs.append(req.req_pool_idx)
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def write(self, indices, values):
        self.req_to_token[indices] = values


class FakeTokenToKVPoolAllocator:
    def __init__(self, start=20, page_size=1):
        self.next_index = start
        self.page_size = page_size
        self.alloc_calls = []
        self.alloc_extend_calls = []
        self.freed = []

    def alloc(self, need_size):
        self.alloc_calls.append(need_size)
        out = torch.arange(
            self.next_index,
            self.next_index + need_size,
            dtype=torch.int64,
        )
        self.next_index += need_size
        return out

    def free(self, free_index):
        self.freed.append(free_index.clone())

    def get_kvcache(self):
        return "fake-kv-cache"

    def alloc_extend(self, *args, **kwargs):
        self.alloc_extend_calls.append((args, kwargs))
        raise AssertionError("UG temp G must not extend into the U prefix page")


class FakeAttentionBackend:
    def __init__(self):
        self.forward_batches = []

    def init_forward_metadata(self, forward_batch):
        self.forward_batches.append(forward_batch)


class FakeModelRunner:
    def __init__(self, model, *, token_to_kv_pool_allocator=None):
        self.device = "cpu"
        self.model = model
        self.req_to_token_pool = FakeReqToTokenPool()
        self.token_to_kv_pool_allocator = (
            token_to_kv_pool_allocator or FakeTokenToKVPoolAllocator()
        )
        self.token_to_kv_pool = "fake-kv-cache"
        self.attn_backend = FakeAttentionBackend()
        self.spec_algorithm = None


class FakeSchedulerWithModelRunner:
    def __init__(self, model_runner):
        self.session_controller = object()
        self.model_worker = SimpleNamespace(model_runner=model_runner)
        self.tree_cache = None

    def is_fully_idle(self):
        return True


class FakeImage:
    def __init__(self, size=(16, 8)):
        self.size = size


class FakeTreeCache:
    def __init__(self):
        self.released_sessions = []

    def release_session(self, session_id):
        self.released_sessions.append(session_id)


class OutputAppendingSRTExecutor:
    def __init__(self, token_id=123):
        self.token_id = token_id
        self.events = []

    def execute_ug_request(self, *, record, req, state):
        del record
        self.events.append((state.value, req.rid, req.sampling_params.max_new_tokens))
        if req.sampling_params.max_new_tokens > 0:
            req.output_ids.append(self.token_id)


class BindingSRTExecutor(OutputAppendingSRTExecutor):
    def __init__(self, token_id=123, start_index=4):
        super().__init__(token_id=token_id)
        self.start_index = start_index

    def get_ug_request_token_binding(self, *, record, req, state):
        del state
        token_count = len(req.origin_input_ids)
        return UGSRTKVTokenBinding(
            session_id=record.session_id,
            request_id=req.rid,
            token_count=token_count,
            token_indices=torch.arange(
                self.start_index,
                self.start_index + token_count,
                dtype=torch.long,
            ),
        )


class DecodingTokenizer:
    bos_token_id = None
    eos_token_id = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [100 + index for index, _ in enumerate(str(text).split())]

    def decode(self, token_ids):
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)


class RecordingBindingSRTExecutor(BindingSRTExecutor):
    def __init__(self, token_id=123, start_index=4):
        super().__init__(token_id=token_id, start_index=start_index)
        self.requests = []

    def execute_ug_request(self, *, record, req, state):
        metadata = getattr(req, "ug_u_forward_metadata", {})
        self.requests.append(
            {
                "state": state.value,
                "rid": req.rid,
                "origin_input_ids": list(req.origin_input_ids),
                "position_ids": list(getattr(req, "ug_position_ids", []) or []),
                "input_text": metadata.get("input_text"),
                "adapter_metadata": dict(metadata.get("adapter_metadata", {})),
            }
        )
        super().execute_ug_request(record=record, req=req, state=state)


class RecordingBAGELSRTUForwardExecutor(BAGELNativeSRTUForwardExecutor):
    def __init__(self):
        self.events = []

    def execute(self, backend, *, session, request, messages):
        self.events.append(
            (
                request.state,
                request.request_id,
                "srt_kv_token_binding" in request.metadata,
            )
        )
        return super().execute(
            backend,
            session=session,
            request=request,
            messages=messages,
        )


class FakeBAGELImageTransform:
    def __init__(self):
        self.resize_calls = []

    def resize_transform(self, image):
        self.resize_calls.append(image)
        return image


class FakeBAGELTokenizer:
    def __init__(self):
        self.bos_token_id = None
        self.eos_token_id = None

    def encode(self, text, *args, **kwargs):
        del args, kwargs
        return [100 + index for index, _ in enumerate(str(text).split())]

    def decode(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


class FakeBAGELInferencer:
    def __init__(self):
        self.model = FakeContextBAGELModel()
        self.vae_model = FakeVAEModel()
        self.tokenizer = FakeBAGELTokenizer()
        self.new_token_ids = {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "start_of_image": 3,
            "end_of_image": 4,
        }
        self.vae_transform = FakeBAGELImageTransform()
        self.vit_transform = FakeBAGELImageTransform()
        self.events = []
        self.decode_image_calls = []

    def init_gen_context(self):
        self.events.append(("init",))
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": {"id": "ctx0"},
        }

    def update_context_text(self, text, gen_context):
        self.events.append(("text", text, gen_context["past_key_values"]["id"]))
        token_count = len(self.tokenizer.encode(text)) + 2
        return {
            "kv_lens": [gen_context["kv_lens"][0] + token_count],
            "ropes": [gen_context["ropes"][0] + token_count],
            "past_key_values": {
                "id": f"{gen_context['past_key_values']['id']}:t{text}"
            },
        }

    def update_context_image(self, image, gen_context, vae=True, vit=True):
        self.events.append(
            ("image", image, vae, vit, gen_context["past_key_values"]["id"])
        )
        return {
            "kv_lens": [gen_context["kv_lens"][0] + 2],
            "ropes": [gen_context["ropes"][0] + 1],
            "past_key_values": {"id": f"{gen_context['past_key_values']['id']}:i"},
        }

    def gen_text(self, gen_context, **kwargs):
        self.events.append(("gen_text", gen_context["past_key_values"]["id"], kwargs))
        return "context_backend_text_after_image"

    def decode_image(self, latent, image_shape):
        self.decode_image_calls.append(
            {
                "latent": latent,
                "image_shape": image_shape,
            }
        )
        return FakeImage(size=(image_shape[1], image_shape[0]))


class FakeNaiveBAGELCache:
    def __init__(self, num_layers):
        self.key_cache = {layer_id: None for layer_id in range(num_layers)}
        self.value_cache = {layer_id: None for layer_id in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        key = self.key_cache[0]
        return 0 if key is None else key.shape[0]


class FakeSRTCacheBAGELInferencer:
    def __init__(self):
        self.model = FakeContextBAGELModel()
        self.vae_model = FakeVAEModel()
        self.tokenizer = FakeBAGELTokenizer()
        self.new_token_ids = {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "start_of_image": 3,
            "end_of_image": 4,
        }
        self.vae_transform = FakeBAGELImageTransform()
        self.vit_transform = FakeBAGELImageTransform()
        self.events = []

    def init_gen_context(self):
        self.events.append(("init",))
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": FakeNaiveBAGELCache(num_layers=2),
        }

    def update_context_text(self, text, gen_context):
        token_count = len(self.tokenizer.encode(text)) + 2
        self.events.append(("text", text, gen_context["past_key_values"].seq_lens))
        return self._append_cache_tokens(gen_context, token_count=token_count)

    def update_context_image(self, image, gen_context, vae=True, vit=True):
        del image, vae, vit
        self.events.append(("image", gen_context["past_key_values"].seq_lens))
        return self._append_cache_tokens(gen_context, token_count=2)

    def gen_text(self, gen_context, **kwargs):
        del kwargs
        return f"srt_cache_text_{gen_context['past_key_values'].seq_lens}"

    def decode_image(self, latent, image_shape):
        del latent
        return FakeImage(size=(image_shape[1], image_shape[0]))

    @staticmethod
    def _append_cache_tokens(gen_context, *, token_count):
        cache = gen_context["past_key_values"]
        old_len = cache.seq_lens
        new_len = old_len + token_count
        for layer_id in range(cache.num_layers):
            key = torch.full(
                (new_len, 1, 2),
                float(layer_id + 1),
                dtype=torch.float32,
            )
            value = torch.full(
                (new_len, 1, 2),
                float(layer_id + 101),
                dtype=torch.float32,
            )
            old_key = cache.key_cache[layer_id]
            old_value = cache.value_cache[layer_id]
            if old_key is not None:
                key[:old_len] = old_key
            if old_value is not None:
                value[:old_len] = old_value
            cache.key_cache[layer_id] = key
            cache.value_cache[layer_id] = value
        gen_context["kv_lens"] = [new_len]
        gen_context["ropes"] = [gen_context["ropes"][0] + 1]
        return gen_context


class FakePagedKVPool:
    def __init__(self, *, size=32, num_layers=2):
        self.key_buffers = [torch.zeros(size, 1, 2) for _ in range(num_layers)]
        self.value_buffers = [torch.zeros(size, 1, 2) for _ in range(num_layers)]

    def get_key_buffer(self, layer_id):
        return self.key_buffers[layer_id]

    def get_value_buffer(self, layer_id):
        return self.value_buffers[layer_id]


class FakePagedAllocator:
    def __init__(self, *, page_size=4):
        self.page_size = page_size
        self.kv_cache = FakePagedKVPool()
        self.next_index = 0
        self.alloc_sizes = []
        self.freed = []

    def get_kvcache(self):
        return self.kv_cache

    def alloc(self, need_size):
        self.alloc_sizes.append(need_size)
        start = self.next_index
        self.next_index += need_size
        return torch.arange(start, start + need_size, dtype=torch.long)

    def free(self, indices):
        self.freed.append(indices.clone())


def _write_required_checkpoint_files(checkpoint_dir: Path) -> None:
    for name in ("llm_config.json", "vit_config.json"):
        (checkpoint_dir / name).write_text("{}", encoding="utf-8")
    for name in ("ae.safetensors", "ema.safetensors"):
        (checkpoint_dir / name).write_bytes(b"fake")


def _fake_bagel_generation_input() -> dict[str, torch.Tensor]:
    return {
        "packed_text_ids": torch.tensor([1, 2]),
        "packed_text_indexes": torch.tensor([0, 3]),
        "packed_init_noises": torch.zeros(8, 64),
        "packed_vae_token_indexes": torch.tensor([1, 2]),
        "packed_vae_position_ids": torch.arange(8),
        "packed_seqlens": torch.tensor([4], dtype=torch.int),
        "packed_position_ids": torch.tensor([0, 0, 0, 0]),
        "packed_indexes": torch.tensor([0, 1, 2, 3]),
        "key_values_lens": torch.tensor([5], dtype=torch.int),
        "packed_key_value_indexes": torch.tensor([0, 1, 2, 3, 4]),
    }


def _fake_bagel_cfg_generation_input() -> dict[str, torch.Tensor]:
    return {
        "cfg_packed_position_ids": torch.tensor([0, 0, 0, 0]),
        "cfg_packed_query_indexes": torch.tensor([5, 6, 7, 8]),
        "cfg_key_values_lens": torch.tensor([5], dtype=torch.int),
        "cfg_packed_key_value_indexes": torch.tensor([0, 1, 2, 3, 4]),
    }


class FakeTokenizer:
    def __init__(self):
        self.special_tokens_map = {}
        self.tokens = {}
        self.added_tokens = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        tokenizer = cls()
        tokenizer.loaded_from = path
        tokenizer.load_kwargs = kwargs
        return tokenizer

    def add_tokens(self, tokens):
        self.added_tokens.extend(tokens)
        for token in tokens:
            self.tokens[token] = len(self.tokens) + 1
        return len(tokens)

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]


class FakeImageTransformForLoader:
    def __init__(self, *args):
        self.args = args


class TestBAGELRealLoader(unittest.TestCase):
    def test_transformers_rope_default_compat(self):
        import transformers.modeling_rope_utils as rope_utils

        original = dict(rope_utils.ROPE_INIT_FUNCTIONS)
        try:
            rope_utils.ROPE_INIT_FUNCTIONS.pop("default", None)
            _ensure_bagel_transformers_compat()
            self.assertIn("default", rope_utils.ROPE_INIT_FUNCTIONS)

            inv_freq, attention_scaling = rope_utils.ROPE_INIT_FUNCTIONS["default"](
                SimpleNamespace(
                    rope_theta=10000.0,
                    hidden_size=8,
                    num_attention_heads=2,
                ),
                device=torch.device("cpu"),
            )

            self.assertEqual(tuple(inv_freq.shape), (2,))
            self.assertEqual(attention_scaling, 1.0)
        finally:
            rope_utils.ROPE_INIT_FUNCTIONS.clear()
            rope_utils.ROPE_INIT_FUNCTIONS.update(original)

    def test_build_native_srt_inferencer_shell_uses_local_loader_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            _write_required_checkpoint_files(checkpoint_dir)
            inferencer = _build_native_srt_bagel_inferencer_shell(
                checkpoint_dir,
                tokenizer_loader=FakeTokenizer,
                image_transform_cls=FakeImageTransformForLoader,
            )

        self.assertIsNone(inferencer.model)
        self.assertIsNone(inferencer.vae_model)
        self.assertEqual(inferencer.tokenizer.loaded_from, str(checkpoint_dir))
        self.assertEqual(inferencer.tokenizer.load_kwargs, {"use_fast": False})
        self.assertEqual(
            set(inferencer.new_token_ids),
            {"bos_token_id", "eos_token_id", "start_of_image", "end_of_image"},
        )
        self.assertEqual(inferencer.vae_transform.args, (1024, 512, 16))
        self.assertEqual(inferencer.vit_transform.args, (980, 224, 14))
        self.assertEqual(inferencer.init_gen_context()["kv_lens"], [0])
        with self.assertRaisesRegex(BAGELAdapterError, "outside SRT"):
            inferencer.update_context_image(None, {})


class TestBAGELSRTKVCacheAdapter(unittest.TestCase):
    def test_in_memory_cache_matches_naive_cache_shape_without_kv_slots(self):
        backing = InMemoryBAGELSRTKVCacheBacking()
        factory = BAGELSRTKVCacheFactory(backing)
        cache = factory.create_cache(
            session_id="srt-cache-session",
            role="full",
            template_cache=FakeNaiveBAGELCache(num_layers=2),
        )

        key = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
        value = key + 100
        cache.key_cache[0] = key
        cache.value_cache[0] = value

        self.assertEqual(cache.num_layers, 2)
        self.assertEqual(cache.seq_lens, 3)
        self.assertTrue(torch.equal(cache.key_cache[0], key))
        self.assertTrue(torch.equal(cache.value_cache[0], value))
        self.assertIsNone(cache.key_cache[1])

        cloned = factory.clone_cache(
            cache,
            session_id="srt-cache-session",
            role="cfg_text",
        )
        cache.key_cache[0] = torch.zeros_like(key)

        self.assertEqual(cloned.handle.role, "cfg_text")
        self.assertTrue(torch.equal(cloned.key_cache[0], key))
        handle_fields = {field.name for field in fields(cache.handle)}
        self.assertTrue(
            handle_fields.isdisjoint({"kv_slot", "slot", "page", "allocator"})
        )

        factory.release_session("srt-cache-session")
        self.assertEqual(backing.released_sessions, ["srt-cache-session"])

    def test_paged_backing_uses_srt_allocator_pages_for_bagel_cache_tensors(self):
        allocator = FakePagedAllocator(page_size=4)
        backing = BAGELPagedKVCacheBacking(allocator)
        factory = BAGELSRTKVCacheFactory(backing)
        cache = factory.create_cache(
            session_id="paged-cache-session",
            role="full",
            template_cache=FakeNaiveBAGELCache(num_layers=2),
        )

        key = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
        value = key + 10
        cache.key_cache[0] = key
        cache.value_cache[0] = value

        self.assertEqual(allocator.alloc_sizes, [4])
        self.assertTrue(torch.equal(cache.key_cache[0], key))
        self.assertTrue(torch.equal(cache.value_cache[0], value))

        bigger_key = torch.arange(10, dtype=torch.float32).reshape(5, 1, 2)
        bigger_value = bigger_key + 20
        cache.key_cache[0] = bigger_key
        cache.value_cache[0] = bigger_value

        self.assertEqual(allocator.alloc_sizes, [4, 8])
        self.assertEqual(len(allocator.freed), 1)
        self.assertEqual(tuple(allocator.freed[0].shape), (4,))
        self.assertTrue(torch.equal(cache.key_cache[0], bigger_key))
        self.assertTrue(torch.equal(cache.value_cache[0], bigger_value))

        factory.release_session("paged-cache-session")
        self.assertEqual(len(allocator.freed), 2)
        self.assertEqual(tuple(allocator.freed[-1].shape), (8,))

    def test_paged_backing_can_bind_to_srt_request_token_indices(self):
        allocator = FakePagedAllocator(page_size=4)
        backing = BAGELPagedKVCacheBacking(allocator)
        factory = BAGELSRTKVCacheFactory(backing)
        cache = factory.create_cache(
            session_id="bound-cache-session",
            role="full",
            template_cache=FakeNaiveBAGELCache(num_layers=2),
        )

        factory.bind_request_tokens(
            UGSRTKVTokenBinding(
                session_id="bound-cache-session",
                request_id="bound-cache-session:u1",
                token_count=4,
                token_indices=torch.tensor([4, 5, 6, 7], dtype=torch.long),
            )
        )
        key = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
        value = key + 10
        cache.key_cache[0] = key
        cache.value_cache[0] = value

        self.assertEqual(allocator.alloc_sizes, [])
        self.assertTrue(torch.equal(cache.key_cache[0], key))
        self.assertTrue(torch.equal(cache.value_cache[0], value))
        self.assertTrue(
            torch.equal(
                allocator.kv_cache.get_key_buffer(0)[5:8],
                key,
            )
        )

        factory.release_session("bound-cache-session")
        self.assertEqual(allocator.freed, [])

    def test_context_backend_binds_srt_forward_view_to_request_tokens(self):
        allocator = FakePagedAllocator(page_size=4)
        backend = BAGELInterleaveContextBackend(
            FakeSRTCacheBAGELInferencer(),
            srt_kv_cache_factory=BAGELSRTKVCacheFactory(
                BAGELPagedKVCacheBacking(allocator)
            ),
            default_image_shape=(32, 32),
        )
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=BindingSRTExecutor(start_index=6),
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="red kite")],
            session_id="bagel-bound-srt-view",
        )

        self.assertEqual(allocator.alloc_sizes, [])
        self.assertEqual(allocator.freed, [])
        self.assertEqual(
            backend.sessions["bagel-bound-srt-view"].gen_context["kv_lens"], [4]
        )
        self.assertIsNotNone(
            backend.sessions["bagel-bound-srt-view"].native_srt_u_context_token_binding
        )
        self.assertEqual(
            runtime.get_debug_counters(handle)["srt_last_request_id"],
            "bagel-bound-srt-view:u1",
        )

    def test_native_srt_text_prefill_uses_bagel_prompt_packing(self):
        inferencer = FakeSRTCacheBAGELInferencer()
        backend = BAGELInterleaveContextBackend(
            inferencer,
            u_forward_bridge=BAGELUForwardBridge(
                srt_u_forward_executor=BAGELNativeSRTUForwardExecutor()
            ),
            native_srt_denoise_executor=BAGELNativeSRTDenoiseExecutor(
                FakeNativeSRTVelocityModel()
            ),
            default_image_shape=(32, 32),
        )
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        executor = RecordingBindingSRTExecutor(start_index=10)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=executor,
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="red kite")],
            session_id="bagel-native-text-pack",
        )
        state = backend.sessions["bagel-native-text-pack"]

        self.assertEqual(
            executor.requests[0]["origin_input_ids"],
            [1, 100, 101, 2],
        )
        self.assertEqual(executor.requests[0]["position_ids"], [0, 1, 2, 3])
        self.assertEqual(state.gen_context["kv_lens"], [4])
        self.assertEqual(state.gen_context["ropes"], [4])
        counters = runtime.get_debug_counters(handle)
        self.assertEqual(counters["context_length"], 4)
        self.assertEqual(counters["ug_model_state"]["bagel"]["logical_kv_lens"], [4])
        self.assertEqual(counters["ug_model_state"]["bagel"]["logical_ropes"], [4])

    def test_native_srt_append_image_uses_runtime_bagel_rope_state(self):
        inferencer = FakeBAGELInferencer()
        native_model = FakeNativeSRTVelocityModel()
        backend = BAGELInterleaveContextBackend(
            inferencer,
            u_forward_bridge=BAGELUForwardBridge(
                srt_u_forward_executor=BAGELNativeSRTUForwardExecutor()
            ),
            native_srt_denoise_executor=BAGELNativeSRTDenoiseExecutor(
                native_model,
                forward_batch_provider=lambda **kwargs: SimpleNamespace(
                    source="native-srt",
                    kwargs=kwargs,
                ),
            ),
            default_image_shape=(32, 32),
        )
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=BindingSRTExecutor(start_index=9),
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="red kite")],
            session_id="bagel-runtime-owned-rope",
        )
        state = backend.sessions["bagel-runtime-owned-rope"]
        bagel_state = runtime.get_debug_counters(handle)["ug_model_state"]["bagel"]
        self.assertEqual(bagel_state["logical_kv_lens"], [4])
        self.assertEqual(bagel_state["logical_ropes"], [4])

        state.gen_context["kv_lens"] = [999]
        state.gen_context["ropes"] = [999]
        handle = runtime.begin_g_denoise(handle)
        handle = runtime.append_generated_image(handle, image=FakeImage(size=(8, 8)))

        self.assertEqual(native_model.prepare_vae_latent_calls[0]["curr_kvlens"], [4])
        self.assertEqual(native_model.prepare_vae_latent_calls[0]["curr_rope"], [4])
        bagel_state = runtime.get_debug_counters(handle)["ug_model_state"]["bagel"]
        self.assertEqual(bagel_state["logical_kv_lens"], [14])
        self.assertEqual(bagel_state["logical_ropes"], [6])

    def test_native_srt_vlm_image_prefill_can_use_vit_only(self):
        inferencer = FakeSRTCacheBAGELInferencer()
        backend = BAGELInterleaveContextBackend(
            inferencer,
            u_forward_bridge=BAGELUForwardBridge(
                srt_u_forward_executor=BAGELNativeSRTUForwardExecutor()
            ),
            native_srt_denoise_executor=BAGELNativeSRTDenoiseExecutor(
                FakeNativeSRTVelocityModel()
            ),
            default_image_shape=(32, 32),
        )
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        executor = RecordingBindingSRTExecutor(start_index=10)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=executor,
        )
        image = FakeImage(size=(8, 8))

        runtime.prefill_interleaved(
            [
                UGInterleavedMessage(
                    type="image",
                    content={"image": image, "vae": False, "vit": True},
                )
            ],
            session_id="bagel-native-vlm-image",
        )

        self.assertEqual(len(executor.requests), 1)
        self.assertEqual(executor.requests[0]["rid"], "bagel-native-vlm-image:u1")
        self.assertEqual(executor.requests[0]["input_text"], "<bagel:vit:image>")
        self.assertEqual(
            executor.requests[0]["adapter_metadata"]["bagel_u_image_stage"], "vit"
        )
        self.assertEqual(inferencer.vae_transform.resize_calls, [image])
        self.assertEqual(inferencer.vit_transform.resize_calls, [])
        self.assertEqual(
            backend.sessions["bagel-native-vlm-image"].gen_context["ropes"], [1]
        )
        self.assertEqual(
            backend.native_srt_denoise_executor.srt_model.embed_grad_enabled, [False]
        )

    def test_native_srt_image_edit_prefill_builds_cfg_img_sidecar(self):
        inferencer = FakeSRTCacheBAGELInferencer()
        native_model = FakeNativeSRTVelocityModel()
        backend = BAGELInterleaveContextBackend(
            inferencer,
            u_forward_bridge=BAGELUForwardBridge(
                srt_u_forward_executor=BAGELNativeSRTUForwardExecutor()
            ),
            native_srt_denoise_executor=BAGELNativeSRTDenoiseExecutor(native_model),
            default_image_shape=(32, 32),
        )
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        executor = RecordingBindingSRTExecutor(start_index=10)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=executor,
        )

        handle = runtime.prefill_interleaved(
            [
                UGInterleavedMessage(type="image", content=FakeImage(size=(8, 8))),
                UGInterleavedMessage(type="text", content="edit this"),
            ],
            session_id="bagel-native-edit-sidecar",
        )

        state = backend.sessions["bagel-native-edit-sidecar"]
        counters = runtime.get_debug_counters(handle)
        self.assertEqual(counters["srt_sidecar_request_count"], 1)
        self.assertEqual(
            counters["srt_sidecar_session_ids"],
            ["bagel-native-edit-sidecar:cfg_img"],
        )
        self.assertIn("sidecar:cfg_img", executor.requests[3]["rid"])
        self.assertEqual(state.cfg_img_context["kv_lens"], [4])
        self.assertEqual(state.cfg_img_context["ropes"], [4])
        self.assertIsNotNone(state.native_srt_cfg_img_token_binding)
        self.assertEqual(
            state.native_srt_cfg_img_token_binding.session_id,
            "bagel-native-edit-sidecar:cfg_img",
        )
        self.assertEqual(state.native_srt_cfg_img_token_binding.token_count, 4)
        self.assertEqual(
            counters["ug_model_state"]["bagel"],
            {
                "logical_kv_lens": [14],
                "logical_ropes": [6],
                "cfg_text_logical_kv_lens": [10],
                "cfg_text_logical_ropes": [2],
                "cfg_text_token_count": 10,
                "cfg_img_logical_kv_lens": [4],
                "cfg_img_logical_ropes": [4],
                "cfg_img_token_count": 4,
                "cfg_img_requires_sidecar": True,
                "cfg_img_sidecar_session_id": "bagel-native-edit-sidecar:cfg_img",
            },
        )

        state.gen_context["kv_lens"] = [999]
        state.gen_context["ropes"] = [999]
        state.cfg_text_context["kv_lens"] = [888]
        state.cfg_text_context["ropes"] = [888]
        state.cfg_img_context["kv_lens"] = [777]
        state.cfg_img_context["ropes"] = [777]
        handle = runtime.begin_g_denoise(handle)
        runtime.prepare_latents(
            UGLatentPrepareRequest(
                session=handle,
                sampling_params=SimpleNamespace(
                    height=32,
                    width=32,
                    cfg_text_scale=1.0,
                    cfg_img_scale=1.0,
                    cfg_interval=(0.0, 1.0),
                    cfg_renorm_min=0.0,
                    cfg_renorm_type="global",
                ),
                seed=123,
            )
        )
        self.assertEqual(native_model.prepare_vae_latent_calls[-1]["curr_kvlens"], [13])
        self.assertEqual(native_model.prepare_vae_latent_calls[-1]["curr_rope"], [6])
        self.assertEqual(
            native_model.prepare_vae_latent_cfg_calls[-2:],
            [
                {"curr_kvlens": [10], "curr_rope": [2], "image_sizes": [(32, 32)]},
                {"curr_kvlens": [4], "curr_rope": [4], "image_sizes": [(32, 32)]},
            ],
        )

    def test_srt_bound_interleave_uses_u_forward_executor(self):
        allocator = FakePagedAllocator(page_size=4)
        u_forward_executor = RecordingBAGELSRTUForwardExecutor()
        backend = BAGELInterleaveContextBackend(
            FakeSRTCacheBAGELInferencer(),
            u_forward_bridge=BAGELUForwardBridge(
                srt_u_forward_executor=u_forward_executor
            ),
            srt_kv_cache_factory=BAGELSRTKVCacheFactory(
                BAGELPagedKVCacheBacking(allocator)
            ),
            native_srt_denoise_executor=BAGELNativeSRTDenoiseExecutor(
                FakeNativeSRTVelocityModel(),
                forward_batch_provider=lambda **kwargs: SimpleNamespace(
                    source="native-srt",
                    kwargs=kwargs,
                ),
            ),
            default_image_shape=(32, 32),
        )
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=BindingSRTExecutor(token_id=88, start_index=6),
            srt_u_decode_max_new_tokens=1,
            tokenizer=DecodingTokenizer(),
            vocab_size=512,
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="red kite")],
            session_id="bagel-strict-srt-u-g-u",
        )
        self.assertEqual(allocator.alloc_sizes, [])
        marker = runtime.decode_next_segment(handle)
        self.assertEqual(marker.type, "image_marker")

        sampling_params = SimpleNamespace(
            height=32,
            width=32,
            cfg_text_scale=1.0,
            cfg_img_scale=1.0,
            cfg_interval=(0.0, 1.0),
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
        )
        prepared_latents = runtime.prepare_latents(
            UGLatentPrepareRequest(
                session=handle,
                sampling_params=sampling_params,
                seed=123,
            )
        )
        response = runtime.predict_velocity(
            UGVelocityRequest(
                session=handle,
                latent_tokens=prepared_latents.latent_tokens,
                timestep=torch.tensor([0.5]),
                latent_position_ids=prepared_latents.latent_position_ids,
                sampling_params=sampling_params,
            )
        )
        generated_image = runtime.decode_latents_to_image(
            UGLatentDecodeRequest(
                session=response.session,
                latent_tokens=response.velocity,
                sampling_params=sampling_params,
            )
        )
        handle = runtime.append_generated_image(response.session, image=generated_image)
        text = runtime.decode_next_segment(handle)

        self.assertEqual(text.type, "text")
        self.assertEqual(text.text, "decoded:1")
        self.assertEqual(
            u_forward_executor.events,
            [
                ("u_prefill", "bagel-strict-srt-u-g-u:u1", True),
                ("u_decode", "bagel-strict-srt-u-g-u:d1", True),
                ("append_image", "bagel-strict-srt-u-g-u:u2:s1", True),
                ("append_image", "bagel-strict-srt-u-g-u:u2:s2", True),
                ("u_decode", "bagel-strict-srt-u-g-u:d2", True),
            ],
        )
        self.assertIsNotNone(
            backend.sessions[
                "bagel-strict-srt-u-g-u"
            ].native_srt_u_context_token_binding
        )
        self.assertEqual(
            backend.sessions["bagel-strict-srt-u-g-u"].append_image_count, 1
        )
        self.assertEqual(
            backend.sessions["bagel-strict-srt-u-g-u"].srt_last_u_decode_output_ids,
            (88,),
        )
        self.assertEqual(
            backend.sessions["bagel-strict-srt-u-g-u"].srt_last_u_decode_text,
            "decoded:1",
        )
        counters = runtime.get_debug_counters(handle)
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 1)
        self.assertEqual(counters["append_image_count"], 1)
        self.assertEqual(counters["decode_count"], 2)
        self.assertEqual(counters["state"], "u_decode")

    def test_native_srt_u_prefill_does_not_call_python_text_update(self):
        inferencer = FakeBAGELInferencer()
        backend = BAGELInterleaveContextBackend(
            inferencer,
            u_forward_bridge=BAGELUForwardBridge(
                srt_u_forward_executor=BAGELNativeSRTUForwardExecutor()
            ),
            default_image_shape=(32, 32),
        )
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=BindingSRTExecutor(start_index=9),
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw a quiet river")],
            session_id="bagel-native-srt-u-prefill",
        )
        state = backend.sessions["bagel-native-srt-u-prefill"]

        self.assertTrue(state.native_srt_u_context)
        self.assertEqual(
            state.native_srt_u_context_request_id,
            "bagel-native-srt-u-prefill:u1",
        )
        self.assertIsNotNone(state.native_srt_u_context_token_binding)
        self.assertEqual(
            state.native_srt_u_context_token_binding.request_id,
            "bagel-native-srt-u-prefill:u1",
        )
        self.assertEqual(
            [event for event in inferencer.events if event[0] == "text"],
            [],
        )
        self.assertEqual(
            state.srt_u_forward_events,
            [("u_prefill", "bagel-native-srt-u-prefill:u1")],
        )

        marker = runtime.decode_next_segment(handle)
        self.assertEqual(marker.type, "image_marker")
        with self.assertRaisesRegex(BAGELAdapterError, "native SRT U context"):
            runtime.prepare_latents(
                UGLatentPrepareRequest(
                    session=handle,
                    sampling_params=SimpleNamespace(height=32, width=32),
                    seed=123,
                )
            )

        counters = runtime.get_debug_counters(handle)
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["state"], "g_denoise")

    def test_native_srt_u_context_uses_native_denoise_executor(self):
        inferencer = FakeBAGELInferencer()
        native_model = FakeNativeSRTVelocityModel()
        backend = BAGELInterleaveContextBackend(
            inferencer,
            u_forward_bridge=BAGELUForwardBridge(
                srt_u_forward_executor=BAGELNativeSRTUForwardExecutor()
            ),
            native_srt_denoise_executor=BAGELNativeSRTDenoiseExecutor(
                native_model,
                forward_batch_provider=lambda **kwargs: SimpleNamespace(
                    source="native-srt", kwargs=kwargs
                ),
            ),
            default_image_shape=(32, 32),
        )
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=BindingSRTExecutor(start_index=9),
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw a quiet river")],
            session_id="bagel-native-srt-g-denoise",
        )
        runtime.decode_next_segment(handle)
        sampling_params = SimpleNamespace(
            height=32,
            width=32,
            cfg_text_scale=1.0,
            cfg_img_scale=1.0,
            cfg_interval=(0.0, 1.0),
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
        )
        prepared_latents = runtime.prepare_latents(
            UGLatentPrepareRequest(
                session=handle,
                sampling_params=sampling_params,
                seed=123,
            )
        )
        response = runtime.predict_velocity(
            UGVelocityRequest(
                session=handle,
                latent_tokens=prepared_latents.latent_tokens,
                timestep=torch.tensor([0.5]),
                latent_position_ids=prepared_latents.latent_position_ids,
                sampling_params=sampling_params,
            )
        )

        self.assertTrue(
            torch.equal(response.velocity, prepared_latents.latent_tokens + 9.0)
        )
        self.assertEqual(inferencer.model.forward_flow_calls, [])
        self.assertEqual(len(native_model.calls), 1)
        native_call = native_model.calls[0]
        self.assertTrue(
            torch.equal(
                native_call["packed_vae_token_indexes"],
                torch.tensor([1, 2]),
            )
        )
        self.assertTrue(
            torch.equal(native_call["packed_text_indexes"], torch.tensor([0, 3]))
        )
        self.assertTrue(
            torch.equal(
                native_call["packed_seqlens"],
                torch.tensor([4], dtype=torch.int32),
            )
        )
        self.assertEqual(
            native_model.prepare_vae_latent_calls[0]["curr_kvlens"],
            [6],
        )
        self.assertEqual(
            native_model.prepare_vae_latent_calls[0]["curr_rope"],
            [6],
        )
        counters = runtime.get_debug_counters(response.session)
        self.assertEqual(counters["prefill_count"], 1)
        self.assertEqual(counters["velocity_count"], 1)

    def test_native_srt_append_image_prepares_embeds_without_python_append(self):
        inferencer = FakeBAGELInferencer()
        backend = BAGELInterleaveContextBackend(
            inferencer,
            u_forward_bridge=BAGELUForwardBridge(
                srt_u_forward_executor=BAGELNativeSRTUForwardExecutor()
            ),
            native_srt_denoise_executor=BAGELNativeSRTDenoiseExecutor(
                FakeNativeSRTVelocityModel(),
                forward_batch_provider=lambda **kwargs: SimpleNamespace(
                    source="native-srt",
                    kwargs=kwargs,
                ),
            ),
            default_image_shape=(32, 32),
        )
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=BindingSRTExecutor(start_index=9),
            srt_u_decode_max_new_tokens=1,
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw")],
            session_id="bagel-native-image-u",
        )
        runtime.decode_next_segment(handle)
        handle = runtime.append_generated_image(handle, image=FakeImage(size=(8, 8)))
        text = runtime.decode_next_segment(handle)

        self.assertEqual(text.type, "text")
        self.assertEqual(text.text, "1")
        self.assertEqual(
            [event for event in inferencer.events if event[0] == "image"],
            [],
        )
        self.assertEqual(len(inferencer.vae_transform.resize_calls), 1)
        state = backend.sessions["bagel-native-image-u"]
        self.assertEqual(state.append_image_count, 1)
        self.assertTrue(state.native_srt_u_context)
        self.assertEqual(state.gen_context["kv_lens"], [15])
        self.assertEqual(state.gen_context["ropes"], [7])
        counters = runtime.get_debug_counters(handle)
        self.assertEqual(counters["append_image_count"], 1)
        self.assertEqual(counters["srt_request_count"], 5)
        self.assertEqual(counters["ug_model_state"]["bagel"]["logical_kv_lens"], [15])
        self.assertEqual(counters["ug_model_state"]["bagel"]["logical_ropes"], [7])
        self.assertIn(":s2", state.native_srt_u_context_request_id)

    def test_scheduler_executor_builds_releasable_temp_g_forward_batch(self):
        native_model = FakeNativeSRTVelocityModel()
        model_runner = FakeModelRunner(native_model)
        scheduler_executor = UGSRTSchedulerExecutor(
            FakeSchedulerWithModelRunner(model_runner)
        )
        denoise_executor = scheduler_executor.create_bagel_native_srt_denoise_executor()
        binding = UGSRTKVTokenBinding(
            session_id="bagel-temp-g",
            request_id="bagel-temp-g:u1",
            token_count=3,
            token_indices=torch.tensor([101, 102, 103], dtype=torch.long),
        )
        prepared = BAGELNativeSRTPreparedDenoise(
            generation_input={
                "packed_text_ids": torch.tensor([1, 2]),
                "packed_text_indexes": torch.tensor([0, 3]),
                "packed_vae_token_indexes": torch.tensor([1, 2]),
                "packed_vae_position_ids": torch.tensor([0, 1]),
                "packed_seqlens": torch.tensor([4], dtype=torch.int32),
                "packed_position_ids": torch.tensor([3, 4, 5, 6]),
                "packed_indexes": torch.tensor([0, 1, 2, 3]),
                "key_values_lens": torch.tensor([3], dtype=torch.int32),
                "packed_key_value_indexes": torch.tensor([0, 1, 2]),
            },
            srt_kv_token_binding=binding,
        )
        latents = torch.zeros(2, 4)

        velocity = denoise_executor.predict_velocity(
            prepared=prepared,
            latent_tokens=latents,
            timestep=torch.tensor([0.5]),
        )

        self.assertTrue(torch.equal(velocity, latents + 9.0))
        self.assertEqual(len(native_model.calls), 1)
        forward_batch = native_model.calls[0]["forward_batch"]
        self.assertIs(forward_batch, model_runner.attn_backend.forward_batches[0])
        self.assertEqual(forward_batch.extend_prefix_lens_cpu, [3])
        self.assertEqual(forward_batch.extend_seq_lens_cpu, [4])
        self.assertTrue(torch.equal(forward_batch.seq_lens.cpu(), torch.tensor([7])))
        self.assertTrue(
            torch.equal(
                forward_batch.out_cache_loc.cpu(), torch.tensor([20, 21, 22, 23])
            )
        )
        self.assertEqual(
            forward_batch.ug_g_forward_metadata,
            {
                "session_id": "bagel-temp-g",
                "request_id": "bagel-temp-g:u1",
                "prefix_len": 3,
                "extend_num_tokens": 4,
                "attention_mode": "non_causal_query",
                "attention_mask_shape": (4, 7),
            },
        )
        self.assertTrue(forward_batch.ug_g_non_causal_query_attention)
        self.assertTrue(
            torch.equal(
                forward_batch.cross_attention_custom_mask,
                torch.ones(28, dtype=torch.bool),
            )
        )
        self.assertEqual(
            model_runner.req_to_token_pool.req_to_token[0, :7].tolist(),
            [101, 102, 103, 20, 21, 22, 23],
        )
        self.assertEqual(model_runner.req_to_token_pool.freed_reqs, [0])
        self.assertEqual(model_runner.req_to_token_pool.free_slots, [1, 0])
        self.assertEqual(model_runner.token_to_kv_pool_allocator.alloc_calls, [4])
        self.assertEqual(
            [x.tolist() for x in model_runner.token_to_kv_pool_allocator.freed],
            [[20, 21, 22, 23]],
        )
        self.assertEqual(scheduler_executor.temp_g_forward_count, 1)
        self.assertEqual(scheduler_executor.temp_g_allocated_token_count, 4)

    def test_native_denoise_executor_disables_grad_tracking(self):
        native_model = FakeNativeSRTVelocityModel()
        denoise_executor = BAGELNativeSRTDenoiseExecutor(native_model)
        prepared = BAGELNativeSRTPreparedDenoise(
            generation_input=_fake_bagel_generation_input(),
        )

        with torch.enable_grad():
            velocity = denoise_executor.predict_velocity(
                prepared=prepared,
                latent_tokens=torch.zeros(2, 4, requires_grad=True),
                timestep=torch.tensor([0.5]),
            )

        self.assertEqual(native_model.grad_enabled, [False])
        self.assertFalse(velocity.requires_grad)

    def test_native_denoise_executor_applies_cfg_branches(self):
        native_model = FakeBranchNativeSRTVelocityModel()
        provider_calls = []

        def forward_batch_provider(**kwargs):
            provider_calls.append(kwargs)
            return SimpleNamespace(
                forward_batch=SimpleNamespace(index=len(provider_calls))
            )

        denoise_executor = BAGELNativeSRTDenoiseExecutor(
            native_model,
            forward_batch_provider=forward_batch_provider,
        )
        base_input = _fake_bagel_generation_input()
        base_input["packed_position_ids"] = torch.full((4,), 10, dtype=torch.long)
        cfg_text_input = _fake_bagel_cfg_generation_input()
        cfg_text_input["cfg_packed_position_ids"] = torch.full(
            (4,),
            1,
            dtype=torch.long,
        )
        cfg_img_input = _fake_bagel_cfg_generation_input()
        cfg_img_input["cfg_packed_position_ids"] = torch.full(
            (4,),
            2,
            dtype=torch.long,
        )
        full_binding = UGSRTKVTokenBinding(
            session_id="bagel-cfg",
            request_id="bagel-cfg:u1",
            token_count=5,
            token_indices=torch.tensor([101, 102, 103, 104, 105]),
        )
        text_binding = UGSRTKVTokenBinding(
            session_id="bagel-cfg",
            request_id="bagel-cfg:u1:cfg_text",
            token_count=0,
            token_indices=torch.tensor([], dtype=torch.long),
        )
        prepared = BAGELNativeSRTPreparedDenoise(
            generation_input=base_input,
            srt_kv_token_binding=full_binding,
            cfg_text_generation_input=cfg_text_input,
            cfg_text_srt_kv_token_binding=text_binding,
            cfg_img_generation_input=cfg_img_input,
            cfg_img_srt_kv_token_binding=full_binding,
            cfg_text_scale=2.0,
            cfg_img_scale=1.5,
            cfg_interval=(0.0, 1.0),
            cfg_renorm_min=1.0,
            cfg_renorm_type="global",
        )

        velocity = denoise_executor.predict_velocity(
            prepared=prepared,
            latent_tokens=torch.zeros(2, 4),
            timestep=torch.tensor([0.5]),
        )

        self.assertTrue(torch.allclose(velocity, torch.full((2, 4), 27.5)))
        self.assertEqual(denoise_executor.velocity_count, 1)
        self.assertEqual(len(native_model.calls), 3)
        self.assertEqual(
            [int(call["packed_position_ids"][0].item()) for call in native_model.calls],
            [10, 1, 2],
        )
        self.assertEqual(
            [
                call["prepared"].srt_kv_token_binding.token_count
                for call in provider_calls
            ],
            [5, 0, 5],
        )

    def test_native_denoise_executor_respects_cfg_interval(self):
        native_model = FakeBranchNativeSRTVelocityModel()
        provider_calls = []
        denoise_executor = BAGELNativeSRTDenoiseExecutor(
            native_model,
            forward_batch_provider=lambda **kwargs: provider_calls.append(kwargs)
            or SimpleNamespace(forward_batch=SimpleNamespace()),
        )
        base_input = _fake_bagel_generation_input()
        base_input["packed_position_ids"] = torch.full((4,), 10, dtype=torch.long)
        prepared = BAGELNativeSRTPreparedDenoise(
            generation_input=base_input,
            srt_kv_token_binding=UGSRTKVTokenBinding(
                session_id="bagel-cfg-interval",
                request_id="bagel-cfg-interval:u1",
                token_count=5,
                token_indices=torch.tensor([101, 102, 103, 104, 105]),
            ),
            cfg_text_generation_input=_fake_bagel_cfg_generation_input(),
            cfg_text_srt_kv_token_binding=UGSRTKVTokenBinding(
                session_id="bagel-cfg-interval",
                request_id="bagel-cfg-interval:u1:cfg_text",
                token_count=0,
                token_indices=torch.tensor([], dtype=torch.long),
            ),
            cfg_img_generation_input=_fake_bagel_cfg_generation_input(),
            cfg_img_srt_kv_token_binding=UGSRTKVTokenBinding(
                session_id="bagel-cfg-interval",
                request_id="bagel-cfg-interval:u1:cfg_img",
                token_count=5,
                token_indices=torch.tensor([101, 102, 103, 104, 105]),
            ),
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_interval=(0.5, 1.0),
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
        )

        velocity = denoise_executor.predict_velocity(
            prepared=prepared,
            latent_tokens=torch.zeros(2, 4),
            timestep=torch.tensor([0.4]),
        )

        self.assertTrue(torch.equal(velocity, torch.full((2, 4), 10.0)))
        self.assertEqual(len(native_model.calls), 1)
        self.assertEqual(len(provider_calls), 1)

    def test_native_denoise_executor_matches_official_img_cfg_gating(self):
        native_model = FakeBranchNativeSRTVelocityModel()
        denoise_executor = BAGELNativeSRTDenoiseExecutor(native_model)
        base_input = _fake_bagel_generation_input()
        base_input["packed_position_ids"] = torch.full((4,), 10, dtype=torch.long)
        cfg_img_input = _fake_bagel_cfg_generation_input()
        cfg_img_input["cfg_packed_position_ids"] = torch.full(
            (4,),
            2,
            dtype=torch.long,
        )
        prepared = BAGELNativeSRTPreparedDenoise(
            generation_input=base_input,
            cfg_img_generation_input=cfg_img_input,
            cfg_text_scale=1.0,
            cfg_img_scale=2.0,
            cfg_interval=(0.0, 1.0),
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
        )

        velocity = denoise_executor.predict_velocity(
            prepared=prepared,
            latent_tokens=torch.zeros(2, 4),
            timestep=torch.tensor([0.5]),
        )

        self.assertTrue(torch.equal(velocity, torch.full((2, 4), 10.0)))
        self.assertEqual(len(native_model.calls), 1)

    def test_native_denoise_executor_applies_text_channel_cfg_renorm(self):
        native_model = FakeBranchNativeSRTVelocityModel()
        denoise_executor = BAGELNativeSRTDenoiseExecutor(native_model)
        base_input = _fake_bagel_generation_input()
        base_input["packed_position_ids"] = torch.full((4,), 10, dtype=torch.long)
        cfg_text_input = _fake_bagel_cfg_generation_input()
        cfg_text_input["cfg_packed_position_ids"] = torch.full(
            (4,),
            1,
            dtype=torch.long,
        )
        cfg_img_input = _fake_bagel_cfg_generation_input()
        cfg_img_input["cfg_packed_position_ids"] = torch.full(
            (4,),
            2,
            dtype=torch.long,
        )
        prepared = BAGELNativeSRTPreparedDenoise(
            generation_input=base_input,
            cfg_text_generation_input=cfg_text_input,
            cfg_img_generation_input=cfg_img_input,
            cfg_text_scale=2.0,
            cfg_img_scale=1.5,
            cfg_interval=(0.0, 1.0),
            cfg_renorm_min=0.0,
            cfg_renorm_type="text_channel",
        )

        velocity = denoise_executor.predict_velocity(
            prepared=prepared,
            latent_tokens=torch.zeros(2, 4),
            timestep=torch.tensor([0.5]),
        )

        self.assertTrue(torch.allclose(velocity, torch.full((2, 4), 14.0)))
        self.assertEqual(len(native_model.calls), 3)

    def test_scheduler_executor_allows_zero_prefix_temp_g_forward_batch(self):
        native_model = FakeNativeSRTVelocityModel()
        model_runner = FakeModelRunner(native_model)
        scheduler_executor = UGSRTSchedulerExecutor(
            FakeSchedulerWithModelRunner(model_runner)
        )
        denoise_executor = scheduler_executor.create_bagel_native_srt_denoise_executor()
        binding = UGSRTKVTokenBinding(
            session_id="bagel-temp-g-zero-prefix",
            request_id="bagel-temp-g-zero-prefix:u1:cfg_text",
            token_count=0,
            token_indices=torch.tensor([], dtype=torch.long),
        )
        prepared = BAGELNativeSRTPreparedDenoise(
            generation_input={
                "packed_text_ids": torch.tensor([1]),
                "packed_text_indexes": torch.tensor([0]),
                "packed_vae_token_indexes": torch.tensor([1, 2]),
                "packed_vae_position_ids": torch.tensor([0, 1]),
                "packed_seqlens": torch.tensor([3], dtype=torch.int32),
                "packed_position_ids": torch.tensor([0, 0, 0]),
                "packed_indexes": torch.tensor([0, 1, 2]),
                "key_values_lens": torch.tensor([0], dtype=torch.int32),
                "packed_key_value_indexes": torch.tensor([], dtype=torch.long),
            },
            srt_kv_token_binding=binding,
        )

        denoise_executor.predict_velocity(
            prepared=prepared,
            latent_tokens=torch.zeros(2, 4),
            timestep=torch.tensor([0.5]),
        )

        forward_batch = native_model.calls[0]["forward_batch"]
        self.assertEqual(forward_batch.extend_prefix_lens_cpu, [0])
        self.assertEqual(forward_batch.extend_seq_lens_cpu, [3])
        self.assertTrue(torch.equal(forward_batch.seq_lens.cpu(), torch.tensor([3])))
        self.assertEqual(
            model_runner.req_to_token_pool.req_to_token[0, :3].tolist(),
            [20, 21, 22],
        )
        self.assertEqual(model_runner.token_to_kv_pool_allocator.alloc_calls, [3])
        self.assertEqual(
            [x.tolist() for x in model_runner.token_to_kv_pool_allocator.freed],
            [[20, 21, 22]],
        )

    def test_scheduler_executor_temp_g_uses_owned_page_aligned_scratch(self):
        native_model = FakeNativeSRTVelocityModel()
        allocator = FakeTokenToKVPoolAllocator(start=20, page_size=4)
        model_runner = FakeModelRunner(
            native_model,
            token_to_kv_pool_allocator=allocator,
        )
        scheduler_executor = UGSRTSchedulerExecutor(
            FakeSchedulerWithModelRunner(model_runner)
        )
        denoise_executor = scheduler_executor.create_bagel_native_srt_denoise_executor()
        binding = UGSRTKVTokenBinding(
            session_id="bagel-temp-g-paged",
            request_id="bagel-temp-g-paged:u1",
            token_count=3,
            token_indices=torch.tensor([101, 102, 103], dtype=torch.long),
        )
        prepared = BAGELNativeSRTPreparedDenoise(
            generation_input={
                "packed_text_ids": torch.tensor([1]),
                "packed_text_indexes": torch.tensor([0]),
                "packed_vae_token_indexes": torch.tensor([1, 2]),
                "packed_vae_position_ids": torch.tensor([0, 1]),
                "packed_seqlens": torch.tensor([3], dtype=torch.int32),
                "packed_position_ids": torch.tensor([1, 1, 1]),
                "packed_indexes": torch.tensor([0, 1, 2]),
                "key_values_lens": torch.tensor([3], dtype=torch.int32),
                "packed_key_value_indexes": torch.tensor([0, 1, 2]),
            },
            srt_kv_token_binding=binding,
        )

        denoise_executor.predict_velocity(
            prepared=prepared,
            latent_tokens=torch.zeros(2, 4),
            timestep=torch.tensor([0.5]),
        )

        forward_batch = native_model.calls[0]["forward_batch"]
        self.assertTrue(
            torch.equal(forward_batch.out_cache_loc.cpu(), torch.tensor([20, 21, 22]))
        )
        self.assertEqual(
            model_runner.req_to_token_pool.req_to_token[0, :6].tolist(),
            [101, 102, 103, 20, 21, 22],
        )
        self.assertEqual(allocator.alloc_calls, [4])
        self.assertEqual(allocator.alloc_extend_calls, [])
        self.assertEqual([x.tolist() for x in allocator.freed], [[20, 21, 22, 23]])


class TestBAGELInterleaveContextBackend(unittest.TestCase):
    def test_native_srt_u_prefill_refuses_fallback_without_srt_result(self):
        backend = BAGELInterleaveContextBackend(
            FakeBAGELInferencer(),
            u_forward_bridge=BAGELUForwardBridge(
                srt_u_forward_executor=BAGELNativeSRTUForwardExecutor()
            ),
        )
        session = SimpleNamespace(
            handle=SimpleNamespace(session_id="native-missing-prefill"),
            srt_last_request_id="missing-u-prefill-result",
        )

        with self.assertRaisesRegex(BAGELAdapterError, "requires SRT-executed"):
            backend.prefill_interleaved(
                session=session,
                messages=[UGInterleavedMessage(type="text", content="draw")],
            )

    def test_native_srt_u_append_refuses_fallback_without_srt_result(self):
        backend = BAGELInterleaveContextBackend(
            FakeBAGELInferencer(),
            u_forward_bridge=BAGELUForwardBridge(
                srt_u_forward_executor=BAGELNativeSRTUForwardExecutor()
            ),
        )
        session = SimpleNamespace(
            handle=SimpleNamespace(session_id="native-missing-append"),
            srt_last_request_id="missing-append-result",
        )

        with self.assertRaisesRegex(BAGELAdapterError, "requires SRT-executed"):
            backend.append_generated_image(
                session=session, image=FakeImage(size=(8, 8))
            )

    def test_context_backend_consumes_prefill_and_append_from_srt_forward_view(self):
        inferencer = FakeBAGELInferencer()
        backend = BAGELInterleaveContextBackend(
            inferencer,
            native_srt_denoise_executor=BAGELNativeSRTDenoiseExecutor(
                FakeNativeSRTVelocityModel(),
                forward_batch_provider=lambda **kwargs: SimpleNamespace(
                    source="native-srt",
                    kwargs=kwargs,
                ),
            ),
            default_image_shape=(32, 32),
        )
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw a quiet river")],
            session_id="bagel-srt-u-forward",
        )
        state = backend.sessions["bagel-srt-u-forward"]

        self.assertEqual(
            state.srt_u_forward_events,
            [("u_prefill", "bagel-srt-u-forward:u1")],
        )
        self.assertEqual(
            len([event for event in inferencer.events if event[0] == "text"]), 0
        )
        self.assertEqual(state.srt_u_forward_results, {})

        runtime.decode_next_segment(handle)
        image = FakeImage(size=(24, 12))
        handle = runtime.append_generated_image(handle, image=image)

        self.assertEqual(
            state.srt_u_forward_events,
            [
                ("u_prefill", "bagel-srt-u-forward:u1"),
                ("append_image", "bagel-srt-u-forward:u2:s1"),
                ("append_image", "bagel-srt-u-forward:u2:s2"),
            ],
        )
        self.assertEqual(
            len([event for event in inferencer.events if event[0] == "image"]),
            0,
        )
        self.assertEqual(inferencer.vae_transform.resize_calls, [image])
        self.assertEqual(state.srt_u_forward_results, {})
        self.assertEqual(runtime.get_debug_counters(handle)["append_image_count"], 1)

    def test_context_backend_observes_srt_u_decode_output_ids(self):
        inferencer = FakeBAGELInferencer()
        backend = BAGELInterleaveContextBackend(inferencer)
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
            srt_request_executor=OutputAppendingSRTExecutor(token_id=77),
            srt_u_decode_max_new_tokens=1,
        )

        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw a tree")],
            session_id="bagel-srt-u-decode",
        )
        runtime.decode_next_segment(handle)
        state = backend.sessions["bagel-srt-u-decode"]

        self.assertEqual(
            state.srt_u_forward_events,
            [
                ("u_prefill", "bagel-srt-u-decode:u1"),
                ("u_decode", "bagel-srt-u-decode:d1"),
            ],
        )
        self.assertEqual(state.srt_last_u_decode_output_ids, (77,))

    def test_context_backend_release_closes_backend_session(self):
        inferencer = FakeBAGELInferencer()
        backend = BAGELInterleaveContextBackend(inferencer)
        adapter = BAGELUGModelAdapter("already-loaded-bagel", backend=backend)
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=SessionController(FakeTreeCache()),
        )
        handle = runtime.prefill_interleaved(
            [UGInterleavedMessage(type="text", content="draw a cat")],
            session_id="bagel-close-session",
        )
        self.assertIn("bagel-close-session", backend.sessions)

        runtime.close_session(handle)

        self.assertNotIn("bagel-close-session", backend.sessions)


if __name__ == "__main__":
    unittest.main()
