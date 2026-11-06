import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.multimodal_gen.runtime.models.dits.ltx_2 as ltx_2
from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2ArchConfig, LTX2Config
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.test_utils import CustomTestCase


def _patch_attention_layers(stack: ExitStack) -> None:
    for name, value in (
        ("get_attn_backend", SDPABackend),
        ("get_compute_dtype", torch.bfloat16),
        ("_resolve_sp_attention_mode", ("ulysses", False)),
        ("_maybe_install_backend_autotune", None),
    ):
        stack.enter_context(patch.object(layer, name, return_value=value))
    stack.enter_context(set_forward_context(0, None))


def _tiny_model(**arch_overrides) -> ltx_2.LTX2VideoTransformer3DModel:
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    torch.manual_seed(0)
    arch = LTX2ArchConfig(
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=32,
        in_channels=16,
        out_channels=16,
        cross_attention_dim=64,
        caption_channels=64,
        audio_num_attention_heads=2,
        audio_attention_head_dim=16,
        audio_in_channels=16,
        audio_out_channels=16,
        audio_cross_attention_dim=32,
        **arch_overrides,
    )
    model = ltx_2.LTX2VideoTransformer3DModel(
        LTX2Config(arch_config=arch), hf_config={}
    ).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.05)
    return model


def _forward_kwargs(text_seed: int) -> dict:
    generator = torch.Generator().manual_seed(text_seed)
    mask = torch.ones(1, 6, dtype=torch.int64)
    mask[:, 4:] = 0
    return dict(
        hidden_states=torch.randn(1, 32, 16, dtype=torch.bfloat16),
        audio_hidden_states=torch.randn(1, 5, 16, dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(
            1, 6, 64, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16),
        audio_encoder_hidden_states=torch.randn(
            1, 6, 64, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16),
        timestep=torch.tensor([500.0]),
        encoder_attention_mask=mask,
        audio_encoder_attention_mask=mask,
        num_frames=2,
        height=4,
        width=4,
        audio_num_frames=5,
        return_latents=False,
    )


def _cached(model, kwargs: dict) -> dict:
    key = model.prepare_prompt_kv(
        kwargs["encoder_hidden_states"], kwargs["audio_encoder_hidden_states"]
    )
    return {
        **kwargs,
        "encoder_hidden_states": None,
        "audio_encoder_hidden_states": None,
        "cached_prompt_kv": key,
    }


class TestLTX2PromptKVCache(CustomTestCase):
    @torch.no_grad()
    def test_cached_prompt_kv_is_bitwise_equal_to_per_step_projection(self):
        with ExitStack() as stack:
            _patch_attention_layers(stack)
            model = _tiny_model()
            kwargs = _forward_kwargs(text_seed=1)
            expected = model(**kwargs)
            actual = model(**_cached(model, kwargs))
            for a, e in zip(actual, expected, strict=True):
                self.assertTrue(torch.equal(a, e))

    @torch.no_grad()
    def test_new_text_is_written_into_the_same_buffers(self):
        with ExitStack() as stack:
            _patch_attention_layers(stack)
            model = _tiny_model()
            first = _cached(model, _forward_kwargs(text_seed=1))
            buffers = model._prompt_kv_buffers[first["cached_prompt_kv"]]
            pointers = [t.data_ptr() for kv in buffers for t in (*kv.video, *kv.audio)]
            second_kwargs = _forward_kwargs(text_seed=2)
            second = _cached(model, second_kwargs)
            self.assertEqual(second["cached_prompt_kv"], first["cached_prompt_kv"])
            self.assertEqual(
                [t.data_ptr() for kv in buffers for t in (*kv.video, *kv.audio)],
                pointers,
            )
            for a, e in zip(model(**second), model(**second_kwargs), strict=True):
                self.assertTrue(torch.equal(a, e))

    def test_prompt_modulated_cross_attention_does_not_cache(self):
        self.assertTrue(_tiny_model().caches_prompt_kv)
        self.assertFalse(_tiny_model(cross_attention_adaln=True).caches_prompt_kv)


class TestLTX2StagePromptKV(CustomTestCase):
    def _model(self, *, caches: bool = True, offloaded: bool = False):
        return SimpleNamespace(
            caches_prompt_kv=caches,
            layerwise_offload_managers=[object()] if offloaded else [],
            prepare_prompt_kv=Mock(return_value=(1, 6, 64, 1, 6, 32)),
        )

    def _use(self, model, ctx, *, cache_dit: bool = False) -> dict:
        stage = SimpleNamespace(_cache_dit_enabled=cache_dit)
        kwargs = {
            "encoder_hidden_states": torch.zeros(1),
            "audio_encoder_hidden_states": torch.zeros(1),
        }
        return LTX2DenoisingStage._use_cached_ltx2_prompt_kv(
            stage, ctx, SimpleNamespace(current_model=model), kwargs
        )

    def test_projects_once_per_loop(self):
        model = self._model()
        ctx = SimpleNamespace(prompt_kv_key=None)
        for _ in range(3):
            kwargs = self._use(model, ctx)
        model.prepare_prompt_kv.assert_called_once()
        self.assertEqual(kwargs["cached_prompt_kv"], (1, 6, 64, 1, 6, 32))
        self.assertIsNone(kwargs["encoder_hidden_states"])
        self.assertIsNone(kwargs["audio_encoder_hidden_states"])

    def test_keeps_projecting_when_the_weights_or_text_may_move(self):
        for model, cache_dit in (
            (self._model(caches=False), False),
            (self._model(offloaded=True), False),
            (self._model(), True),
        ):
            kwargs = self._use(
                model, SimpleNamespace(prompt_kv_key=None), cache_dit=cache_dit
            )
            self.assertNotIn("cached_prompt_kv", kwargs)
            model.prepare_prompt_kv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
