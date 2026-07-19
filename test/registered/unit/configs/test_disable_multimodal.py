import argparse
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.models import qwen3_omni_moe, qwen3_vl, qwen3_vl_moe
from sglang.srt.models.interns1pro import InternS1ProForConditionalGeneration
from sglang.srt.models.interns2preview import InternS2PreviewForConditionalGeneration
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from sglang.srt.server_args import ServerArgs
from sglang.srt.server_args_config_parser import ConfigArgumentMerger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeLanguageModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


class _FakeOmniThinker(nn.Module):
    pad_input_ids = None


def _qwen35_config(*, enable_multimodal: bool):
    return SimpleNamespace(
        enable_multimodal=enable_multimodal,
        encoder_only=False,
        language_only=False,
        tie_word_embeddings=False,
        text_config=SimpleNamespace(
            rms_norm_eps=1e-6,
            rope_parameters={"mrope_section": [16, 24, 24]},
            rope_scaling={},
            tie_word_embeddings=False,
            pad_token_id=None,
        ),
        audio_config=SimpleNamespace(),
        vision_config=SimpleNamespace(deepstack_visual_indexes=[8, 16, 24]),
    )


class TestMultimodalConfiguration(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def test_enable_multimodal_cli_is_tristate(self):
        for cli_args, expected in (
            ([], None),
            (["--enable-multimodal"], True),
            (["--disable-multimodal"], False),
        ):
            with self.subTest(cli_args=cli_args):
                parsed = self.parser.parse_args(["--model", "dummy", *cli_args])
                server_args = ServerArgs.from_cli_args(parsed)

                self.assertIs(server_args.enable_multimodal, expected)

        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                [
                    "--model",
                    "dummy",
                    "--enable-multimodal",
                    "--disable-multimodal",
                ]
            )

    def test_multimodal_yaml_config_is_tristate_and_cli_wins(self):
        for config_value, cli_args, expected in (
            (True, [], True),
            (False, [], False),
            (False, ["--enable-multimodal"], True),
            (True, ["--disable-multimodal"], False),
            (False, ["--enable-multim"], True),
            (True, ["--disable-multim"], False),
        ):
            with self.subTest(config_value=config_value, cli_args=cli_args):
                merger = ConfigArgumentMerger(self.parser)
                with patch.object(
                    merger,
                    "_parse_yaml_config",
                    return_value={"enable-multimodal": config_value},
                ):
                    merged = merger.merge_config_with_args(
                        ["--config", "config.yaml", "--model", "dummy", *cli_args]
                    )

                parsed = self.parser.parse_args(merged)
                self.assertIs(parsed.enable_multimodal, expected)

    def test_qwen35_skips_vision_when_multimodal_is_disabled(self):
        config = _qwen35_config(enable_multimodal=False)
        pp_group = SimpleNamespace(is_last_rank=False, world_size=1)
        server_args = SimpleNamespace(mm_enable_dp_encoder=False)

        with (
            patch.object(qwen3_vl, "get_pp_group", return_value=pp_group),
            patch.object(qwen3_vl, "get_server_args", return_value=server_args),
            patch.object(qwen3_vl, "Qwen3VLMoeVisionModel") as vision_model,
            patch.object(qwen3_vl, "LogitsProcessor", return_value=nn.Identity()),
            patch.object(qwen3_vl, "Pooler", return_value=nn.Identity()),
        ):
            model = Qwen3_5ForConditionalGeneration(
                config,
                language_model_cls=_FakeLanguageModel,
            )

        vision_model.assert_not_called()
        self.assertIsNone(model.visual)
        self.assertTrue(model.is_mrope_enabled)
        self.assertEqual(model.deepstack_visual_indexes, [8, 16, 24])

    def test_qwen35_constructs_vision_when_multimodal_is_enabled(self):
        config = _qwen35_config(enable_multimodal=True)
        pp_group = SimpleNamespace(is_last_rank=False, world_size=1)
        server_args = SimpleNamespace(mm_enable_dp_encoder=False)

        with (
            patch.object(qwen3_vl, "get_pp_group", return_value=pp_group),
            patch.object(qwen3_vl, "get_server_args", return_value=server_args),
            patch.object(qwen3_vl, "Qwen3VLMoeVisionModel") as vision_model,
            patch.object(qwen3_vl, "LogitsProcessor", return_value=nn.Identity()),
            patch.object(qwen3_vl, "Pooler", return_value=nn.Identity()),
        ):
            vision_model.return_value.deepstack_visual_indexes = [8, 16, 24]
            model = Qwen3_5ForConditionalGeneration(
                config,
                language_model_cls=_FakeLanguageModel,
            )

        vision_model.assert_called_once()
        self.assertIs(model.visual, vision_model.return_value)

    def test_intern_models_inherit_disabled_vision_construction(self):
        config = _qwen35_config(enable_multimodal=False)
        pp_group = SimpleNamespace(is_last_rank=False, world_size=1)
        server_args = SimpleNamespace(mm_enable_dp_encoder=False)

        with (
            patch.object(qwen3_vl, "get_pp_group", return_value=pp_group),
            patch.object(qwen3_vl, "get_server_args", return_value=server_args),
            patch.object(qwen3_vl, "Qwen3VLMoeVisionModel") as vision_model,
            patch.object(qwen3_vl, "LogitsProcessor", return_value=nn.Identity()),
            patch.object(qwen3_vl, "Pooler", return_value=nn.Identity()),
        ):
            for model_cls in (
                InternS1ProForConditionalGeneration,
                InternS2PreviewForConditionalGeneration,
            ):
                with self.subTest(model_cls=model_cls):
                    model = model_cls(
                        config,
                        language_model_cls=_FakeLanguageModel,
                    )
                    self.assertIsNone(model.visual)

        vision_model.assert_not_called()

    def test_qwen3_vl_moe_skips_disabled_visual_weights_without_warnings(self):
        model = Qwen3VLMoeForConditionalGeneration.__new__(
            Qwen3VLMoeForConditionalGeneration
        )
        nn.Module.__init__(model)
        model.config = SimpleNamespace(num_experts=1)
        model.enable_multimodal = False

        with patch.object(qwen3_vl_moe.logger, "warning") as warning:
            model.load_weights([("model.visual.patch_embed.proj.weight", None)])

        warning.assert_not_called()

    def test_qwen35_loaders_skip_disabled_visual_weights(self):
        for model_cls in (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
        ):
            with self.subTest(model_cls=model_cls):
                model = model_cls.__new__(model_cls)
                nn.Module.__init__(model)
                model.config = SimpleNamespace(num_experts=1)
                model.enable_multimodal = False
                model.enable_shared_expert_fusion = False

                loaded = model.load_weights(
                    [("model.visual.patch_embed.proj.weight", None)]
                )

                self.assertEqual(loaded, set())

    def test_qwen3_omni_propagates_runtime_config_to_thinker(self):
        for runtime_config, expected_enable_multimodal in (
            ({}, True),
            (
                {
                    "enable_multimodal": False,
                    "encoder_only": False,
                    "language_only": False,
                },
                False,
            ),
        ):
            with self.subTest(runtime_config=runtime_config):
                thinker_config = SimpleNamespace()
                config = SimpleNamespace(
                    thinker_config=thinker_config,
                    **runtime_config,
                )

                with (
                    patch.object(
                        qwen3_omni_moe.PreTrainedModel,
                        "__init__",
                        lambda self, config: nn.Module.__init__(self),
                    ),
                    patch.object(
                        qwen3_omni_moe,
                        "Qwen3OmniMoeThinkerForConditionalGeneration",
                        return_value=_FakeOmniThinker(),
                    ),
                ):
                    model = Qwen3OmniMoeForConditionalGeneration(config)

                self.assertIs(model.enable_multimodal, expected_enable_multimodal)
                self.assertIs(
                    thinker_config.enable_multimodal, expected_enable_multimodal
                )
                self.assertFalse(thinker_config.encoder_only)
                self.assertFalse(thinker_config.language_only)

    def test_qwen3_omni_enabled_towers_use_text_norm_eps(self):
        config = _qwen35_config(enable_multimodal=True)
        config.text_config.rms_norm_eps = 1e-5
        pp_group = SimpleNamespace(is_last_rank=False, world_size=1)
        server_args = SimpleNamespace(mm_enable_dp_encoder=False)

        with (
            patch.object(qwen3_vl, "get_pp_group", return_value=pp_group),
            patch.object(qwen3_vl, "get_server_args", return_value=server_args),
            patch.object(qwen3_omni_moe, "Qwen3MoeLLMModel", _FakeLanguageModel),
            patch.object(qwen3_vl, "Qwen3VLMoeVisionModel"),
            patch.object(qwen3_vl, "LogitsProcessor", return_value=nn.Identity()),
            patch.object(qwen3_vl, "Pooler", return_value=nn.Identity()),
            patch.object(qwen3_omni_moe, "Qwen3OmniMoeAudioEncoder") as audio_model,
            patch.object(qwen3_omni_moe, "Qwen3OmniMoeVisionEncoder") as vision_model,
        ):
            model = Qwen3OmniMoeThinkerForConditionalGeneration(config)

        audio_model.assert_called_once_with(config.audio_config, None)
        vision_model.assert_called_once_with(
            config.vision_config,
            quant_config=None,
            norm_eps=config.text_config.rms_norm_eps,
            prefix="visual",
        )
        self.assertIs(model.visual, vision_model.return_value)

    def test_qwen3_omni_skips_disabled_multimodal_towers_and_weights(self):
        config = _qwen35_config(enable_multimodal=False)
        pp_group = SimpleNamespace(is_last_rank=False, world_size=1)
        server_args = SimpleNamespace(mm_enable_dp_encoder=False)

        with (
            patch.object(qwen3_vl, "get_pp_group", return_value=pp_group),
            patch.object(qwen3_vl, "get_server_args", return_value=server_args),
            patch.object(qwen3_omni_moe, "Qwen3MoeLLMModel", _FakeLanguageModel),
            patch.object(qwen3_vl, "LogitsProcessor", return_value=nn.Identity()),
            patch.object(qwen3_vl, "Pooler", return_value=nn.Identity()),
            patch.object(qwen3_omni_moe, "Qwen3OmniMoeAudioEncoder") as audio_model,
            patch.object(qwen3_omni_moe, "Qwen3OmniMoeVisionEncoder") as vision_model,
        ):
            model = Qwen3OmniMoeThinkerForConditionalGeneration(config)

        audio_model.assert_not_called()
        vision_model.assert_not_called()
        self.assertIsNone(model.audio_tower)
        self.assertIsNone(model.visual)

        outer_model = Qwen3OmniMoeForConditionalGeneration.__new__(
            Qwen3OmniMoeForConditionalGeneration
        )
        nn.Module.__init__(outer_model)
        outer_model.config = SimpleNamespace(num_experts=1)
        outer_model.enable_talker = False
        outer_model.enable_multimodal = False
        weights = [
            ("thinker.visual.patch_embed.proj.weight", None),
            ("thinker.audio_tower.conv1.weight", None),
        ]

        with patch.object(qwen3_omni_moe.logger, "warning") as warning:
            outer_model.load_weights(weights)

        warning.assert_not_called()


class TestDisabledMultimodalRequests(CustomTestCase):
    def test_rejects_multimodal_input_at_request_boundary(self):
        for request in (
            EmbeddingReqInput(image_data="image.png"),
            EmbeddingReqInput(
                text=["first", "second"],
                image_data=["first.png", "second.png"],
            ),
        ):
            with self.subTest(is_batch=isinstance(request.text, list)):
                manager = TokenizerManager.__new__(TokenizerManager)
                manager.auto_create_handle_loop = lambda: None
                manager.model_config = SimpleNamespace(enable_multimodal=False)

                with self.assertRaisesRegex(
                    ValueError, "Multimodal inputs are disabled"
                ):
                    asyncio.run(manager.generate_request(request).__anext__())


if __name__ == "__main__":
    unittest.main()
