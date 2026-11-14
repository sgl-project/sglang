"""Each draft architecture's window from its config (`sliding_layer_window_size`),
used to place its storage before it loads, equals what its built model resolves."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import _validate_draft_window_honored
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    resolve_sliding_layer_window_size,
    resolve_sliding_window_size,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _built(cls, **attrs):
    """A built model of ``cls`` as far as its window rule reads it."""
    model = object.__new__(cls)
    for name, value in attrs.items():
        object.__setattr__(model, name, value)
    return model


def _model_config(hf_config, stated):
    return SimpleNamespace(
        hf_config=hf_config,
        is_hybrid_swa=True,
        sliding_window_size=stated,
        attention_chunk_size=None,
    )


class TestDraftSlidingWindowRules(CustomTestCase):
    def _check(self, cls, hf_config, stated, expected, **attrs):
        model_config = _model_config(hf_config, stated)
        model = _built(cls, config=hf_config, **attrs)
        self.assertEqual(resolve_sliding_window_size(model, model_config), expected)
        self.assertEqual(resolve_sliding_layer_window_size(cls, model_config), expected)

    def test_gemma4_assistant_attends_over_one_less(self):
        from sglang.srt.models.gemma4_mtp import (
            Gemma4AssistantForCausalLM,
            Gemma4UnifiedAssistantForCausalLM,
        )

        for cls in (Gemma4AssistantForCausalLM, Gemma4UnifiedAssistantForCausalLM):
            with self.subTest(cls=cls.__name__):
                self._check(cls, SimpleNamespace(sliding_window=512), 512, 511)
                # Before load the config still wraps the text config the
                # built model keeps.
                text = SimpleNamespace(sliding_window=512)
                wrapped = SimpleNamespace(text_config=text)
                self.assertEqual(
                    resolve_sliding_layer_window_size(cls, _model_config(wrapped, 512)),
                    resolve_sliding_window_size(
                        _built(cls, config=text), _model_config(text, 512)
                    ),
                )

    def test_dflash_attends_over_one_less(self):
        from sglang.srt.models.dflash import DFlash2DraftModel, DFlashDraftModel

        hf_config = SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=2048,
            text_config=None,
            get_text_config=None,
        )
        for cls in (DFlashDraftModel, DFlash2DraftModel):
            with self.subTest(cls=cls.__name__):
                self._check(cls, hf_config, 2048, 2047)

    def test_inkling_banded_depth_attends_over_one_less_than_its_extent(self):
        from sglang.srt.models.inkling import InklingForConditionalGenerationMTP

        text_config = SimpleNamespace(
            mtp_local_layer_ids=[1], mtp_local_extent=64, sliding_window_size=128
        )
        hf_config = SimpleNamespace(text_config=text_config)
        # The runner of the banded depth is the one whose layers slide.
        self._check(
            InklingForConditionalGenerationMTP,
            hf_config,
            64,
            63,
            text_config=text_config,
            draft_model_idx=1,
        )

    def test_dots3_nextn_follows_its_trunk(self):
        from sglang.srt.models.dots3_common.modeling import (
            get_attention_sliding_window_size,
        )
        from sglang.srt.models.dots3_common.nextn import Dots3NoteForCausalLMNextN

        hf_config = SimpleNamespace(sliding_window_size=4096)
        self._check(
            Dots3NoteForCausalLMNextN,
            hf_config,
            4096,
            get_attention_sliding_window_size(hf_config),
        )

    def test_eagle3_draft_window_is_the_server_argument(self):
        from sglang.srt.models import llama_eagle3
        from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3

        spec = SimpleNamespace(speculative_draft_window_size=4096)
        with patch.object(llama_eagle3, "get_spec", return_value=spec):
            self._check(
                LlamaForCausalLMEagle3,
                SimpleNamespace(),
                None,
                4096,
                _draft_window_size=4096,
            )

    def test_a_class_without_a_rule_attends_over_its_stated_window(self):
        class _Plain:
            pass

        self._check(_Plain, SimpleNamespace(), 300, 300)


class _PlainModel:
    pass


class _NativeWindowModel:
    @staticmethod
    def get_attention_sliding_window_size(self):
        return 32

    @staticmethod
    def sliding_layer_window_size(config):
        return 32


class TestDraftWindowValidation(CustomTestCase):
    def _validate(self, model_cls, **kwargs):
        options = dict(
            speculative_algorithm="EAGLE",
            speculative_draft_model_path="draft",
            speculative_draft_attention_backend="fa3",
            speculative_num_steps=1,
            speculative_draft_window_size=64,
        )
        options.update(kwargs)
        args = ServerArgs(model_path="dummy", **options)
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(),
            is_hybrid_swa=False,
            sliding_window_size=None,
            attention_chunk_size=None,
        )
        with (
            patch(
                "sglang.srt.configs.model_config.ModelConfig.from_server_args",
                return_value=model_config,
            ),
            patch(
                "sglang.srt.model_loader.utils.get_model_architecture",
                return_value=(model_cls, None),
            ),
        ):
            _validate_draft_window_honored(args)

    def test_validation_uses_backend_steps_and_native_window(self):
        from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3

        cases = (
            (
                "triton_window_and_sink",
                _PlainModel,
                {
                    "speculative_draft_attention_backend": "triton",
                    "speculative_num_steps": 2,
                    "speculative_draft_sink_size": 16,
                },
                None,
            ),
            (
                "unsupported_backend",
                _PlainModel,
                {"speculative_num_steps": 2},
                "draft-window-size",
            ),
            (
                "single_step",
                _PlainModel,
                {"speculative_draft_attention_backend": "triton"},
                "draft-window-size",
            ),
            ("native_window", _NativeWindowModel, {}, None),
            (
                "native_window_sink",
                _NativeWindowModel,
                {"speculative_draft_sink_size": 8},
                "draft-sink-size",
            ),
            ("eagle3_native_window", LlamaForCausalLMEagle3, {}, None),
            ("dflash_window", _PlainModel, {"speculative_algorithm": "DFLASH"}, None),
        )
        for name, model_cls, options, error in cases:
            with self.subTest(name=name):
                if error is None:
                    self._validate(model_cls, **options)
                else:
                    with self.assertRaisesRegex(ValueError, error):
                        self._validate(model_cls, **options)


if __name__ == "__main__":
    unittest.main()
