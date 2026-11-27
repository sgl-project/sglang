# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import PIL.Image
import pytest
import requests
import transformers

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines import (
    lingbot_video_moe as pipeline_lingbot_video_moe,
)
from sglang.multimodal_gen.runtime.pipelines.lingbot_video_moe import (
    LingBotVideoPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.auto_negative import (
    LingBotVideoAutoNegativeStage,
    prune_negative,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter import (
    LingBotVideoPromptRewriteStage,
    build_expand_prompt,
    build_map_prompt,
    needs_rewrite,
    parse_caption,
    resolve_mode,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter_backends import (
    HTTPRewriterBackend,
    TransformersRewriterBackend,
    build_rewriter_backend,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter_prompts import (
    VIDEO_DURATION_EN,
    VIDEO_DURATION_ZH,
)


def test_rewrite_mode_follows_the_request_shape():
    assert resolve_mode(SimpleNamespace(num_frames=1, image_path=None)) == "t2i"
    assert resolve_mode(SimpleNamespace(num_frames=121, image_path=None)) == "t2v"
    assert resolve_mode(SimpleNamespace(num_frames=121, image_path="f.png")) == "ti2v"


def test_structured_captions_skip_rewriting():
    assert needs_rewrite("a red fox in the snow")
    assert not needs_rewrite('{"comprehensive_description": "..."}')
    assert not needs_rewrite('  {"caption": {}}')


def test_expand_prompt_carries_the_duration_for_video_modes():
    video = build_expand_prompt("t2v", "a red fox", 5)
    assert "Video Duration: 5 seconds" in video
    assert video.endswith("Video Duration: 5 seconds")

    assert "Duration" not in build_expand_prompt("t2i", "a glass of tea", 5)


def test_expand_prompt_matches_the_caption_language():
    cjk_prompt = VIDEO_DURATION_ZH.format(duration=3)
    expanded = build_expand_prompt("t2v", cjk_prompt, 5)

    assert VIDEO_DURATION_ZH.format(duration=5) in expanded
    assert VIDEO_DURATION_EN.format(duration=5) not in expanded


def test_map_prompt_feeds_the_expansion_back():
    mapped = build_map_prompt("t2v", "A red fox trots through snow.", 5)

    assert "DETAILED CAPTION:\nA red fox trots through snow." in mapped
    assert mapped.endswith("Output the JSON now.")


def test_caption_parsing_tolerates_fences_and_prose():
    assert parse_caption('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_caption('Here you go: {"a": 1} hope that helps') == {"a": 1}
    assert parse_caption("no json here") is None
    assert parse_caption('["not", "an", "object"]') is None


class _FakeBackend:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def generate(self, text, image, use_lora):
        self.calls.append((use_lora, image))
        return self.replies.pop(0)


def _rewrite_stage(replies):
    stage = object.__new__(LingBotVideoPromptRewriteStage)
    stage.backend = _FakeBackend(replies)
    return stage


def test_rewriting_replaces_the_prompt_with_a_compact_caption():
    stage = _rewrite_stage(["A red fox trots through snow.", '{"b": 2, "a": 1}'])
    batch = SimpleNamespace(prompt="a red fox", num_frames=121, fps=24, image_path=None)

    stage.forward(batch, None)

    assert batch.prompt == '{"b":2,"a":1}'
    # Only the mapping turn enables the adapter.
    assert [use_lora for use_lora, _ in stage.backend.calls] == [False, True]
    assert all(image is None for _, image in stage.backend.calls)


def test_rewriting_refuses_to_fall_back_to_the_free_text_prompt():
    stage = _rewrite_stage(["expanded", "sorry, I cannot"])
    batch = SimpleNamespace(prompt="a red fox", num_frames=121, fps=24, image_path=None)

    with pytest.raises(ValueError, match="structured caption"):
        stage.forward(batch, None)


def test_rewriting_leaves_a_structured_caption_alone():
    stage = _rewrite_stage([])
    caption = '{"comprehensive_description": "..."}'
    batch = SimpleNamespace(prompt=caption, num_frames=121, fps=24, image_path=None)

    stage.forward(batch, None)

    assert batch.prompt == caption
    assert stage.backend.calls == []


def test_http_rewriter_routes_both_turns_with_the_reference_image(monkeypatch):
    calls = []
    replies = iter(["expanded motion", '{"caption":"motion"}'])

    def post(url, *, json, timeout):
        calls.append((url, json, timeout))
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": next(replies)}}]
        }
        return response

    monkeypatch.setattr(requests, "post", post)
    config = LingBotVideoMoEPipelineConfig(
        rewriter_url="http://base/",
        rewriter_map_url="http://adapter/",
        rewriter_expand_model="base",
        rewriter_map_model="adapter",
        rewriter_timeout=7,
    )
    stage = LingBotVideoPromptRewriteStage(
        build_rewriter_backend(config, trust_remote_code=False)
    )
    batch = SimpleNamespace(
        prompt="a runner",
        num_frames=17,
        fps=16,
        image_path="frame.png",
        condition_image=[PIL.Image.new("RGB", (3, 2))],
    )
    assert stage.forward(batch, None).prompt == '{"caption":"motion"}'
    assert [(url, body["model"], timeout) for url, body, timeout in calls] == [
        ("http://base/v1/chat/completions", "base", 7),
        ("http://adapter/v1/chat/completions", "adapter", 7),
    ]
    first, second = (body["messages"][0]["content"] for _, body, _ in calls)
    assert first[0] == second[0]
    assert first[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "expanded motion" in second[1]["text"]


def test_rewriter_backend_is_off_until_a_url_or_local_model_is_set(monkeypatch):
    config = LingBotVideoMoEPipelineConfig()
    assert build_rewriter_backend(config, trust_remote_code=False) is None

    config.rewriter_url = "http://host:30000"
    assert isinstance(
        build_rewriter_backend(config, trust_remote_code=False), HTTPRewriterBackend
    )

    local = LingBotVideoMoEPipelineConfig()
    local.rewriter_model_path = "/models/base"
    with pytest.raises(ValueError, match="rewriter_adapter_path"):
        build_rewriter_backend(local, trust_remote_code=False)

    local.rewriter_adapter_path = "/models/adapter"
    processor, model = Mock(), Mock()
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", processor)
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText, "from_pretrained", model
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=Mock()))
    for trust in (False, True):
        backend = build_rewriter_backend(local, trust_remote_code=trust)
        assert isinstance(backend, TransformersRewriterBackend)
        backend._load()
        assert processor.call_args.kwargs["trust_remote_code"] is trust
        assert model.call_args.kwargs["trust_remote_code"] is trust


def _added_stages(**config_overrides) -> list:
    config = LingBotVideoMoEPipelineConfig()
    for name, value in config_overrides.items():
        setattr(config, name, value)
    added = []
    pipeline = object.__new__(LingBotVideoPipeline)
    pipeline.add_stage = lambda stage, *a, **k: added.append(stage)
    pipeline.get_module = lambda name, default=None: SimpleNamespace(
        modules=lambda: iter(())
    )
    pipeline.add_standard_latent_preparation_stage = lambda *a, **k: None
    pipeline.add_standard_timestep_preparation_stage = lambda *a, **k: None
    pipeline.add_standard_decoding_stage = lambda *a, **k: None

    pipeline.create_pipeline_stages(
        SimpleNamespace(pipeline_config=config, trust_remote_code=False)
    )
    return added


def _added_stage_names(**config_overrides) -> list[str]:
    return [type(stage).__name__ for stage in _added_stages(**config_overrides)]


def test_pipeline_skips_the_rewriter_stages_by_default(monkeypatch):
    # add_stage_if evaluates its argument eagerly, so the guard has to come first.
    def refuse(*args, **kwargs):
        raise AssertionError("the rewriter stages must not be constructed")

    monkeypatch.setattr(
        pipeline_lingbot_video_moe, "LingBotVideoPromptRewriteStage", refuse
    )
    monkeypatch.setattr(
        pipeline_lingbot_video_moe, "LingBotVideoAutoNegativeStage", refuse
    )
    assert LingBotVideoMoEPipelineConfig().rewriter_url is None

    added = _added_stage_names()

    assert "InputValidationStage" in added


def test_auto_negative_follows_the_rewriter_only_when_asked():
    served = dict(rewriter_url="http://host:30000")

    assert "LingBotVideoPromptRewriteStage" in _added_stage_names(**served)
    assert "LingBotVideoAutoNegativeStage" not in _added_stage_names(**served)

    both = _added_stage_names(**served, rewriter_auto_negative=True)
    assert both.index("LingBotVideoAutoNegativeStage") == (
        both.index("LingBotVideoPromptRewriteStage") + 1
    )


def test_rewriter_stages_share_one_backend():
    stages = _added_stages(
        rewriter_model_path="/models/base",
        rewriter_adapter_path="/models/adapter",
        rewriter_auto_negative=True,
    )

    backends = {
        id(stage.backend)
        for stage in stages
        if isinstance(
            stage, (LingBotVideoPromptRewriteStage, LingBotVideoAutoNegativeStage)
        )
    }
    assert len(backends) == 1


_DEFAULT_NEGATIVE = {
    "universal_negative": {
        "visual_quality": ["low quality", "underexposed", "crushed blacks"],
        "artistic_style": ["painting", "cartoon"],
        "physical_plausibility": ["objects defying gravity"],
        "temporal_and_motion_stability": ["motion blur", "warping"],
    }
}


def test_auto_negative_keeps_the_default_order_and_drops_only_deletions():
    pruned = {
        "universal_negative": {
            # Reordered, with an invented term the model must not be able to add.
            "visual_quality": ["invented", "low quality"],
            "artistic_style": ["painting", "cartoon"],
            "physical_plausibility": ["objects defying gravity"],
            "temporal_and_motion_stability": ["warping", "motion blur"],
        }
    }

    out = prune_negative(_DEFAULT_NEGATIVE, pruned, "a bright meadow at noon")

    assert out["universal_negative"]["visual_quality"] == ["low quality"]
    assert out["universal_negative"]["temporal_and_motion_stability"] == [
        "motion blur",
        "warping",
    ]


def test_auto_negative_restores_a_block_the_caption_never_asked_to_drop():
    emptied = {
        "universal_negative": {"physical_plausibility": [], "artistic_style": []}
    }

    live_action = prune_negative(_DEFAULT_NEGATIVE, emptied, "a cinematic car chase")
    assert live_action["universal_negative"]["physical_plausibility"] == [
        "objects defying gravity"
    ]
    assert live_action["universal_negative"]["artistic_style"] == [
        "painting",
        "cartoon",
    ]

    surreal = prune_negative(
        _DEFAULT_NEGATIVE, emptied, "a surreal dreamlike watercolor painting"
    )
    assert surreal["universal_negative"]["physical_plausibility"] == []
    assert surreal["universal_negative"]["artistic_style"] == []


def test_auto_negative_deletes_terms_the_caption_clearly_wants():
    kept = {"universal_negative": dict(_DEFAULT_NEGATIVE["universal_negative"])}

    out = prune_negative(
        _DEFAULT_NEGATIVE, kept, "a moody night scene with deep shadows"
    )

    assert out["universal_negative"]["visual_quality"] == ["low quality"]


def test_auto_negative_turns_dynamic_batching_off():
    # A merged request carries one negative for several captions, so pruning it
    # per request is only possible while merging is off.
    config = LingBotVideoMoEPipelineConfig()
    assert config.supports_dynamic_batching()

    config.rewriter_auto_negative = True
    assert not config.supports_dynamic_batching()


def test_auto_negative_needs_a_rewriter_backend():
    config = LingBotVideoMoEPipelineConfig()
    config.check_pipeline_config()

    config.rewriter_auto_negative = True
    with pytest.raises(ValueError, match="rewriter_auto_negative"):
        config.check_pipeline_config()

    config.rewriter_url = "http://host:30000"
    config.check_pipeline_config()

    local = LingBotVideoMoEPipelineConfig()
    local.rewriter_auto_negative = True
    local.rewriter_model_path = "/models/base"
    local.rewriter_adapter_path = "/models/adapter"
    local.check_pipeline_config()


@pytest.mark.parametrize(
    "negative_prompt",
    [
        "blurry, low quality",
        '{"universal_negative": []}',
        '{"universal_negative": "bad"}',
        '{"universal_negative": {"visual_quality": "not a list"}}',
        '{"universal_negative": {"visual_quality": [1, 2]}}',
    ],
)
def test_auto_negative_skips_a_negative_it_cannot_prune(negative_prompt):
    # A request may send any string here, so an unprunable one is passed through
    # rather than indexed as categories.
    stage = object.__new__(LingBotVideoAutoNegativeStage)
    stage.backend = _FakeBackend([])
    batch = SimpleNamespace(
        prompt='{"comprehensive_description": "a night scene"}',
        negative_prompt=negative_prompt,
        num_frames=121,
        fps=24,
        image_path=None,
    )

    stage.forward(batch, None)

    assert batch.negative_prompt == negative_prompt
    assert stage.backend.calls == []


def test_auto_negative_hints_match_whole_words_only():
    default = {
        "universal_negative": {
            "visual_quality": [
                "low quality",
                "underexposed",
                "subject hidden in darkness",
                "crushed blacks",
            ]
        }
    }

    def survivors(caption):
        return prune_negative(default, default, caption)["universal_negative"][
            "visual_quality"
        ]

    # "knight" contains "night" and "three-dimensional" contains "dim".
    assert (
        survivors("A knight in bright daylight")
        == default["universal_negative"]["visual_quality"]
    )
    assert (
        survivors("A bright three-dimensional sculpture")
        == default["universal_negative"]["visual_quality"]
    )

    assert survivors("A dim night scene") == ["low quality"]
    assert survivors("dimly lit room") == ["low quality"]
    # Multi-word and hyphenated hints still match.
    assert survivors("shot in low-light") == ["low quality"]
    assert survivors("a low light interior") == ["low quality"]
