"""Workflow-to-API recipe: payloads match the nodes, output is runnable and key-free."""

import json
import shlex
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

# Import by path: the sglang package root pulls in the LLM serving stack.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import PKG, FakeResponse, load_plugin  # noqa: E402

SERVER_API, UTILS, NODES = load_plugin()
RECIPE = sys.modules[f"{PKG}.recipe"]

SECRET = "sk-do-not-leak"
# Keys the node's client adds that carry the server's own defaults.
CLIENT_ONLY_KEYS = {
    "output_path",
    "generator_device",
    "n",
    "quality",
    "style",
    "background",
}


def _graph(class_type, **inputs):
    return {
        "1": {
            "class_type": "SGLDiffusionServerModel",
            "inputs": {"base_url": "http://gpu-box:30010", "api_key": SECRET},
        },
        "2": {
            "class_type": class_type,
            "inputs": {"sgld_client": ["1", 0], **inputs},
        },
    }


def _recipe(graph, node_id="2"):
    return RECIPE.build_recipe(graph, node_id)


def _real_payload(class_type, **node_kwargs):
    """What the real node sends for the same settings, with only HTTP mocked."""
    captured = {}
    client = SERVER_API.SGLDiffusionServerAPI(base_url="http://gpu-box:30010")

    def fake_post(url, json=None, headers=None, timeout=None, **kw):
        captured.update(json)
        captured["_url"] = url
        return FakeResponse({"id": "j", "data": [{"b64_json": ""}]})

    def fake_get(url, headers=None, timeout=None):
        return FakeResponse({"status": "completed", "file_path": "/tmp/x.mp4"})

    node = getattr(NODES, class_type)()
    with (
        mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post),
        mock.patch(f"{PKG}.core.server_api.requests.get", side_effect=fake_get),
        mock.patch(f"{PKG}.nodes.convert_b64_to_tensor_image", return_value=None),
        mock.patch(f"{PKG}.nodes.extract_audio", return_value=None),
        mock.patch(f"{PKG}.nodes.get_image_path", return_value="input.png"),
        mock.patch(f"{PKG}.nodes.get_video_path", return_value="input.mp4"),
    ):
        try:
            if class_type == "SGLDiffusionGenerateImage":
                node.generate_image(sgld_client=client, **node_kwargs)
            elif class_type == "SGLDiffusionGenerateVideo":
                node.generate_video(sgld_client=client, **node_kwargs)
            else:
                node.generate_audio_video(sgld_client=client, **node_kwargs)
        except RuntimeError:
            pass  # the fake image has no pixels; the request was already captured
    return captured


def _widget_kwargs(inputs):
    mapping = {"image": torch.rand(1, 8, 8, 3), "video": object()}
    return {k: mapping.get(k, v) for k, v in inputs.items()}


@pytest.mark.parametrize(
    "class_type, inputs",
    [
        ("SGLDiffusionGenerateImage", {"positive_prompt": "a fox"}),
        (
            "SGLDiffusionGenerateImage",
            {
                "positive_prompt": "a fox",
                "negative_prompt": "blurry",
                "seed": -1,
                "steps": 20,
                "cfg": 3.5,
                "width": 512,
                "height": 768,
                "enable_teacache": True,
                "enhance_prompt": True,
            },
        ),
        ("SGLDiffusionGenerateVideo", {"positive_prompt": "waves"}),
        (
            "SGLDiffusionGenerateVideo",
            {
                "positive_prompt": "waves",
                "seed": 5,
                "width": 832,
                "height": 480,
                "num_frames": 81,
                "fps": 16,
                "seconds": 4,
                "enhance_prompt": True,
            },
        ),
        (
            "SGLDiffusionGenerateAudioVideo",
            {
                "positive_prompt": "a drummer",
                "extra_request_json": '{"flow_shift": 12.0}',
            },
        ),
    ],
)
def test_recipe_payload_matches_what_the_node_sends(class_type, inputs):
    real = _real_payload(class_type, **_widget_kwargs(inputs))
    recipe = _recipe(_graph(class_type, **inputs))
    real.pop("_url")

    assert recipe.payload == {
        k: v for k, v in real.items() if k not in CLIENT_ONLY_KEYS
    }
    assert set(real) - set(recipe.payload) <= CLIENT_ONLY_KEYS


def test_video_inputs_become_reference_fields():
    inputs = {"positive_prompt": "restyle", "image": ["9", 0], "video": ["8", 0]}
    graph = _graph("SGLDiffusionGenerateVideo", **inputs)
    graph.update(
        {
            "8": {"class_type": "LoadVideo", "inputs": {}},
            "9": {"class_type": "LoadImage", "inputs": {}},
        }
    )

    real = _real_payload("SGLDiffusionGenerateVideo", **_widget_kwargs(inputs))
    recipe = _recipe(graph)

    assert recipe.payload["video_path"] == real["video_path"] == "input.mp4"
    assert recipe.payload["input_reference"] == real["input_reference"]
    assert any("replace" in note for note in recipe.notes)


def test_image_input_switches_to_the_edit_endpoint():
    graph = _graph("SGLDiffusionGenerateImage", positive_prompt="x", image=["9", 0])
    graph["9"] = {"class_type": "LoadImage", "inputs": {}}

    recipe = _recipe(graph)

    assert recipe.endpoint == "/images/edits"
    assert recipe.files == {"image": "input.png"}
    assert "-F image=@input.png" in RECIPE.render(recipe, "curl")


def test_linked_prompt_is_read_from_a_primitive_node():
    graph = _graph("SGLDiffusionGenerateVideo", positive_prompt=["5", 0])
    graph["5"] = {"class_type": "PrimitiveString", "inputs": {"value": "a red kite"}}

    assert _recipe(graph).payload["prompt"] == "a red kite"


def test_unresolvable_inputs_fall_back_and_are_flagged():
    graph = _graph(
        "SGLDiffusionGenerateVideo", positive_prompt=["5", 0], steps=["6", 0]
    )
    graph["5"] = {"class_type": "SomeLLMNode", "inputs": {}}
    graph["6"] = {"class_type": "SomeMath", "inputs": {}}

    recipe = _recipe(graph)

    assert recipe.payload["prompt"] == "<your prompt>"
    assert recipe.payload["num_inference_steps"] == 6
    assert any("'steps'" in note for note in recipe.notes)


def test_server_url_comes_from_the_connected_server_node():
    recipe = _recipe(_graph("SGLDiffusionGenerateImage", positive_prompt="x"))

    assert recipe.base_url == "http://gpu-box:30010/v1"
    assert (
        _recipe(
            {"2": {"class_type": "SGLDiffusionGenerateImage", "inputs": {}}}
        ).base_url
        == RECIPE.DEFAULT_BASE_URL
    )


@pytest.mark.parametrize("fmt", RECIPE.FORMATS)
@pytest.mark.parametrize(
    "class_type",
    ["SGLDiffusionGenerateImage", "SGLDiffusionGenerateVideo"],
)
def test_api_key_is_never_written_out(fmt, class_type):
    text = RECIPE.render(_recipe(_graph(class_type, positive_prompt="x")), fmt)

    assert SECRET not in text


def test_curl_command_round_trips_through_a_shell_parser():
    prompt = 'it\'s a "quoted" prompt; rm -rf /'
    text = RECIPE.render(
        _recipe(_graph("SGLDiffusionGenerateVideo", positive_prompt=prompt)), "curl"
    )

    words = shlex.split(text.split("\n# the job")[0].replace("\\\n", " "))
    body = json.loads(words[words.index("-d") + 1])
    assert body["prompt"] == prompt
    assert words[:3] == ["curl", "-X", "POST"]
    assert words[3] == "http://gpu-box:30010/v1/videos"
    assert "$SGLANG_API_KEY" in text and "/videos/<id" in text


@pytest.mark.parametrize(
    "class_type, with_image",
    [
        ("SGLDiffusionGenerateImage", False),
        ("SGLDiffusionGenerateImage", True),
        ("SGLDiffusionGenerateVideo", False),
    ],
)
def test_python_snippet_is_valid_python(class_type, with_image):
    inputs = {"positive_prompt": "x"}
    graph = _graph(class_type, **inputs, **({"image": ["9", 0]} if with_image else {}))
    graph["9"] = {"class_type": "LoadImage", "inputs": {}}

    text = RECIPE.render(_recipe(graph), "python")

    compile(text, "recipe.py", "exec")
    assert "SGLANG_API_KEY" in text


def test_cli_command_uses_real_flags_and_notes_what_it_cannot_express():
    graph = _graph(
        "SGLDiffusionGenerateVideo",
        positive_prompt="waves",
        width=832,
        height=480,
        enhance_prompt=True,
        enable_teacache=True,
    )

    text = RECIPE.render(_recipe(graph), "sglang generate")
    command = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    words = shlex.split(command.replace("\\\n", " "))

    assert words[:2] == ["sglang", "generate"]
    for flag in (
        "--prompt",
        "--width",
        "--height",
        "--num-frames",
        "--enable-teacache",
    ):
        assert flag in words
    assert words[words.index("--width") + 1] == "832"
    assert "no CLI flag for: enhance_prompt" in text


def test_target_selection():
    graph = _graph("SGLDiffusionGenerateImage", positive_prompt="x")
    assert RECIPE.find_target(graph, "") == "2"

    graph["3"] = {"class_type": "SGLDiffusionGenerateVideo", "inputs": {}}
    with pytest.raises(ValueError, match="set node_id"):
        RECIPE.find_target(graph, "")
    assert RECIPE.find_target(graph, "3") == "3"
    with pytest.raises(ValueError, match="not an SGLDiffusion generate node"):
        RECIPE.find_target(graph, "1")
    with pytest.raises(ValueError, match="No SGLDiffusion generate node"):
        RECIPE.find_target({"1": graph["1"]}, "")


def test_node_returns_text_for_display_and_always_reruns():
    node = NODES.SGLDiffusionApiRecipe()
    graph = _graph("SGLDiffusionGenerateImage", positive_prompt="x")

    out = node.make_recipe(format="curl", prompt=graph)

    assert out["ui"]["text"] == list(out["result"])
    assert "images/generations" in out["result"][0]
    assert (
        NODES.SGLDiffusionApiRecipe.IS_CHANGED()
        != NODES.SGLDiffusionApiRecipe.IS_CHANGED()
    )
    assert NODES.SGLDiffusionApiRecipe.INPUT_TYPES()["hidden"] == {"prompt": "PROMPT"}
