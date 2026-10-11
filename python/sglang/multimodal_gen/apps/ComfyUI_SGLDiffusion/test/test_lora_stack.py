"""LoRA browser and stacking: list_loras dropdown, per-LoRA strength, merge mode."""

import sys
from pathlib import Path
from unittest import mock

import pytest
import requests

# Import by path: the sglang package root pulls in the LLM serving stack.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import PKG, FakeResponse, load_plugin  # noqa: E402

SERVER_API, UTILS, NODES = load_plugin()

LISTING = {
    "loaded_adapters": [
        {"nickname": "anime", "path": "/loras/anime.safetensors"},
        {"nickname": "film", "path": "org/film-lora"},
    ],
    "active": {},
}


def _client():
    return SERVER_API.SGLDiffusionServerAPI(base_url="http://127.0.0.1:30010")


def _set_lora(call):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        assert url.endswith("/v1/set_lora")
        captured.update(json)
        return FakeResponse({"status": "ok"})

    with mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post):
        result = call(_client())
    return captured, result


def _stack(**kwargs):
    return _set_lora(
        lambda client: NODES.SGLDiffusionServerLoraStack().set_lora_stack(
            sgld_client=client, **kwargs
        )
    )


def test_stack_sends_aligned_lists_and_skips_empty_slots():
    payload, (_, summary) = _stack(
        lora_1="anime",
        strength_1=0.8,
        lora_2="none",
        lora_path_2="/loras/new-style.safetensors",
        strength_2=0.5,
        lora_3="none",
        lora_4="film",
        strength_4=1.25,
    )

    assert payload["lora_nickname"] == ["anime", "new-style", "film"]
    assert payload["lora_path"] == [None, "/loras/new-style.safetensors", None]
    assert payload["strength"] == [0.8, 0.5, 1.25]
    assert payload["merge_mode"] == "merge"
    assert summary == "anime@0.8, new-style@0.5, film@1.25"


def test_stack_server_default_omits_merge_mode():
    payload, _ = _stack(lora_1="anime", merge_mode="server default")

    assert "merge_mode" not in payload


def test_stack_uses_one_target_for_every_adapter():
    payload, _ = _stack(lora_1="anime", lora_2="film", target="transformer")

    assert payload["target"] == "transformer"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "at least one"),
        ({"lora_1": "anime", "lora_2": "anime"}, "twice"),
        ({"lora_1": "anime", "lora_2": "film", "merge_mode": "dynamic"}, "merge_mode"),
    ],
)
def test_stack_rejects_bad_input_before_calling_the_server(kwargs, message):
    with mock.patch(f"{PKG}.core.server_api.requests.post") as post:
        with pytest.raises(ValueError, match=message):
            NODES.SGLDiffusionServerLoraStack().set_lora_stack(
                sgld_client=_client(), **kwargs
            )
    post.assert_not_called()


def test_single_dynamic_lora_is_allowed():
    payload, _ = _stack(lora_1="anime", merge_mode="dynamic")

    assert payload["merge_mode"] == "dynamic"


def test_set_lora_node_sends_strength_and_merge_mode():
    payload, _ = _set_lora(
        lambda client: NODES.SGLDiffusionServerSetLora().set_lora(
            sgld_client=client,
            lora_name="/loras/anime.safetensors",
            lora_nickname="anime",
            strength=0.6,
            merge_mode="merge",
        )
    )

    assert payload["strength"] == 0.6
    assert payload["merge_mode"] == "merge"
    assert payload["lora_nickname"] == "anime"


def test_client_set_lora_payload_is_unchanged_for_existing_callers():
    payload, _ = _set_lora(
        lambda client: client.set_lora(lora_nickname="a", lora_path="/p/a.safetensors")
    )

    assert payload == {
        "lora_nickname": "a",
        "lora_path": "/p/a.safetensors",
        "target": "all",
    }


def test_list_node_returns_nicknames_and_status():
    with mock.patch(
        f"{PKG}.core.server_api.requests.get", return_value=FakeResponse(LISTING)
    ) as get:
        nicknames, status = NODES.SGLDiffusionServerListLoras().list_loras(_client())

    assert get.call_args.args[0].endswith("/v1/list_loras")
    assert nicknames == "anime\nfilm"
    assert "org/film-lora" in status


def test_list_loras_explains_a_server_without_lora():
    response = FakeResponse({"detail": "Lora is not enabled"}, 500)
    response.raise_for_status = mock.Mock(
        side_effect=requests.exceptions.HTTPError("500", response=response)
    )

    with mock.patch(f"{PKG}.core.server_api.requests.get", return_value=response):
        with pytest.raises(RuntimeError, match="Lora is not enabled"):
            _client().list_loras()


def test_dropdown_lists_adapters_of_the_connected_server():
    client = _client()
    with mock.patch.object(client, "get_model_info", return_value={}):
        with mock.patch.object(NODES, "SGLDiffusionServerAPI", return_value=client):
            NODES.SGLDiffusionServerModel().load_server("http://x:1/v1", "k")

    with mock.patch.object(client, "list_loras", return_value=LISTING):
        optional = NODES.SGLDiffusionServerLoraStack.INPUT_TYPES()["optional"]

    assert optional["lora_1"][0] == ["none", "anime", "film"]
    assert optional["lora_4"][0] == ["none", "anime", "film"]


def test_dropdown_falls_back_to_none_when_the_server_is_unreachable():
    client = _client()
    NODES._last_client = client

    with mock.patch.object(client, "list_loras", side_effect=RuntimeError("down")):
        optional = NODES.SGLDiffusionServerLoraStack.INPUT_TYPES()["optional"]

    assert optional["lora_1"][0] == ["none"]


def test_stale_dropdown_values_are_not_rejected_by_validation():
    assert NODES.SGLDiffusionServerLoraStack.VALIDATE_INPUTS(
        lora_1="gone", lora_2="none", lora_3="none", lora_4="none"
    )


def test_validation_tolerates_graphs_that_omit_slots():
    assert NODES.SGLDiffusionServerLoraStack.VALIDATE_INPUTS()
    assert NODES.SGLDiffusionServerLoraStack.VALIDATE_INPUTS(lora_2="anime")
