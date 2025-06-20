# sglang/test/srt/openai/test_server.py
import requests

from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST as MODEL_ID


def test_health(openai_server: str):
    r = requests.get(f"{openai_server}/health")
    assert r.status_code == 200
    # FastAPI returns an empty body → r.text == ""
    assert r.text == ""


def test_models_endpoint(openai_server: str):
    r = requests.get(f"{openai_server}/v1/models")
    assert r.status_code == 200, r.text
    payload = r.json()

    # Basic contract
    assert "data" in payload and isinstance(payload["data"], list) and payload["data"]

    # Validate fields of the first model card
    first = payload["data"][0]
    for key in ("id", "root", "max_model_len"):
        assert key in first, f"missing {key} in {first}"

    # max_model_len must be positive
    assert isinstance(first["max_model_len"], int) and first["max_model_len"] > 0

    # The server should report the same model id we launched it with
    ids = {m["id"] for m in payload["data"]}
    assert MODEL_ID in ids


def test_get_model_info(openai_server: str):
    r = requests.get(f"{openai_server}/get_model_info")
    assert r.status_code == 200, r.text
    info = r.json()

    expected_keys = {"model_path", "tokenizer_path", "is_generation"}
    assert expected_keys.issubset(info.keys())

    # model_path must end with the one we passed on the CLI
    assert info["model_path"].endswith(MODEL_ID)

    # is_generation is documented as a boolean
    assert isinstance(info["is_generation"], bool)


def test_unknown_route_returns_404(openai_server: str):
    r = requests.get(f"{openai_server}/definitely-not-a-real-route")
    assert r.status_code == 404
