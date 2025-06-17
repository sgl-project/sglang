# sglang/test/srt/openai/test_server.py
import pytest
import requests


def test_health(openai_server: str):
    r = requests.get(f"{openai_server}/health")
    assert r.status_code == 200, r.text
    assert r.text == ""


@pytest.mark.xfail(reason="Endpoint skeleton not implemented yet")
def test_models_endpoint(openai_server: str):
    r = requests.get(f"{openai_server}/v1/models")
    # once implemented this should be 200
    assert r.status_code == 200
