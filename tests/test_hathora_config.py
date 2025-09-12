import os
import json

from hathora_config import DeploymentConfig


def test_config_from_json_env(monkeypatch):
    cfg = {
        "hf_token": "hf_123",
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "dtype": "auto",
        "tp_size": 2,
        "enable_metrics": True,
        "h100_only": True,
    }
    monkeypatch.setenv("DEPLOYMENT_CONFIG_JSON", json.dumps(cfg))

    parsed = DeploymentConfig(**json.loads(os.environ["DEPLOYMENT_CONFIG_JSON"]))
    assert parsed.hf_token == "hf_123"
    assert parsed.model_id.endswith("8B-Instruct")
    assert parsed.tp_size == 2
    assert parsed.enable_metrics is True
    assert parsed.h100_only is True


def test_tp_size_validation():
    ok = DeploymentConfig(model_id="mistralai/Mistral-7B-Instruct-v0.3", tp_size=4)
    assert ok.tp_size == 4

    try:
        DeploymentConfig(model_id="mistralai/Mistral-7B-Instruct-v0.3", tp_size=3)
        assert False, "Expected validation error"
    except Exception:
        pass


