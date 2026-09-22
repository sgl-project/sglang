"""Unit tests for the static local multi-model supervisor configuration."""

import pytest

from sglang_router.multi_model import MultiModelConfigError, parse_multi_model_config


def _valid_config():
    return {
        "router": {"host": "0.0.0.0", "port": 30000, "policy": "round_robin"},
        "worker_base_port": 31000,
        "models": [
            {
                "model_id": "qwen-chat",
                "model_path": "/models/Qwen3-0.6B",
                "gpu_groups": [[0], [1]],
                "server_args": {"tp_size": 1},
            },
            {
                "model_id": "glm-reason",
                "model_path": "/models/GLM-5.3-Flash",
                "gpu_groups": [[2, 3]],
                "server_args": {"tp_size": 2, "mem_fraction_static": 0.8},
            },
        ],
    }


def test_parse_multi_model_config_expands_gpu_groups_into_replicas():
    config = parse_multi_model_config(_valid_config())

    assert config.router_args == {
        "host": "0.0.0.0",
        "port": 30000,
        "policy": "round_robin",
    }
    assert config.worker_host == "127.0.0.1"
    assert config.model_resolver is None
    assert [(replica.model_id, replica.gpu_ids) for replica in config.replicas] == [
        ("qwen-chat", (0,)),
        ("qwen-chat", (1,)),
        ("glm-reason", (2, 3)),
    ]


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda config: config["models"][1].update({"gpu_groups": [[1, 2]]}),
            "reuses GPU IDs",
        ),
        (
            lambda config: config["models"][1]["server_args"].update(
                {"dp_size": 2}
            ),
            "supervisor-owned fields",
        ),
        (
            lambda config: config["models"][1]["server_args"].update(
                {"enable_dp_attention": True}
            ),
            "MVP-unsupported fields",
        ),
        (
            lambda config: config["models"][1].update({"gpu_groups": [[2]]}),
            r"tp_size \* pp_size requires 2",
        ),
    ],
)
def test_parse_multi_model_config_rejects_unsafe_topologies(mutate, error):
    config = _valid_config()
    mutate(config)

    with pytest.raises(MultiModelConfigError, match=error):
        parse_multi_model_config(config)


def test_parse_multi_model_config_rejects_disabled_igw():
    config = _valid_config()
    config["router"]["enable_igw"] = False

    with pytest.raises(MultiModelConfigError, match="cannot be false"):
        parse_multi_model_config(config)


def test_parse_multi_model_config_allows_explicit_gpu_sharing():
    config = _valid_config()
    config["allow_gpu_sharing"] = True
    config["models"][1]["gpu_groups"] = [[0, 2]]

    parsed = parse_multi_model_config(config)

    assert [(replica.model_id, replica.gpu_ids) for replica in parsed.replicas] == [
        ("qwen-chat", (0,)),
        ("qwen-chat", (1,)),
        ("glm-reason", (0, 2)),
    ]


def test_parse_multi_model_config_accepts_resource_aware_model_resolver():
    config = _valid_config()
    config["router"]["port"] = 30001
    config["model_resolver"] = {
        "host": "0.0.0.0",
        "port": 30000,
        "refresh_interval_secs": 0.5,
        "stale_after_secs": 2.0,
        "profiles": [
            {
                "model_id": "general-chat",
                "candidates": ["qwen-chat", "glm-reason"],
                "max_kv_utilization": 0.85,
                "max_waiting_requests": 4,
                "min_free_tokens": 128,
            }
        ],
    }

    parsed = parse_multi_model_config(config)

    assert parsed.model_resolver is not None
    assert parsed.model_resolver.port == 30000
    assert parsed.model_resolver.profiles[0].model_id == "general-chat"
    assert parsed.model_resolver.profiles[0].candidates == (
        "qwen-chat",
        "glm-reason",
    )


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda config: config["model_resolver"].update({"port": 30001}),
            "must differ",
        ),
        (
            lambda config: config["model_resolver"]["profiles"][0].update(
                {"candidates": ["missing-model"]}
            ),
            "unknown concrete models",
        ),
    ],
)
def test_parse_multi_model_config_rejects_invalid_resource_resolver(mutate, error):
    config = _valid_config()
    config["router"]["port"] = 30001
    config["model_resolver"] = {
        "port": 30000,
        "profiles": [
            {"model_id": "general-chat", "candidates": ["qwen-chat"]}
        ],
    }
    mutate(config)

    with pytest.raises(MultiModelConfigError, match=error):
        parse_multi_model_config(config)
