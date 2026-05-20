"""Unit-level tests for the smoke fixture's router config TOML.

These do NOT launch the binary; they validate the *shape* of the TOML the
``router`` fixture would write. The router's Rust ``Config`` struct
requires a ``[discovery]`` section (``DiscoveryConfig`` has no
``#[serde(default)]``), and there is no top-level ``[[workers]]`` field.
A previous version of the fixture emitted ``[[workers]]`` directly,
which made the smoke router fail to start; this test guards that
regression at the unit level so the contract is checked without a GPU.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from conftest import build_smoke_router_config


def test_build_smoke_router_config_produces_parseable_toml(tmp_path: Path) -> None:
    workers_path = tmp_path / "workers.toml"
    main_cfg, workers_cfg = build_smoke_router_config(
        host="127.0.0.1",
        port=8090,
        model="Qwen/Qwen3-0.6B",
        tokenizer_path="/tmp/tok.json",
        sglang_url="http://localhost:30000",
        workers_path=workers_path,
    )

    main = tomllib.loads(main_cfg)
    assert "server" in main, "missing required [server] section"
    assert "models" in main, "missing required [[models]] array"
    assert (
        "discovery" in main
    ), "missing required [discovery] section (Config requires it)"
    assert (
        "workers" not in main
    ), "[[workers]] must NOT be inline in the main config; lives in the static_file path"
    assert main["discovery"]["backend"] == "static_file"
    assert main["discovery"]["static_file"]["path"] == str(
        workers_path
    ), "discovery.static_file.path must point at the workers file"

    workers = tomllib.loads(workers_cfg)
    assert "workers" in workers, "workers file must contain [[workers]] entries"
    assert workers["workers"][0]["url"] == "http://localhost:30000"
