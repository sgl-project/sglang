import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "playground" / "disaggregation" / "pd_flip_docker"
HTTP_SERVER = ROOT / "python" / "sglang" / "srt" / "entrypoints" / "http_server.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def env_values() -> dict[str, str]:
    values = {}
    for raw_line in read(HARNESS / "env.example").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip("'\"")
    return values


def route_block(path: str) -> str:
    source = read(HTTP_SERVER)
    match = re.search(
        rf'@app\.(?:get|post)\("{re.escape(path)}"\)(?P<body>.{{0,160}})',
        source,
        re.DOTALL,
    )
    assert match, f"missing HTTP route {path}"
    return match.group(0)


def test_progressive_endpoints_are_admin_authenticated():
    endpoints = (
        "/pd_flip/migration/source/start",
        "/pd_flip/migration/source/delta",
        "/pd_flip/migration/source/finish",
        "/pd_flip/migration/target/prepare",
        "/pd_flip/migration/target/delta/prepare",
        "/pd_flip/migration/target/commit",
        "/pd_flip/migration/target/activate",
        "/pd_flip/migration/target/abort",
        "/pd_flip/migration/abort",
        "/pd_flip/migration/status",
        "/pd_flip/runtime_role/status",
        "/pd_flip/runtime_role/set",
        "/pd_flip/runtime_role/admission",
    )
    for endpoint in endpoints:
        assert "@auth_level(AuthLevel.ADMIN_OPTIONAL)" in route_block(endpoint)


def test_example_environment_is_one_prefill_three_decode_and_progressive_safe():
    env = env_values()
    assert [env[f"NODE{i}_ROLE"] for i in range(4)] == [
        "prefill",
        "decode",
        "decode",
        "decode",
    ]
    assert env["PD_FLIP_FIRST_MIGRATION_RATIO"] == "0.5"
    assert env["PD_FLIP_OBSERVATION_SECONDS"] == "10"
    assert env["PD_FLIP_SLO_THRESHOLD"] == "0.9"
    assert env["PD_FLIP_MIN_PREFILL_SLO_SAMPLES"] == "20"
    assert env["PD_FLIP_MIN_DECODE_SLO_SAMPLES"] == "20"
    assert env["HICACHE_WRITE_POLICY"] == "write_through"
    assert env["MOONCAKE_MASTER"]
    assert env["MOONCAKE_TE_META_DATA_SERVER"].endswith("/metadata")
    assert env["MOONCAKE_GLOBAL_SEGMENT_SIZE"]
    assert env["ADMIN_API_KEY"]
    assert env["PD_FLIP_ARTIFACT_DIR"]


def test_worker_and_controller_forward_the_deployment_contract():
    worker = read(HARNESS / "run_worker.sh")
    controller = read(HARNESS / "run_controller.sh")
    for flag in (
        "--enable-pd-flip-state-machine",
        "--enable-pd-runtime-role-switch",
        "--enable-pd-flip-hicache-stitch",
        "--disaggregation-decode-enable-radix-cache",
        "--enable-hierarchical-cache",
        "--hicache-storage-backend",
        "--hicache-write-policy",
        "--admin-api-key",
    ):
        assert flag in worker
    for name in (
        "MOONCAKE_MASTER",
        "MOONCAKE_TE_META_DATA_SERVER",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE",
    ):
        assert name in worker
    assert 'extra_docker_args+=(-e "${name}=${!name}")' in worker
    for flag in (
        "--api-key",
        "--first-migration-ratio",
        "--observation-seconds",
        "--slo-threshold",
        "--min-prefill-slo-samples",
        "--min-decode-slo-samples",
        "--session-journal-path",
    ):
        assert flag in controller


def test_runbook_is_executable_and_defines_acceptance_and_artifacts():
    runbook = read(HARNESS / "README.md")
    required_contract = (
        "node0: prefill",
        "node1: decode",
        "node2: decode source selected for D-to-P",
        "node3: decode target",
        "node0/node2 prefill, node1/node3 decode",
        "full_prefix_stitch",
        "partial_prefix_stitch",
        "source_decode_full_fallback",
        "SLO recovery without role flip",
        "persistent prefill risk with successful D-to-P",
        "Authorization: Bearer ${ADMIN_API_KEY}",
        "/pd_flip/migration/abort",
        "reconcile_session",
        "migration_status_samples.csv",
        "migration_request_samples.jsonl",
        "mooncake_bytes_available",
        "configuration archive",
    )
    for text in required_contract:
        assert text in runbook
