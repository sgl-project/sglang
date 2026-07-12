import re
import shutil
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "playground" / "disaggregation" / "pd_flip_docker"
HTTP_SERVER = ROOT / "python" / "sglang" / "srt" / "entrypoints" / "http_server.py"
ROUTER_PD_RUNTIME = (
    ROOT / "experimental" / "sgl-router" / "src" / "server" / "routes" / "pd_runtime.rs"
)
WINDOWS_HELPER = HARNESS / "windows_four_node.ps1"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    return subprocess.check_output(
        ["bash", "-lc", f"wslpath -a -u {shlex.quote(str(path))}"], text=True
    ).strip()


def powershell_path(path: Path) -> str:
    if os.name == "nt":
        return str(path)
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


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


def harness_env(tmp_path: Path, *, admin_key: str) -> tuple[Path, dict[str, str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    env_file = tmp_path / "env.local"
    env_file.write_bytes(
        textwrap.dedent(
            f"""\
            SGLANG_REPO=/repo
            IMAGE=test-image
            MODEL_PATH=/model
            MODEL_ID=test-model
            TOKENIZER_PATH=
            TP_SIZE=1
            DP_SIZE=1
            PORT=30000
            BOOTSTRAP_PORT=8998
            MEM_FRACTION_STATIC=0.5
            TRANSFER_BACKEND=mooncake
            IB_DEVICE=mlx5_0
            MOONCAKE_GLOBAL_SEGMENT_SIZE=0
            ROUTER_HOST=127.0.0.1
            ROUTER_PORT=8000
            NODE0=http://node0:30000
            NODE1=http://node1:30000
            NODE2=http://node2:30000
            NODE3=http://node3:30000
            TTFT_SLO_SECONDS=0.2
            TPOT_SLO_SECONDS=0.02
            PD_FLIP_WINDOW_SECONDS=30
            PD_FLIP_ENTER_THRESHOLD=0.9
            PD_FLIP_EXIT_THRESHOLD=0.95
            PD_FLIP_COMMIT_THRESHOLD=0.9
            PD_FLIP_MONITOR_ITERATIONS=1
            PD_FLIP_MONITOR_POLL_INTERVAL=1
            PD_FLIP_FIRST_MIGRATION_RATIO=0.5
            PD_FLIP_OBSERVATION_SECONDS=10
            PD_FLIP_SLO_THRESHOLD=0.9
            PD_FLIP_MIN_PREFILL_SLO_SAMPLES=20
            PD_FLIP_MIN_DECODE_SLO_SAMPLES=20
            PD_FLIP_ARTIFACT_DIR=/default-artifacts
            ADMIN_API_KEY={admin_key}
            EXTRA_SGLANG_ARGS=
            EXTRA_DOCKER_ARGS=
            EXTRA_ROUTER_ARGS=
            """
        ).encode("utf-8"),
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_bytes(b'#!/usr/bin/env bash\nprintf "ARG=%s\\n" "$@"\n')
    docker.chmod(0o755)
    env = os.environ.copy()
    env.update(
        ENV_FILE=str(env_file),
        PATH=f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
    )
    return env_file, env


def run_harness(tmp_path: Path, script: str, *args: str, admin_key: str, **overrides):
    env_file, env = harness_env(tmp_path, admin_key=admin_key)
    script_path = HARNESS / script
    command = ["bash", str(script_path), *args]
    if os.name == "nt":
        env["ENV_FILE"] = bash_path(env_file)
        env["PATH"] = f"{bash_path(tmp_path / 'bin')}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        script_path = bash_path(script_path)
        assignments = {"ENV_FILE": env["ENV_FILE"], "PATH": env["PATH"]}
        assignments.update({key: str(value) for key, value in overrides.items()})
        shell = "exec env " + " ".join(
            f"{key}={shlex.quote(value)}" for key, value in assignments.items()
        )
        shell += " bash " + " ".join(shlex.quote(value) for value in (script_path, *args))
        command = ["bash", "-c", shell]
    else:
        env.update({key: str(value) for key, value in overrides.items()})
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )


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
    assert env["MOONCAKE_GLOBAL_SEGMENT_SIZE"] == "0"
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


def test_harnesses_fail_closed_for_empty_or_placeholder_admin_keys(tmp_path):
    for script, args in (
        ("run_worker.sh", ("decode", "0.0.0.0")),
        ("run_controller.sh", ("metrics",)),
        ("run_router.sh", ()),
    ):
        for key in ("", "replace-with-a-strong-admin-secret"):
            result = run_harness(tmp_path / f"{script}-{len(key)}", script, *args, admin_key=key)
            assert result.returncode != 0, (script, key, result.stdout, result.stderr)
            assert "ADMIN_API_KEY" in result.stderr


def test_valid_admin_key_is_forwarded_to_worker_controller_and_router(tmp_path):
    worker = run_harness(
        tmp_path / "worker", "run_worker.sh", "decode", "0.0.0.0", admin_key="secret"
    )
    controller = run_harness(
        tmp_path / "controller", "run_controller.sh", "metrics", admin_key="secret"
    )
    router = run_harness(tmp_path / "router", "run_router.sh", admin_key="secret")
    for result in (worker, controller, router):
        assert result.returncode == 0, result.stderr
    assert "--admin-api-key secret" in worker.stdout
    assert "ARG=--api-key\nARG=secret" in controller.stdout
    assert "secret" not in router.stdout
    assert "ARG=-e\nARG=PD_FLIP_ROUTER_ADMIN_API_KEY" in router.stdout


def test_router_rejects_a_key_the_controller_cannot_use(tmp_path):
    result = run_harness(
        tmp_path,
        "run_router.sh",
        admin_key="worker-secret",
        PD_FLIP_ROUTER_ADMIN_API_KEY="different-secret",
    )
    assert result.returncode != 0
    assert "must match ADMIN_API_KEY" in result.stderr


def test_controller_command_prefix_overrides_env_file(tmp_path):
    result = run_harness(
        tmp_path,
        "run_controller.sh",
        "monitor",
        admin_key="secret",
        PD_FLIP_MONITOR_ITERATIONS=120,
        PD_FLIP_ARTIFACT_DIR="/sgl-workspace/sglang/pd-flip-artifacts/per-run",
    )
    assert result.returncode == 0, result.stderr
    assert "ARG=--iterations\nARG=120" in result.stdout
    assert (
        "ARG=--session-journal-path\n"
        "ARG=/sgl-workspace/sglang/pd-flip-artifacts/per-run/pd_flip_session.json"
    ) in result.stdout


def test_router_auth_contract_is_fail_closed_in_rust_source():
    source = read(ROUTER_PD_RUNTIME)
    assert "AUTHORIZATION" in source
    assert "Bearer " in source
    assert "ApiError::Unauthorized" in source
    for handler in ("list_workers", "set_worker_drain", "set_worker_role"):
        block = re.search(rf"pub async fn {handler}\b(?P<body>.*?\n\}})", source, re.DOTALL)
        assert block and "require_admin" in block.group("body")
    for case in (
        "missing_admin_key_is_unauthorized",
        "missing_bearer_is_unauthorized",
        "wrong_bearer_is_unauthorized_without_mutation",
        "correct_bearer_authorizes_controls",
    ):
        assert case in source
    assert "duplicate_authorization_headers_are_rejected" in source
    assert "bearer_scheme_is_ascii_case_insensitive" in source
    assert "malformed_or_whitespace_bearers_are_rejected" in source
    types_source = read(ROOT / "experimental/sgl-router/src/config/types.rs")
    cli_source = read(ROOT / "experimental/sgl-router/src/config/cli.rs")
    assert "SecretString" in types_source and "[REDACTED]" in types_source
    assert "load_pd_flip_router_admin_key" in cli_source
    assert '"PD_FLIP_ROUTER_ADMIN_API_KEY"' in cli_source
    assert "pub pd_flip_router_admin_api_key" not in cli_source
    cargo = read(ROOT / "experimental/sgl-router/Cargo.toml")
    assert 'subtle = "2"' in cargo
    assert "ConstantTimeEq" in source


def test_windows_helper_uses_1p3d_secrets_and_authenticated_status():
    source = read(WINDOWS_HELPER)
    assert '@{ Name = "node1"; Host = "cloud-100"; Role = "decode"' in source
    assert "[string]$AdminApiKey" in source
    assert '"ADMIN_API_KEY=$(Quote-ShValue $AdminApiKey)"' in source
    assert '"PD_FLIP_ROUTER_ADMIN_API_KEY=$(Quote-ShValue $AdminApiKey)"' in source
    assert '"MOONCAKE_GLOBAL_SEGMENT_SIZE=0"' in source
    assert "Authorization: Bearer" in source
    assert "PD_FLIP_ROUTER_ADMIN_API_KEY" in source


def test_windows_helper_round_trips_shell_metacharacter_secret(tmp_path):
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if not powershell:
        pytest.skip("PowerShell is unavailable")
    secret = "p@$$&'quoted"
    env_file = tmp_path / "windows.env"
    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            powershell_path(WINDOWS_HELPER),
            "-Action",
            "write-env",
            "-AdminApiKey",
            secret,
            "-EnvFile",
            powershell_path(env_file),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    verify_script = tmp_path / "verify-secret.sh"
    verify_script.write_text(
        "set -eu\nset -a\nsource %s\nprintf '%%s\\n%%s' \"$ADMIN_API_KEY\" "
        '"$PD_FLIP_ROUTER_ADMIN_API_KEY"\n' % shlex.quote(bash_path(env_file)),
        encoding="utf-8",
        newline="\n",
    )
    sourced = subprocess.run(
        ["bash", bash_path(verify_script)], text=True, capture_output=True
    )
    assert sourced.returncode == 0, sourced.stderr
    assert sourced.stdout.splitlines() == [secret, secret]


def test_readme_has_clean_shell_setup_tracked_workload_and_paired_summary(tmp_path):
    runbook = read(HARNESS / "README.md")
    block = re.search(
        r"# task11-clean-shell-smoke-begin\n(?P<body>.*?)\n# task11-clean-shell-smoke-end",
        runbook,
        re.DOTALL,
    )
    assert block
    smoke = tmp_path / "clean-shell-smoke.sh"
    smoke.write_bytes(block.group("body").encode("utf-8"))
    parsed = subprocess.run(
        ["bash", "-n", bash_path(smoke)], text=True, capture_output=True
    )
    assert parsed.returncode == 0, parsed.stderr
    env_file, env = harness_env(tmp_path, admin_key="secret")
    if os.name == "nt":
        command = [
            "bash",
            "-c",
            "exec env ENV_FILE=%s bash %s"
            % (shlex.quote(bash_path(env_file)), shlex.quote(bash_path(smoke))),
        ]
    else:
        env["ENV_FILE"] = str(env_file)
        command = ["bash", str(smoke)]
    executed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert executed.returncode == 0, executed.stderr
    assert "pd_flip_progressive_workload.py" in runbook
    assert "pd_flip_progressive_matrix.py" in runbook
    matrix_source = read(
        ROOT / "scripts/playground/disaggregation/pd_flip_progressive_matrix.py"
    )
    assert 'MODES = ("full", "partial", "zero")' in matrix_source
    assert 'PATHS = ("recovery", "commit")' in matrix_source
    assert "--decision-path \"$PATH_KIND\"" in runbook
    assert "--controller-log" in runbook
    for tool in (
        ROOT / "scripts/playground/disaggregation/pd_flip_progressive_workload.py",
        ROOT / "scripts/playground/disaggregation/pd_flip_progressive_matrix.py",
        ROOT / "scripts/playground/disaggregation/pd_flip_migration_measure.py",
    ):
        executed_tool = subprocess.run(
            [sys.executable, str(tool), "--help"],
            cwd=HARNESS,
            text=True,
            capture_output=True,
        )
        assert executed_tool.returncode == 0, (tool, executed_tool.stderr)
    for index, body in enumerate(re.findall(r"```bash\n(.*?)\n```", runbook, re.DOTALL)):
        snippet = tmp_path / f"runbook-{index}.sh"
        snippet.write_bytes(body.encode("utf-8"))
        parsed = subprocess.run(
            ["bash", "-n", bash_path(snippet)], text=True, capture_output=True
        )
        assert parsed.returncode == 0, (index, parsed.stderr)


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
