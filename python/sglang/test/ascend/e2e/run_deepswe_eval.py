"""Run an evalscope DeepSWE evaluation inside the x86 eval pod.

Launched by ``run_deepswe_eval.sh`` in the ``eval`` container of the amd64
pod, next to the ``docker:dind`` sidecar. This module deliberately avoids
importing ``sglang``: the pod has no NPU runtime, and only needs the isolated
evalscope venv plus the docker CLI.

Config is passed in through environment variables by
``test_npu_accuracy_utils.run_evalscope_on_x86_pod``:

* ``DEEPSWE_API_URL``   router base URL, e.g. ``http://<node-ip>:6677/v1``
* ``DEEPSWE_MODEL``     model name handed to evalscope (ignored by SGLang)
* ``DEEPSWE_DATASETS``  comma separated dataset list (default ``deep_swe``)
* ``DEEPSWE_DATASET_ARGS``  JSON blob forwarded as evalscope ``dataset_args``
* ``DEEPSWE_LIMIT``     number of tasks to evaluate
* ``DEEPSWE_TIMEOUT``   dataset-level timeout forwarded to evalscope
* ``DEEPSWE_API_KEY``   API key sent to the router (``EMPTY``)
* ``METRICS_DATA_FILE`` shared-PVC directory for the evalscope report

On success the last line printed is ``[DEEPSWE_METRICS] {"accuracy": <float>}``,
which the router pod forwards to the CI pod-log monitor.
"""

import glob
import json
import os
import re
import subprocess
import sys

# The eval image may bundle torch_npu, but the amd64 eval node has no Ascend
# driver; torch auto-loading it crashes evalscope. DeepSWE talks to the router
# over HTTP and needs no NPU, so disable device-backend auto-loading. Belt and
# braces with the pod env set by test_npu_accuracy_utils.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
# The dind daemon is old (no BuildKit) and can't reach Docker Hub; force the
# classic image builder so `docker compose build` does not try to pull
# moby/buildkit.
os.environ.setdefault("DOCKER_BUILDKIT", "0")
os.environ.setdefault("COMPOSE_BAKE", "false")

E2E_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = "/tmp/deepswe_eval"
VENV_PYTHON = os.path.join(WORK_DIR, "test_env_evalscope", "bin", "python")

METRICS_MARKER = "[DEEPSWE_METRICS]"
PIP_MIRROR = os.environ.get(
    "EVALSCOPE_PIP_MIRROR", "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
)

# Agent-style reports name the binary metric ``acc``; older schemas use
# ``accuracy`` or the top-level ``score``.
ACCURACY_METRIC_NAMES = ("accuracy", "acc")


def install_eval_env():
    """Create/refresh the isolated evalscope + pier venv (idempotent)."""
    os.makedirs(WORK_DIR, exist_ok=True)
    env = os.environ.copy()
    env["EVALSCOPE_WITH_PIER"] = "1"
    env["EVALSCOPE_PIP_MIRROR"] = PIP_MIRROR
    script = os.path.join(E2E_DIR, "run_evalscope.sh")
    subprocess.run(["bash", script], cwd=WORK_DIR, env=env, check=True)


# pier's mini-swe-agent install step fetches the uv binary from GitHub releases,
# which this network cannot reach. Swap that download for the OBS mirror and
# point uv at a reachable PyPI index (via ~/.local/bin/env, which pier sources
# right after).
UV_TARBALL_URL = os.environ.get(
    "DEEPSWE_UV_TARBALL_URL",
    "https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com/ci/tools/uv-x86_64-unknown-linux-gnu.tar.gz",
)
UV_INDEX_URL = os.environ.get("DEEPSWE_UV_INDEX_URL", PIP_MIRROR)
_UV_INSTALL_PATTERN = re.compile(
    r"curl\s+-LsSf\s+https://astral\.sh/uv/\S*install\.sh\s*\|\s*sh"
)
_UV_INSTALL_REPLACEMENT = (
    "mkdir -p $HOME/.local/bin && "
    f"curl -fsSL {UV_TARBALL_URL} -o /tmp/uv.tgz && "
    "tar -xzf /tmp/uv.tgz -C $HOME/.local/bin --strip-components=1 && "
    "printf 'export PATH=$HOME/.local/bin:$PATH\\n"
    f"export UV_DEFAULT_INDEX={UV_INDEX_URL}\\n"
    f"export UV_INDEX_URL={UV_INDEX_URL}\\n' > $HOME/.local/bin/env"
)
_PATCH_SUFFIXES = (
    ".py",
    ".sh",
    ".dockerfile",
    ".j2",
    ".jinja",
    ".yaml",
    ".yml",
    ".tmpl",
)


def patch_agent_install():
    """Replace pier's GitHub uv download with the OBS mirror (idempotent)."""
    site_packages = glob.glob(
        os.path.join(WORK_DIR, "test_env_evalscope", "lib", "python*", "site-packages")
    )
    patched = []
    for sp in site_packages:
        for package in ("pier", "evalscope"):
            for dirpath, _, filenames in os.walk(os.path.join(sp, package)):
                for name in filenames:
                    if not name.endswith(_PATCH_SUFFIXES):
                        continue
                    path = os.path.join(dirpath, name)
                    try:
                        with open(path, encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        new_text = _UV_INSTALL_PATTERN.sub(_UV_INSTALL_REPLACEMENT, text)
                        if new_text == text:
                            continue
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(new_text)
                        patched.append(path)
                    except OSError as e:  # noqa: BLE001 - diagnostics only
                        print(f"WARNING: failed to patch {path}: {e}")
    print(f"[env] uv install patched in {len(patched)} file(s): {patched}")


def log_eval_env():
    """Print the interpreter and key package versions for triage."""
    print(f"[env] python: {VENV_PYTHON}")
    for module in ("evalscope", "pier"):
        try:
            result = subprocess.run(
                [
                    VENV_PYTHON,
                    "-c",
                    f"import {module}; print(getattr({module}, '__version__', 'unknown'))",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            lines = (result.stdout or result.stderr).strip().splitlines()
            print(f"[env] {module}: {lines[0] if lines else 'unknown'}")
        except Exception as e:  # noqa: BLE001 - diagnostics only
            print(f"[env] {module} version check failed: {e}")


def build_config():
    metrics_dir = os.environ.get("METRICS_DATA_FILE", WORK_DIR)
    config = {
        "model": os.environ.get("DEEPSWE_MODEL", "deepswe"),
        "api_url": os.environ["DEEPSWE_API_URL"],
        "api_key": os.environ.get("DEEPSWE_API_KEY", "EMPTY"),
        "eval_type": "openai_api",
        "datasets": [
            name for name in os.environ.get("DEEPSWE_DATASETS", "deep_swe").split(",")
            if name
        ],
        # One Docker environment per task.
        "eval_batch_size": 1,
        "limit": int(os.environ.get("DEEPSWE_LIMIT", "10")),
        "timeout": int(os.environ.get("DEEPSWE_TIMEOUT", "7200")),
        "work_dir": os.path.join(metrics_dir, "deepswe_evalscope"),
    }
    dataset_args = os.environ.get("DEEPSWE_DATASET_ARGS")
    if dataset_args:
        config["dataset_args"] = json.loads(dataset_args)
    return config


# DeepSWE tasks set ``allow_internet = false``. With that, pier runs the agent
# container on an internal Docker network whose only egress is a squid proxy
# limited to ports 80/443 and ``dstdomain`` matches. Our router is
# ``http://<pod-ip>:6677/v1`` (a bare IP on a non-standard port), so squid
# rejects every request and no inference traffic ever reaches the router. The
# dedicated eval pod already restricts egress at cluster level, so force public
# networking for these trials instead of silently hanging inside the proxy.
_NETWORK_PATCH = (
    "try:\n"
    "    from pier.environments.docker.docker import DockerEnvironment\n"
    "    _orig_start = DockerEnvironment.start\n"
    "    async def _start(self, force_build):\n"
    "        self.task_env_config.allow_internet = True\n"
    "        return await _orig_start(self, force_build)\n"
    "    DockerEnvironment.start = _start\n"
    "    print('[env] pier network policy patched: allow_internet=True')\n"
    "except Exception as exc:\n"
    "    raise RuntimeError(f'failed to patch pier network policy: {exc}')\n"
)


def write_runner(config):
    config_json = json.dumps(config, ensure_ascii=False, indent=2)
    escaped = config_json.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")
    script = (
        "import json\n"
        "from evalscope import TaskConfig, run_task\n\n"
        f"{_NETWORK_PATCH}\n"
        f"config = json.loads('''{escaped}''')\n"
        "run_task(task_cfg=TaskConfig(**config))\n"
    )
    path = os.path.join(WORK_DIR, "run_deepswe.py")
    with open(path, "w") as f:
        f.write(script)
    return path


def extract_accuracy(report):
    score = report.get("score")
    if isinstance(score, (int, float)):
        return float(score)

    primary_identity = report.get("primary_metric_identity")
    primary = (
        primary_identity.get("name") if isinstance(primary_identity, dict) else None
    )
    metrics = report.get("metrics")
    if not isinstance(metrics, list):
        metrics = []
    for name in (primary, *ACCURACY_METRIC_NAMES):
        if not name:
            continue
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            identity = metric.get("identity")
            if not isinstance(identity, dict) or identity.get("name") != name:
                continue
            value = metric.get("score")
            if isinstance(value, (int, float)):
                return float(value)
    return None


def run_evalscope(config):
    runner = write_runner(config)
    process = subprocess.Popen(
        [VENV_PYTHON, runner],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        # Keep any relative paths evalscope writes (caches, report links) under
        # a writable directory instead of the image's default WORKDIR.
        cwd=WORK_DIR,
    )
    output_lines = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
        output_lines.append(line.rstrip("\n"))
    process.wait()
    return process.returncode, output_lines


def find_report_path(output_lines, work_dir):
    """Locate the evalscope report.

    Prefer the "Dump report to:" line; otherwise fall back to the newest
    ``report.json`` under ``<work_dir>/reports``.
    """
    report_path = None
    for line in output_lines:
        match = re.search(r"Dump report to:\s*(\S+)", line)
        if match:
            report_path = match.group(1)
    if report_path and os.path.isfile(report_path):
        return report_path

    candidates = glob.glob(os.path.join(work_dir, "reports", "**", "report.json"), recursive=True)
    if candidates:
        return max(candidates, key=os.path.getmtime)
    return None


def main():
    install_eval_env()
    patch_agent_install()
    log_eval_env()
    config = build_config()
    print(f"DeepSWE evalscope config: {json.dumps(config, ensure_ascii=False)}")

    returncode, output_lines = run_evalscope(config)
    if returncode != 0:
        print(f"ERROR: DeepSWE evalscope exited with code {returncode}")
        return returncode or 1

    report_path = find_report_path(output_lines, config["work_dir"])
    print(f"[env] evalscope report: {report_path}")
    accuracy = None
    if report_path and os.path.isfile(report_path):
        try:
            with open(report_path) as f:
                accuracy = extract_accuracy(json.load(f))
        except Exception as e:  # noqa: BLE001 - report schema is external
            print(f"WARNING: failed to parse report {report_path}: {e}")

    if accuracy is None:
        print("ERROR: could not find the DeepSWE accuracy in the evalscope report")
        return 1

    print(f"DeepSWE accuracy: {accuracy}")
    print(f"{METRICS_MARKER} {json.dumps({'accuracy': accuracy})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
