"""Every flag used in our YAML configs must exist in the in-repo sglang source.

Pure text scan — sglang/torch are never imported, so this stays
hardware-free and CI-safe.  Server flags are dataclass-driven in sglang
0.5.x (``cuda_graph_bs_decode`` etc. under ``srt/arg_groups/``), bench
flags are literal ``add_argument`` strings in ``sglang/benchmark/serving.py``.
Skipped when this directory is relocated outside a sglang checkout.
"""

import re
from pathlib import Path

import pytest
import yaml

ASCEND_BENCH = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

SERVER_ARGS_DIRS = [REPO_ROOT / "python" / "sglang" / "srt" / "arg_groups"]
SERVER_ARGS_FILES = [REPO_ROOT / "python" / "sglang" / "srt" / "server_args.py"]
BENCH_SERVING = REPO_ROOT / "python" / "sglang" / "benchmark" / "serving.py"
GSM8K_SCRIPT = REPO_ROOT / "benchmark" / "gsm8k" / "bench_sglang.py"
GSM8K_COMMON_ARGS = REPO_ROOT / "python" / "sglang" / "test" / "test_utils.py"

MODEL_FLAGS = {"--model-path", "--dtype", "--quantization", "--trust-remote-code"}
RUNNER_BENCH_FLAGS = {
    "--backend",
    "--host",
    "--port",
    "--dataset-name",
    "--num-prompts",
    "--warmup-requests",
    "--seed",
    "--output-file",
    "--tag",
}
RUNNER_GSM8K_FLAGS = {
    "--host",
    "--port",
    "--num-questions",
    "--num-shots",
    "--data-path",
}


def _read_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _server_args_text() -> str:
    chunks = [_read_if_exists(p) for p in SERVER_ARGS_FILES]
    for directory in SERVER_ARGS_DIRS:
        if directory.is_dir():
            chunks.extend(
                p.read_text(encoding="utf-8", errors="replace")
                for p in directory.rglob("*.py")
            )
    return "\n".join(chunks)


def _config_flags() -> "dict[Path, tuple[list[str], list[str]]]":
    """Per config: (server_flags, workload_flags)."""
    out: dict[Path, tuple[list[str], list[str]]] = {}
    for path in sorted((ASCEND_BENCH / "configs").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        server_flags: list[str] = list(MODEL_FLAGS)

        def _collect(where: dict, bucket: list[str]) -> None:
            for key, value in (where or {}).items():
                bucket.append(key)
                if isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict):
                            bucket.extend(entry)

        _collect((data.get("server") or {}).get("args"), server_flags)
        _collect((data.get("server") or {}).get("axes"), server_flags)
        workload_flags: list[str] = []
        _collect((data.get("workload") or {}).get("args"), workload_flags)
        _collect((data.get("workload") or {}).get("axes"), workload_flags)
        out[path] = (server_flags, workload_flags)
    return out


@pytest.mark.skipif(
    not SERVER_ARGS_DIRS[0].is_dir(),
    reason="ascend_bench relocated outside a sglang checkout",
)
def test_config_flags_exist_in_sglang_source():
    server_text = _server_args_text()
    bench_text = _read_if_exists(BENCH_SERVING)
    gsm_text = _read_if_exists(GSM8K_SCRIPT) + _read_if_exists(GSM8K_COMMON_ARGS)

    missing: list[str] = []
    for path, (server_flags, workload_flags) in _config_flags().items():
        rel = path.relative_to(REPO_ROOT)
        for flag in server_flags:
            field = flag.lstrip("-").replace("-", "_")
            if not re_search(server_text, field):
                missing.append(f"{rel}: server flag {flag} not in sglang server args")
        for flag in workload_flags:
            if flag not in bench_text:
                missing.append(f"{rel}: workload flag {flag} not in bench_serving")

    for flag in RUNNER_BENCH_FLAGS:
        if flag not in bench_text:
            missing.append(f"runner bench flag {flag} not in bench_serving")
    if "sglang-oai" not in bench_text:
        missing.append("runner backend choice 'sglang-oai' not in bench_serving")
    for flag in RUNNER_GSM8K_FLAGS:
        if flag not in gsm_text:
            missing.append(f"runner gsm8k flag {flag} not in gsm8k bench sources")

    assert not missing, "unknown flags:\n" + "\n".join(missing)


def re_search(text: str, word: str) -> bool:
    return re.search(rf"\b{word}\b", text) is not None
