#!/usr/bin/env python3
"""Analyze srtslurm logs with opencode inside a Modal sandbox.

This script accepts either:
- a local log directory
- a `.tar.gz` bundle such as `multinode_server_logs.tar.gz`

It uploads the logs into an ephemeral Modal sandbox, installs and runs
opencode with an analysis prompt, and prints the resulting markdown.

Example:
  uv run --with modal python scripts/ci/slurm/analyze_logs_with_modal.py \
    --tarball /tmp/multinode_server_logs.tar.gz \
    --job-id 4645
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path

try:
    import modal
except ImportError:  # pragma: no cover - runtime guard for local usage
    modal = None


logger = logging.getLogger("slurm_log_analysis")

SANDBOX_TIMEOUT = 600
DEFAULT_MODAL_SECRET_NAME = "or"
DEFAULT_MODEL = "openrouter/minimax/minimax-m2.7"
DEFAULT_REPOS = [
    "https://github.com/sgl-project/sglang.git",
]
PROMPT_PATH = Path(__file__).with_name("log_analysis_prompt.md")

# Matches common API key / token patterns (sk-..., ak-..., as-..., key-..., etc.)
_SECRET_PATTERN = re.compile(
    r"""(?:"""
    r"""(?:sk|ak|as|key|token|secret|bearer)[-_][A-Za-z0-9_\-]{16,}"""
    r"""|"""
    r"""(?:OPENROUTER_API_KEY|MODAL_TOKEN_ID|MODAL_TOKEN_SECRET|ANTHROPIC_API_KEY)"""
    r"""[=:]\s*\S+"""
    r""")""",
    re.IGNORECASE,
)


def sanitize(text: str) -> str:
    """Redact strings that look like API keys or secrets."""
    if not text:
        return text
    sanitized = _SECRET_PATTERN.sub("[REDACTED]", text)
    # Also redact any env var values we know are secrets
    for var in ("OPENROUTER_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        val = os.environ.get(var)
        if val and len(val) > 8:
            sanitized = sanitized.replace(val, "[REDACTED]")
    return sanitized


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def extract_tarball(tarball: Path, destination: Path) -> None:
    with tarfile.open(tarball, "r:gz") as archive:
        # Python 3.14 changes the default extraction behavior. Use the
        # data filter when available so extraction remains explicit.
        if "data" in tarfile._NAMED_FILTERS:  # type: ignore[attr-defined]
            archive.extractall(destination, filter="data")
        else:
            archive.extractall(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a srtslurm log bundle with opencode in Modal."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--tarball",
        type=Path,
        help="Path to a local multinode_server_logs.tar.gz bundle.",
    )
    source.add_argument(
        "--log-dir",
        type=Path,
        help="Path to an unpacked log directory.",
    )
    parser.add_argument(
        "--job-id",
        default="unknown",
        help="Job identifier used in the report header and logs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model selector to pass to opencode run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the markdown analysis.",
    )
    parser.add_argument(
        "--repo-url",
        action="append",
        dest="repo_urls",
        help=(
            "Optional extra repo URL to clone into the sandbox for context. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=SANDBOX_TIMEOUT,
        help="Sandbox lifetime in seconds.",
    )
    parser.add_argument(
        "--modal-secret-name",
        default=DEFAULT_MODAL_SECRET_NAME,
        help="Modal secret name that provides OPENROUTER_API_KEY to the sandbox.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def build_sandbox_image() -> "modal.Image":
    if modal is None:
        raise RuntimeError(
            "The 'modal' package is required. Run this script with "
            "`uv run --with modal python ...` or install modal locally."
        )

    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("bash", "curl", "git", "gh")
        .run_commands(
            "curl -fsSL https://opencode.ai/install | bash",
        )
        .env(
            {
                "PATH": "/root/.opencode/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            }
        )
    )


def prepare_log_dir(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.log_dir:
        if not args.log_dir.is_dir():
            raise FileNotFoundError(f"log directory not found: {args.log_dir}")
        return args.log_dir.resolve(), None

    assert args.tarball is not None
    if not args.tarball.is_file():
        raise FileNotFoundError(f"tarball not found: {args.tarball}")

    temp_dir = Path(tempfile.mkdtemp(prefix="sglang_logs_"))
    extract_tarball(args.tarball, temp_dir)
    return temp_dir.resolve(), temp_dir


def build_prompt(job_id: str, repo_urls: list[str]) -> str:
    skill_content = PROMPT_PATH.read_text()
    repo_lines = []
    for repo_url in repo_urls:
        repo_name = repo_url.rsplit("/", 1)[-1].removesuffix(".git")
        repo_lines.append(f"- **{repo_name} repo**: `/workspace/repos/{repo_name}/`")
    repo_section = (
        "\n".join(repo_lines) if repo_lines else "- No extra repos were requested."
    )

    return f"""{skill_content}

---

## Your Environment

- **Logs**: `/workspace/logs/`
- **GitHub CLI**: `gh` is installed and authenticated.
{repo_section}

## Job

You are analyzing job `{job_id}`. Follow Steps 1–5 in the prompt above.

**You MUST write the final markdown report to `/workspace/logs/ai_analysis.md`.**
This is a hard requirement. Do not just print the report to stdout. Use your
file-writing tool to create `/workspace/logs/ai_analysis.md` with the full
analysis. The downstream pipeline reads this file.

**You MUST file GitHub issues when the root cause is clear (Category A or B).**
Do not skip issue filing. The whole point of this system is automated triage.
"""


def upload_tree(sandbox: "modal.Sandbox", log_dir: Path) -> None:
    log_files = [path for path in log_dir.rglob("*") if path.is_file()]
    logger.info("Uploading %d log files into the sandbox", len(log_files))
    for index, log_file in enumerate(log_files, start=1):
        rel_path = log_file.relative_to(log_dir)
        remote_path = str(Path("/workspace/logs") / rel_path)
        sandbox.mkdir(str(Path(remote_path).parent), parents=True)
        sandbox.filesystem.write_bytes(log_file.read_bytes(), remote_path)
        if index % 10 == 0 or index == len(log_files):
            logger.info("Uploaded %d/%d files", index, len(log_files))


def clone_context_repos(sandbox: "modal.Sandbox", repo_urls: list[str]) -> None:
    if not repo_urls:
        return

    sandbox.mkdir("/workspace/repos", parents=True)

    for repo_url in repo_urls:
        repo_name = repo_url.rsplit("/", 1)[-1].removesuffix(".git")
        logger.info("Cloning %s into the sandbox", repo_name)
        sandbox.exec(
            "git",
            "clone",
            "--depth",
            "100",
            repo_url,
            f"/workspace/repos/{repo_name}",
        ).wait()


def read_optional_file(sandbox: "modal.Sandbox", path: str) -> str | None:
    try:
        return sandbox.filesystem.read_text(path)
    except Exception:
        return None


def run_opencode_analysis(
    *,
    log_dir: Path,
    job_id: str,
    model: str,
    repo_urls: list[str],
    timeout_seconds: int,
    modal_secret_name: str,
) -> str:
    prompt = build_prompt(job_id, repo_urls)
    app = modal.App.lookup("sglang-log-analyzer", create_if_missing=True)
    sandbox = modal.Sandbox.create(
        app=app,
        image=build_sandbox_image(),
        timeout=timeout_seconds,
        secrets=[modal.Secret.from_name(modal_secret_name)],
    )
    logger.info("Created Modal sandbox %s", sandbox.object_id)

    try:
        sandbox.mkdir("/workspace/logs", parents=True)
        sandbox.mkdir("/workspace/repos", parents=True)

        clone_context_repos(sandbox, repo_urls)
        upload_tree(sandbox, log_dir)

        sandbox.filesystem.write_text(prompt, "/workspace/prompt.txt")

        runner_script = f"""#!/bin/bash
set -uo pipefail
cd /workspace
if opencode run \\
  --dangerously-skip-permissions \\
  --dir /workspace/logs \\
  -m {shlex.quote(model)} \\
  "$(cat /workspace/prompt.txt)" \\
  < /dev/null \\
  > /workspace/logs/opencode.stdout \\
  2> /workspace/logs/opencode.stderr; then
  echo 0 > /workspace/logs/opencode.exitcode
else
  echo $? > /workspace/logs/opencode.exitcode
fi
ls -la /workspace/logs > /workspace/logs/log_dir_listing.txt
"""
        sandbox.filesystem.write_text(runner_script, "/workspace/run_opencode.sh")
        sandbox.exec("chmod", "+x", "/workspace/run_opencode.sh").wait()

        logger.info("Running opencode analysis")
        process = sandbox.exec(
            "bash",
            "/workspace/run_opencode.sh",
        )
        process.wait()

        stderr = process.stderr.read()
        if stderr:
            logger.warning("runner stderr: %s", sanitize(stderr[:500]))

        exitcode = read_optional_file(sandbox, "/workspace/logs/opencode.exitcode")
        opencode_stdout = (
            read_optional_file(sandbox, "/workspace/logs/opencode.stdout") or ""
        )
        opencode_stderr = read_optional_file(sandbox, "/workspace/logs/opencode.stderr")
        log_dir_listing = read_optional_file(
            sandbox, "/workspace/logs/log_dir_listing.txt"
        )
        try:
            ai_analysis = read_optional_file(sandbox, "/workspace/logs/ai_analysis.md")
            if ai_analysis and ai_analysis.strip():
                return sanitize(ai_analysis)

            if opencode_stdout.strip():
                return sanitize(opencode_stdout)

            raise RuntimeError("opencode completed without producing analysis output")
        except Exception as exc:
            stdout = process.stdout.read()
            if stdout:
                return sanitize(stdout)
            details = [
                f"opencode analysis did not produce a usable report: {exc}",
                f"exitcode={exitcode!r}",
                f"stdout_preview={sanitize(opencode_stdout[:500])!r}",
                f"stderr_preview={sanitize((opencode_stderr or '')[:500])!r}",
                f"log_dir_listing={sanitize((log_dir_listing or '')[:500])!r}",
            ]
            raise RuntimeError(" ".join(details)) from exc
    finally:
        sandbox.terminate()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    repo_urls = list(DEFAULT_REPOS)
    if args.repo_urls:
        repo_urls.extend(args.repo_urls)

    log_dir, cleanup_dir = prepare_log_dir(args)
    try:
        analysis = run_opencode_analysis(
            log_dir=log_dir,
            job_id=args.job_id,
            model=args.model,
            repo_urls=repo_urls,
            timeout_seconds=args.timeout_seconds,
            modal_secret_name=args.modal_secret_name,
        )
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    print(analysis)
    if args.output:
        args.output.write_text(analysis)
        logger.info("Wrote analysis to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
