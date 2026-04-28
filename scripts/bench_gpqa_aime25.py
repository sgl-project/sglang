# This script should be used inside the container. Before testing anything, please
# 1. install typer
# 2. set the following environment variables:
# - HOST: the host to connect to (default 127.0.0.1)
# - PORT: the port to connect to (default 30010)
# - HF_TOKEN: needed for `setup-ns`
# 3. checkout to the commit you want to test

# Caution!!!
# This script assumes that thinking mode can be controlled from SGLang side. (with an environ or argument)
# e.g. ++chat_template_kwargs.thinking=true is not included in the nemo skills command

# Test GPQA:
#    python bench_gpqa_aime25.py setup-ns
#    python bench_gpqa_aime25.py run-gpqa --num-repeats 16 --temperature 1.0 --max-tokens 400000 --max-concurrency 512

# Test AIME25:
#    python bench_gpqa_aime25.py setup-ns
#    python bench_gpqa_aime25.py run-aime25 --num-repeats 16 --temperature 1.0 --max-tokens 400000 --max-concurrency 512\
#    python bench_gpqa_aime25.py regrade-aime25 <log_folder> # Post process that bypasses box limitation


import os
import random
import subprocess
import time
from typing import Annotated

import typer

app = typer.Typer()

# Some manually set configs:
MODEL_PATH = "deepseek-ai/DeepSeek-V4-Pro"
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "30000"))
LOG_DIR = "/sgl-workspace/logs"

NS_VENV = "/sgl-workspace/ns-venv"

info_msg = f"""
Using configurations:
MODEL_PATH: {MODEL_PATH}
HOST: {HOST}
PORT: {PORT}
LOG_DIR: {LOG_DIR}
"""

# input(info_msg + "\nPress Enter to continue...")
print(info_msg)


def _venv_cmd(venv_dir: str, cmd: str) -> str:
    return f"source {venv_dir}/bin/activate && {cmd}"


def get_timestamp():
    return time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))


def get_random_int():
    return random.randint(0, 10000)


@app.command()
def setup_ns():
    HF_TOKEN = os.getenv("HF_TOKEN", None)
    if HF_TOKEN is None:
        raise ValueError("Please set HF_TOKEN for nemo skill setup")
    exec_command(f"uv venv {NS_VENV}")
    exec_command(
        _venv_cmd(
            NS_VENV,
            "uv pip install git+https://github.com/NVIDIA-NeMo/Skills.git@d77caab 'tree_sitter_language_pack<1.0' --reinstall-package blinker",
        )
    )
    exec_command(_venv_cmd(NS_VENV, f"HF_TOKEN={HF_TOKEN} ns prepare_data aime25"))
    # User might be asked for access of GPQA dataset. Just click in the hugging face website to grant access.
    exec_command(
        _venv_cmd(NS_VENV, f"HF_TOKEN={HF_TOKEN} ns prepare_data gpqa --split diamond")
    )


@app.command()
def run_gpqa(
    num_repeats: Annotated[int, typer.Option()] = 16,
    temperature: Annotated[float, typer.Option()] = 1.0,
    max_tokens: Annotated[int, typer.Option()] = 60000,
    max_concurrency: Annotated[int, typer.Option()] = 64,
):
    if not os.path.exists(f"{LOG_DIR}/gpqa_logs"):
        exec_command(f"mkdir -p {LOG_DIR}/gpqa_logs")

    random_seed = get_random_int()
    gpqa_log_folder = f"{LOG_DIR}/gpqa_logs/{get_timestamp()}_{random_seed}"
    exec_command(f"mkdir -p {gpqa_log_folder}")
    print(f"Running GPQA, log folder: {gpqa_log_folder}")

    exec_command(
        _venv_cmd(
            NS_VENV,
            f"nohup ns eval "
            f"--server_type=openai "
            f"--model={MODEL_PATH} "
            f"--server_address=http://{HOST}:{PORT}/v1 "
            f"--benchmarks=gpqa:{num_repeats} "
            f"--output_dir={gpqa_log_folder} "
            f"++inference.tokens_to_generate={max_tokens} "
            f"++max_concurrent_requests={max_concurrency} "
            f"++inference.temperature={temperature} "
            f"++inference.top_p=1.0 "
            f"++inference.timeout=25000000 "
            f"--starting_seed {random_seed} "
            f"> {gpqa_log_folder}/output.log 2>&1 &",
        )
    )


@app.command()
def run_aime25(
    num_repeats: Annotated[int, typer.Option()] = 16,
    temperature: Annotated[float, typer.Option()] = 1.0,
    max_tokens: Annotated[int, typer.Option()] = 60000,
    max_concurrency: Annotated[int, typer.Option()] = 64,
):
    if not os.path.exists(f"{LOG_DIR}/aime25_logs"):
        exec_command(f"mkdir -p {LOG_DIR}/aime25_logs")

    random_seed = get_random_int()
    aime25_log_folder = f"{LOG_DIR}/aime25_logs/{get_timestamp()}_{random_seed}"
    exec_command(f"mkdir -p {aime25_log_folder}")
    print(f"Running AIME25, log folder: {aime25_log_folder}")

    exec_command(
        _venv_cmd(
            NS_VENV,
            f"nohup ns eval "
            f"--server_type=openai "
            f"--model={MODEL_PATH} "
            f"--server_address=http://{HOST}:{PORT}/v1 "
            f"--benchmarks=aime25:{num_repeats} "
            f"--output_dir={aime25_log_folder} "
            f"++inference.tokens_to_generate={max_tokens} "
            f"++max_concurrent_requests={max_concurrency} "
            f"++inference.temperature={temperature} "
            f"++inference.top_p=1.0 "
            f"++inference.timeout=25000000 "
            f"--starting_seed {random_seed} "
            f"> {aime25_log_folder}/output.log 2>&1 &",
        )
    )


@app.command()
def exec_command(cmd: str, capture_output: bool = False) -> str | None:
    print(f"EXEC: {cmd}", flush=True)
    return subprocess.run(
        ["bash", "-c", cmd],
        shell=False,
        check=True,
        capture_output=capture_output,
        **(dict(text=True) if capture_output else {}),
    )


# ---------------------------------------------------------------------------
# Post-eval relaxed grade for AIME25
# ---------------------------------------------------------------------------
# Why this exists: nemo-skills' default extractor uses `\boxed{}` only, and the
# DeepSeek-V4-Pro generations sometimes finish with prose like
# "**Answer:** 336" or "the final answer is 821" instead of `\boxed{}`. Those
# get scored as no-answer. We can't pass a relaxed regex via CLI because
# nemo-run rebuilds the inner shell command and unquotes / strips backslashes,
# so the parens land in bash bare and break it. Instead, run this command
# *after* `run-aime25` finishes to re-extract via a fallback regex *only* when
# the boxed extractor returned None, then regenerate metrics.json via
# `ns summarize_results`.
#
# Usage:
#    python bench_gpqa_aime25.py regrade-aime25 <log_folder>
#    # e.g. /sgl-workspace/logs/aime25_logs/20260427005929_3235


@app.command()
def regrade_aime25(
    log_folder: Annotated[
        str,
        typer.Argument(help="The aime25_logs/<timestamp>_<seed> dir from run-aime25."),
    ],
):
    import glob
    import json
    import sys

    sys.path.insert(0, f"{NS_VENV}/lib/python3.12/site-packages")
    from nemo_skills.evaluation.math_grader import extract_answer, math_equal

    FALLBACK_REGEX = (
        r"(?:\*\*Answer\*\*[^0-9\-]{0,30}"
        r"|(?i:final answer)[^0-9\-]{0,30}"
        r"|(?i:answer)\s*(?:is|=|:)[^0-9\-]{0,30})(-?\d+)"
    )

    eval_dir = f"{log_folder}/eval-results/aime25"
    files = sorted(glob.glob(f"{eval_dir}/output-rs*.jsonl"))
    if not files:
        raise typer.Exit(f"No output-rs*.jsonl files in {eval_dir}")
    print(f"Re-extracted {len(files)} files in {eval_dir}")

    total = recovered = 0
    for f in files:
        lines_out = []
        changed = False
        with open(f) as fp:
            for line in fp:
                r = json.loads(line)
                total += 1
                if r.get("predicted_answer") is None:
                    new_pred = extract_answer(
                        r["generation"],
                        extract_from_boxed=False,
                        extract_regex=FALLBACK_REGEX,
                    )
                    if new_pred is not None:
                        r["predicted_answer"] = new_pred
                        r["symbolic_correct"] = bool(
                            math_equal(r["expected_answer"], new_pred)
                        )
                        recovered += 1
                        changed = True
                lines_out.append(json.dumps(r))
        if changed:
            with open(f, "w") as fp:
                fp.write("\n".join(lines_out) + "\n")
    print(f"Re-extracted {recovered} / {total} previously-no-answer records.")

    # Regenerate metrics.json via ns summarize_results (runs inside the venv).
    exec_command(_venv_cmd(NS_VENV, f"ns summarize_results {log_folder}"))
    print(f"Updated metrics: {eval_dir}/metrics.json")


if __name__ == "__main__":
    app()
