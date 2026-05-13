"""branch_compare CLI: record / verify phases.

Usage (run from sglang-source/test/manual/):

    SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \\
    python -m branch_compare.run \\
        --mode record --artifact-dir DIR \\
        --model-path MODEL [--tp-size N] [...any server args...] \\
        --eval-name gpqa --num-examples 16

    SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \\
    python -m branch_compare.run \\
        --mode verify --artifact-dir DIR --record-dir RECORD_DIR \\
        --model-path MODEL [--tp-size N] [...any server args...]

The script registers ServerArgs.add_cli_args so every server flag is
accepted as pass-through and forwarded to popen_launch_server. If
--base-url is set the script attaches to a running server instead of
launching one (server args become advisory and are recorded in meta.json).
"""

from __future__ import annotations

import argparse
import datetime
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import requests
from branch_compare import artifacts
from branch_compare import prompts as prompts_mod

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_server

DEFAULT_PORT = 30000
TIMEOUT_LAUNCH = 600
TIMEOUT_REQUEST = 3600
LARGE_ARTIFACT_GB = 5.0


# --------------------------- argparse ---------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="branch_compare: record / verify a server's per-step "
        "logprobs for divergence comparison."
    )

    # Mode + I/O
    g = parser.add_argument_group("branch_compare")
    g.add_argument("--mode", choices=["record", "verify"], required=True)
    g.add_argument("--artifact-dir", required=True)
    g.add_argument(
        "--record-dir",
        default=None,
        help="(verify mode) Directory containing main's record artifact.",
    )
    g.add_argument(
        "--base-url",
        default=None,
        help="Attach to a running server instead of launching one.",
    )

    # Prompts
    g.add_argument(
        "--eval-name",
        default=None,
        choices=[
            "mmlu",
            "gpqa",
            "math",
            "gsm8k",
            "humaneval",
            "mgsm",
            "aime25",
            "longbench_v2",
        ],
    )
    g.add_argument("--num-examples", type=int, default=16)
    g.add_argument(
        "--prompts-file",
        default=None,
        help="Alternative to --eval-name: JSONL file with {'prompt': '...'} per line.",
    )
    g.add_argument(
        "--api",
        choices=["chat", "completion"],
        default="chat",
        help="chat applies the chat template client-side; completion sends raw text.",
    )

    # Generation
    g.add_argument("--max-new-tokens", type=int, default=1024)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=None)
    g.add_argument("--min-p", type=float, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument(
        "--topk-logprobs",
        type=int,
        default=128,
        help="K for top_logprobs_num. PD-disagg caps at 128; raise if you're "
        "not in disagg mode.",
    )

    g.add_argument(
        "--ack-large-artifacts",
        action="store_true",
        help="Required when projected artifact size exceeds 5 GB.",
    )
    g.add_argument(
        "--num-threads",
        type=int,
        default=16,
        help="Concurrent in-flight requests to the server. The server's "
        "scheduler batches them internally, so higher values overlap "
        "prompts and dramatically reduce wall time on multi-prompt runs.",
    )

    # Server pass-through. Last so server's own --help text reads naturally.
    ServerArgs.add_cli_args(parser)
    return parser


# --------------------------- helpers ---------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _git_sha(cwd: str = ".") -> Tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True
        ).strip()
        return sha, bool(diff)
    except Exception:
        return "unknown", False


def _server_args_passthrough(args: argparse.Namespace) -> List[str]:
    """Reconstruct the CLI list to forward to popen_launch_server.

    Walk a fresh parser populated only by ServerArgs.add_cli_args and emit
    flags whose value on `args` differs from the parser default. We can't go
    through ServerArgs(...) construction because __post_init__ derives fields
    (model_config, random_seed, kt_*, ...) that don't map back to CLI flags.
    """
    sa_parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(sa_parser)

    out: List[str] = []
    for action in sa_parser._actions:
        dest = action.dest
        if dest in ("help", "model_path"):
            continue  # model_path is passed positionally to popen_launch_server
        if not action.option_strings:
            continue
        if not hasattr(args, dest):
            continue
        cur = getattr(args, dest)
        default = action.default
        if cur == default:
            continue
        cli = action.option_strings[0]
        if isinstance(cur, bool):
            if cur and not default:
                out.append(cli)
            # bool flags rarely have a paired --no-X; leave it alone otherwise
        elif isinstance(cur, list):
            for item in cur:
                out += [cli, str(item)]
        else:
            out += [cli, str(cur)]
    return out


def _project_artifact_size_gb(n_prompts: int, max_new_tokens: int, topk: int) -> float:
    # bf16 logprobs (2B) + int32 token_ids (4B) per (step, k), plus int64 chosen_logprob (negligible)
    bytes_per = (2 + 4) * topk
    total = n_prompts * max_new_tokens * bytes_per
    return total / (1024**3)


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# --------------------------- HTTP ---------------------------


def _request(
    base_url: str,
    text: str,
    sampling_params: Dict[str, Any],
    topk_logprobs: int,
) -> Dict[str, Any]:
    payload = {
        "text": text,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": topk_logprobs,
        "return_text_in_logprobs": False,
        "stream": False,
    }
    r = requests.post(f"{base_url}/generate", json=payload, timeout=TIMEOUT_REQUEST)
    r.raise_for_status()
    return r.json()


def _parse_response(
    resp: Dict[str, Any],
) -> Tuple[List[int], List[float], List[List[int]], List[List[float]], str]:
    meta = resp["meta_info"]
    output_token_logprobs = meta["output_token_logprobs"]  # [(lp, tid, None), ...]
    output_top_logprobs = meta["output_top_logprobs"]  # [[(lp, tid, None), ... K], ...]

    output_ids: List[int] = []
    chosen_logprob: List[float] = []
    for lp, tid, _ in output_token_logprobs:
        output_ids.append(int(tid))
        chosen_logprob.append(float(lp))

    top_k_token_ids: List[List[int]] = []
    top_k_logprobs: List[List[float]] = []
    for step in output_top_logprobs:
        top_k_token_ids.append([int(t[1]) for t in step])
        top_k_logprobs.append([float(t[0]) for t in step])

    finish_reason = (meta.get("finish_reason") or {}).get("type", "unknown")
    return output_ids, chosen_logprob, top_k_token_ids, top_k_logprobs, finish_reason


# --------------------------- record / verify ---------------------------


def _build_sampling_params(
    args: argparse.Namespace, max_new: int, forced_path: Optional[str] = None
) -> Dict[str, Any]:
    sp: Dict[str, Any] = {
        "max_new_tokens": max_new,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.top_k is not None:
        sp["top_k"] = args.top_k
    if args.min_p is not None:
        sp["min_p"] = args.min_p
    if args.seed is not None:
        sp["sampling_seed"] = args.seed
    if forced_path is not None:
        sp["forced_token_ids_path"] = forced_path
    return sp


def cmd_record(args: argparse.Namespace, base_url: str) -> None:
    tokenizer = _load_tokenizer(args.model_path) if args.api == "chat" else None
    prompts_list = prompts_mod.load_prompts(
        eval_name=args.eval_name,
        num_examples=args.num_examples,
        prompts_file=args.prompts_file,
        api=args.api,
        tokenizer=tokenizer,
    )
    n = len(prompts_list)
    proj_gb = _project_artifact_size_gb(n, args.max_new_tokens, args.topk_logprobs)
    print(
        f"[branch_compare] {n} prompts, projected artifact size ≤ {proj_gb:.2f} GB",
        flush=True,
    )
    if proj_gb > LARGE_ARTIFACT_GB and not args.ack_large_artifacts:
        raise SystemExit(
            f"Projected artifact size ({proj_gb:.2f} GB) exceeds "
            f"{LARGE_ARTIFACT_GB} GB. Re-run with --ack-large-artifacts to confirm."
        )

    os.makedirs(args.artifact_dir, exist_ok=True)
    sha, dirty = _git_sha()
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    server_args_list = _server_args_passthrough(args)

    def _record_one(i_p: Tuple[int, Any]) -> Dict[str, Any]:
        i, p = i_p
        sp = _build_sampling_params(args, args.max_new_tokens)
        t0 = time.time()
        resp = _request(base_url, p.text, sp, args.topk_logprobs)
        out_ids, chosen_lp, top_ids, top_lps, finish = _parse_response(resp)
        dt = time.time() - t0
        artifacts.write_prompt_artifact(
            args.artifact_dir, i, out_ids, top_ids, top_lps, chosen_lp
        )
        print(
            f"  [{i+1}/{n}] {len(out_ids)} steps, finish={finish}, {dt:.1f}s",
            flush=True,
        )
        return {
            "idx": i,
            "prompt": p.text,
            "messages": p.messages,
            "logprobs_file": artifacts.prompt_filename(i),
            "n_steps": len(out_ids),
            "stop_reason": finish,
        }

    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        prompt_records = list(ex.map(_record_one, enumerate(prompts_list)))

    meta = {
        "phase": "record",
        "git_commit": sha,
        "git_dirty": dirty,
        "model_path": args.model_path,
        "server_args": server_args_list,
        "server_mode": "attached" if args.base_url else "launched",
        "topk_logprobs": args.topk_logprobs,
        "max_new_tokens": args.max_new_tokens,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "seed": args.seed,
        },
        "eval": {
            "name": args.eval_name,
            "num_examples": args.num_examples,
            "api": args.api,
            "prompts_file": args.prompts_file,
        },
        "timestamp": timestamp,
        "prompts": prompt_records,
    }
    artifacts.write_meta(args.artifact_dir, meta)
    print(f"[branch_compare] record artifact written to {args.artifact_dir}")


def cmd_verify(args: argparse.Namespace, base_url: str) -> None:
    if args.record_dir is None:
        raise SystemExit("--record-dir is required for --mode verify")
    record_meta = artifacts.read_meta(args.record_dir)
    record_dir_abs = os.path.abspath(args.record_dir)
    n = len(record_meta["prompts"])

    proj_gb = _project_artifact_size_gb(n, args.max_new_tokens, args.topk_logprobs)
    print(
        f"[branch_compare] {n} prompts, projected artifact size ≤ {proj_gb:.2f} GB",
        flush=True,
    )
    if proj_gb > LARGE_ARTIFACT_GB and not args.ack_large_artifacts:
        raise SystemExit(
            f"Projected artifact size ({proj_gb:.2f} GB) exceeds "
            f"{LARGE_ARTIFACT_GB} GB. Re-run with --ack-large-artifacts to confirm."
        )

    os.makedirs(args.artifact_dir, exist_ok=True)
    sha, dirty = _git_sha()
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    server_args_list = _server_args_passthrough(args)

    def _verify_one(
        entry: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        idx = entry["idx"]
        text = entry["prompt"]
        n_steps_record = entry["n_steps"]
        forced_path = artifacts.absolute_logprobs_path(record_dir_abs, idx)
        sp = _build_sampling_params(
            args, max_new=n_steps_record, forced_path=forced_path
        )
        t0 = time.time()
        resp = _request(base_url, text, sp, args.topk_logprobs)
        out_ids, chosen_lp, top_ids, top_lps, finish = _parse_response(resp)
        dt = time.time() - t0
        artifacts.write_prompt_artifact(
            args.artifact_dir, idx, out_ids, top_ids, top_lps, chosen_lp
        )

        # Sanity: forcing must actually have taken.
        record_artifact = artifacts.read_prompt_artifact(record_dir_abs, idx)
        record_out_ids = record_artifact["output_ids"].tolist()
        mismatch: Optional[Dict[str, Any]] = None
        if out_ids != record_out_ids:
            mismatch_step = next(
                (k for k, (a, b) in enumerate(zip(out_ids, record_out_ids)) if a != b),
                min(len(out_ids), len(record_out_ids)),
            )
            mismatch = {"prompt_idx": idx, "first_mismatch_step": mismatch_step}
            print(
                f"  [{idx+1}/{n}] FORCED-TOKEN MISMATCH at step "
                f"{mismatch_step}: forcing did not take.",
                flush=True,
            )
        else:
            print(
                f"  [{idx+1}/{n}] {len(out_ids)} steps OK, finish={finish}, "
                f"{dt:.1f}s",
                flush=True,
            )

        record = {
            "idx": idx,
            "prompt": text,
            "messages": entry.get("messages"),
            "logprobs_file": artifacts.prompt_filename(idx),
            "n_steps": len(out_ids),
            "stop_reason": finish,
        }
        return record, mismatch

    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        results = list(ex.map(_verify_one, record_meta["prompts"]))
    prompt_records = [r for r, _ in results]
    failed_steps = [m for _, m in results if m is not None]

    meta = {
        "phase": "verify",
        "git_commit": sha,
        "git_dirty": dirty,
        "model_path": args.model_path,
        "server_args": server_args_list,
        "server_mode": "attached" if args.base_url else "launched",
        "topk_logprobs": args.topk_logprobs,
        "max_new_tokens": args.max_new_tokens,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "seed": args.seed,
        },
        "record_dir": record_dir_abs,
        "record_git_commit": record_meta.get("git_commit"),
        "timestamp": timestamp,
        "prompts": prompt_records,
        "failed_steps": failed_steps,
    }
    artifacts.write_meta(args.artifact_dir, meta)
    print(
        f"[branch_compare] verify artifact written to {args.artifact_dir}; "
        f"{len(failed_steps)} prompts had forced-token mismatches."
    )


# --------------------------- main ---------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.model_path:
        raise SystemExit("--model-path is required")

    # Establish base_url and (if needed) launch the server.
    proc = None
    if args.base_url:
        base_url = args.base_url.rstrip("/")
    else:
        port = args.port or _free_port() or DEFAULT_PORT
        base_url = f"http://127.0.0.1:{port}"
        # Pass through every server arg the user set (excluding our own group).
        server_other_args = _server_args_passthrough(args)
        # popen_launch_server passes the model as a positional + appends --port,
        # so strip those if they snuck in.
        server_other_args = [a for a in server_other_args if a not in ("--model-path",)]
        if "--port" in server_other_args:
            i = server_other_args.index("--port")
            del server_other_args[i : i + 2]
        env = os.environ.copy()
        # Surface the env-var requirement so users who forget get a clear error.
        if env.get("SGLANG_ENABLE_FORCED_TOKEN_IDS") != "1":
            print(
                "[branch_compare] WARNING: SGLANG_ENABLE_FORCED_TOKEN_IDS is "
                "not set in the environment. The server will reject "
                "forced_token_ids requests on the verify phase.",
                file=sys.stderr,
            )
        proc = popen_launch_server(
            args.model_path,
            base_url,
            timeout=TIMEOUT_LAUNCH,
            other_args=server_other_args,
            env=env,
        )
        print(f"[branch_compare] server launched at {base_url}", flush=True)

    try:
        if args.mode == "record":
            cmd_record(args, base_url)
        elif args.mode == "verify":
            cmd_verify(args, base_url)
        else:
            raise SystemExit(f"Unknown mode: {args.mode}")
    finally:
        if proc is not None:
            kill_process_tree(proc.pid)


if __name__ == "__main__":
    main()
