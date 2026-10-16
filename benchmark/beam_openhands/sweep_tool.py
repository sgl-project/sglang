"""Preflight checks and bookkeeping for the OpenHands sweep.

usage: sweep_tool.py client-check    --requirements FILE --out FILE
       sweep_tool.py tokenizer-view  --model-path DIR --view DIR [--revision REV] --out FILE
       sweep_tool.py dataset-check   --dataset FILE --tokenizer-manifest FILE --chat-template-kwargs JSON
                                     --need N [--render-sample K] --out FILE
       sweep_tool.py server-check    --base-url URL --profile FILE --model-path DIR --served-model-name NAME
                                     --n-gpus N [--cc N] --out FILE
       sweep_tool.py wait-idle       --base-url URL [--timeout-s S]
       sweep_tool.py sample-loads    --base-url URL --interval-s S --out FILE
       sweep_tool.py accept-run      --results-dir DIR --attempt-dir DIR --package-dir DIR --config JSON

Everything runs on the standard library except tokenizer-view and dataset-check --render-sample, which load the
tokenizer from the client venv. Exit status 1 means a check failed, and 3 means the run identity changed.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

EVALSCOPE_COMMIT = "acd09b44384d53174768bb1063f675420f76fae9"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
DATASET_FORMAT = "openhands-padded/v2"
# Files whose content changes what a rung measures or how it is judged; a resumed run must match them byte for byte.
IDENTITY_FILES = (
    "run_sweep.sh",
    "common.sh",
    "sweep_tool.py",
    "summarize_rung.py",
    "evalscope_client.py",
    "client-requirements.txt",
    "server_profile.json",
    "curve_config.py",
    "launch_server.py",
    "guarded_node.py",
    "run_curve.py",
)
# /server_info keys that change while one server instance runs (live scheduler state) or per boot without a
# config change (startup timings). Every other top-level key is part of the resume identity.
VOLATILE_SERVER_KEYS = ("internal_states", "startup_time")
# Preflight outputs that describe an accepted run; a refused resume never replaces them.
RUN_FILES = (
    "client-manifest.json",
    "tokenizer-view.json",
    "dataset-stats.json",
    "server-info-start.json",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def special_token_pattern(tokenizer_dir):
    """Declared special tokens and Beam protocol controls that cannot be replayed as plain text."""
    with open(os.path.join(tokenizer_dir, "tokenizer.json")) as f:
        added = json.load(f).get("added_tokens", [])
    # Released Beam checkpoints declare these protocol controls as ordinary
    # added tokens. They must still be removed from plain-text replay history.
    harmony_controls = {
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|channel|>",
        "<|constrain|>",
        "<|call|>",
        "<|return|>",
    }
    spellings = sorted(
        {
            t["content"]
            for t in added
            if t.get("special") or t["content"] in harmony_controls
        },
        key=len,
        reverse=True,
    )
    if not spellings:
        raise ValueError(f"{tokenizer_dir}/tokenizer.json declares no special tokens")
    return re.compile("|".join(re.escape(t) for t in spellings))


# --------------------------------------------------------------------------- client venv


def parse_requirements(path):
    """{canonical name: version or git commit} of a closed `name==version` / `name @ git+URL@commit` file."""
    pins = {}
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "==" in line:
                name, version = line.split("==", 1)
            elif " @ git+" in line and "@" in line.rsplit("/", 1)[-1]:
                name, version = line.split(" @ ", 1)[0], line.rsplit("@", 1)[1]
            else:
                raise ValueError(f"not an exact pin: {line!r}")
            pins[_canonical(name.strip())] = version.strip()
    return pins


def _installed_distributions():
    return {
        _canonical(d.metadata["Name"]): d for d in importlib.metadata.distributions()
    }


def _vcs_commit(dist):
    text = dist.read_text("direct_url.json")
    return json.loads(text).get("vcs_info", {}).get("commit_id", "") if text else ""


def check_client(pins, installed):
    """Problems of an installed distribution set against the pins: missing, wrong version or commit, or extra."""
    problems = []
    for name, want in sorted(pins.items()):
        dist = installed.get(name)
        if dist is None:
            problems.append(f"missing {name}")
            continue
        got = _vcs_commit(dist) if HEX40.match(want) else dist.version
        if got != want:
            problems.append(f"{name}: installed {got or '?'}, pinned {want}")
    extra = sorted(set(installed) - set(pins) - {"pip"})
    if extra:
        problems.append(f"not in the closed set: {', '.join(extra)}")
    if pins.get("evalscope") != EVALSCOPE_COMMIT:
        problems.append(f"evalscope must be pinned to {EVALSCOPE_COMMIT}")
    return problems


def cmd_client_check(args):
    pins = parse_requirements(args.requirements)
    installed = _installed_distributions()
    problems = check_client(pins=pins, installed=installed)
    evalscope_bin = os.path.join(os.path.dirname(sys.executable), "evalscope")
    for module in (
        "evalscope.perf.main",
        "evalscope.perf.plugin.datasets.swe_smith",
        "transformers",
        "datasets",
    ):
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"], capture_output=True, text=True
        )
        if proc.returncode:
            problems.append(
                f"import {module} failed: {proc.stderr.strip().splitlines()[-1:]}"
            )
    proc = subprocess.run(
        [evalscope_bin, "perf", "--help"], capture_output=True, text=True
    )
    if proc.returncode:
        problems.append(f"`evalscope perf --help` exited {proc.returncode}")
    for p in problems:
        print(f"client-check: {p}", file=sys.stderr)
    if problems:
        return 1
    manifest = {
        "created_at": _utc_now(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "requirements_sha256": _sha256_file(args.requirements),
        "evalscope_commit": _vcs_commit(installed["evalscope"]),
        "packages": {name: installed[name].version for name in sorted(pins)},
    }
    _write_json(args.out, manifest)
    print(
        f"client-check: {len(pins)} pinned distributions match; evalscope {manifest['evalscope_commit']}"
    )
    return 0


# --------------------------------------------------------------------------- tokenizer view


def _hf_download_revision(model_path, name):
    path = os.path.join(
        model_path, ".cache", "huggingface", "download", f"{name}.metadata"
    )
    try:
        with open(path) as f:
            return f.readline().strip()
    except FileNotFoundError:
        return None


def ensure_tokenizer_view(model_path, view):
    """Create or verify a directory of symlinks to the checkpoint's tokenizer files; never rewrite a foreign one.

    The checkpoint's config.json is kept out on purpose: AutoTokenizer on the full directory parses it, and this
    transformers version rejects its rope config.
    """
    os.makedirs(view, exist_ok=True)
    extra = sorted(set(os.listdir(view)) - set(TOKENIZER_FILES))
    if extra:
        raise SystemExit(
            f"tokenizer-view: {view} holds unexpected entries {extra}; refusing to use it"
        )
    for name in TOKENIZER_FILES:
        target = os.path.join(os.path.abspath(model_path), name)
        link = os.path.join(view, name)
        if not os.path.isfile(target):
            raise SystemExit(f"tokenizer-view: missing {target}")
        if os.path.lexists(link):
            if not os.path.islink(link) or os.readlink(link) != target:
                raise SystemExit(
                    f"tokenizer-view: {link} is not a symlink to {target}; refusing to modify it"
                )
        else:
            os.symlink(target, link)


def cmd_tokenizer_view(args):
    ensure_tokenizer_view(model_path=args.model_path, view=args.view)
    revisions = {
        name: _hf_download_revision(args.model_path, name) for name in TOKENIZER_FILES
    }
    found = set(revisions.values())
    if None in found:
        if not args.revision:
            print(
                "tokenizer-view: no download metadata and no MODEL_REVISION; revision unknown",
                file=sys.stderr,
            )
            return 1
        revision, source = (
            args.revision,
            "site.env (no download metadata to check it against)",
        )
    elif len(found) != 1:
        print(
            f"tokenizer-view: tokenizer files come from different revisions {revisions}",
            file=sys.stderr,
        )
        return 1
    else:
        revision, source = found.pop(), "huggingface download metadata"
        if args.revision and args.revision != revision:
            print(
                f"tokenizer-view: checkpoint is at {revision}, site.env says {args.revision}",
                file=sys.stderr,
            )
            return 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.view)
    manifest = {
        "created_at": _utc_now(),
        "view": os.path.abspath(args.view),
        "model_path": os.path.abspath(args.model_path),
        "revision": revision,
        "revision_source": source,
        "files": {
            name: _sha256_file(os.path.join(args.view, name))
            for name in TOKENIZER_FILES
        },
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
    }
    _write_json(args.out, manifest)
    print(
        f"tokenizer-view: {args.view} -> {args.model_path} @ {revision} ({manifest['tokenizer_class']})"
    )
    return 0


# --------------------------------------------------------------------------- dataset


def _metadata_without_stats(metadata):
    return {k: v for k, v in metadata.items() if k != "conversation_stats"}


def check_dataset(data, tokenizer_manifest, chat_template_kwargs, need, shape):
    """Problems of a built dataset against the current tokenizer, request kwargs and workload shape."""
    meta = data.get("metadata", {})
    convs = data.get("conversations", [])
    stats = meta.get("conversation_stats", [])
    problems = []
    if meta.get("format") != DATASET_FORMAT:
        problems.append(f"format {meta.get('format')!r}, expected {DATASET_FORMAT!r}")
    for key, want in shape.items():
        if meta.get(key) != want:
            problems.append(f"metadata {key}={meta.get(key)!r}, expected {want!r}")
    if meta.get("chat_template_kwargs") != chat_template_kwargs:
        problems.append(
            f"built with chat_template_kwargs {meta.get('chat_template_kwargs')!r}, "
            f"requests send {chat_template_kwargs!r}"
        )
    tok = meta.get("tokenizer", {})
    if tok.get("files") != tokenizer_manifest["files"]:
        problems.append(
            "tokenizer files differ from the ones the dataset was built with"
        )
    if tok.get("revision") != tokenizer_manifest["revision"]:
        problems.append(
            f"tokenizer revision {tok.get('revision')!r} != {tokenizer_manifest['revision']!r}"
        )
    for key in ("dataset_revision", "pad_dataset_revision"):
        if not HEX40.match(str(meta.get(key, ""))):
            problems.append(f"{key} is not a pinned commit: {meta.get(key)!r}")
    if meta.get("budget_check", {}).get("violations") != 0:
        problems.append(
            f"token budget violations recorded by the builder: {meta.get('budget_check')}"
        )
    if len(convs) != meta.get("num_conversations") or len(stats) != len(convs):
        problems.append(
            f"{len(convs)} conversations, {len(stats)} stats, metadata says {meta.get('num_conversations')}"
        )
    if len(convs) < need:
        problems.append(f"{len(convs)} conversations < {need} needed")
    # A length mismatch is already reported above; compare the common prefix.
    for i, (conv, stat) in enumerate(zip(convs, stats, strict=False)):
        roles = [[m.get("role") for m in turn.get("messages", [])] for turn in conv]
        if roles != [["system", "user"]] + [["user"]] * (shape["num_turns"] - 1):
            problems.append(f"conversation {i}: turn roles {roles[:3]}...")
            break
        system_sha = hashlib.sha256(
            conv[0]["messages"][0]["content"].encode()
        ).hexdigest()
        if system_sha != stat.get("system_sha256"):
            problems.append(
                f"conversation {i}: system message does not match its recorded sha256"
            )
            break
    if len({s.get("system_sha256") for s in stats}) != len(stats):
        problems.append("conversations share a system message; prefixes are not unique")
    return problems


def _render_counts(data, tokenizer_view, chat_template_kwargs, indices):
    """Turn-1 prompt tokens as the server counts them (serving_chat.py renders to text, then encodes)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_view)
    counts = {}
    for i in indices:
        text = tokenizer.apply_chat_template(
            data["conversations"][i][0]["messages"],
            tokenize=False,
            add_generation_prompt=True,
            tools=None,
            **chat_template_kwargs,
        )
        counts[i] = len(tokenizer.encode(text, add_special_tokens=False))
    return counts


def sample_indices(n, k):
    """k indices spread evenly over range(n), always including the first and the last."""
    if n <= 0 or k <= 0:
        return []
    if k == 1 or n == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def cmd_dataset_check(args):
    tokenizer_manifest = _load_json(args.tokenizer_manifest)
    kwargs = json.loads(args.chat_template_kwargs)
    with open(args.dataset) as f:
        data = json.load(f)
    shape = {
        "num_turns": args.turns,
        "first_turn_length": args.first_turn_length,
        "subsequent_turn_length": args.subsequent_turn_length,
        "osl_estimate": args.osl,
    }
    problems = check_dataset(
        data=data,
        tokenizer_manifest=tokenizer_manifest,
        chat_template_kwargs=kwargs,
        need=args.need,
        shape=shape,
    )
    stats = data["metadata"].get("conversation_stats", [])
    if not problems and args.render_sample:
        rendered = _render_counts(
            data=data,
            tokenizer_view=tokenizer_manifest["view"],
            chat_template_kwargs=kwargs,
            indices=sample_indices(len(stats), args.render_sample),
        )
        for i, count in rendered.items():
            if count != stats[i]["first_turn_prompt_tokens"]:
                problems.append(
                    f"conversation {i}: renders to {count} tokens, recorded {stats[i]['first_turn_prompt_tokens']}"
                )
    for p in problems:
        print(f"dataset-check: {p}", file=sys.stderr)
    if problems:
        return 1
    meta = _metadata_without_stats(data["metadata"])
    out = {
        "path": os.path.abspath(args.dataset),
        "sha256": _sha256_file(args.dataset),
        "num_conversations": len(stats),
        "metadata_sha256": _sha256_json(meta),
        "metadata": meta,
        "conversation_stats": stats,
    }
    _write_json(args.out, out)
    worst = max(s["final_turn_prompt_tokens_est"] for s in stats)
    print(
        f"dataset-check: {len(stats)} conversations, sha256 {out['sha256'][:16]}, max final-turn prompt ~{worst}"
    )
    return 0


# --------------------------------------------------------------------------- server


def _get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def fetch_server_info(base_url):
    try:
        return _get_json(f"{base_url}/server_info")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        return _get_json(f"{base_url}/get_server_info")


def serving_gpus(info):
    """GPUs one server instance occupies: TP x PP, times DP unless DP attention shares the TP group."""
    dp = 1 if info.get("enable_dp_attention") else (info.get("dp_size") or 1)
    return (info.get("tp_size") or 1) * (info.get("pp_size") or 1) * dp


def check_server(info, profile, model_path, served_model_name, cc, n_gpus):
    """Mismatches of a /server_info payload against the profile, the site settings and a rung's concurrency."""
    info = info.get("server_args", info)
    want = dict(
        profile["resolved_args"],
        model_path=model_path,
        served_model_name=served_model_name,
    )
    problems = [
        f"{k}: server has {info.get(k)!r}, profile wants {v!r}"
        for k, v in want.items()
        if info.get(k) != v
    ]
    if cc is not None and (info.get("max_running_requests") or 0) < cc:
        problems.append(
            f"max_running_requests {info.get('max_running_requests')} < concurrency {cc}"
        )
    if n_gpus != serving_gpus(info):
        problems.append(
            f"N_GPUS={n_gpus}, but the server spans {serving_gpus(info)} GPUs (per-GPU columns divide by it)"
        )
    return problems


def cmd_server_check(args):
    info = fetch_server_info(args.base_url)
    _write_json(args.out, info)
    problems = check_server(
        info=info,
        profile=_load_json(args.profile),
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        cc=args.cc,
        n_gpus=args.n_gpus,
    )
    for p in problems:
        print(f"server-check: {p}", file=sys.stderr)
    if problems:
        return 1
    print(
        f"server-check: {info.get('served_model_name')} sglang {info.get('version')} matches profile; "
        f"enable_cache_report={info.get('enable_cache_report')}"
    )
    return 0


def load_counts(payload):
    """(running, waiting) requests over all DP ranks of a /v1/loads payload."""
    loads = payload.get("loads", [])
    running = sum(int(r.get("num_running_reqs", 0)) for r in loads)
    waiting = sum(int(r.get("num_waiting_reqs", 0)) for r in loads)
    for r in loads:
        waiting += sum(
            int(v) for k, v in (r.get("queues") or {}).items() if k != "waiting"
        )
    return running, waiting


def cmd_wait_idle(args):
    deadline = time.monotonic() + args.timeout_s
    while True:
        running, waiting = load_counts(_get_json(f"{args.base_url}/v1/loads"))
        if running == 0 and waiting == 0:
            print("wait-idle: server is idle")
            return 0
        if time.monotonic() > deadline:
            print(
                f"wait-idle: still {running} running / {waiting} queued after {args.timeout_s}s",
                file=sys.stderr,
            )
            return 1
        time.sleep(2)


def cmd_sample_loads(args):
    parent = os.getppid()
    with open(args.out, "a") as out:
        # Stops by itself once the sweep that started it is gone.
        while os.getppid() == parent:
            try:
                payload = _get_json(f"{args.base_url}/v1/loads", timeout=10)
                running, waiting = load_counts(payload)
                token_usage = max(
                    (r.get("token_usage", 0.0) for r in payload.get("loads", [])),
                    default=0.0,
                )
                row = {
                    "t": time.time(),
                    "running": running,
                    "waiting": waiting,
                    "token_usage": token_usage,
                }
                try:
                    with urllib.request.urlopen(
                        f"{args.base_url}/metrics", timeout=3
                    ) as response:
                        lines = response.read().decode().splitlines()
                    row["retraction_metrics"] = [
                        line
                        for line in lines
                        if "retract" in line.lower() and not line.startswith("#")
                    ]
                except OSError as error:
                    row["retraction_metrics_error"] = str(error)[:200]
            except (OSError, ValueError) as e:
                row = {"t": time.time(), "error": str(e)[:200]}
            out.write(json.dumps(row) + "\n")
            out.flush()
            time.sleep(args.interval_s)
    return 0


# --------------------------------------------------------------------------- run identity


def server_identity(info):
    """The static part of /server_info: every top-level key but the live scheduler state and boot timings."""
    return {k: v for k, v in info.items() if k not in VOLATILE_SERVER_KEYS}


def build_identity(client_manifest, dataset_stats, server_info, package_dir, config):
    return {
        "config": config,
        "dataset": {
            "sha256": dataset_stats["sha256"],
            "num_conversations": dataset_stats["num_conversations"],
            "metadata_sha256": dataset_stats["metadata_sha256"],
        },
        "client": {
            "requirements_sha256": client_manifest["requirements_sha256"],
            "evalscope_commit": client_manifest["evalscope_commit"],
            "python": client_manifest["python"],
            "machine": client_manifest["machine"],
        },
        "tokenizer": {"revision": dataset_stats["metadata"]["tokenizer"]["revision"]},
        "server": server_identity(server_info),
        "package": {
            name: _sha256_file(os.path.join(package_dir, name))
            for name in IDENTITY_FILES
        },
    }


def diff_identity(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new)):
            out += diff_identity(old.get(key), new.get(key), f"{prefix}{key}.")
        return out
    return [] if old == new else [f"{prefix.rstrip('.')}: {old!r} -> {new!r}"]


def accept_run(results_dir, attempt_dir, package_dir, config):
    """(exit status, messages) for a preflight in attempt_dir against the run already accepted in results_dir.

    A new run promotes the attempt's files into results_dir. A resume with the same identity leaves results_dir
    untouched. A changed identity is refused, and results_dir stays byte-identical.
    """
    identity = build_identity(
        client_manifest=_load_json(os.path.join(attempt_dir, "client-manifest.json")),
        dataset_stats=_load_json(os.path.join(attempt_dir, "dataset-stats.json")),
        server_info=_load_json(os.path.join(attempt_dir, "server-info-start.json")),
        package_dir=package_dir,
        config=config,
    )
    _write_json(os.path.join(attempt_dir, "run-identity.json"), identity)
    accepted = os.path.join(results_dir, "run-identity.json")
    if not os.path.exists(accepted):
        for name in RUN_FILES:
            shutil.copy2(
                os.path.join(attempt_dir, name), os.path.join(results_dir, name)
            )
        shutil.copy2(os.path.join(attempt_dir, "run-identity.json"), accepted)
        return 0, [f"new run {_sha256_json(identity)[:16]}"]
    changes = diff_identity(_load_json(accepted), identity)
    if changes:
        return 3, [f"identity changed: {c}" for c in changes]
    return 0, [f"resuming run {_sha256_json(identity)[:16]}"]


def cmd_accept_run(args):
    rc, messages = accept_run(
        results_dir=args.results_dir,
        attempt_dir=args.attempt_dir,
        package_dir=args.package_dir,
        config=json.loads(args.config),
    )
    for m in messages:
        print(f"accept-run: {m}", file=sys.stderr if rc else sys.stdout)
    return rc


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("client-check")
    p.add_argument("--requirements", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("tokenizer-view")
    p.add_argument("--model-path", required=True)
    p.add_argument("--view", required=True)
    p.add_argument("--revision", default="")
    p.add_argument("--out", required=True)

    p = sub.add_parser("dataset-check")
    p.add_argument("--dataset", required=True)
    p.add_argument("--tokenizer-manifest", required=True)
    p.add_argument("--chat-template-kwargs", required=True)
    p.add_argument("--need", type=int, required=True)
    p.add_argument("--turns", type=int, required=True)
    p.add_argument("--first-turn-length", type=int, required=True)
    p.add_argument("--subsequent-turn-length", type=int, required=True)
    p.add_argument("--osl", type=int, required=True)
    p.add_argument("--render-sample", type=int, default=0)
    p.add_argument("--out", required=True)

    p = sub.add_parser("server-check")
    p.add_argument("--base-url", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--served-model-name", required=True)
    p.add_argument("--cc", type=int, default=None)
    p.add_argument("--n-gpus", type=int, required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("wait-idle")
    p.add_argument("--base-url", required=True)
    p.add_argument("--timeout-s", type=float, default=120)

    p = sub.add_parser("sample-loads")
    p.add_argument("--base-url", required=True)
    p.add_argument("--interval-s", type=float, default=5)
    p.add_argument("--out", required=True)

    p = sub.add_parser("accept-run")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--attempt-dir", required=True)
    p.add_argument("--package-dir", required=True)
    p.add_argument("--config", required=True)

    args = parser.parse_args()
    handlers = {
        "client-check": cmd_client_check,
        "tokenizer-view": cmd_tokenizer_view,
        "dataset-check": cmd_dataset_check,
        "server-check": cmd_server_check,
        "wait-idle": cmd_wait_idle,
        "sample-loads": cmd_sample_loads,
        "accept-run": cmd_accept_run,
    }
    sys.exit(handlers[args.cmd](args))


if __name__ == "__main__":
    main()
