# SPDX-License-Identifier: Apache-2.0
"""CLI Guide: `sglang serve MODEL` runs the Cookbook command for the visible GPUs."""

import json
import os
import shlex
import sys
from importlib import resources

from sglang.srt.utils.device import get_device_count, get_device_name, is_hip

# Picked in this order when the user does not choose a mode.
MODES = ("balanced", "low-latency", "high-throughput")
NO_MODE_MESSAGE = (
    "This model has no mode options in the Cookbook; "
    "using the recommended default configuration."
)
NO_RECIPE_MESSAGE = (
    "The Cookbook has no command for {model} yet; launching it with the default flags."
)
NO_MATCH_MESSAGE = (
    "The Cookbook has no command for {model} on {machine} (it has {available}); "
    "launching it with the default flags. To pick fewer GPUs, set {devices_env}."
)
EDIT_PROMPT = (
    'Change a flag ("--port 8000"), remove one ("!--port"), or press Enter to launch: '
)

# Cookbook hardware ids by device name, most specific first;
_HARDWARE = (
    ("GB300", "gb300"),
    ("B300", "b300"),
    ("GB200", "gb200"),
    ("B200", "b200"),
    ("GB10", "dgx-spark"),
    ("RTX PRO 6000", "rtx6000"),
    ("GH200", None),
    ("H200", "h200"),
    ("H100", "h100"),
    ("H800", "h800"),
    ("MI355X", "mi355x"),
    ("MI350X", "mi350x"),
    ("MI325X", "mi325x"),
    ("MI300X", "mi300x"),
)


def recipes_for_this_machine(model: str) -> list[dict]:
    """Cookbook commands for `model` on the visible GPUs; says so when there are none."""
    recipes = _load_recipes(model)
    if not recipes:
        _log(NO_RECIPE_MESSAGE.format(model=model))
        return []
    device_name, gpus = _detect_gpus()
    hw = _hardware_id(device_name)
    matching = [r for r in recipes if r["hw"] == hw and r["gpus"] == gpus]
    if not matching:
        machine = f"{gpus}x {device_name}" if gpus else "this machine (no GPU)"
        available = ", ".join(sorted({f"{r['gpus']}x {r['hw']}" for r in recipes}))
        devices_env = "HIP_VISIBLE_DEVICES" if is_hip() else "CUDA_VISIBLE_DEVICES"
        _log(
            NO_MATCH_MESSAGE.format(
                model=model,
                machine=machine,
                available=available,
                devices_env=devices_env,
            )
        )
    return matching


def run_guide(model: str, recipes: list[dict], dry_run: bool) -> None:
    """Print the Cookbook command picked from `recipes`, then exec it."""
    interactive = _is_terminal()
    recipe = _pick_mode(recipes=recipes, interactive=interactive)
    _log(f"Cookbook recipe: {recipe['url']}")
    if recipe["verified"] != "verified":
        _log(
            "Warning: the Cookbook has not verified this command yet "
            f"({recipe['verified']})."
        )
    env, argv = recipe["env"], command_argv(recipe["command"])
    if dry_run:
        print(recipe["command"])
        return
    if interactive:
        argv = _ask_flag_edits(env=env, argv=argv, command=recipe["command"])
    else:
        print(recipe["command"])
    yours = [key for key in env if os.environ.get(key, env[key]) != env[key]]
    if yours:
        _log(f"Using your exported {', '.join(yours)} instead of the Cookbook's.")

    _log(
        f'To change flags, run "sglang serve {model} --dry-run", '
        "edit the command without running."
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "sglang.cli.main", "serve", *argv],
        {**env, **os.environ},
    )


def command_argv(command: str) -> list[str]:
    """Arguments after `sglang serve` in a Cookbook command."""
    lines = [line.rstrip(" \\") for line in command.strip().splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    start = next(i for i, line in enumerate(lines) if line.startswith("sglang serve"))
    return shlex.split(" ".join(lines[start:]))[2:]


def format_command(env: dict[str, str], argv: list[str]) -> str:
    """Render a command in the Cookbook's one-flag-per-line style."""
    lines = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    lines.append("sglang serve")
    lines += [f"  {shlex.join(group)}" for group in _flag_groups(argv)]
    return " \\\n".join(lines)


def edit_flags(argv: list[str], edits: str) -> list[str]:
    """Apply `--flag value` (set or add) and `!--flag` (remove) to argv."""
    tokens = [
        part
        for token in shlex.split(edits)
        for part in (token.split("=", 1) if token.startswith("--") else [token])
    ]
    groups = _flag_groups(argv)
    for edit in _flag_groups(tokens):
        flag = edit[0]
        if flag.startswith("!--"):
            if len(edit) > 1:
                raise ValueError(f"{flag} takes no value.")
            kept = [group for group in groups if group[0] != flag[1:]]
            if len(kept) == len(groups):
                raise ValueError(f"{flag[1:]} is not in the command.")
            groups = kept
        elif flag.startswith("--"):
            flags = [group[0] for group in groups]
            if flag in flags:
                groups[flags.index(flag)] = edit
            else:
                groups.append(edit)
        else:
            raise ValueError(f'Expected a flag such as "--port 8000", got {flag!r}.')
    return [token for group in groups for token in group]


def _flag_groups(argv: list[str]) -> list[list[str]]:
    groups = []
    for token in argv:
        if token.startswith(("--", "!--")) or not groups:
            groups.append([token])
        else:
            groups[-1].append(token)
    return groups


def _pick_mode(recipes: list[dict], interactive: bool) -> dict:
    by_mode = {}
    for recipe in recipes:
        by_mode.setdefault(recipe["mode"], recipe)
    if None in by_mode:
        _log(NO_MODE_MESSAGE)
        return by_mode[None]

    modes = sorted(by_mode, key=lambda m: MODES.index(m) if m in MODES else len(MODES))
    if len(modes) == 1 or not interactive:
        _log(f"Mode: {modes[0]} (default; available: {', '.join(modes)})")
        return by_mode[modes[0]]

    _log("Modes:\n" + "\n".join(f"  {i}) {m}" for i, m in enumerate(modes, 1)))
    while True:
        answer = _ask(f"Mode [{modes[0]}]: ").strip()
        if not answer:
            return by_mode[modes[0]]
        if answer in by_mode:
            return by_mode[answer]
        if answer.isdigit() and 1 <= int(answer) <= len(modes):
            return by_mode[modes[int(answer) - 1]]
        _log(f"Pick 1-{len(modes)} or a mode name.")


def _ask_flag_edits(env: dict[str, str], argv: list[str], command: str) -> list[str]:
    print(command)
    while True:
        edits = _ask(EDIT_PROMPT)
        if not edits.strip():
            return argv
        try:
            argv = edit_flags(argv=argv, edits=edits)
        except ValueError as e:
            _log(str(e))
            continue
        print(format_command(env=env, argv=argv))


def _ask(prompt: str) -> str:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        _log("\nCancelled; nothing launched.")
        raise SystemExit(130)


def _is_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _load_recipes(model: str) -> list[dict]:
    path = (
        resources.files("sglang.cli") / "_commands" / f"{model.replace('/', '--')}.json"
    )
    return json.loads(path.read_text()) if path.is_file() else []


def _detect_gpus() -> tuple[str | None, int]:
    count = get_device_count()
    return (get_device_name() if count else None), count


def _hardware_id(device_name: str | None) -> str | None:
    for name, hw in _HARDWARE:
        if device_name and name in device_name.upper():
            return hw
    return None


def _log(message: str) -> None:
    print(message, file=sys.stderr)
