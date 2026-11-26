"""Every LLM launch command in the docs must parse with the real ServerArgs parser.

The test extracts each `python -m sglang.launch_server` and `sglang serve`
command from fenced code blocks in docs/**/*.md(x), from notebook cells and from
`launch_server_cmd("...")` strings, and parses it the way `prepare_server_args`
does. A command fails on an unknown option, a value the parser rejects, a
removed or deprecated option, or a stray argument. Placeholders such as
`$MODEL` or `<port>` in an option value are replaced with a value the option
accepts.

Not checked: diffusion and VLA pages and diffusion sections (another CLI),
pages whose commands are frozen to a validated 0.5.x image (they carry a
"NOTE(agent): ... frozen" marker), commands that set SGLANG_PLATFORM
(out-of-tree platform plugins), and the config-driven <Deployment> commands in
docs/src/snippets/configs, which are JSX rather than fenced code.

Each problem is reported with its page and line; a removed flag
(server_args.REMOVED_FLAGS) or a deprecated alias also gets the spelling to use
instead. Problems left in place on purpose are listed in _KNOWN_ISSUES with
their number of occurrences per page, so one more occurrence on a listed page
still fails. A PR that removes or deprecates an option the docs still use
should update those pages in the same PR.
"""

import argparse
import json
import re
import shlex
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.arg_groups import argparse_actions
from sglang.srt.server_args import REMOVED_FLAGS, ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DOCS = Path(__file__).resolve().parents[4] / "docs"

# "<page>: <problem>" -> occurrences, for problems left in place on purpose.
# Each needs a reason; delete the entry once the page is fixed. A count is
# only an upper bound: a docs-only PR does not run this test, so fixing a page
# must not make it fail.
_KNOWN_ISSUES = {
    # Fixed by the open #51137.
    "docs/advanced_features/hicache_best_practices.mdx: space after a "
    "line-continuation backslash": 1,
    # The alias still runs and keeps the cap of 2 states per path. The suggested
    # --hybrid-cache-checkpoints-level 2 (the default) caps at one, so switching
    # would change these tuned recipes.
    "cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx: deprecated "
    "--mamba-max-states-per-path": 1,
    "cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx: deprecated "
    "--mamba-max-states-per-path": 1,
}

# Pages whose commands use another CLI: diffusion (`sglang generate` and the
# diffusion serve backend) and VLA (`--model-type` backends with their own flags).
_SKIPPED_PATH_PARTS = ("diffusion", "cookbook/vla/")
# Pages validated on a pinned 0.5.x image, which ask to be left out of flag
# renames (see #52851).
_FROZEN_MARKER_RE = re.compile(r"NOTE\(agent\):[^\n]*\bfrozen\b")

_LAUNCH_RE = re.compile(
    r"(?:\bpython3?\s+-m\s+sglang\.launch_server|\bsglang\s+serve)(?=\s|$)"
)
_FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n(?P<body>.*?)^[ \t]*(?P=fence)[ \t]*$",
    re.M | re.S,
)
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.M)
_LAUNCH_SERVER_CMD_RE = re.compile(
    r"launch_server_cmd\(\s*(?P<quote>\"\"\"|'''|\"|')(?P<body>.*?)(?P=quote)", re.S
)
# A shell operator or redirection ends the command.
_COMMAND_END_RE = re.compile(r"\s(?:&&|\|\||\||;|&)(?:\s|$)|\s\d*>")
# `KEY=value` prefixes that select an out-of-tree platform plugin, which can
# register its own backends (e.g. the SGLang-TorchTPU commands on TPU pages).
_PLATFORM_PREFIX_RE = re.compile(r"\bSGLANG_PLATFORM=")
_SPACE_AFTER_BACKSLASH = "space after a line-continuation backslash"


def _is_deprecated(action: argparse.Action) -> bool:
    # The Deprecated*Action classes, plus plain flags whose help starts with
    # "Deprecated" or "(Deprecated". A help text that only calls one *value*
    # deprecated (e.g. --moe-a2a-backend) does not count.
    help_text = (action.help or "").lstrip("(")
    return type(action).__name__.startswith("Deprecated") or help_text.startswith(
        "Deprecated"
    )


def _is_removed(action: argparse.Action) -> bool:
    # A flag in server_args.REMOVED_FLAGS: still registered, but only to stop
    # the launch with a message that names the replacement.
    return getattr(action, "error_message", None) is not None


def _record_deprecated_use(self, parser, namespace, values, option_string=None):
    if _is_removed(self):
        # Record instead of stopping, so one command reports every removed flag.
        namespace._removed_used.append(option_string)
        return
    namespace._deprecated_used.append(option_string)
    type(self).__mro__[1].__call__(self, parser, namespace, values, option_string)


def _raise(message):
    raise argparse.ArgumentError(None, message)


def _build_parser() -> argparse.ArgumentParser:
    # Same parser as prepare_server_args, except that errors raise and nothing
    # is required, so a command that omits --model-path still parses.
    parser = argparse.ArgumentParser(
        prog="sglang serve", add_help=False, exit_on_error=False
    )
    ServerArgs.add_cli_args(parser)
    parser.error = _raise
    for action in parser._actions:
        action.required = False
        if _is_deprecated(action):
            cls = type(action)
            action.__class__ = type(
                f"_Recorded{cls.__name__}",
                (cls,),
                {"__call__": _record_deprecated_use},
            )
    return parser


def _fenced_blocks(text: str):
    """(line of the first code line, code) for each fenced code block, except
    those under a heading that names diffusion (an LLM page's diffusion section
    uses the diffusion CLI)."""
    headings = []  # (level, title) of the enclosing sections
    end = 0
    for match in _FENCE_RE.finditer(text):
        for level, title in _HEADING_RE.findall(text, end, match.start()):
            headings = [h for h in headings if h[0] < len(level)]
            headings.append((len(level), title))
        end = match.end()
        if not any("diffusion" in title.lower() for _, title in headings):
            yield text.count("\n", 0, match.start("body")) + 1, match.group("body")


def _code_blocks(path: Path):
    """(cell, line of the first code line, code) for each code block, with cell
    "" for a page and " cell N" for a notebook cell."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if _FROZEN_MARKER_RE.search(text):
        return
    if path.suffix == ".ipynb":
        for n, cell in enumerate(json.loads(text).get("cells", []), 1):
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "code":
                yield f" cell {n}", 1, source
            else:
                for line, code in _fenced_blocks(source):
                    yield f" cell {n}", line, code
    else:
        for line, code in _fenced_blocks(text):
            yield "", line, code


def _shell_commands(code: str, first_line: int = 0):
    """Yield (line, env prefix, launcher, arguments) for each launch command in
    a block, with `line` counted from `first_line` for the block's first line."""
    for match in _LAUNCH_SERVER_CMD_RE.finditer(code):
        line = first_line + code.count("\n", 0, match.start("body"))
        yield from _shell_commands(match.group("body"), line)
    # Blank out those strings, keeping their newlines so line numbers still hold.
    code = _LAUNCH_SERVER_CMD_RE.sub(lambda m: "\n" * m.group(0).count("\n"), code)
    lines = [line.removesuffix("\r") for line in code.split("\n")]
    i = 0
    while i < len(lines):
        start, command = i, lines[i]
        # Only a backslash right before the newline continues the line, as in bash.
        while command.endswith("\\") and i + 1 < len(lines):
            i += 1
            command = command[:-1] + " " + lines[i]
        i += 1
        match = _LAUNCH_RE.search(command)
        if match:
            rest = _COMMAND_END_RE.split(command[match.end() :])[0]
            yield first_line + start, command[: match.start()], match.group(0), rest


def _is_placeholder(value: str) -> bool:
    if "$" in value or "<" in value or ">" in value or "..." in value:
        return True
    if "{" in value or "}" in value or value.startswith("["):
        try:
            json.loads(value)
        except ValueError:
            return True
    return False


def _stand_in(action) -> str:
    """A value the option accepts, to stand in for a `$VAR` or `<value>`."""
    for value in [*(action.choices or []), "1", "x", "{}"]:
        try:
            converted = action.type(value) if callable(action.type) else value
        except Exception:
            continue
        if not action.choices or converted in action.choices:
            return value
    return "x"


def _action_for(parser: argparse.ArgumentParser, option: str):
    """The action an option selects, including by an unambiguous prefix (argparse
    accepts `--tp` for `--tp-size`)."""
    actions = parser._option_string_actions
    if option in actions:
        return actions[option]
    matches = {actions[o] for o in actions if o.startswith(option)}
    return matches.pop() if option.startswith("--") and len(matches) == 1 else None


def _launch_argv(parser: argparse.ArgumentParser, launcher: str, rest: str):
    """The ServerArgs argv of a command, or None if it is not checkable.

    A placeholder that stands for an option's value is replaced with a value
    the option accepts, so the option names are still checked. A command with
    any other placeholder is not checked.
    """
    try:
        tokens = shlex.split(rest, comments=True)
    except ValueError:
        return None
    if "-h" in tokens or "--help" in tokens:
        return None
    is_serve = launcher.startswith("sglang")
    argv = []
    for i, token in enumerate(tokens):
        option, eq, value = token.partition("=")
        if token.startswith("--") and eq:
            if _is_placeholder(value):
                action = _action_for(parser, option)
                token = f"{option}={_stand_in(action) if action else 'x'}"
        elif _is_placeholder(token):
            action = _action_for(parser, argv[-1]) if argv else None
            if action is not None and action.nargs != 0:
                token = _stand_in(action)
            elif is_serve and i == 0:
                token = "model"  # `sglang serve <model>`
            elif argv and argv[-1].startswith("--") and action is None:
                token = "x"  # the value of an unknown option, reported anyway
            else:
                return None
        argv.append(token)
    if is_serve:
        # `sglang serve` strips these before it hands argv to ServerArgs.
        argv = [a for a in argv if a != "--dry-run"]
        model_type = "llm"
        for i, token in enumerate(argv):
            if token == "--model-type" and i + 1 < len(argv):
                model_type = argv[i + 1]
                del argv[i : i + 2]
                break
            if token.startswith("--model-type="):
                model_type = token.split("=", 1)[1]
                del argv[i]
                break
        if model_type != "llm":
            return None
        if argv and not argv[0].startswith("-"):
            argv = ["--model-path", *argv]
    return argv


def _problems(parser: argparse.ArgumentParser, argv: list[str]) -> list[str]:
    namespace = argparse.Namespace(_deprecated_used=[], _removed_used=[])
    problems = []
    try:
        with patch.object(argparse_actions, "print_deprecated_warning"):
            _, unknown = parser.parse_known_args(argv, namespace)
        for i, token in enumerate(unknown):
            if token.startswith("-"):
                problems.append(f"unknown {token.split('=')[0]}")
            elif not token.strip():
                # `\ # comment` escapes the space instead of the newline: bash
                # passes " ", the comment ends the command and the next lines
                # run as separate commands.
                problems.append(_SPACE_AFTER_BACKSLASH)
            elif i == 0 or not unknown[i - 1].startswith("-"):
                problems.append(f"unexpected argument {token!r}")
    except argparse.ArgumentError as e:
        message = str(e)
        value = re.search(r"invalid (?:choice|\w+ value): '([^']*)'", message)
        if e.argument_name and value:
            problems.append(f"invalid {e.argument_name} {value.group(1)}")
        else:
            problems.append(f"error {message.split(' (choose from')[0]}")
    problems += [f"removed {o}" for o in namespace._removed_used]
    problems += [f"deprecated {o}" for o in namespace._deprecated_used]
    return problems


def _hint(parser: argparse.ArgumentParser, problem: str) -> str:
    """What to write instead, for a removed or deprecated flag that names it."""
    kind, _, option = problem.partition(" ")
    if kind == "removed" and option in REMOVED_FLAGS:
        return f" ({REMOVED_FLAGS[option]})"
    if kind == "deprecated":
        new_flag = getattr(parser._option_string_actions.get(option), "new_flag", None)
        if new_flag:
            return f" (use {new_flag})"
    return ""


def _problem_line(code_lines: list[str], index: int, problem: str) -> int:
    """Offset of the line, from the command's first line at `index`, that holds
    the option or the stray backslash a problem names, else 0."""
    if problem == _SPACE_AFTER_BACKSLASH:
        pattern = re.compile(r"\\[ \t]+(?:#|$)")
    else:
        option = next((w for w in problem.split() if w.startswith("--")), None)
        if option is None:
            return 0
        pattern = re.compile(rf"(?<![\w-]){re.escape(option)}(?![\w-])")
    for offset, text in enumerate(code_lines[index:]):
        if pattern.search(text):
            return offset
    return 0


def scan_docs(docs: Path = _DOCS):
    """Return (number of checked commands, {"page: problem": [locations]})."""
    parser = _build_parser()
    checked, found = 0, {}
    for path in sorted(docs.rglob("*")):
        rel = path.relative_to(docs).as_posix()
        if path.suffix not in {".md", ".mdx", ".ipynb"} or "node_modules" in rel:
            continue
        if any(part in rel for part in _SKIPPED_PATH_PARTS):
            continue
        for cell, first_line, code in _code_blocks(path):
            code_lines = code.split("\n")
            for line, prefix, launcher, rest in _shell_commands(code, first_line):
                if _PLATFORM_PREFIX_RE.search(prefix):
                    continue
                argv = _launch_argv(parser, launcher, rest)
                if argv is None:
                    continue
                checked += 1
                for problem in _problems(parser, argv):
                    at = line + _problem_line(code_lines, line - first_line, problem)
                    found.setdefault(f"{rel}: {problem}", []).append(
                        f"docs/{rel}{cell}:{at}: {problem}{_hint(parser, problem)}"
                    )
    return checked, found


def _location_order(location: str):
    # "docs/<page>[ cell N]:<line>: <problem>", by page, then by line.
    page, line, _ = location.split(":", 2)
    return page, int(line)


class TestDocsLaunchCommands(CustomTestCase):
    longMessage = False

    @classmethod
    def setUpClass(cls):
        cls.checked, cls.found = scan_docs()

    def test_scanner_finds_commands(self):
        # Guards against an extraction bug that silently checks nothing.
        self.assertGreater(self.checked, 500)

    def test_no_new_problems(self):
        new = sorted(
            (
                location
                for entry, locations in self.found.items()
                if len(locations) > _KNOWN_ISSUES.get(entry, 0)
                for location in locations
            ),
            key=_location_order,
        )
        self.assertFalse(
            new,
            "these documented launch commands do not parse with the current "
            "ServerArgs parser. Fix the command in the page. If this PR removes "
            "or deprecates the option on purpose, update those pages in the "
            "same PR:\n  " + "\n  ".join(new),
        )


class TestScanner(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = _build_parser()

    def _check(self, rest, launcher="python -m sglang.launch_server"):
        argv = _launch_argv(self.parser, launcher, rest)
        return None if argv is None else _problems(self.parser, argv)

    def test_valid_command(self):
        self.assertEqual(self._check("--model-path m --tp 2 --port 30000"), [])

    def test_unknown_option(self):
        self.assertEqual(
            self._check("--model-path m --no-such-flag 1"), ["unknown --no-such-flag"]
        )

    def test_invalid_choice(self):
        self.assertEqual(
            self._check("--model-path m --tool-call-parser nope"),
            ["invalid --tool-call-parser nope"],
        )

    def test_removed_options_are_all_reported(self):
        self.assertEqual(
            self._check(
                "--model-path m --enable-dp-attention --dp-size 8 "
                "--disable-cuda-graph --tp 8"
            ),
            ["removed --enable-dp-attention", "removed --disable-cuda-graph"],
        )
        self.assertIn(
            "--attn-dp-size", _hint(self.parser, "removed --enable-dp-attention")
        )

    def test_deprecated_option(self):
        problems = self._check("--model-path m --mamba-max-states-per-path 2")
        self.assertEqual(problems, ["deprecated --mamba-max-states-per-path"])
        self.assertIn(
            "--hybrid-cache-checkpoints-level", _hint(self.parser, problems[0])
        )

    def test_deprecated_value_in_help_is_not_a_deprecated_option(self):
        # --moe-a2a-backend's help calls one *value* a deprecated alias.
        self.assertEqual(self._check("--model-path m --moe-a2a-backend deepep"), [])

    def test_comment_after_continuation(self):
        self.assertEqual(
            self._check("--model-path m \\ # a comment"),
            ["space after a line-continuation backslash"],
        )

    def test_sglang_serve_positional_model_and_model_type(self):
        self.assertEqual(self._check("Qwen/Qwen3-8B --tp 2", "sglang serve"), [])
        self.assertEqual(self._check("m --model-type llm --tp 2", "sglang serve"), [])
        self.assertIsNone(self._check("m --model-type diffusion", "sglang serve"))

    def test_placeholder_values_are_replaced(self):
        self.assertEqual(self._check("--model-path $MODEL --tp ${TP}"), [])
        self.assertEqual(self._check("<model> --port <port>", "sglang serve"), [])
        self.assertEqual(self._check("--model-path m --grammar-backend [a|b]"), [])
        self.assertEqual(
            self._check("--model-path m --no-such-flag $X --tp=$TP"),
            ["unknown --no-such-flag"],
        )
        self.assertEqual(
            self._check("--model-path m --json-model-override-args '{\"a\": 1}'"), []
        )

    def test_other_placeholders_are_skipped(self):
        self.assertIsNone(self._check("--model-path m --tp {2 or 4}"))
        self.assertIsNone(self._check("--model-path m [OPTIONS]"))
        self.assertIsNone(self._check("--model-path m ..."))

    def test_extraction_and_line_numbers(self):
        code = (
            "export A=1\n"
            "SGLANG_X=1 python3 -m sglang.launch_server --model-path m \\\n"
            "  --tp 2 2>&1 | tee log\n"
            'p, port = launch_server_cmd("python -m sglang.launch_server --model-path m")\n'
            "sglang serve m\n"
        )
        self.assertEqual(
            [(line, rest.split()) for line, _, _, rest in _shell_commands(code, 10)],
            [
                (13, ["--model-path", "m"]),
                (11, ["--model-path", "m", "--tp", "2"]),
                (14, ["m"]),
            ],
        )

    def test_diffusion_sections_are_skipped(self):
        text = (
            "## LLM\n```bash\na\n```\n## Diffusion\n```bash\nb\n```\n"
            "### Sub\n```bash\nc\n```\n## Other\n```bash\nd\n```\n"
        )
        self.assertEqual(
            [(line, b.strip()) for line, b in _fenced_blocks(text)],
            [(3, "a"), (15, "d")],
        )

    def test_frozen_pages_are_skipped(self):
        self.assertIsNotNone(
            _FROZEN_MARKER_RE.search(
                "{/* NOTE(agent): The image version and scripts in this file are "
                "frozen (validated best practice). */}"
            )
        )


if __name__ == "__main__":
    unittest.main()
