#!/usr/bin/env python3
"""Check that the rust-ext cache key agrees across the sites that build it.

A consumer derives its key prefix from the runner it lands on; a producer is told
one by its caller. Nothing at runtime compares a producer's prefix with a
consumer's until an entry is looked up and missed - and a miss is not an error,
just a slower job that source-builds, which is why this is a lint.

The producer's own prefix is checked against the modules it built, by
`stage_rust_ext_modules.sh` via EXPECT_PREFIX. That check is the authority; this
one runs the same comparisons over the workflow files, before any runner is
queued, plus the two it cannot see:

1. Every producer's prefix has the shape the action derives, over an arch this
   repo builds for and the interpreters the compile matrix actually runs.
2. Each producer's runner labels are that same arch. The compile job has to be,
   to produce loadable modules; the collect job has to be, because its GLIBC gate
   reads the .so with objdump and the images ship a single-target binutils.
3. A consumer stage takes its artifact from a producer of the stage's OWN arch,
   and no arch a stage runs on is left with no producer at all.
4. The action's restore key really interpolates the derived prefix, and every
   lookup / save / restore site hashes the same inputs.
"""

import pathlib
import re
import sys

import yaml

BUILD_WORKFLOW = ".github/workflows/_pr-test-rust-ext-build.yml"
DOWNLOAD_ACTION = ".github/actions/download-rust-ext/action.yml"
STAGE_WORKFLOW = "_pr-test-stage.yml"
RUNNER_CONFIGS = "scripts/ci/runner_configs.yml"
WORKFLOW_GLOB = ".github/workflows/*.yml"

# `uname -m` per runner label. Declared rather than sniffed from the label text:
# `4-gpu-gb300` is Grace and says nothing about arm, and `arm-kernel-build-node`
# would match a substring test only by luck. A label missing here is an error
# telling the author to add it, not a guess.
RUNNER_ARCHES = {
    "ubuntu-latest": "x86_64",
    "ubuntu-22.04": "x86_64",
    "ubuntu-24.04": "x86_64",
    "ubuntu-22.04-arm": "aarch64",
    "ubuntu-24.04-arm": "aarch64",
    "x64-kernel-build-node": "x86_64",
    "arm-kernel-build-node": "aarch64",
    "1-gpu-5090": "x86_64",
    "1-gpu-h100": "x86_64",
    "2-gpu-h100": "x86_64",
    "4-gpu-h100": "x86_64",
    "8-gpu-h20": "x86_64",
    "8-gpu-h200": "x86_64",
    "8-gpu-b200": "x86_64",
    "8-gpu-b300": "x86_64",
    # check-changes substitutes a 4-gpu-b200 variant for this, all of them x86_64.
    "$b200_runner": "x86_64",
    "4-gpu-b200": "x86_64",
    "4-gpu-gb300": "aarch64",
}

_HASH_FILES = re.compile(r"hashFiles\(([^)]*)\)")
_QUOTED = re.compile(r"'([^']*)'")
# The one place the action spells the prefix out, e.g.
#   resolved="rust-ext-$(uname -m)-cp310-cp312"
_DERIVED_PREFIX = re.compile(r'"([^"]*\$\(uname -m\)[^"]*)"')
_UNAME = "$(uname -m)"
_NEEDS_JOB = re.compile(r"needs\.([A-Za-z0-9_.-]+)\.outputs")
_MATRIX_REF = re.compile(r"^\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}$")


class Unparsable(Exception):
    """A YAML file this check has to read does not parse."""


def _load(path: pathlib.Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as error:
        raise Unparsable(f"ERROR: cannot read {path}: {error}") from error


def _jobs(workflow: dict) -> dict:
    jobs = workflow.get("jobs") or {}
    return {name: job for name, job in jobs.items() if isinstance(job, dict)}


def _triggers(workflow: dict) -> dict:
    # yaml 1.1 parses the `on:` key as boolean True
    return workflow.get("on", workflow.get(True)) or {}


def hashed_inputs(path: pathlib.Path) -> list[tuple[str, ...]]:
    """The argument tuple of every ``hashFiles(...)`` cache key in a file."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return [tuple(_QUOTED.findall(args)) for args in _HASH_FILES.findall(text)]


def derived_prefix(action: pathlib.Path) -> str | None:
    """The prefix literal the action derives, `$(uname -m)` still in it."""
    with open(action, encoding="utf-8") as f:
        found = _DERIVED_PREFIX.findall(f.read())
    return found[0] if len(found) == 1 else None


def prefix_pattern(literal: str) -> "re.Pattern[str]":
    """Matches a producer prefix, capturing its arch and its interpreter tail.

    The tail is captured rather than fixed so that a prefix naming the wrong
    interpreters still yields its arch: the two halves are separate mistakes, and
    reporting the interpreter one should not also strand every consumer of that
    producer behind an "unknown job" cascade.
    """
    head, _, tail = literal.partition(_UNAME)
    arches = "|".join(re.escape(arch) for arch in sorted(set(RUNNER_ARCHES.values())))
    return re.compile(
        f"^{re.escape(head)}({arches})-(.+)$"
        if tail.startswith("-")
        else f"^{re.escape(head)}({arches}){re.escape(tail)}$"
    )


def restore_key_uses_derived_prefix(action: pathlib.Path) -> bool:
    """Whether the cache restore key interpolates the resolve step's output.

    The derived literal existing in the file proves nothing on its own: a key
    reverted to an input, or pointed at another step, would leave it dead and
    every consumer looking up a prefix nothing saved.
    """
    workflow = _load(action)
    resolve_ids = [
        step.get("id")
        for step in workflow.get("runs", {}).get("steps", [])
        if isinstance(step, dict) and _DERIVED_PREFIX.search(str(step.get("run", "")))
    ]
    if len(resolve_ids) != 1 or not resolve_ids[0]:
        return False
    wanted = f"steps.{resolve_ids[0]}.outputs"
    return any(
        wanted in str((step.get("with") or {}).get("key", ""))
        for step in workflow["runs"]["steps"]
        if isinstance(step, dict) and "cache/restore" in str(step.get("uses", ""))
    )


def interpreter_tokens(root: pathlib.Path) -> str:
    """The compile matrix's interpreters as the prefix spells them, e.g. cp310-cp312."""
    compile_job = _jobs(_load(root / BUILD_WORKFLOW))["compile"]
    versions = compile_job["strategy"]["matrix"]["python-version"]
    # Lexicographic, matching how stage_rust_ext_modules.sh sorts the suffix set.
    return "-".join(sorted(f"cp{str(v).replace('.', '')}" for v in versions))


def producers(root: pathlib.Path) -> list[dict]:
    """Every caller of the build workflow, with the prefix and labels it passes."""
    inputs = _triggers(_load(root / BUILD_WORKFLOW))["workflow_call"]["inputs"]
    build_workflow_name = BUILD_WORKFLOW.rsplit("/", 1)[-1]

    found = []
    for path in sorted(root.glob(WORKFLOW_GLOB)):
        for name, job in _jobs(_load(path)).items():
            if not str(job.get("uses", "")).endswith(build_workflow_name):
                continue
            with_ = job.get("with") or {}
            found.append(
                {
                    "file": str(path.relative_to(root)),
                    "job": name,
                    "prefix": with_.get(
                        "cache_key_prefix", inputs["cache_key_prefix"]["default"]
                    ),
                    "labels": {
                        field: str(
                            with_.get(field, inputs.get(field, {}).get("default", ""))
                        )
                        for field in ("runs_on", "collect_runs_on")
                    },
                }
            )
    return found


def matrix_values(job: dict, key: str) -> list[str]:
    """Every value `matrix.<key>` can take, across `include:` rows and plain lists."""
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    values = [str(value) for value in matrix.get(key, [])]
    values += [
        str(row[key])
        for row in matrix.get("include", [])
        if isinstance(row, dict) and key in row
    ]
    return values


def runner_configs_of(job: dict, with_: dict) -> list[str]:
    """The runner_configs one caller job resolves to, expanding a matrix reference."""
    declared = str(with_["runner_config"])
    reference = _MATRIX_REF.match(declared)
    if reference is None:
        return [declared] if "${{" not in declared else []
    return matrix_values(job, reference.group(1))


def stage_callers(root: pathlib.Path) -> list[dict]:
    """Every job calling a test stage, with the runner and artifact it names.

    A matrix row is its own caller here: one job can serve several pools, and it
    is the pool's arch that decides which prefix that install looks up.
    """
    found = []
    for path in sorted(root.glob(WORKFLOW_GLOB)):
        for name, job in _jobs(_load(path)).items():
            with_ = job.get("with") or {}
            if not str(job.get("uses", "")).endswith(STAGE_WORKFLOW):
                continue
            if "runner_config" not in with_:
                continue
            resolved = runner_configs_of(job, with_)
            if not resolved:
                found.append(
                    {
                        "file": str(path.relative_to(root)),
                        "job": name,
                        "runner_config": None,
                        "declared": str(with_["runner_config"]),
                        "producer_job": None,
                        "skips": False,
                    }
                )
                continue
            producer = _NEEDS_JOB.search(str(with_.get("rust_ext_artifact", "")))
            for runner_config in resolved:
                found.append(
                    {
                        "file": str(path.relative_to(root)),
                        "job": name,
                        "runner_config": runner_config,
                        "declared": runner_config,
                        "producer_job": producer.group(1) if producer else None,
                        "skips": bool(with_.get("skip_prebuilt_rust_ext")),
                    }
                )
    return found


def runner_config_labels(root: pathlib.Path) -> dict[str, str]:
    configs = _load(root / RUNNER_CONFIGS)["runner_configs"]
    return {name: str(config["runs_on"]) for name, config in configs.items()}


def _arch_of(label: str, where: str, problems: list[str]) -> str | None:
    arch = RUNNER_ARCHES.get(label)
    if arch is None:
        problems.append(
            f"ERROR: {where}: runner label '{label}' has no architecture on record. "
            f"Add it to RUNNER_ARCHES in {__file__.rsplit('/', 1)[-1]} - the arch is "
            f"half the cache key and cannot be read off the label text."
        )
    return arch


def check(root: pathlib.Path) -> list[str]:
    """Every problem found, one message per entry; empty means the sites agree."""
    problems: list[str] = []
    action = root / DOWNLOAD_ACTION

    try:
        literal = derived_prefix(action)
        if literal is None:
            return [
                f"ERROR: expected exactly one `rust-ext-{_UNAME}-...` literal in "
                f"{DOWNLOAD_ACTION}; this check cannot tell what consumers restore."
            ]
        if not restore_key_uses_derived_prefix(action):
            problems.append(
                f"ERROR: {DOWNLOAD_ACTION} derives '{literal}' but its cache restore "
                f"key does not interpolate that step's output, so consumers look up "
                f"a prefix no producer saves."
            )

        pattern = prefix_pattern(literal)
        abis = interpreter_tokens(root)
        found = producers(root)
        callers = stage_callers(root)
        labels = runner_config_labels(root)
        sites = [
            (BUILD_WORKFLOW, args) for args in hashed_inputs(root / BUILD_WORKFLOW)
        ]
        sites += [(DOWNLOAD_ACTION, args) for args in hashed_inputs(action)]
    except Unparsable as error:
        return problems + [str(error)]
    except (KeyError, TypeError) as error:
        return problems + [
            f"ERROR: {BUILD_WORKFLOW} or {RUNNER_CONFIGS} is not shaped as this "
            f"check expects ({error!r}); update the check with the change."
        ]

    if not found:
        return problems + [
            f"ERROR: no caller of {BUILD_WORKFLOW} found; this check is dead."
        ]

    producer_arches: dict[str, str] = {}
    for producer in found:
        where = f"{producer['file']}: job '{producer['job']}'"
        prefix = producer["prefix"]
        match = pattern.match(prefix)
        if not match:
            problems.append(
                f"ERROR: {where} saves under '{prefix}', which no consumer can "
                f"derive ({pattern.pattern}). Producer and consumer must agree, or "
                f"that pool source-builds."
            )
            continue
        arch = match.group(1)
        producer_arches[producer["job"]] = arch
        if match.lastindex == 2 and match.group(2) != abis:
            problems.append(
                f"ERROR: {where} saves under '{prefix}', but the compile matrix "
                f"builds {abis}. The interpreter set is part of the key because no "
                f"crate sets abi3, so no consumer would derive this one."
            )
        for field, label in producer["labels"].items():
            label_arch = _arch_of(label, f"{where} ({field})", problems)
            if label_arch is not None and label_arch != arch:
                problems.append(
                    f"ERROR: {where} saves under '{prefix}' but {field} is "
                    f"'{label}' ({label_arch}). The compile job must be the arch it "
                    f"builds for, and the collect job too - its GLIBC gate runs "
                    f"objdump on the .so."
                )

    consumer_arches = set()
    for caller in callers:
        where = f"{caller['file']}: job '{caller['job']}'"
        if caller["runner_config"] is None:
            problems.append(
                f"ERROR: {where}: cannot resolve runner_config "
                f"'{caller['declared']}' to a pool, so its architecture is unknown. "
                f"Pass a literal or a matrix reference this check can expand."
            )
            continue
        label = labels.get(caller["runner_config"])
        if label is None:
            problems.append(
                f"ERROR: {where}: runner_config '{caller['runner_config']}' is not in "
                f"{RUNNER_CONFIGS}."
            )
            continue
        arch = _arch_of(label, where, problems)
        if arch is None:
            continue
        if not caller["skips"]:
            consumer_arches.add(arch)
        producer_job = caller["producer_job"]
        if producer_job is None:
            continue
        producer_arch = producer_arches.get(producer_job)
        if producer_arch is None:
            problems.append(
                f"ERROR: {where} takes its artifact from '{producer_job}', which is "
                f"not a producer in that file."
            )
        elif producer_arch != arch:
            problems.append(
                f"ERROR: {where} runs on {arch} ({label}) but takes its artifact "
                f"from '{producer_job}', which builds {producer_arch}. The modules "
                f"would be ignored and the stage would compile during install."
            )

    orphaned = consumer_arches - set(producer_arches.values())
    if orphaned:
        problems.append(
            f"ERROR: stages run on {sorted(orphaned)} with no producer for it, so "
            f"they can never get a prebuild. Add a caller of {BUILD_WORKFLOW} for "
            f"that arch, or set skip_prebuilt_rust_ext on those stages."
        )

    # Adding a file to one key alone permanently misses the other's entries.
    if not sites:
        problems.append("ERROR: no hashFiles(...) cache key found; this check is dead.")
    elif len({args for _, args in sites}) > 1:
        detail = "\n".join(f"  {path}: {list(args)}" for path, args in sites)
        problems.append(
            "ERROR: rust-ext cache key inputs do not match. Every lookup/save/"
            f"restore site must hash the same inputs.\n{detail}"
        )

    return problems


def main() -> int:
    problems = check(pathlib.Path("."))
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
