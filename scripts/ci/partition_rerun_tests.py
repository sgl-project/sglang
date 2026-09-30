import ast
import functools
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

# Run test has a 60-minute timeout; reserve a third for runtime variance/retries.
PARTITION_SECONDS = 40 * 60
# A single test estimated above this cannot finish inside the Run test step.
STEP_TIMEOUT_SECONDS = 60 * 60
# CPU suites whose per-commit job runs on ubuntu-latest, the only CPU pool here.
UBUNTU_CPU_SUITES = {"base-a-test-cpu"}
# rerun-test never builds sgl-kernel, so `$b200_runner` resolves to this pool.
B200_RERUN_RUNNER = "4-gpu-b200"

_REPO_ROOT = Path(__file__).resolve().parents[2]


@functools.cache
def _lpt():
    """Load partitioning.py by path; it has no sglang imports."""
    path = _REPO_ROOT / "python/sglang/multimodal_gen/test/partitioning.py"
    spec = importlib.util.spec_from_file_location("rerun_test_partitioning", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_runs_on(cfg):
    runs_on = cfg.get("runs_on")
    return B200_RERUN_RUNNER if runs_on == "$b200_runner" else runs_on


@functools.cache
def _runner_config_labels():
    try:
        sys.path.insert(0, str(_REPO_ROOT / "scripts/ci"))
        import runner_configs
    except ImportError:  # no PyYAML: fall back to the largest registration
        return {}
    return {name: resolve_runs_on(cfg) for name, cfg in runner_configs.load().items()}


def _registrations(path, register_name):
    """Return [(est_time, runner_config, suite)] or None if any estimate is unusable."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None
    found = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == register_name
        ):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        value = kwargs.get("est_time", node.args[0] if node.args else None)
        try:
            estimate = ast.literal_eval(value)
        except (ValueError, TypeError):
            return None
        if (
            type(estimate) not in (int, float)
            or not math.isfinite(estimate)
            or estimate <= 0
        ):
            return None
        tags = []
        for key in ("runner_config", "suite"):
            try:
                tags.append(ast.literal_eval(kwargs[key]))
            except (KeyError, ValueError, TypeError):
                tags.append(None)
        found.append((estimate, *tags))
    return found


@functools.cache
def _multimodal_estimates(root):
    sys.path.insert(0, str(_REPO_ROOT / "scripts/ci/utils/diffusion"))
    import diffusion_case_parser as parser

    run_suite = root / parser.RUN_SUITE_REL_PATH
    try:
        suites = parser.collect_diffusion_suites(
            parser.resolve_case_config_path(root, run_suite),
            run_suite,
            root / parser.BASELINE_REL_PATH,
        )
    except Exception:
        return {}, {}
    cases, standalone = {}, {}
    for info in suites.values():
        for case in info.cases:
            cases[case.case_id] = max(case.est_time, cases.get(case.case_id, 0))
        for name, est in info.standalone_est_times.items():
            standalone[name] = max(est, standalone.get(name, 0))
    return cases, standalone


def estimate_seconds(command, root, mode, runs_on=""):
    if mode == "multimodal_gen":
        filename, _, selector = command.split()[0].partition("::")
        cases, standalone = _multimodal_estimates(root)
        case = re.search(r"\[([^\]]+)\]", selector)
        if case:
            return cases.get(case.group(1))
        return standalone.get(filename, standalone.get(Path(filename).name))

    register_name = {"cuda": "register_cuda_ci", "cpu": "register_cpu_ci"}.get(mode)
    if register_name is None:
        return None
    filename = command.split()[0].split("::", 1)[0]
    found = _registrations(root / "test" / filename, register_name)
    if not found:
        return None
    if mode == "cuda":
        labels = _runner_config_labels()
        matching = [est for est, rc, _ in found if rc and labels.get(rc) == runs_on]
    else:
        matching = [est for est, _, suite in found if suite in UBUNTU_CPU_SUITES]
    # A file may register on multiple pools; without a match, stay conservative.
    return max(matching or [est for est, _, _ in found])


def _pack(items):
    count = math.ceil(sum(item.est_time for item in items) / PARTITION_SECONDS)
    while True:
        partitions = _lpt().partition_items_by_lpt(items, count)
        if all(sum(i.est_time for i in p) <= PARTITION_SECONDS for p in partitions):
            return [p for p in partitions if p]
        count += 1


def partition_commands(commands, root, mode, runs_on=""):
    order = [c.strip() for c in commands.splitlines() if c.strip()]
    if not order:
        raise ValueError("No test commands supplied")
    alone, items = [], []
    for idx, command in enumerate(order):
        estimate = estimate_seconds(command, root, mode, runs_on)
        if estimate is None or estimate > PARTITION_SECONDS:
            alone.append([idx])
        else:
            items.append(
                _lpt().PartitionItem(kind="test", item_id=str(idx), est_time=estimate)
            )
    batches = alone + [[int(i.item_id) for i in p] for p in _pack(items)]
    if len(batches) > 256:
        raise ValueError("Test commands exceed the 256-job matrix limit")
    batches = sorted(sorted(batch) for batch in batches)
    return {
        "include": [
            {"partition": n + 1, "test_command": "\n".join(order[i] for i in batch)}
            for n, batch in enumerate(batches)
        ]
    }


if __name__ == "__main__":
    matrix = partition_commands(
        os.environ["TEST_COMMAND"],
        Path(os.environ["TEST_ROOT"]),
        os.environ["MODE"],
        os.environ.get("RUNS_ON", ""),
    )
    for partition in matrix["include"]:
        print(f"Partition {partition['partition']}:\n{partition['test_command']}")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"matrix={json.dumps(matrix, separators=(',', ':'))}\n")
