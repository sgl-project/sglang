import ast
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

__all__ = [
    "HWBackend",
    "CIRegistry",
    "collect_tests",
    "auto_partition",
    "register_cpu_ci",
    "register_cuda_ci",
    "register_amd_ci",
    "register_npu_ci",
    "ut_parse_one_file",
]

# `suite` stays in positional slot 2 for backward compat with existing
# `register_cpu_ci(5, "stage-a-test-cpu")` style positional calls. New fields
# (`stage`, `runner_config`) are kwarg-only.
_PARAM_ORDER = ("est_time", "suite", "nightly", "disabled")
_KWARG_ONLY = ("stage", "runner_config")
_ALL_PARAMS = _PARAM_ORDER + _KWARG_ONLY
_UNSET = object()


class HWBackend(Enum):
    CPU = auto()
    CUDA = auto()
    AMD = auto()
    NPU = auto()


@dataclass
class CIRegistry:
    backend: HWBackend
    filename: str
    est_time: float
    stage: Optional[str] = None
    runner_config: Optional[str] = None
    # Legacy single-string suite; kept for nightly/stress/weekly + AMD/CPU/NPU
    # suites whose names don't follow `{stage}-test-{runner_config}` shape.
    suite: Optional[str] = None
    nightly: bool = False
    disabled: Optional[str] = None

    @property
    def effective_suite(self) -> Optional[str]:
        if self.stage is not None and self.runner_config is not None:
            return f"{self.stage}-test-{self.runner_config}"
        return self.suite


def register_cpu_ci(
    est_time: float,
    suite: Optional[str] = None,
    nightly: bool = False,
    disabled: Optional[str] = None,
    *,
    stage: Optional[str] = None,
    runner_config: Optional[str] = None,
):
    """Marker for CPU CI registration (parsed via AST; runtime no-op)."""
    return None


def register_cuda_ci(
    est_time: float,
    suite: Optional[str] = None,
    nightly: bool = False,
    disabled: Optional[str] = None,
    *,
    stage: Optional[str] = None,
    runner_config: Optional[str] = None,
):
    """Marker for CUDA CI registration (parsed via AST; runtime no-op)."""
    return None


def register_amd_ci(
    est_time: float,
    suite: Optional[str] = None,
    nightly: bool = False,
    disabled: Optional[str] = None,
    *,
    stage: Optional[str] = None,
    runner_config: Optional[str] = None,
):
    """Marker for AMD CI registration (parsed via AST; runtime no-op)."""
    return None


def register_npu_ci(
    est_time: float,
    suite: Optional[str] = None,
    nightly: bool = False,
    disabled: Optional[str] = None,
    *,
    stage: Optional[str] = None,
    runner_config: Optional[str] = None,
):
    """Marker for NPU CI registration (parsed via AST; runtime no-op)."""
    return None


REGISTER_MAPPING = {
    "register_cpu_ci": HWBackend.CPU,
    "register_cuda_ci": HWBackend.CUDA,
    "register_amd_ci": HWBackend.AMD,
    "register_npu_ci": HWBackend.NPU,
}


class RegistryVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.registries: list[CIRegistry] = []
        self.has_main_entry: bool = False

    def _constant_value(self, node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        return _UNSET

    def _parse_call_args(self, func_call: ast.Call) -> dict:
        args = {name: _UNSET for name in _ALL_PARAMS}
        seen = set()

        if any(isinstance(arg, ast.Starred) for arg in func_call.args):
            raise ValueError(
                f"{self.filename}: starred arguments are not supported in {func_call.func.id}()"
            )
        if len(func_call.args) > len(_PARAM_ORDER):
            raise ValueError(
                f"{self.filename}: too many positional arguments in {func_call.func.id}()"
            )

        for name, arg in zip(_PARAM_ORDER, func_call.args):
            seen.add(name)
            args[name] = self._constant_value(arg)

        for kw in func_call.keywords:
            if kw.arg is None:
                raise ValueError(
                    f"{self.filename}: **kwargs are not supported in {func_call.func.id}()"
                )
            if kw.arg not in args:
                raise ValueError(
                    f"{self.filename}: unknown argument '{kw.arg}' in {func_call.func.id}()"
                )
            if kw.arg in seen:
                raise ValueError(
                    f"{self.filename}: duplicated argument '{kw.arg}' in {func_call.func.id}()"
                )
            seen.add(kw.arg)
            args[kw.arg] = self._constant_value(kw.value)

        if args["est_time"] is _UNSET:
            raise ValueError(
                f"{self.filename}: est_time is a required constant in {func_call.func.id}()"
            )

        # The only valid (stage, runner_config, suite) shapes are:
        #   (set,   set,   unset)  -> new-style pair
        #   (unset, unset, set)    -> legacy single-string
        # Any other combination is rejected with the actual triple in the error.
        stage_set = args["stage"] is not _UNSET
        runner_set = args["runner_config"] is not _UNSET
        suite_set = args["suite"] is not _UNSET
        valid_shape = (stage_set and runner_set and not suite_set) or (
            not stage_set and not runner_set and suite_set
        )
        if not valid_shape:
            raise ValueError(
                f"{self.filename}: {func_call.func.id}() must specify exactly one of "
                f"(stage, runner_config) pair or suite; got stage={stage_set}, "
                f"runner_config={runner_set}, suite={suite_set}"
            )

        est_time = args["est_time"]
        if not isinstance(est_time, (int, float)):
            raise ValueError(
                f"{self.filename}: est_time must be a number in {func_call.func.id}()"
            )

        suite = args["suite"] if suite_set else None
        if suite is not None and not isinstance(suite, str):
            raise ValueError(
                f"{self.filename}: suite must be a string in {func_call.func.id}()"
            )

        stage = args["stage"] if stage_set else None
        runner_config = args["runner_config"] if runner_set else None
        for name, value in (("stage", stage), ("runner_config", runner_config)):
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"{self.filename}: {name} must be a string in {func_call.func.id}()"
                )

        nightly_value = args["nightly"]
        if nightly_value is _UNSET:
            nightly = False
        elif isinstance(nightly_value, bool):
            nightly = nightly_value
        else:
            raise ValueError(
                f"{self.filename}: nightly must be a boolean in {func_call.func.id}()"
            )

        disabled = args["disabled"] if args["disabled"] is not _UNSET else None
        if disabled is not None and not isinstance(disabled, str):
            raise ValueError(
                f"{self.filename}: disabled must be a string in {func_call.func.id}()"
            )

        return {
            "est_time": float(est_time),
            "stage": stage,
            "runner_config": runner_config,
            "suite": suite,
            "nightly": nightly,
            "disabled": disabled,
        }

    def _collect_ci_registry(self, func_call: ast.Call):
        if not isinstance(func_call.func, ast.Name):
            return None

        backend = REGISTER_MAPPING.get(func_call.func.id)
        if backend is None:
            return None

        parsed = self._parse_call_args(func_call)
        return CIRegistry(
            backend=backend,
            filename=self.filename,
            **parsed,
        )

    @staticmethod
    def _is_main_block_with_call(stmt: ast.If) -> bool:
        """True iff `stmt` is `if __name__ == "__main__":` with a body that
        contains at least one call (i.e. actually runs something, not just
        `pass`). This is what makes `python3 file.py` execute tests."""
        test = stmt.test
        if not isinstance(test, ast.Compare):
            return False
        if not (isinstance(test.left, ast.Name) and test.left.id == "__name__"):
            return False
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            return False
        if len(test.comparators) != 1:
            return False
        rhs = test.comparators[0]
        if not (isinstance(rhs, ast.Constant) and rhs.value == "__main__"):
            return False
        for child in ast.walk(ast.Module(body=stmt.body, type_ignores=[])):
            if isinstance(child, ast.Call):
                return True
        return False

    def visit_Module(self, node):
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                cr = self._collect_ci_registry(stmt.value)
                if cr is not None:
                    self.registries.append(cr)
            elif isinstance(stmt, ast.If) and self._is_main_block_with_call(stmt):
                self.has_main_entry = True

        self.generic_visit(node)


def ut_parse_one_file(filename: str) -> Tuple[List[CIRegistry], bool]:
    """Parse a test file and return (registries, has_main_entry).

    `has_main_entry` is True iff the file has `if __name__ == "__main__":`
    with a call in its body -- required for `python3 file.py` to actually
    run tests (the CI runner's invocation pattern).
    """
    with open(filename, "r") as f:
        file_content = f.read()
    tree = ast.parse(file_content, filename=filename)
    visitor = RegistryVisitor(filename=filename)
    visitor.visit(tree)
    return visitor.registries, visitor.has_main_entry


def auto_partition(files: List[CIRegistry], rank: int, size: int) -> List[CIRegistry]:
    """Partition files into `size` sublists with approximately equal sums of
    estimated times using a greedy algorithm (LPT heuristic), and return the
    partition for the specified rank.
    """
    if not files or size <= 0:
        return []

    # Sort by estimated_time descending; filename as tie-breaker for
    # deterministic partitioning regardless of glob ordering.
    sorted_files = sorted(files, key=lambda f: (-f.est_time, f.filename))

    partitions: List[List[CIRegistry]] = [[] for _ in range(size)]
    partition_sums = [0.0] * size

    # Greedily assign each file to the partition with the smallest current total time
    for file in sorted_files:
        min_sum_idx = min(range(size), key=partition_sums.__getitem__)
        partitions[min_sum_idx].append(file)
        partition_sums[min_sum_idx] += file.est_time

    if rank < size:
        return partitions[rank]
    return []


def collect_tests(files: list[str], sanity_check: bool = True) -> List[CIRegistry]:
    ci_tests = []
    for file in files:
        registries, has_main_entry = ut_parse_one_file(file)
        if len(registries) == 0:
            msg = f"No CI registry found in {file}"
            if sanity_check:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                continue

        # Every file with at least one enabled registry must have an
        # executable `if __name__ == "__main__":` block; otherwise
        # `python3 file.py -f` (how run_unittest_files invokes tests)
        # silently exits and the file shows green without running.
        has_enabled = any(r.disabled is None for r in registries)
        if sanity_check and has_enabled and not has_main_entry:
            raise ValueError(
                f'{file}: missing `if __name__ == "__main__":` entry. '
                f"Pytest-style tests in this file will silently skip under "
                f"`python3 file.py -f`. Add `unittest.main()` (for "
                f"unittest.TestCase) or `sys.exit(pytest.main([__file__, "
                f'"-v"]))` (for pytest-style).'
            )

        ci_tests.extend(registries)

    return ci_tests
