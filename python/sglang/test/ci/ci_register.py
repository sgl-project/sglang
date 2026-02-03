import ast
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

__all__ = [
    "HWBackend",
    "CIRegistry",
    "collect_tests",
    "register_cpu_ci",
    "register_cuda_ci",
    "register_amd_ci",
    "register_npu_ci",
    "ut_parse_one_file",
]

_PARAM_ORDER = ("est_time", "suite", "nightly", "disabled")
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
    suite: str
    nightly: bool = False
    disabled: Optional[str] = None  # None = enabled, string = disabled with reason


def register_cpu_ci(
    est_time: float, suite: str, nightly: bool = False, disabled: Optional[str] = None
):
    """Marker for CPU CI registration (parsed via AST; runtime no-op)."""
    return None


def register_cuda_ci(
    est_time: float, suite: str, nightly: bool = False, disabled: Optional[str] = None
):
    """Marker for CUDA CI registration (parsed via AST; runtime no-op)."""
    return None


def register_amd_ci(
    est_time: float,
    suite: str,
    nightly: bool = False,
    disabled: Optional[str] = None,
):
    """Marker for AMD CI registration (parsed via AST; runtime no-op)."""
    return None


def register_npu_ci(
    est_time: float,
    suite: str,
    nightly: bool = False,
    disabled: Optional[str] = None,
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

    def _constant_value(self, node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        return _UNSET

    def _parse_call_args(
        self, func_call: ast.Call
    ) -> tuple[float, str, bool, Optional[str]]:
        args = {name: _UNSET for name in _PARAM_ORDER}
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

        if args["est_time"] is _UNSET or args["suite"] is _UNSET:
            raise ValueError(
                f"{self.filename}: est_time and suite are required constants in {func_call.func.id}()"
            )

        est_time, suite = args["est_time"], args["suite"]
        nightly_value = args["nightly"]

        if not isinstance(est_time, (int, float)):
            raise ValueError(
                f"{self.filename}: est_time must be a number in {func_call.func.id}()"
            )
        if not isinstance(suite, str):
            raise ValueError(
                f"{self.filename}: suite must be a string in {func_call.func.id}()"
            )

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

        return float(est_time), suite, nightly, disabled

    def _collect_ci_registry(self, func_call: ast.Call):
        if not isinstance(func_call.func, ast.Name):
            return None

        backend = REGISTER_MAPPING.get(func_call.func.id)
        if backend is None:
            return None

        est_time, suite, nightly, disabled = self._parse_call_args(func_call)
        return CIRegistry(
            backend=backend,
            filename=self.filename,
            est_time=est_time,
            suite=suite,
            nightly=nightly,
            disabled=disabled,
        )

    def visit_Module(self, node):
        for stmt in node.body:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue

            cr = self._collect_ci_registry(stmt.value)
            if cr is not None:
                self.registries.append(cr)

        self.generic_visit(node)


def ut_parse_one_file(filename: str) -> List[CIRegistry]:
    with open(filename, "r") as f:
        file_content = f.read()
    tree = ast.parse(file_content, filename=filename)
    visitor = RegistryVisitor(filename=filename)
    visitor.visit(tree)
    return visitor.registries


def collect_tests(files: list[str], sanity_check: bool = True) -> List[CIRegistry]:
    ci_tests = []
    for file in files:
        registries = ut_parse_one_file(file)
        if len(registries) == 0:
            msg = f"No CI registry found in {file}"
            if sanity_check:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                continue

        ci_tests.extend(registries)

    return ci_tests
