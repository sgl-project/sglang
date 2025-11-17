import ast
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class HWBackend(Enum):
    CPU = auto()
    CUDA = auto()
    AMD = auto()


@dataclass
class CIRegistry:
    backend: HWBackend
    filename: str
    est_time: float
    suite: str


def register_cpu_ci(est_time: float, suite: str):
    pass


def register_cuda_ci(est_time: float, suite: str):
    pass


def register_amd_ci(est_time: float, suite: str):
    pass


REGISTER_MAPPING = {
    "register_cpu_ci": HWBackend.CPU,
    "register_cuda_ci": HWBackend.CUDA,
    "register_amd_ci": HWBackend.AMD,
}


class RegistryVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.registries: list[CIRegistry] = []

    def _collect_ci_registry(self, func_call: ast.Call):
        if not isinstance(func_call.func, ast.Name):
            return None

        if func_call.func.id not in REGISTER_MAPPING:
            return None

        hw = REGISTER_MAPPING[func_call.func.id]
        est_time, suite = None, None
        for kw in func_call.keywords:
            if kw.arg == "est_time":
                if isinstance(kw.value, ast.Constant):
                    est_time = kw.value.value
            elif kw.arg == "suite":
                if isinstance(kw.value, ast.Constant):
                    suite = kw.value.value

        for i, arg in enumerate(func_call.args):
            if isinstance(arg, ast.Constant):
                if i == 0:
                    est_time = arg.value
                elif i == 1:
                    suite = arg.value
        assert (
            est_time is not None
        ), "esimation_time is required and should be a constant"
        assert suite is not None, "suite is required and should be a constant"
        return CIRegistry(
            backend=hw, filename=self.filename, est_time=est_time, suite=suite
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
