import ast
import logging
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class HWBackend(Enum):
    SKIP = auto()
    CUDA = auto()
    AMD = auto()


@dataclass
class CIRegistry:
    backend: HWBackend
    estimation_time: float
    stage: str


@dataclass
class CITest:
    filename: str
    testname: str
    ci_registry: list[CIRegistry] = field(default_factory=list)


REGISTER_MAPPING = {
    "skip_ci": HWBackend.SKIP,
    "register_cuda_ci": HWBackend.CUDA,
    "register_amd_ci": HWBackend.AMD,
}


def skip_ci():
    def wrapper(fn):
        return fn

    return wrapper


def register_cuda_ci(esimation_time: float, ci_stage: str):
    def wrapper(fn):
        return fn

    return wrapper


def register_amd_ci(esimation_time: float, ci_stage: str):
    def wrapper(fn):
        return fn

    return wrapper


class TestCaseVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.ci_tests: list[CITest] = []

    def _collect_ci_registry(self, dec):
        if not isinstance(dec, ast.Call):
            logger.info(f"skip non-call decorator: {ast.dump(dec)}")
            return None

        if not isinstance(dec.func, ast.Name):
            logger.info(f"skip non-name decorator: {ast.dump(dec)}")
            return None

        if dec.func.id not in REGISTER_MAPPING:
            return None

        hw = REGISTER_MAPPING[dec.func.id]
        if hw == HWBackend.SKIP:
            return CIRegistry(backend=hw, estimation_time=0, stage="")

        # parse arguments
        est_time = None
        ci_stage = None
        for kw in dec.keywords:
            if kw.arg == "esimation_time":
                if isinstance(kw.value, ast.Constant):
                    est_time = kw.value.value
            elif kw.arg == "ci_stage":
                if isinstance(kw.value, ast.Constant):
                    ci_stage = kw.value.value

        for i, arg in enumerate(dec.args):
            if isinstance(arg, ast.Constant):
                if i == 0:
                    est_time = arg.value
                elif i == 1:
                    ci_stage = arg.value

        assert (
            est_time is not None
        ), "esimation_time is required and should be a constant"
        assert ci_stage is not None, "ci_stage is required and should be a constant"
        return CIRegistry(backend=hw, estimation_time=est_time, stage=ci_stage)

    def visit_ClassDef(self, node):
        for base in node.bases:
            is_ci_test_case = isinstance(base, ast.Name) and base.id == "CustomTestCase"

            if not is_ci_test_case:
                continue

            ci_test = CITest(filename=self.filename, testname=node.name)
            self.ci_tests.append(ci_test)

            for dec in node.decorator_list:
                ci_registry = self._collect_ci_registry(dec)
                if ci_registry is not None:
                    ci_test.ci_registry.append(ci_registry)

        self.generic_visit(node)


def ut_parse_one_file(file_path: str):
    with open(file_path, "r") as f:
        file_content = f.read()
    tree = ast.parse(file_content, filename=file_path)
    visitor = TestCaseVisitor(file_path)
    visitor.visit(tree)
    return visitor


def collect_all_tests(files: list[str], sanity_check: bool = True) -> list[CITest]:
    ci_tests = []
    for file in files:
        visitor = ut_parse_one_file(file)
        if len(visitor.ci_tests) == 0:
            msg = f"No test cases found in {file}"
            if sanity_check:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                continue

        for reg in visitor.ci_tests:
            if len(reg.ci_registry) == 0:
                msg = f"No CI registry found in CustomTestCase {reg.testname} in {file}"
                if sanity_check:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)
                    continue

            if len(reg.ci_registry) > 1 and any(
                r.backend == HWBackend.SKIP for r in reg.ci_registry
            ):
                msg = f"Conflicting CI registry found in CustomTestCase {reg.testname} in {file}"
                if sanity_check:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)
                    continue

        ci_tests.extend(visitor.ci_tests)
