import argparse
import ast
import logging
from dataclasses import dataclass, field
from enum import Enum, auto

from sglang.test.test_utils import CustomTestCase

logger = logging.getLogger(__name__)


class HWBackend(Enum):
    SKIP = auto()
    CUDA = auto()
    AMD = auto()


@dataclass
class HWConfig:
    backend: HWBackend
    estimation_time: float
    stage: str


@dataclass
class CITest:
    filename: str
    testname: str
    hw_configs: list[HWConfig] = field(default_factory=list)


REGISTER_MAPPING = {
    "skip_ci": HWBackend.SKIP,
    "register_cuda_ci": HWBackend.CUDA,
    "register_amd_ci": HWBackend.AMD,
}


def skip_ci(reason: str):
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
    def __init__(self):
        self.has_custom_test_case = False
        self.ut_registries = []

    def _collect_ci(self, dec):
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
            return HWConfig(backend=hw, estimation_time=0, stage="")

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
        return HWConfig(backend=hw, estimation_time=est_time, stage=ci_stage)

    def visit_ClassDef(self, node):
        for base in node.bases:
            is_ci_test_case = isinstance(base, ast.Name) and base.id == "CustomTestCase"

            if not is_ci_test_case:
                continue

            self.has_custom_test_case = True
            for dec in node.decorator_list:
                hw_config = self._collect_ci(dec)
                if hw_config is not None:
                    self.ut_registries.append(hw_config)

        self.generic_visit(node)


@register_cuda_ci(esimation_time=300, ci_stage="1-gpu")
class TestGGUF(CustomTestCase):
    def test_models(self):
        pass


def test_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    with open(args.file, "r") as f:
        file_content = f.read()
    tree = ast.parse(file_content, filename=args.file)
    visitor = TestCaseVisitor()
    visitor.visit(tree)

    print(f"{visitor.has_custom_test_case=}")
    for reg in visitor.ut_registries:
        print(f"{reg=}")


if __name__ == "__main__":
    test_main()
