import os

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

BOOTSTRAP_PTH = "src/sglang_simulator_bootstrap.pth"


class BuildPyWithBootstrap(build_py):
    def run(self):
        super().run()
        self.copy_file(
            BOOTSTRAP_PTH,
            os.path.join(self.build_lib, os.path.basename(BOOTSTRAP_PTH)),
        )

    def get_outputs(self, include_bytecode=1):
        return [
            *super().get_outputs(include_bytecode),
            os.path.join(self.build_lib, os.path.basename(BOOTSTRAP_PTH)),
        ]

    def get_source_files(self):
        return [*super().get_source_files(), BOOTSTRAP_PTH]


def get_version():
    version = "0.1.0"
    with open("src/sglang_simulator/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip(' \n"')
    return version


setup(
    name="sglang-simulator",
    version=get_version(),
    url="https://github.com/sgl-project/sglang.git",
    description="A High-Fidelity LLM inference simulator for SGLang",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmdclass={"build_py": BuildPyWithBootstrap},
    package_data={
        "sglang_simulator.tools.collect_baseline": ["README.md"],
        "sglang_simulator.tools.train_ml_time_predictor": [
            "README.md",
            "example_steps.jsonl",
        ],
    },
)
