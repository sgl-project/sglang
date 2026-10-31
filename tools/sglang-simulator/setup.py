from setuptools import find_packages, setup


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
    python_requires=">=3.10",
    url="https://github.com/sgl-project/sglang.git",
    description="A High-Fidelity LLM inference simulator for SGLang",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["usercustomize"],
    entry_points={
        "console_scripts": [
            "sglang-simulator-afd=sglang_simulator.afd.cli:main",
            "sglang-simulator-collect=sglang_simulator.tools.collect_baseline.collect:main",
            "sglang-simulator-train=sglang_simulator.tools.train_ml_time_predictor.train:main",
        ],
    },
    package_data={
        "sglang_simulator.tools.collect_baseline": ["README.md"],
        "sglang_simulator.tools.train_ml_time_predictor": [
            "README.md",
            "example_steps.jsonl",
        ],
    },
)
