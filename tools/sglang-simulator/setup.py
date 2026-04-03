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
    url="https://github.com/sgl-project/sglang.git",
    description="A High-Fidelity LLM inference simulator for SGLang",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
