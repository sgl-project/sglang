import importlib
import os
import pkgutil
import subprocess
import sys
import warnings
from collections.abc import Sequence
from typing import Any

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

EXCLUDE_MODELS = {
    "python.sglang.srt.models.registry",
    "python.sglang.srt.models.torch_native_llama",
}

DOC = "docs/references/supported_models.md"


def get_edited_model_files() -> set[str]:
    cmd = ("git", "diff", "--staged", "--name-only", "--raw")
    all_files = cmd_output(*cmd).splitlines()
    model_files = set()
    for file in all_files:
        if file.startswith("python/sglang/srt/models"):
            model_files.add(file[:-3].replace("/", "."))
    return model_files


def cmd_output(*cmd: str, retcode: int | None = 0, **kwargs: Any) -> str:
    kwargs.setdefault("stdout", subprocess.PIPE)
    kwargs.setdefault("stderr", subprocess.PIPE)
    proc = subprocess.Popen(cmd, **kwargs)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode()
    if retcode is not None and proc.returncode != retcode:
        raise RuntimeError(cmd, retcode, proc.returncode, stdout, stderr)
    return stdout


def get_supported_models_in_package() -> set[str]:
    model_names = set()
    edited_model_files = get_edited_model_files()
    if edited_model_files:
        package_name = "python.sglang.srt.models"
        package = importlib.import_module(package_name)

        for _, name, ispkg in pkgutil.iter_modules(
            package.__path__, package_name + "."
        ):
            if not ispkg:
                if name in EXCLUDE_MODELS or name not in edited_model_files:
                    continue
                try:
                    module = importlib.import_module(name)
                except Exception as e:
                    continue
                if hasattr(module, "EntryClass"):
                    entry = module.EntryClass
                    if isinstance(entry, list):
                        for tmp in entry:
                            assert isinstance(
                                tmp, tuple
                            ), f"Please add model name for EntryClass {tmp}"
                            model_names.add(tmp[1])
                    else:
                        assert isinstance(
                            entry, tuple
                        ), f"Please add model name for EntryClass {entry}"
                        model_names.add(entry[1])

    return model_names


def format(model_name: str) -> str:
    """Handle special format int doc"""
    model_name = model_name.strip()
    if model_name.startswith("llava 1.5"):
        return "llava"
    elif model_name and model_name[0] == "[":
        model_name = model_name.split("]")[0][1:]
    return model_name.strip()


def get_supported_models_in_doc() -> set[str]:
    models = []
    with open(DOC, "r") as f:
        for line in f:
            if line.startswith("- "):
                models.extend(list(map(format, line[2:].lower().split("/"))))
            elif line == "## How to Support a New Language Model\n":
                break
    return set(models)


def check_if_doc_update_to_date(
    models_in_pkg: set[str], models_in_doc: set[str]
) -> bool:
    """True if doc includes all models in package"""
    models_diff = []
    for name in models_in_pkg:
        name_case = name.lower()
        if name_case not in models_in_doc:
            models_diff.append(name)
    if models_diff:
        print(f"Please add the following missing models in {DOC}")
        for model in models_diff:
            print(model)
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    models_in_pkg = get_supported_models_in_package()
    models_in_doc = get_supported_models_in_doc()
    return not check_if_doc_update_to_date(models_in_pkg, models_in_doc)


if __name__ == "__main__":
    raise SystemExit(main())
