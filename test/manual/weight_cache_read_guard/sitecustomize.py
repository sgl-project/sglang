# SPDX-License-Identifier: Apache-2.0
"""Opt-in manual-test subprocess guard; never on production PYTHONPATH.

Allow metadata/header inspection, reject cached checkpoint tensor retrieval.
Each fresh launcher/spawned interpreter records installation and tensor accesses.
This covers the safetensors/Torch loading APIs used by the admitted native loader,
not arbitrary native reads or a security boundary against hostile code.
"""

import json
import os
from pathlib import Path


def install():
    import safetensors
    import safetensors.torch
    import torch

    targets = {
        str(Path(path).resolve())
        for path in json.loads(os.environ["WC_TEST_BLOCKED_FILES"])
    }
    directory = Path(os.environ["WC_TEST_GUARD_LOG_DIR"])

    def record(event, **fields):
        with (directory / f"{os.getpid()}.jsonl").open("a") as out:
            out.write(json.dumps({"event": event, "pid": os.getpid(), **fields}) + "\n")

    def access(path, api):
        resolved = str(Path(path).resolve())
        blocked = resolved in targets
        record(
            "blocked_tensor" if blocked else "uncached_tensor", path=resolved, api=api
        )
        if blocked:
            raise RuntimeError(f"WC_TEST_CACHED_TENSOR_READ: {api}: {resolved}")

    original = safetensors.safe_open

    class GuardedSlice:
        def __init__(self, value, path):
            self.value = value
            self.path = path

        def __getattr__(self, name):
            return getattr(self.value, name)

        def __getitem__(self, index):
            access(self.path, "safe_open.get_slice[]")
            return self.value[index]

    class GuardedFile:
        def __init__(self, filename, *args, **kwargs):
            self.path = filename
            self.value = original(filename, *args, **kwargs)

        def __enter__(self):
            self.value.__enter__()
            return self

        def __exit__(self, *args):
            return self.value.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.value, name)

        def get_tensor(self, key):
            access(self.path, "safe_open.get_tensor")
            return self.value.get_tensor(key)

        def get_slice(self, key):
            return GuardedSlice(self.value.get_slice(key), self.path)

    safetensors.safe_open = GuardedFile
    safetensors.torch.safe_open = GuardedFile
    original_torch_load = torch.load

    def guarded_torch_load(file, *args, **kwargs):
        path = (
            file
            if isinstance(file, (str, os.PathLike))
            else getattr(file, "name", None)
        )
        if isinstance(path, (str, os.PathLike)):
            access(path, "torch.load")
        return original_torch_load(file, *args, **kwargs)

    torch.load = guarded_torch_load
    record("installed")


if os.environ.get("WC_TEST_BLOCKED_FILES"):
    # A sitecustomize exception normally only prints a warning and continues.
    # An incomplete guard must instead make the test subprocess fail closed.
    try:
        install()
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(91)
