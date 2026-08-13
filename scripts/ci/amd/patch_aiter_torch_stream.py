#!/usr/bin/env python3
"""Teach AITER's ctypes converter to accept a base torch.Stream.

On torch 2.11 Dynamo reconstructs a stream as a base torch.Stream across a graph
break, which AITER's torch_to_c_types rejects because it only accepts the cuda
subclass, so every --enable-torch-compile run dies with
"Unsupported type: torch.Stream". Only the subclass carries the raw handle --
stream_id is a pool index, not a pointer -- so rebuild the subclass from it
instead of passing it through.

Both docker/rocm.Dockerfile and scripts/ci/amd/amd_ci_install_dependency.sh run
this: the CI installer re-clones AITER when it rebuilds, which discards the
image's patched copy. Self-skips once ROCm/aiter carries the fix (unfixed on
main @ 4db400a9).
"""

import pathlib
import sys

OLD = """        elif isinstance(arg, torch.cuda.Stream):
            c_args.append(ctypes.cast(arg.cuda_stream, ctypes.c_void_p))
"""

NEW = """        elif isinstance(arg, torch.Stream):
            handle = getattr(arg, "cuda_stream", None)
            if handle is None:
                handle = torch.cuda.Stream(
                    stream_id=arg.stream_id,
                    device_index=arg.device_index,
                    device_type=arg.device_type,
                ).cuda_stream
            c_args.append(ctypes.cast(handle, ctypes.c_void_p))
"""

DEFAULT_TARGET = "/sgl-workspace/aiter/csrc/cpp_itfs/torch_utils.py"


def main(argv: list[str]) -> int:
    path = pathlib.Path(argv[1] if len(argv) > 1 else DEFAULT_TARGET)
    source = path.read_text()

    if OLD in source:
        path.write_text(source.replace(OLD, NEW))
        print(f"patched {path} (base torch.Stream handle recovery)")
        return 0

    assert (
        "isinstance(arg, torch.Stream)" in source
    ), f"FATAL: {path} no longer matches the stream patch"
    print(f"{path} already accepts a base torch.Stream; no patch needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
