import argparse
import os
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser(
    description="Bind an unmodified vLLM CopySFA adapter to its installed OPP."
)
parser.add_argument("--vllm-csrc", type=Path, required=True)
parser.add_argument("--build-dir", type=Path, required=True)
args = parser.parse_args()
task = Path(__file__).parent
source = args.vllm_csrc.resolve()
build = args.build_dir.resolve()
build.mkdir(parents=True, exist_ok=True)
npu = Path(torch_npu.__file__).parent
os.environ["MAX_JOBS"] = "4"
library = load(
    name="vllm_copy_reference",
    sources=[
        str(task / "vllm_copy_reference.cpp"),
        str(source / "aclnn_torch_adapter/NPUBridge.cpp"),
        str(source / "aclnn_torch_adapter/NPUStorageImpl.cpp"),
    ],
    extra_include_paths=[str(npu / "include"), str(source)],
    extra_cflags=["-O2", "-std=c++17"],
    extra_ldflags=["-L" + str(npu / "lib"), "-ltorch_npu", "-ldl"],
    is_python_module=False,
    with_cuda=False,
    build_directory=str(build),
    verbose=True,
)
print(library)
