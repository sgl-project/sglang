"""Build this checkout's exact Top-K dependency with the measured compiler flags."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".dosa_topk"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    import os
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
    os.environ.setdefault("MAX_JOBS", "2")
    import sgl_kernel
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    production = root / "python/sglang/kernels/aot/csrc/elementwise/topk.cu"
    text = production.read_text()
    source = output / "topk.cu"
    source.write_text(text.replace("namespace {", "namespace dosa_exact_topk {\nnamespace {", 1)
                      + "\n}  // namespace dosa_exact_topk\n")
    bindings = output / "bindings.cpp"
    bindings.write_text('''#include <ATen/core/Tensor.h>
#include <torch/library.h>
#include <optional>
namespace dosa_exact_topk {
void fast_topk_interface(const at::Tensor&, at::Tensor&, const at::Tensor&, std::optional<at::Tensor>);
void fast_topk_transform_interface(const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&, const at::Tensor&, std::optional<at::Tensor>);
void fast_topk_transform_ragged_interface(const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&, std::optional<at::Tensor>);
}
TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("fast_topk", &dosa_exact_topk::fast_topk_interface);
  m.impl("fast_topk_transform_fused", &dosa_exact_topk::fast_topk_transform_interface);
  m.impl("fast_topk_transform_ragged_fused", &dosa_exact_topk::fast_topk_transform_ragged_interface);
}
''')
    build = output / "build"
    build.mkdir(exist_ok=True)
    binary = load(
        name="dosa_exact_topk_main", sources=[str(source), str(bindings)],
        build_directory=str(build), with_cuda=True, is_python_module=False,
        extra_cflags=["-O3", "-std=c++20", "-fvisibility=hidden"],
        extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "-lineinfo", "-Xptxas=-v",
                           "-Xcompiler=-fvisibility=hidden"], verbose=True,
    )
    # Preserve the installed package; only the three committed CUDA selectors
    # are registered after it establishes the normal operator schemas.
    package = Path(sgl_kernel.__file__).resolve().parent
    overlay = output / "python/sgl_kernel"
    overlay.mkdir(parents=True, exist_ok=True)
    for item in package.iterdir():
        if item.name in ("__init__.py", "__pycache__"):
            continue
        target = overlay / item.name
        if not target.exists():
            target.symlink_to(item, target_is_directory=item.is_dir())
    (overlay / "__init__.py").write_text(
        (package / "__init__.py").read_text()
        + "\nimport torch as _dosa_torch\n_dosa_torch.ops.load_library(" + repr(str(binary)) + ")\n"
    )
    print(f"Exact Top-K built from {production}")
    print(f"Add {overlay.parent} to PYTHONPATH for both baseline and DoSA.")


if __name__ == "__main__":
    main()
