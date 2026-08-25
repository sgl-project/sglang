"""Transitional local actions for the existing sgl-kernel wheel builders."""

load("//bazel/rocm:toolchain.bzl", "AmdgpuTargetInfo")

_ROCM_TOOLCHAIN_TYPE = "//bazel/rocm:toolchain_type"
_UV_TOOLCHAIN_TYPE = "@rules_python//python/uv:uv_toolchain_type"

def _dependency_root(dep):
    files = dep[DefaultInfo].files.to_list()
    if not files:
        fail("{} does not provide any source files".format(dep.label))

    root = dep.label.workspace_root
    if not root:
        fail("{} is not in an external repository".format(dep.label))
    for file in files:
        if not file.path.startswith(root + "/"):
            fail("{} contains a file outside {}: {}".format(dep.label, root, file.path))
    return root

def _kernel_wheel_impl(ctx):
    wheel_dir = ctx.actions.declare_directory(ctx.label.name)
    dependency_args = sorted([
        "{}={}".format(fetchcontent_name, _dependency_root(dep))
        for dep, fetchcontent_name in ctx.attr.cmake_source_dirs.items()
    ])
    dependency_inputs = [
        dep[DefaultInfo].files
        for dep in ctx.attr.cmake_source_dirs.keys()
    ]
    action_env = {}
    toolchain_inputs = []
    if ctx.attr.backend == "rocm":
        amdgpu_target = ctx.attr._amdgpu_target[AmdgpuTargetInfo].value
        rocm_toolchain = ctx.toolchains[_ROCM_TOOLCHAIN_TYPE]
        uv_toolchain = ctx.toolchains[_UV_TOOLCHAIN_TYPE]
        uv_target = uv_toolchain.uv_toolchain_info.uv
        uv_executable = uv_target[DefaultInfo].files_to_run.executable
        action_env = {
            "AMDGPU_TARGET": amdgpu_target,
            "CXX": rocm_toolchain.cxx,
            "PYTHON_BIN_PATH": rocm_toolchain.python,
            "ROCM_HOME": rocm_toolchain.rocm_root,
            "SGL_KERNEL_BUILD_FRONTEND": uv_executable.path,
        }
        toolchain_inputs = [
            rocm_toolchain.inputs,
            uv_target[DefaultInfo].files,
        ]

    ctx.actions.run_shell(
        command = "exec /bin/bash \"$@\"",
        arguments = [
            ctx.file._builder.path,
            ctx.attr.backend,
            wheel_dir.path,
            ctx.attr.source_root,
        ] + dependency_args,
        inputs = depset(
            direct = ctx.files.srcs + [ctx.file._builder],
            transitive = dependency_inputs + toolchain_inputs,
        ),
        outputs = [wheel_dir],
        env = action_env,
        mnemonic = "SglKernelWheel",
        progress_message = "Building %{label} with the selected accelerator toolchain",
        use_default_shell_env = True,
        execution_requirements = {
            "block-network": "1",
            "local": "1",
            "no-cache": "1",
            "no-remote": "1",
            "no-sandbox": "1",
        },
    )
    return [DefaultInfo(files = depset([wheel_dir]))]

_kernel_wheel_attrs = {
    "backend": attr.string(
        mandatory = True,
        values = ["cpu", "cuda", "rocm"],
    ),
    "cmake_source_dirs": attr.label_keyed_string_dict(
        allow_files = True,
    ),
    "source_root": attr.string(mandatory = True),
    "srcs": attr.label_list(allow_files = True),
    "_builder": attr.label(
        allow_single_file = True,
        default = Label("//python/sglang/kernels/aot:bazel_build_wheel.sh"),
        cfg = "exec",
    ),
}

_kernel_wheel = rule(
    implementation = _kernel_wheel_impl,
    attrs = _kernel_wheel_attrs,
)

_rocm_kernel_wheel_attrs = dict(_kernel_wheel_attrs)
_rocm_kernel_wheel_attrs["_amdgpu_target"] = attr.label(
    default = Label("//bazel/rocm:amdgpu_target"),
    providers = [AmdgpuTargetInfo],
)

_rocm_kernel_wheel = rule(
    implementation = _kernel_wheel_impl,
    attrs = _rocm_kernel_wheel_attrs,
    toolchains = [
        _ROCM_TOOLCHAIN_TYPE,
        _UV_TOOLCHAIN_TYPE,
    ],
)

def kernel_wheel(name, backend, **kwargs):
    if backend == "rocm":
        _rocm_kernel_wheel(
            name = name,
            backend = backend,
            **kwargs
        )
    else:
        _kernel_wheel(
            name = name,
            backend = backend,
            **kwargs
        )

def _shell_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"

def _kernel_wheel_import_test_impl(ctx):
    wheel_files = ctx.attr.wheel[DefaultInfo].files.to_list()
    if len(wheel_files) != 1:
        fail("{} must provide exactly one wheel directory".format(ctx.attr.wheel.label))

    rocm_toolchain = ctx.toolchains[_ROCM_TOOLCHAIN_TYPE]
    launcher = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = launcher,
        content = """#!/usr/bin/env bash
set -euo pipefail
runfile() {{
  local path="$1"
  if [[ "${{path}}" == ../* ]]; then
    printf '%s/%s\\n' "${{TEST_SRCDIR}}" "${{path#../}}"
  else
    printf '%s/%s/%s\\n' "${{TEST_SRCDIR}}" "${{TEST_WORKSPACE}}" "${{path}}"
  fi
}}
export CXX={cxx}
export PYTHON_BIN_PATH={python}
export ROCM_HOME={rocm_root}
exec /bin/bash "$(runfile {tester})" "$(runfile {wheel})"
""".format(
            cxx = _shell_quote(rocm_toolchain.cxx),
            python = _shell_quote(rocm_toolchain.python),
            rocm_root = _shell_quote(rocm_toolchain.rocm_root),
            tester = _shell_quote(ctx.file._tester.short_path),
            wheel = _shell_quote(wheel_files[0].short_path),
        ),
        is_executable = True,
    )
    return [
        DefaultInfo(
            executable = launcher,
            runfiles = ctx.runfiles(
                files = [
                    ctx.file._tester,
                    wheel_files[0],
                ],
                transitive_files = rocm_toolchain.inputs,
            ),
        ),
    ]

kernel_wheel_import_test = rule(
    implementation = _kernel_wheel_import_test_impl,
    attrs = {
        "wheel": attr.label(mandatory = True),
        "_tester": attr.label(
            allow_single_file = True,
            default = Label("//python/sglang/kernels/aot:wheel_import_test.sh"),
        ),
    },
    test = True,
    toolchains = [_ROCM_TOOLCHAIN_TYPE],
)
