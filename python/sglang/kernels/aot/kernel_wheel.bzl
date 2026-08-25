"""Transitional local actions for the existing sgl-kernel wheel builders."""

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
            transitive = dependency_inputs,
        ),
        outputs = [wheel_dir],
        mnemonic = "SglKernelWheel",
        progress_message = "Building %{label} with the local PyTorch toolchain",
        use_default_shell_env = True,
        execution_requirements = {
            "local": "1",
            "no-cache": "1",
            "no-remote": "1",
            "no-sandbox": "1",
        },
    )
    return [DefaultInfo(files = depset([wheel_dir]))]

kernel_wheel = rule(
    implementation = _kernel_wheel_impl,
    attrs = {
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
    },
)
