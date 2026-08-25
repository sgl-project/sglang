"""Transitional local actions for the existing sgl-kernel wheel builders."""

def _kernel_wheel_impl(ctx):
    wheel_dir = ctx.actions.declare_directory(ctx.label.name)
    ctx.actions.run_shell(
        command = "exec /bin/bash \"$1\" \"$2\" \"$3\" \"$4\"",
        arguments = [
            ctx.file._builder.path,
            ctx.attr.backend,
            wheel_dir.path,
            ctx.attr.source_root,
        ],
        inputs = depset(ctx.files.srcs + [ctx.file._builder]),
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
        "source_root": attr.string(mandatory = True),
        "srcs": attr.label_list(allow_files = True),
        "_builder": attr.label(
            allow_single_file = True,
            default = Label("//python/sglang/kernels/aot:bazel_build_wheel.sh"),
            cfg = "exec",
        ),
    },
)
