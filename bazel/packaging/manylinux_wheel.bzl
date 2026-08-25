"""Repair and validate a direct Bazel wheel for a manylinux policy."""

def _manylinux_wheel_impl(ctx):
    wheel = ctx.file.wheel
    input_tag = "{}-{}-{}".format(
        ctx.attr.python_tag,
        ctx.attr.abi,
        ctx.attr.input_platform,
    )
    output_tag = "{}-{}-{}".format(
        ctx.attr.python_tag,
        ctx.attr.abi,
        ctx.attr.output_platform,
    )
    input_suffix = "-{}.whl".format(input_tag)
    if not wheel.basename.endswith(input_suffix):
        fail(
            "{} does not end in expected tag {}".format(
                wheel.basename,
                input_suffix,
            ),
        )

    output = ctx.actions.declare_file(
        wheel.basename.removesuffix(input_suffix) + "-{}.whl".format(output_tag),
    )
    audit = ctx.actions.declare_file(output.basename + ".audit.json")

    args = ctx.actions.args()
    args.add("--input", wheel.path)
    args.add("--output", output.path)
    args.add("--audit-report", audit.path)
    args.add("--input-tag", input_tag)
    args.add("--output-tag", output_tag)
    args.add("--policy", ctx.attr.output_platform)
    args.add("--extension-suffix", ctx.attr.extension_suffix)
    args.add("--max-glibc", ctx.attr.max_glibc)
    args.add("--expect-auditwheel", ctx.attr.auditwheel_version)
    args.add("--expect-patchelf", ctx.attr.patchelf_version)
    args.add_all(
        sorted(ctx.attr.native_modules),
        before_each = "--native-module",
    )

    ctx.actions.run(
        executable = ctx.executable._repairer,
        arguments = [args],
        inputs = [wheel],
        outputs = [output, audit],
        mnemonic = "SglangManylinuxWheel",
        progress_message = "Repairing and auditing manylinux wheel %{label}",
        tools = [ctx.executable._repairer],
    )
    return [
        DefaultInfo(files = depset([output])),
        OutputGroupInfo(audit = depset([audit])),
    ]

manylinux_wheel = rule(
    implementation = _manylinux_wheel_impl,
    attrs = {
        "abi": attr.string(mandatory = True),
        "auditwheel_version": attr.string(mandatory = True),
        "extension_suffix": attr.string(mandatory = True),
        "input_platform": attr.string(mandatory = True),
        "max_glibc": attr.string(mandatory = True),
        "native_modules": attr.string_list(mandatory = True),
        "output_platform": attr.string(mandatory = True),
        "patchelf_version": attr.string(mandatory = True),
        "python_tag": attr.string(mandatory = True),
        "wheel": attr.label(
            allow_single_file = [".whl"],
            mandatory = True,
        ),
        "_repairer": attr.label(
            cfg = "exec",
            default = Label("//bazel/packaging:repair_manylinux_wheel"),
            executable = True,
        ),
    },
)
