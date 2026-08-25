"""Hermetic assembly of the main SGLang wheel from Bazel-owned artifacts."""

def _source_arg(file):
    return "{}::={}".format(file.short_path, file.path)

def _main_wheel_impl(ctx):
    version = ctx.var.get("SGLANG_WHEEL_VERSION", "")
    if not version:
        fail(
            "SGLANG_WHEEL_VERSION is required; pass " +
            "--define=SGLANG_WHEEL_VERSION=<PEP-440-version>",
        )
    if "/" in version:
        fail("SGLANG_WHEEL_VERSION may not contain '/'")

    tag = "{}-{}-{}".format(
        ctx.attr.python_tag,
        ctx.attr.abi,
        ctx.attr.platform,
    )
    output = ctx.actions.declare_file(
        "{}-{}-{}.whl".format(ctx.attr.distribution, version, tag),
    )

    args = ctx.actions.args()
    args.add("--output", output.path)
    args.add("--distribution", ctx.attr.distribution)
    args.add("--version", version)
    args.add("--tag", tag)
    args.add("--extension-suffix", ctx.attr.extension_suffix)
    args.add("--pyproject", ctx.file.pyproject.path)
    args.add("--readme", ctx.file.readme.path)
    args.add_all(
        ctx.files.srcs,
        before_each = "--source",
        map_each = _source_arg,
    )

    native_inputs = []
    for target, module in ctx.attr.native_extensions.items():
        files = target[DefaultInfo].files.to_list()
        if len(files) != 1:
            fail(
                "{} must provide exactly one native library, got {}".format(
                    target.label,
                    files,
                ),
            )
        native_inputs.append(files[0])
        args.add(
            "--native-extension",
            "{}::={}".format(module, files[0].path),
        )

    args.use_param_file("@%s", use_always = True)
    args.set_param_file_format("multiline")
    ctx.actions.run(
        executable = ctx.executable._builder,
        arguments = [args],
        inputs = depset(
            ctx.files.srcs +
            native_inputs +
            [
                ctx.file.pyproject,
                ctx.file.readme,
            ],
        ),
        outputs = [output],
        mnemonic = "SglangMainWheel",
        progress_message = "Assembling Bazel-owned main wheel %{label}",
        tools = [ctx.executable._builder],
    )
    return [DefaultInfo(files = depset([output]))]

main_wheel = rule(
    implementation = _main_wheel_impl,
    attrs = {
        "abi": attr.string(mandatory = True),
        "distribution": attr.string(mandatory = True),
        "extension_suffix": attr.string(mandatory = True),
        "native_extensions": attr.label_keyed_string_dict(allow_files = True),
        "platform": attr.string(mandatory = True),
        "pyproject": attr.label(
            allow_single_file = [".toml"],
            mandatory = True,
        ),
        "python_tag": attr.string(mandatory = True),
        "readme": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "srcs": attr.label_list(allow_files = True),
        "_builder": attr.label(
            cfg = "exec",
            default = Label("//bazel/packaging:build_main_wheel"),
            executable = True,
        ),
    },
)
