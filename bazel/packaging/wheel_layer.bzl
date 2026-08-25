"""Build an OCI filesystem layer by installing one wheel."""

def _wheel_layer_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".tar")
    args = ctx.actions.args()
    args.add("--wheel", ctx.file.wheel.path)
    args.add("--output", output.path)
    args.add("--site-packages", ctx.attr.site_packages)

    ctx.actions.run(
        executable = ctx.executable._builder,
        arguments = [args],
        inputs = [ctx.file.wheel],
        mnemonic = "SglangWheelLayer",
        outputs = [output],
        progress_message = "Installing %{label} into an OCI layer",
        tools = [ctx.executable._builder],
    )
    return [DefaultInfo(files = depset([output]))]

wheel_layer = rule(
    implementation = _wheel_layer_impl,
    attrs = {
        "site_packages": attr.string(mandatory = True),
        "wheel": attr.label(
            allow_single_file = [".whl"],
            mandatory = True,
        ),
        "_builder": attr.label(
            cfg = "exec",
            default = Label("//bazel/packaging:build_wheel_layer"),
            executable = True,
        ),
    },
)
