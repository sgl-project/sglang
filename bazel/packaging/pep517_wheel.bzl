"""Transitional, explicitly non-hermetic PEP 517 wheel action."""

def _pep517_wheel_impl(ctx):
    wheel_dir = ctx.actions.declare_directory(ctx.label.name)
    arguments = ctx.actions.args()
    arguments.add(wheel_dir.path)
    arguments.add_all(ctx.files.srcs)
    arguments.use_param_file("@%s", use_always = True)
    arguments.set_param_file_format("multiline")

    ctx.actions.run(
        executable = ctx.executable.builder,
        arguments = [arguments],
        inputs = depset(ctx.files.srcs),
        outputs = [wheel_dir],
        tools = [ctx.executable.builder],
        env = {
            "SGLANG_PEP517_TRANSITIONAL_ACTION": "1",
        },
        execution_requirements = {
            "local": "1",
            "no-cache": "1",
            "no-remote-cache": "1",
            "no-remote-exec": "1",
            "no-sandbox": "1",
        },
        mnemonic = "TransitionalPep517Wheel",
        progress_message = "Building non-hermetic PEP 517 wheel %{label}",
        use_default_shell_env = True,
    )

    return [DefaultInfo(files = depset([wheel_dir]))]

pep517_wheel = rule(
    implementation = _pep517_wheel_impl,
    attrs = {
        "builder": attr.label(
            cfg = "exec",
            executable = True,
            mandatory = True,
        ),
        "srcs": attr.label_list(allow_files = True),
    },
)
