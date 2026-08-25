"""ROCm build settings and the transitional local ROCm toolchain."""

SUPPORTED_AMDGPU_TARGETS = [
    "gfx942",
    "gfx950",
    "gfx1250",
]

AmdgpuTargetInfo = provider(
    doc = "The AMD GPU architecture selected for a ROCm build.",
    fields = {
        "value": "A validated AMDGPU_TARGET value.",
    },
)

def _amdgpu_target_impl(ctx):
    value = ctx.build_setting_value
    if not value:
        fail(
            "AMDGPU_TARGET is required for ROCm builds; pass " +
            "--//bazel/rocm:amdgpu_target=<{}>".format(
                "|".join(SUPPORTED_AMDGPU_TARGETS),
            ),
        )
    if value not in SUPPORTED_AMDGPU_TARGETS:
        fail(
            "unsupported AMDGPU_TARGET '{}'; expected one of {}".format(
                value,
                ", ".join(SUPPORTED_AMDGPU_TARGETS),
            ),
        )
    return [AmdgpuTargetInfo(value = value)]

amdgpu_target = rule(
    implementation = _amdgpu_target_impl,
    build_setting = config.string(flag = True),
)

def _rocm_toolchain_impl(ctx):
    return [
        platform_common.ToolchainInfo(
            cxx = ctx.attr.cxx,
            inputs = depset(ctx.files.inputs),
            python = ctx.attr.python,
            rocm_root = ctx.attr.rocm_root,
        ),
    ]

rocm_toolchain = rule(
    implementation = _rocm_toolchain_impl,
    attrs = {
        "cxx": attr.string(mandatory = True),
        "inputs": attr.label_list(allow_files = True),
        "python": attr.string(mandatory = True),
        "rocm_root": attr.string(mandatory = True),
    },
)
