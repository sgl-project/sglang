# SPDX-License-Identifier: Apache-2.0
"""Role definitions for diffusion pipeline disaggregation."""

from enum import Enum

_ROLE_ALIASES = {"denoising": "denoiser"}


class RoleType(str, Enum):
    MONOLITHIC = "monolithic"
    ENCODER = "encoder"
    DENOISER = "denoiser"
    DECODER = "decoder"
    SERVER = "server"  # Head node (no GPU, routes requests)

    @classmethod
    def from_string(cls, value: str) -> "RoleType":
        v = _ROLE_ALIASES.get(value.lower(), value.lower())
        try:
            return cls(v)
        except ValueError:
            raise ValueError(
                f"Invalid role: {value}. Must be one of: {', '.join([r.value for r in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        return [role.value for role in cls] + sorted(_ROLE_ALIASES)


def get_module_role(module_name: str) -> "RoleType | None":
    """Classify a module name to its primary role. Returns None for shared modules."""
    encoder_prefixes = (
        "text_encoder",
        "tokenizer",
        "image_encoder",
        "image_processor",
        "processor",
        "connectors",
        "vision_language_encoder",
    )
    if any(
        module_name == p or module_name.startswith(p + "_") for p in encoder_prefixes
    ):
        return RoleType.ENCODER

    if module_name in {"hy3dshape_conditioner", "hy3dshape_image_processor"}:
        return RoleType.ENCODER

    denoising_prefixes = (
        "transformer",
        "unconditional_transformer",
        "video_dit",
        "audio_dit",
        "dual_tower_bridge",
    )
    if any(
        module_name == p or module_name.startswith(p + "_") for p in denoising_prefixes
    ):
        return RoleType.DENOISER

    if module_name == "hy3dshape_model":
        return RoleType.DENOISER

    decoder_prefixes = ("vae", "audio_vae", "video_vae", "vocoder")
    if any(
        module_name == p or module_name.startswith(p + "_") for p in decoder_prefixes
    ):
        return RoleType.DECODER

    if module_name == "hy3dshape_vae":
        return RoleType.DECODER

    return None


def filter_modules_for_role(
    module_names: list[str],
    role: "RoleType",
    *,
    extra_allowed_modules: set[str] | None = None,
) -> list[str]:
    """Filter module names to only those needed by the given role."""
    if role in (RoleType.MONOLITHIC, RoleType.SERVER):
        return module_names

    extra_allowed_modules = extra_allowed_modules or set()
    filtered = []
    for name in module_names:
        module_role = get_module_role(name)

        if module_role is None:
            filtered.append(name)
        elif module_role == role:
            filtered.append(name)
        elif name in extra_allowed_modules:
            filtered.append(name)

    return filtered
