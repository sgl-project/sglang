# SPDX-License-Identifier: Apache-2.0

__all__ = ["DM05PolicyModel", "Pi05PolicyModel"]


def __getattr__(name: str):
    if name == "DM05PolicyModel":
        from sglang.multimodal_gen.runtime.models.vlas.dm05_policy import (
            DM05PolicyModel,
        )

        return DM05PolicyModel
    if name == "Pi05PolicyModel":
        from sglang.multimodal_gen.runtime.models.vlas.pi05_policy import (
            Pi05PolicyModel,
        )

        return Pi05PolicyModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
