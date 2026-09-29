"""Patching a name the communicator package's modules look up.

Each module of ``sglang.srt.layers.communicator`` binds the names it imports
in its own namespace, so a test double has to go into every module that reads
the name, not only into the package.
"""

import contextlib
from unittest import mock

from sglang.srt.layers import communicator as communicator
from sglang.srt.layers.communicator import (
    boundary,
    construction,
    exit,
    factories,
    layer,
    layout,
    legacy_stage,
    ops,
    output,
)
from sglang.srt.layers.communicator import residual as residual_contract
from sglang.srt.layers.communicator import (
    stage,
)
from sglang.srt.layers.communicator.adapters import attention, branch, lora
from sglang.srt.layers.communicator.fusions import allreduce
from sglang.srt.layers.communicator.residual import (
    access,
    add_norm,
    batch,
    mhc,
    stream,
)

COMMUNICATOR_MODULES = (
    layer,
    legacy_stage,
    communicator,
    allreduce,
    exit,
    branch,
    lora,
    layout,
    output,
    attention,
    residual_contract,
    access,
    add_norm,
    mhc,
    ops,
    boundary,
    construction,
    factories,
    stage,
    batch,
    stream,
)


@contextlib.contextmanager
def patch_communicator(name, new=mock.DEFAULT, **kwargs):
    """``mock.patch.object(module, name, new, **kwargs)`` on every communicator
    module that has ``name``, all with the same replacement; with
    ``create=True`` and no module having it, on every module. Yields the
    replacement."""
    targets = [m for m in COMMUNICATOR_MODULES if hasattr(m, name)]
    if not targets:
        if not kwargs.get("create"):
            raise AttributeError(f"no communicator module has {name!r}")
        targets = list(COMMUNICATOR_MODULES)
    with contextlib.ExitStack() as stack:
        replacement = stack.enter_context(
            mock.patch.object(targets[0], name, new, **kwargs)
        )
        for module in targets[1:]:
            stack.enter_context(
                mock.patch.object(
                    module, name, replacement, create=kwargs.get("create", False)
                )
            )
        yield replacement
