"""Patching a name the communicator package's modules look up.

Each module of ``sglang.srt.layers.layer_boundary`` binds the names it imports
in its own namespace, so a test double has to go into every module that reads
the name, not only into the package.
"""

import contextlib
from unittest import mock

from sglang.srt.layers import layer_boundary as communicator
from sglang.srt.layers.layer_boundary import (
    boundary,
    construction,
    contracts,
    exit,
    factories,
    layout,
    ops,
    output,
    prepare,
)
from sglang.srt.layers.layer_boundary import residual as residual_contract
from sglang.srt.layers.layer_boundary import (
    stage,
)
from sglang.srt.layers.layer_boundary.adapters import attention, branch, lora, overlap
from sglang.srt.layers.layer_boundary.fusions import allreduce
from sglang.srt.layers.layer_boundary.residual import (
    access,
    add_norm,
    batch,
    mhc,
    stream,
)

COMMUNICATOR_MODULES = (
    communicator,
    allreduce,
    exit,
    branch,
    lora,
    overlap,
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
    contracts,
    prepare,
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
