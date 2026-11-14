# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Share Kimi K3 attention projection storage with the megakernel.

The stock model must remain usable for shapes the megakernel does not serve.
Consequently this module keeps every SGLang ``Parameter`` object and only
repoints its data to a row view of one module-owned tensor.  The megakernel
consumes that tensor directly instead of allocating a second concatenation.

This runs after weight loading and final device placement.  A later module
``_apply``/``to`` or assign-style parameter replacement is outside the serving
lifecycle; a repeated call validates and rejects any broken alias instead of
silently rebuilding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_STORAGE_ATTR = "_nvidia_mega_projection_storage"
_ALIGNMENT = 16
_MISSING = object()


@dataclass(frozen=True)
class _Part:
    path: str
    rows: int


@dataclass(frozen=True)
class _Layout:
    name: str
    parts: tuple[_Part, ...]
    columns: int
    dtype: torch.dtype
    alias_name: str | None = None
    alias_part: int | None = None
    padding_rows: int = 0

    @property
    def rows(self) -> int:
        return sum(part.rows for part in self.parts) + self.padding_rows


_MLA_LAYOUT = _Layout(
    name="MLA",
    parts=(
        _Part("fused_qkv_a_proj_with_mqa.weight", 2112),
        _Part("g_proj.weight", 1536),
    ),
    columns=7168,
    dtype=torch.bfloat16,
)

_KDA_LAYOUT = _Layout(
    name="KDA",
    parts=(
        _Part("fused_qkvg_proj.weight", 6144),
        _Part("f_a_proj.weight", 128),
        _Part("b_proj.weight", 12),
    ),
    columns=7168,
    dtype=torch.bfloat16,
    alias_name="_bfa_w",
    alias_part=1,
    padding_rows=4,
)


def _resolve_parameter(owner: Any, path: str) -> torch.nn.Parameter:
    value = owner
    for component in path.split("."):
        if not hasattr(value, component):
            raise ValueError(f"projection component {path!r} is absent")
        value = getattr(value, component)
    if not isinstance(value, torch.nn.Parameter):
        raise TypeError(
            f"projection component {path!r} is {type(value).__name__}, not Parameter"
        )
    return value


def _registered(owner: Any, name: str) -> bool:
    return any(
        name in getattr(owner, collection, {})
        for collection in ("_parameters", "_buffers", "_modules")
    )


def _validate_layout(layout: _Layout) -> None:
    if not layout.name or not layout.parts:
        raise ValueError("projection layout needs a name and at least one part")
    if layout.columns <= 0 or any(part.rows <= 0 for part in layout.parts):
        raise ValueError(f"invalid {layout.name} projection dimensions")
    paths = [part.path for part in layout.parts]
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate {layout.name} projection component")
    if layout.padding_rows < 0:
        raise ValueError(f"negative {layout.name} projection padding")
    if (layout.alias_name is None) != (layout.alias_part is None):
        raise ValueError(f"incomplete {layout.name} projection alias contract")
    if layout.alias_part is not None and not 0 <= layout.alias_part < len(layout.parts):
        raise ValueError(f"invalid {layout.name} projection alias start")


def _validate_tensor(
    value: torch.Tensor,
    *,
    label: str,
    shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{label} shape {tuple(value.shape)} != {shape}")
    if value.dtype != dtype:
        raise ValueError(f"{label} dtype {value.dtype} != {dtype}")
    if value.device != device:
        raise ValueError(f"{label} device {value.device} != {device}")
    if value.stride() != (shape[1], 1):
        raise ValueError(f"{label} stride {value.stride()} is not row-major")
    if value.data_ptr() % _ALIGNMENT:
        raise ValueError(f"{label} is not {_ALIGNMENT}-byte aligned")


def _parts(
    owner: Any, layout: _Layout, *, device_type: str
) -> tuple[torch.nn.Parameter, ...]:
    parameters = tuple(_resolve_parameter(owner, part.path) for part in layout.parts)
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError(f"{layout.name} projection reuses a Parameter object")
    device = parameters[0].device
    if device.type != device_type:
        raise ValueError(
            f"{layout.name} projection device type {device.type!r} != {device_type!r}"
        )
    for part, parameter in zip(layout.parts, parameters):
        _validate_tensor(
            parameter,
            label=f"{layout.name} {part.path}",
            shape=(part.rows, layout.columns),
            dtype=layout.dtype,
            device=device,
        )
        if parameter.requires_grad:
            raise ValueError(
                f"{layout.name} {part.path} unexpectedly requires gradients"
            )
    return parameters


def _same_view(
    value: torch.Tensor,
    root: torch.Tensor,
    *,
    row: int,
    rows: int,
    columns: int,
) -> bool:
    return bool(
        value.device == root.device
        and value.untyped_storage().data_ptr() == root.untyped_storage().data_ptr()
        and value.storage_offset() == root.storage_offset() + row * columns
        and tuple(value.shape) == (rows, columns)
        and value.stride() == (columns, 1)
    )


def _alias_start_row(layout: _Layout) -> int:
    assert layout.alias_part is not None
    return sum(part.rows for part in layout.parts[: layout.alias_part])


def _validate_alias(
    owner: Any,
    layout: _Layout,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    root: torch.Tensor | None,
) -> None:
    if layout.alias_name is None:
        return
    if _registered(owner, layout.alias_name):
        raise ValueError(
            f"{layout.name} projection alias {layout.alias_name!r} is registered"
        )
    alias = owner.__dict__.get(layout.alias_name, _MISSING)
    if not isinstance(alias, torch.Tensor):
        raise TypeError(
            f"{layout.name} projection alias {layout.alias_name!r} is absent"
        )
    assert layout.alias_part is not None
    alias_rows = (
        sum(part.rows for part in layout.parts[layout.alias_part :])
        + layout.padding_rows
    )
    _validate_tensor(
        alias,
        label=f"{layout.name} {layout.alias_name}",
        shape=(alias_rows, layout.columns),
        dtype=layout.dtype,
        device=parameters[0].device,
    )
    alias_row = 0
    for index in range(layout.alias_part, len(layout.parts)):
        part, parameter = layout.parts[index], parameters[index]
        if not _same_view(
            parameter,
            alias,
            row=alias_row,
            rows=part.rows,
            columns=layout.columns,
        ):
            raise ValueError(
                f"{layout.name} {part.path} is not the expected {layout.alias_name} view"
            )
        alias_row += part.rows
    if root is not None and not _same_view(
        alias,
        root,
        row=_alias_start_row(layout),
        rows=alias_rows,
        columns=layout.columns,
    ):
        raise ValueError(f"{layout.name} projection alias is not a root-storage view")


def _validate_installed(
    owner: Any,
    layout: _Layout,
    parameters: tuple[torch.nn.Parameter, ...],
    root: torch.Tensor,
) -> None:
    _validate_tensor(
        root,
        label=f"{layout.name} projection storage",
        shape=(layout.rows, layout.columns),
        dtype=layout.dtype,
        device=parameters[0].device,
    )
    if isinstance(root, torch.nn.Parameter) or root.requires_grad:
        raise ValueError(f"{layout.name} projection storage must be a plain tensor")
    if root.storage_offset() != 0:
        raise ValueError(f"{layout.name} projection storage is itself a view")
    if root.untyped_storage().nbytes() != root.numel() * root.element_size():
        raise ValueError(f"{layout.name} projection storage has a non-exact allocation")
    row = 0
    for part, parameter in zip(layout.parts, parameters):
        if not _same_view(
            parameter,
            root,
            row=row,
            rows=part.rows,
            columns=layout.columns,
        ):
            raise ValueError(
                f"{layout.name} {part.path} is not the expected projection-storage view"
            )
        row += part.rows
    _validate_alias(
        owner,
        layout,
        parameters,
        root=root,
    )


def _coalesce_projection_storage(
    owner: Any, layout: _Layout, *, device_type: str
) -> torch.Tensor:
    """Install or validate one row-major projection allocation on ``owner``."""

    _validate_layout(layout)
    if _registered(owner, _STORAGE_ATTR):
        raise ValueError(f"{layout.name} projection storage is registered on the model")
    if hasattr(owner, _STORAGE_ATTR) and _STORAGE_ATTR not in owner.__dict__:
        raise ValueError(f"{layout.name} projection storage attribute is shadowed")
    parameters = _parts(owner, layout, device_type=device_type)
    existing = owner.__dict__.get(_STORAGE_ATTR, _MISSING)
    if existing is not _MISSING:
        if not isinstance(existing, torch.Tensor):
            raise ValueError(f"{layout.name} projection storage is not a tensor")
        _validate_installed(owner, layout, parameters, existing)
        return existing

    _validate_alias(
        owner,
        layout,
        parameters,
        root=None,
    )
    root = parameters[0].detach().new_empty((layout.rows, layout.columns))
    row = 0
    with torch.no_grad():
        for part, parameter in zip(layout.parts, parameters):
            root[row : row + part.rows].copy_(parameter.detach())
            row += part.rows
        if layout.padding_rows:
            root[row:].zero_()

    original_data = tuple(parameter.data for parameter in parameters)
    original_alias = (
        owner.__dict__.get(layout.alias_name, _MISSING)
        if layout.alias_name is not None
        else _MISSING
    )
    try:
        row = 0
        for part, parameter in zip(layout.parts, parameters):
            parameter.data = root[row : row + part.rows]
            row += part.rows
        if layout.alias_name is not None:
            owner.__dict__[layout.alias_name] = root[_alias_start_row(layout) :]
        # This is derived runtime storage, not checkpoint state.  Bypass
        # nn.Module.__setattr__ so it cannot become a parameter or buffer.
        owner.__dict__[_STORAGE_ATTR] = root
        _validate_installed(owner, layout, parameters, root)
    except BaseException:
        for parameter, data in zip(parameters, original_data):
            parameter.data = data
        owner.__dict__.pop(_STORAGE_ATTR, None)
        if layout.alias_name is not None:
            if original_alias is _MISSING:
                owner.__dict__.pop(layout.alias_name, None)
            else:
                owner.__dict__[layout.alias_name] = original_alias
        raise
    return root


def coalesce_mla_projection_storage(attention: Any) -> torch.Tensor:
    """Return MLA's [qkv-a | gate] weight without retaining a second copy."""

    return _coalesce_projection_storage(attention, _MLA_LAYOUT, device_type="cuda")


def coalesce_kda_projection_storage(attention: Any) -> torch.Tensor:
    """Return KDA's [qkvg | f-a | beta | pad] weight as shared model storage."""

    return _coalesce_projection_storage(attention, _KDA_LAYOUT, device_type="cuda")
