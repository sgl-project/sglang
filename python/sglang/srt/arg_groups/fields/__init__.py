"""Namespace declarations for input and derived configuration.

``ServerArgs`` collects input fields from classes carrying ``_NS_PATH``.
Derived declarations stay on those classes for publication into config bags.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Tuple, get_type_hints

import msgspec
import msgspec.structs

from sglang.srt.arg_groups.field_order import POSITIONAL_FIELD_ORDER


def collect_input_fields(
    sources: List[type],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    """Return resolved annotations, defaults, and a field-to-namespace map.

    Resolve annotations in their declaring module so imported type names remain
    available. Preserve ``POSITIONAL_FIELD_ORDER`` for constructor compatibility;
    append new fields in declaration order.
    """
    annotations: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {}
    for source in sources:
        hints = get_type_hints(source, include_extras=True)
        for field in msgspec.structs.fields(source):
            if field.name in annotations:
                raise ValueError(
                    f"{field.name!r} is declared by both "
                    f"{annotations[field.name][0].__name__} and {source.__name__}; "
                    "a field belongs to exactly one namespace"
                )
            annotations[field.name] = (source, hints[field.name])
            if field.default is not msgspec.NODEFAULT:
                defaults[field.name] = field.default
            elif field.default_factory is not msgspec.NODEFAULT:
                defaults[field.name] = msgspec.field(
                    default_factory=field.default_factory
                )
    known = [n for n in POSITIONAL_FIELD_ORDER if n in annotations]
    rest = [n for n in annotations if n not in set(POSITIONAL_FIELD_ORDER)]
    ordered = known + rest
    return (
        {name: annotations[name][1] for name in ordered},
        {name: defaults[name] for name in ordered if name in defaults},
        {name: annotations[name][0]._NS_PATH for name in ordered},
    )
