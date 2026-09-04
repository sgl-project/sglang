"""Config field declarations, one module per top-level namespace.

Each class carries the ``_NS_PATH`` it stands for, so the module a field is
declared in *is* its namespace. ``ServerArgs`` is assembled from the classes
that declare **operator input**; the derived fields, which nobody can type,
are declared beside them but are not collected into the record.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Tuple, get_type_hints

from sglang.srt.arg_groups.field_order import POSITIONAL_FIELD_ORDER


def collect_input_fields(
    sources: List[type],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    """The annotations and defaults of every field these classes declare.

    Each source's annotations are resolved **in its own module** and handed on
    as type objects. Strings never travel: an annotation carried across as text
    would be re-evaluated where it lands, and the composing module does not
    import the names the declarations use -- that is the whole point of them
    living where they do.

    Returns the resolved annotations, the defaults, and ``{field: namespace}``
    -- the last because the assembled record has no base classes to read a
    ``_NS_PATH`` off, and the namespace is still the class that declared it.

    Which classes are passed here is the record's contract: the record holds
    what the operator asked for, so a class of derived fields is simply not in
    the list, and the rule is one readable call rather than an invariant spread
    across a base-class list.

    The result is ordered by ``POSITIONAL_FIELD_ORDER``, not by namespace: a
    dataclass turns field order into a positional constructor signature, and
    grouping fields that used to be interleaved would silently move
    ``ServerArgs(model_path, tokenizer_path)``'s second argument onto another
    field. Anything the record declares that the frozen order does not name
    goes after it, in declaration order -- the only backward-compatible place
    for a new field.
    """
    annotations: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {}
    for source in sources:
        hints = get_type_hints(source, include_extras=True)
        for field in dataclasses.fields(source):
            if field.name in annotations:
                raise ValueError(
                    f"{field.name!r} is declared by both "
                    f"{annotations[field.name][0].__name__} and {source.__name__}; "
                    "a field belongs to exactly one namespace"
                )
            annotations[field.name] = (source, hints[field.name])
            if field.default is not dataclasses.MISSING:
                defaults[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                defaults[field.name] = dataclasses.field(
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
