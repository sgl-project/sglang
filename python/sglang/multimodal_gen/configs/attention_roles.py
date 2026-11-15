# SPDX-License-Identifier: Apache-2.0
"""Attention roles.

A "role" distinguishes the type of attention a layer performs (self-attention
vs cross-attention) so a different backend can be selected per role. The value
strings double as the role token accepted in ``--component-attention-backends``
keys, e.g. ``transformer.cross``; ``ServerArgs`` splits those entries out of
``component_attention_backends`` into ``component_attention_backend_roles`` so a
component lookup by name never sees a role-qualified key.
"""

import enum


class AttentionRole(str, enum.Enum):
    SELF = "self"
    CROSS = "cross"


# Separator between the component name and the role token in a
# ``--component-attention-backends`` key (e.g. ``transformer.cross``).
ROLE_KEY_SEPARATOR = "."


def make_component_role_key(component: str, role: AttentionRole) -> str:
    """Build a role-qualified config key from a component name and role."""
    return f"{component}{ROLE_KEY_SEPARATOR}{role.value}"


def split_component_role_key(key: str) -> tuple[str, AttentionRole | None]:
    """Split a ``component`` or ``component.role`` config key.

    Returns the component part and the parsed ``AttentionRole`` (or ``None`` when
    the key carries no role suffix). Raises ``ValueError`` if a role suffix is
    present but is not a valid role token.
    """
    component_part, sep, role_part = key.rpartition(ROLE_KEY_SEPARATOR)
    if not sep:
        return key, None
    try:
        role = AttentionRole(role_part.strip().lower())
    except ValueError:
        valid_roles = sorted(r.value for r in AttentionRole)
        raise ValueError(
            f"Invalid attention role '{role_part}' in component key '{key}'. "
            f"Available roles are: {valid_roles}"
        ) from None
    return component_part, role
