"""Shared recipe YAML loader with `extends:` inheritance.

Both scripts/ci/slurm/launch_mi355x.sh (inline python) and
scripts/ci/slurm/process_result.py load recipe files, so the extends-resolution
lives here to guarantee they agree on the merged result.

A recipe may declare `extends: <path>` (or a list of paths) pointing at parent
recipe(s) relative to the child's own directory. The child is deep-merged over
its parent(s): nested dicts merge recursively; scalars and lists are replaced
wholesale by the child.
"""

import os

import yaml


class RecipeCycleError(RuntimeError):
    """Raised when `extends:` forms a cycle."""


def deep_merge(base, over):
    """Return base with over layered on top. Dicts merge recursively; scalars
    and lists are overridden wholesale (lists are never concatenated)."""
    if not isinstance(base, dict) or not isinstance(over, dict):
        return over
    merged = dict(base)
    for k, v in over.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_recipe(path, _seen=None):
    """Load a recipe YAML, resolving `extends:` parents relative to the child's
    directory and deep-merging them under the child."""
    real = os.path.realpath(path)
    _seen = set() if _seen is None else _seen
    if real in _seen:
        raise RecipeCycleError(f"extends cycle detected at {real}")
    _seen = _seen | {real}

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    parents = raw.pop("extends", None)
    if parents is None:
        return raw
    if isinstance(parents, str):
        parents = [parents]

    base_dir = os.path.dirname(real)
    merged = {}
    for parent in parents:
        parent_path = parent if os.path.isabs(parent) else os.path.join(base_dir, parent)
        merged = deep_merge(merged, load_recipe(parent_path, _seen))
    return deep_merge(merged, raw)
