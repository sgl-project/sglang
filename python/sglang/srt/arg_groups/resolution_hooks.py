"""Out-of-tree replacement for one named step of the resolution pipeline.

`run_resolution_pipeline` calls its steps by name, hardcoded, with no
per-step dispatch through `self` -- there is nothing on `ServerArgs` left to
subclass in order to change how one step decides. This is the replacement
for that: a decorator that wraps whatever currently runs under a given name,
and a dispatcher the pipeline calls instead of the bare function.

Whitelisted names only, the same discipline `Arg(resolvable=True)` uses for
declarable fields: this project has already been bitten once by an
unqualified name collision in this exact pipeline (`_parse_cuda_graph_config`
and `_handle_cuda_graph_config` merged under one rename and the dispatcher
called itself). A name that is not on the list fails loudly at import time,
not silently at the call site three modules away.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List

# The only step open to replacement so far. Add a name here only alongside
# the call site's own switch to `run_hook` -- an entry with no matching
# `run_hook(...)` call is a name nothing will ever look up.
_OVERRIDABLE_HOOKS: FrozenSet[str] = frozenset({"handle_cuda_graph_config"})

# name -> registered overrides, oldest first. Each takes `(server_args,
# previous)`, where `previous` is the callable it wraps -- the built-in on
# the first registration, the previous registrant's override on every one
# after. Process-global as a `dict[str, list]` so a test isolates it the way
# `test_model_overrides.py` isolates `_MODEL_OVERRIDE_FNS`: `patch.dict(...,
# clear=True)`.
_HOOKS: Dict[str, List[Callable[[Any, Callable[[Any], None]], None]]] = {}


def register_resolution_hook(name: str):
    """Replace (or wrap) the pipeline step named ``name``.

    The decorated function is called as ``fn(server_args, previous)``.
    ``previous`` is a plain ``server_args -> None`` callable: the built-in
    step on the first registration for this name, or the previous
    registrant's own wrapper on every registration after that. Call it to
    run what would have run without this override -- the `super().handle_x()`
    shape, expressed as an explicit argument instead of a method-resolution
    lookup, because there is no class hierarchy here for `super()` to walk.
    Not calling it is a full replacement.

    Registering twice for the same name does not replace the first
    registration; it wraps it. The **last** registration is outermost --
    runs first, and decides whether/when its `previous` (everything
    registered before it, down to the built-in) runs at all. Two downstream
    packages that both target the same name compose in whichever order they
    happened to import in; if that order matters to you, make one of them
    import the other first.

    This changes *what* runs at the step's existing position in the
    pipeline, never *when*: the call site in `pipeline.py` is unmoved, so
    every other step keeps the order it already had. A wrapped step's own
    declarations reach `resolution_result` the same way any declaration
    does -- only readers from this position onward see them; a step that
    already ran and read the old value before this one's `previous` (or the
    built-in) declared its replacement has already made its decision on it.
    """
    if name not in _OVERRIDABLE_HOOKS:
        raise ValueError(
            f"{name!r} is not an overridable resolution hook; the "
            f"overridable set is {sorted(_OVERRIDABLE_HOOKS)}. A new entry "
            "needs a matching `run_hook(...)` call at the step's site in "
            "pipeline.py, not just a name here."
        )

    def decorator(fn):
        _HOOKS.setdefault(name, []).append(fn)
        return fn

    return decorator


def run_hook(name: str, builtin: Callable[[Any], None], server_args: Any) -> None:
    """Run the pipeline step named ``name`` -- the registered chain if
    anything overrode it, ``builtin`` otherwise.

    Called from the step's fixed position in `run_resolution_pipeline`;
    `builtin` is that position's own local import, so this never needs the
    built-in registered anywhere in advance. The chain is rebuilt from the
    registry on every call rather than cached at registration time, because
    at registration time (import time, before any `ServerArgs` exists) there
    is no `server_args` yet and the first registrant's `previous` cannot be
    bound to anything real until a call actually happens.
    """
    step = builtin
    for fn in _HOOKS.get(name, ()):
        step = _bind(fn, step)
    step(server_args)


def _bind(fn, previous):
    def wrapped(server_args):
        fn(server_args, previous)

    return wrapped
