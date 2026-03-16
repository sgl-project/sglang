"""
IOChain plugin loader.

Discovers and instantiates IOFilter plugins from two sources, in order:

1. **Installed packages** (``sglang.general_plugins`` entry-point group)
   Any Python package that declares an ``sglang.general_plugins`` entry point
   pointing to an ``IOFilter`` subclass is automatically discovered and loaded
   at server startup — no CLI flags required.

   Example ``pyproject.toml``::

       [project.entry-points."sglang.general_plugins"]
       my_filter = "mypackage.filters:MyFilter"

2. **CLI paths** (``--iochain-filter``)
   Explicit ``module.path:ClassName`` strings passed on the command line.
   Loaded after entry-point filters; order of ``--iochain-filter`` flags is
   preserved.

Both sources add filters to the **default chain** (``get_default_chain()``).
If no filters are registered by either mechanism the chain stays empty and
the serving layer incurs zero overhead.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from sglang.srt.iochain.base import IOChain, IOFilter

try:
    from importlib.metadata import entry_points
except ImportError:  # pragma: no cover
    entry_points = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "sglang.general_plugins"


def _load_class(dotted_path: str) -> type:
    """
    Import and return a class from a ``module.path:ClassName`` string.

    Raises ``ValueError`` for malformed paths and ``ImportError`` /
    ``AttributeError`` for missing modules or names.
    """
    if ":" not in dotted_path:
        raise ValueError(
            f"Invalid filter path {dotted_path!r}. "
            "Expected format: 'module.path:ClassName'  "
            "(e.g. 'mypackage.filters:ContentPolicy')"
        )
    module_path, class_name = dotted_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, IOFilter)):
        raise TypeError(
            f"{dotted_path!r} must be a subclass of IOFilter, got {cls!r}"
        )
    return cls


def _load_entry_points(chain: IOChain) -> None:
    """Discover and register filters declared via entry points."""
    if entry_points is None:  # pragma: no cover
        logger.debug("importlib.metadata not available; skipping entry-point discovery")
        return

    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except Exception as exc:
        logger.warning("Failed to query entry points: %s", exc)
        return

    for ep in eps:
        try:
            cls = ep.load()
            if not (isinstance(cls, type) and issubclass(cls, IOFilter)):
                logger.warning(
                    "Entry point %r (%s) is not an IOFilter subclass — skipped",
                    ep.name,
                    ep.value,
                )
                continue
            chain.add(cls())
            logger.info("Loaded IOFilter from entry point: %s (%s)", ep.name, ep.value)
        except Exception:
            logger.exception("Failed to load IOFilter entry point %r", ep.name)
            raise


def _load_cli_filters(chain: IOChain, filter_paths: list[str]) -> None:
    """Load filters specified via ``--iochain-filter`` CLI flags."""
    for path in filter_paths:
        try:
            cls = _load_class(path)
            chain.add(cls())
            logger.info("Loaded IOFilter from CLI: %s", path)
        except Exception:
            logger.exception("Failed to load IOFilter from CLI path %r", path)
            raise


def load_iochain(server_args: "ServerArgs") -> IOChain:
    """
    Build the server's IOChain from all configured sources.

    Called once at server startup (inside the FastAPI lifespan handler).
    Returns the populated default chain; callers should pass it to each
    ``OpenAIServingBase`` handler via ``handler.set_iochain(chain)``.

    If no filters are registered the returned chain is empty and the serving
    layer adds no overhead.
    """
    from sglang.srt.iochain import get_default_chain

    chain = get_default_chain()

    # 1. Auto-discover installed plugins
    _load_entry_points(chain)

    # 2. Explicit CLI paths (appended after entry-point filters)
    cli_filters: list[str] = getattr(server_args, "iochain_filters", []) or []
    _load_cli_filters(chain, cli_filters)

    if chain._filters:
        logger.info(
            "IOChain initialised with %d filter(s): %s",
            len(chain._filters),
            ", ".join(type(f).__name__ for f in chain._filters),
        )
    else:
        logger.debug("IOChain: no filters registered")

    return chain
