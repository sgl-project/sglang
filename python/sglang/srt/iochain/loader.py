"""
IOChain plugin loader.

Discovers and instantiates ``IOProcessor`` plugins from two sources, then
returns a configured ``IOChain`` ready to be wired to the serving layer.

Source 1 — entry-point group ``sglang.io_processor_plugins``
-------------------------------------------------------------
Any installed Python package can register processors that are loaded
automatically at server startup.  Add to your ``pyproject.toml``::

    [project.entry-points."sglang.io_processor_plugins"]
    my_processor = "mypackage.processors:MyProcessor"

After ``pip install -e .`` (or a normal install), SGLang discovers the
processor via ``importlib.metadata.entry_points`` without requiring any
CLI flags.

Source 2 — ``--io-processor`` CLI flag
----------------------------------------
Explicit ``module.path:ClassName`` strings passed at launch time.  Useful
for development or one-off deployments where installing a package is
impractical::

    python -m sglang.launch_server --model-path ... \\
        --io-processor mypackage.processors:MyProcessor

Multiple flags are accepted; processors are appended in the order given,
*after* any entry-point processors.

Ordering
--------
Entry-point processors are added first (discovery order within the group
is unspecified), followed by CLI processors in left-to-right order.
Ingress runs 0 → N; egress runs N → 0.

Error handling
--------------
* A processor whose entry point fails to load raises and aborts startup,
  so misconfigured plugins are caught early.
* A plugin that is not an ``IOProcessor`` subclass is skipped with a
  warning (entry-point path) or raises ``TypeError`` (CLI path).
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from sglang.srt.iochain.base import IOChain, IOProcessor

try:
    from importlib.metadata import entry_points
except ImportError:  # pragma: no cover
    entry_points = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "sglang.io_processor_plugins"


def _load_class(path: str) -> type:
    """
    Import and return an ``IOProcessor`` subclass from a dotted path string.

    Parameters
    ----------
    path:
        ``"module.path:ClassName"`` — the part before ``:`` is passed to
        ``importlib.import_module``; the part after is looked up as an
        attribute on the resulting module.

    Raises
    ------
    ValueError:
        If *path* does not contain ``:``.
    TypeError:
        If the resolved object is not an ``IOProcessor`` subclass.
    """
    if ":" not in path:
        raise ValueError(
            f"Invalid processor path {path!r}. Expected 'module.path:ClassName'."
        )
    module_path, class_name = path.rsplit(":", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    if not (isinstance(cls, type) and issubclass(cls, IOProcessor)):
        raise TypeError(f"{path!r} is not an IOProcessor subclass, got {cls!r}")
    return cls


def load_iochain(server_args: "ServerArgs") -> IOChain:
    """
    Build and return an ``IOChain`` populated from entry points and CLI args.

    Called once at server startup by ``http_server.py``.  The returned chain
    is then wired to every ``OpenAIServingBase`` handler via
    ``handler.set_iochain(chain)``.

    Parameters
    ----------
    server_args:
        Parsed ``ServerArgs`` instance.  Reads ``server_args.io_processors``
        (list of ``module:Class`` strings supplied via ``--io-processor``).
    """
    chain = IOChain()

    # 1. Entry-point plugins
    if entry_points is not None:
        try:
            for ep in entry_points(group=_ENTRY_POINT_GROUP):
                try:
                    cls = ep.load()
                    if not (isinstance(cls, type) and issubclass(cls, IOProcessor)):
                        logger.warning(
                            "Entry point %r is not an IOProcessor subclass — skipped",
                            ep.name,
                        )
                        continue
                    chain.add(cls())
                    logger.info("IOChain: loaded %s from entry point", ep.name)
                except Exception:
                    logger.exception("IOChain: failed to load entry point %r", ep.name)
                    raise
        except Exception as exc:
            logger.warning("IOChain: entry-point discovery failed: %s", exc)

    # 2. CLI plugins
    for path in getattr(server_args, "io_processors", []) or []:
        cls = _load_class(path)
        chain.add(cls())
        logger.info("IOChain: loaded %s from CLI", path)

    if chain._processors:
        logger.info(
            "IOChain: %d processor(s) active: %s",
            len(chain._processors),
            ", ".join(type(p).__name__ for p in chain._processors),
        )

    return chain
