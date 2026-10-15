"""Shared module-site protocol for request-scoped diffusion fast paths."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from torch import nn

RejectReason = Callable[[nn.Module], str | None]


class QualityGatedFusion:
    """Track and toggle one family of opt-in fusion sites.

    The marker metadata describes the model attribute(s) owned by a site. The
    enabled flag remains a plain module attribute so compiled model forwards
    can read it without depending on this helper object.
    """

    __slots__ = ("enabled_attr", "marker_attr", "name")

    def __init__(self, *, name: str, marker_attr: str, enabled_attr: str) -> None:
        self.name = name
        self.marker_attr = marker_attr
        self.enabled_attr = enabled_attr

    def mark(self, module: nn.Module, metadata: Any = True) -> None:
        setattr(module, self.marker_attr, metadata)
        setattr(module, self.enabled_attr, False)

    def metadata(self, module: nn.Module, default: Any = None) -> Any:
        return getattr(module, self.marker_attr, default)

    def is_enabled(self, module: nn.Module) -> bool:
        return bool(getattr(module, self.enabled_attr, False))

    def iter_sites(self, root: nn.Module) -> Iterator[nn.Module]:
        for module in root.modules():
            if hasattr(module, self.marker_attr):
                yield module

    def mount(
        self,
        root: nn.Module,
        *,
        reject_reason: RejectReason | None = None,
        logger: logging.Logger | None = None,
    ) -> bool:
        """Enable every eligible site, or leave the whole family disabled."""
        sites = list(self.iter_sites(root))
        if not sites:
            return False

        if reject_reason is not None:
            for site in sites:
                reason = reject_reason(site)
                if reason is None:
                    continue
                self._set_enabled(sites, False)
                if logger is not None:
                    logger.info(
                        "%s: %s site failed static guards (%s); keeping the "
                        "whole model on the reference path",
                        self.name,
                        type(site).__name__,
                        reason,
                    )
                return False

        self._set_enabled(sites, True)
        return True

    def unmount(self, root: nn.Module) -> None:
        self._set_enabled(self.iter_sites(root), False)

    def _set_enabled(
        self, sites: Iterator[nn.Module] | list[nn.Module], enabled: bool
    ) -> None:
        for site in sites:
            setattr(site, self.enabled_attr, enabled)
