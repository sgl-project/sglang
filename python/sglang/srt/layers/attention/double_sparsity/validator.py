"""Server-args validator for standalone Double Sparsity.

Mirrors the role of ``arg_groups.hisparse_hook.validate_hisparse`` but lives
inside the Double Sparsity package per the upstream-shaped path budget
(``arg_groups/`` is intentionally out of scope for this feature).

Real schema / content-hash / capability checks land alongside the channel
mask file loader and selection kernels in later milestones; this minimal
validator only enforces the startup-time rules needed to keep AC-1
honest:

* mutual-exclusion with ``--enable-hisparse`` (DEC-8),
* presence of ``--double-sparsity-config`` with a parseable
  ``channel_mask_path``,
* page-size pairing (the JSON ``page_size`` must equal ``server_args.page_size``),
* placeholder-guard sentinel: warn (and refuse to serve in production) when
  the placeholder selector is still wired.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def validate_double_sparsity(server_args: "ServerArgs") -> None:
    """Validate ``--enable-double-sparsity`` constraints.

    Called from ``ServerArgs._handle_other_validations`` after
    ``validate_hisparse``. The early-return semantics keep this safe to call
    unconditionally.
    """

    if not getattr(server_args, "enable_double_sparsity", False):
        return

    if getattr(server_args, "enable_hisparse", False):
        raise ValueError(
            "Double Sparsity and HiSparse are mutually exclusive; there are no plans "
            "to integrate them. Pick exactly one of --enable-double-sparsity or "
            "--enable-hisparse."
        )

    if getattr(server_args, "disaggregation_mode", None) not in (None, "null"):
        raise ValueError(
            "Standalone Double Sparsity does not support --disaggregation-mode. "
            f"Got --disaggregation-mode={server_args.disaggregation_mode!r}. "
            "Drop the disaggregation flag or use HiSparse instead "
            "(HiSparse is the PD-disaggregated sparsity path)."
        )

    payload = getattr(server_args, "double_sparsity_config", None)
    if payload is None or (isinstance(payload, str) and not payload.strip()):
        raise ValueError(
            "--enable-double-sparsity requires --double-sparsity-config to be set "
            "with at least 'channel_mask_path'. Example: --double-sparsity-config "
            '\'{"top_k": 2048, "page_size": 64, '
            '"channel_mask_path": "/path/to/channel_mask.safetensors", '
            '"device_buffer_size": 4096}\'.'
        )

    from sglang.srt.layers.attention.double_sparsity.config import (
        parse_double_sparsity_config,
    )

    config = parse_double_sparsity_config(payload)

    if not config.channel_mask_path:
        raise ValueError(
            "Double Sparsity requires 'channel_mask_path' in --double-sparsity-config."
        )

    server_page_size = getattr(server_args, "page_size", None)
    if server_page_size is not None and config.page_size != server_page_size:
        raise ValueError(
            f"Double Sparsity config page_size={config.page_size} does not match "
            f"--page-size={server_page_size}. The two must agree at startup."
        )

    if os.environ.get("SGLANG_DS_ALLOW_PLACEHOLDER") != "1":
        logger.warning(
            "Double Sparsity selector is still the placeholder implementation. "
            "Set SGLANG_DS_ALLOW_PLACEHOLDER=1 for explicit test runs; production "
            "serving will be refused until the real selection kernels land."
        )

    setattr(server_args, "_double_sparsity_parsed_config", config)
