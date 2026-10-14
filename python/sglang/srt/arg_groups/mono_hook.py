from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.model_override_base import model_config_of
from sglang.srt.arg_groups.overrides import resolving_view

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def handle_layer_backend(server_args: ServerArgs) -> None:
    """--layer-backend mono: refuse at startup (every reason at once) when the model has no mono spec or the resolved
    deployment uses a feature the spec does not support. Runs after every other resolution."""
    cfg = resolving_view(server_args)
    from sglang.srt.layers.mono import guards, registry

    if not guards.mono_enabled(cfg):
        return

    hf_config = model_config_of(server_args).hf_config
    arch = (hf_config.architectures or ["?"])[0]
    spec = registry.load_spec(arch)
    if spec is None:
        raise guards.MonoRefused(f"--layer-backend mono: no mono spec for {arch}")
    guards.raise_if_refused(
        "config", spec.name, guards.config_refusals(cfg, hf_config, spec)
    )
    guards.warn_eager(cfg, spec)
    logger.info(
        "--layer-backend mono: %s mono spec accepted the configuration", spec.name
    )
