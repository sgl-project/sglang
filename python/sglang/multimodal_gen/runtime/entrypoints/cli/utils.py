# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import argparse

from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RaiseNotImplementedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        raise NotImplementedError(f"The {option_string} option is not yet implemented")


def add_resolved_pipeline_config_cli_args(parser: argparse.ArgumentParser) -> None:
    """Let the selected or detected pipeline's config contribute extra CLI args.

    The base ``PipelineConfig`` CLI args are registered before the model is known,
    so model-specific fields (e.g. FLUX.3's ``--flux3-alpha``) can't be exposed the
    normal way. This does a lightweight pre-parse for the explicit pipeline
    selector (including its ``--pipeline`` alias), falling back to ``--config``
    and then to the pipeline the registry detects for ``--model-path`` without
    downloading it, and lets the selected config register its own flags. No-op
    for pipelines that don't provide the hook (unchanged behavior for every
    existing model)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--pipeline",
        "--pipeline-class-name",
        dest="pipeline_class_name",
        default=None,
    )
    pre.add_argument("--config", default=None)
    pre.add_argument("--model-path", dest="model_path", default=None)
    try:
        known, _ = pre.parse_known_args()
    except SystemExit:
        return
    pipeline_class_name = known.pipeline_class_name
    model_path = known.model_path
    if not pipeline_class_name and known.config:
        try:
            config_args = ServerArgs.load_config_file(known.config) or {}
            pipeline_class_name = config_args.get(
                "pipeline_class_name"
            ) or config_args.get("pipeline")
            model_path = model_path or config_args.get("model_path")
        except Exception as exc:
            logger.debug("Skipping config-selected CLI args: %s", exc)
            return
    try:
        from sglang.multimodal_gen.registry import (
            get_non_diffusers_pipeline_name,
            get_pipeline_config_classes,
        )

        if not pipeline_class_name and model_path:
            pipeline_class_name = get_non_diffusers_pipeline_name(model_path)
        if not pipeline_class_name:
            return
        classes = get_pipeline_config_classes(pipeline_class_name)
    except Exception as exc:  # pragma: no cover - registry hiccup shouldn't break CLI
        logger.debug("Skipping resolved-config CLI args: %s", exc)
        return
    if classes and hasattr(classes[0], "add_pipeline_cli_args"):
        classes[0].add_pipeline_cli_args(parser)
