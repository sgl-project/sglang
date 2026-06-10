# SPDX-License-Identifier: Apache-2.0


def _get_pipeline_class_name(extra_argv):
    for i, arg in enumerate(extra_argv):
        if arg == "--pipeline-class-name" and i + 1 < len(extra_argv):
            return extra_argv[i + 1]
        if arg.startswith("--pipeline-class-name="):
            return arg.split("=", 1)[1]
    return None


def has_registered_pipeline_class(extra_argv):
    pipeline_class_name = _get_pipeline_class_name(extra_argv)
    if not pipeline_class_name:
        return False
    try:
        from sglang.multimodal_gen.registry import (
            _PIPELINE_REGISTRY,
            _discover_and_register_pipelines,
        )
    except ImportError:
        return False
    _discover_and_register_pipelines()
    return pipeline_class_name in _PIPELINE_REGISTRY
