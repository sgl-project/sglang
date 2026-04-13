import inspect

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _filter_supported_scheduler_init_kwargs(
    scheduler_cls: type, config: dict
) -> tuple[dict, list[str]]:
    """Drop config fields unsupported by the resolved scheduler class."""
    try:
        signature = inspect.signature(scheduler_cls.__init__)
    except (TypeError, ValueError):
        return config, []

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return config, []

    supported_keys = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    filtered_config = {key: value for key, value in config.items() if key in supported_keys}
    ignored_keys = sorted(key for key in config if key not in supported_keys)
    return filtered_config, ignored_keys


class SchedulerLoader(ComponentLoader):
    """Loader for scheduler."""

    component_names = ["scheduler"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        """Load the scheduler based on the model path, and inference args."""
        config = get_diffusers_component_config(component_path=component_model_path)

        class_name = config.pop("_class_name")
        assert (
            class_name is not None
        ), "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        scheduler_cls, _ = ModelRegistry.resolve_model_cls(class_name)
        scheduler_config, ignored_keys = _filter_supported_scheduler_init_kwargs(
            scheduler_cls, config
        )
        if ignored_keys:
            logger.warning(
                "Ignoring unsupported scheduler config keys for %s: %s",
                class_name,
                ", ".join(ignored_keys),
            )

        scheduler = scheduler_cls(**scheduler_config)
        if server_args.pipeline_config.flow_shift is not None:
            scheduler.set_shift(server_args.pipeline_config.flow_shift)

        return scheduler
