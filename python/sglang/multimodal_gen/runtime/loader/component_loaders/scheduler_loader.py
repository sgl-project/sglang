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


def _supported_init_kwargs(scheduler_cls, config: dict) -> dict:
    """Drop scheduler_config keys the constructor can't accept.

    Some checkpoints (e.g. distilled fixed-step schedulers) carry keys in
    ``scheduler_config.json`` that are consumed downstream, not by the
    scheduler ``__init__``. Constructors taking ``**kwargs`` accept anything
    and are left untouched; otherwise unknown non-private keys would raise
    ``TypeError``, so they are dropped (private ``_``-keys are handled by
    ``register_to_config``).
    """
    params = inspect.signature(scheduler_cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return config
    dropped = [k for k in config if k not in params and not k.startswith("_")]
    if dropped:
        logger.debug("Ignoring unsupported scheduler config keys: %s", dropped)
        return {k: v for k, v in config.items() if k not in dropped}
    return config


class SchedulerLoader(ComponentLoader):
    """Loader for scheduler."""

    component_names = ["scheduler"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        """Load the scheduler based on the model path, and inference args."""
        config = get_diffusers_component_config(component_path=component_model_path)

        checkpoint_class_name = config.pop("_class_name", None)
        class_name = (
            getattr(server_args.pipeline_config, "scheduler_class_override", None)
            or checkpoint_class_name
        )
        assert (
            class_name is not None
        ), "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        if checkpoint_class_name is not None and class_name != checkpoint_class_name:
            logger.info(
                "Overriding scheduler class from %s to %s",
                checkpoint_class_name,
                class_name,
            )

        scheduler_cls, _ = ModelRegistry.resolve_model_cls(class_name)

        scheduler = scheduler_cls(**_supported_init_kwargs(scheduler_cls, config))
        if server_args.pipeline_config.flow_shift is not None:
            scheduler.set_shift(server_args.pipeline_config.flow_shift)

        return scheduler
