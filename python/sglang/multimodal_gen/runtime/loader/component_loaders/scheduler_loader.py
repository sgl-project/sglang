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


class SchedulerLoader(ComponentLoader):
    """Loader for scheduler."""

    component_names = ["scheduler"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        """Load the scheduler based on the model path, and inference args."""
        config = get_diffusers_component_config(model_path=component_model_path)

        class_name = config.pop("_class_name")
        assert (
            class_name is not None
        ), "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        scheduler_cls, _ = ModelRegistry.resolve_model_cls(class_name)

        scheduler = scheduler_cls(**config)
        if server_args.pipeline_config.flow_shift is not None:
            scheduler.set_shift(server_args.pipeline_config.flow_shift)

        return scheduler
