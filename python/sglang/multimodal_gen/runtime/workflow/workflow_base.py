# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from abc import ABC, abstractmethod
from typing import Any, Optional

from sglang.multimodal_gen.runtime.pipelines import ComposedPipelineBase, build_pipeline
from sglang.multimodal_gen.runtime.pipelines.pipeline_registry import PipelineType
from sglang.multimodal_gen.runtime.server_args import ExecutionMode, ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WorkflowBase(ABC):
    """
    Abstract base class for defining video processing workflows.

    A workflow serves as the top-level orchestrator that coordinates multiple pipelines
    and components to accomplish a specific video processing task. The workflow pattern
    provides several key benefits:

    1. **Separation of Concerns**: Workflows separate high-level orchestration logic
       from low-level processing implementations in pipelines.

    2. **Modularity**: Different workflows can be created for different execution modes
       (preprocess, inference, etc.) while sharing common pipeline components.

    3. **Configuration Management**: Workflows manage the configuration and initialization
       of multiple related pipelines and components in a centralized manner.

    4. **Environment Setup**: Workflows handle system-level setup and resource
       allocation before pipeline execution begins.

    5. **Lifecycle Management**: Workflows control the complete lifecycle from
       initialization through execution to cleanup.

    The workflow acts as a factory and coordinator, creating the appropriate pipelines
    based on configuration, setting up the execution environment, and orchestrating
    the overall processing flow.
    """

    def __init__(self, server_args: ServerArgs):
        """
        Initialize the workflow with configuration arguments.

        Args:
            server_args: Configuration object containing all parameters
                          needed for workflow and pipeline setup.
        """
        self.server_args = server_args

        # TODO: pipeline_config should be: dict[str, PipelineConfig]
        # pipeline_type should be included in the PipelineConfig
        # pipeline_config[pipeline_name] = (pipeline_type, server_args)
        self._pipeline_configs: dict[str, tuple[PipelineType, ServerArgs]] = {}
        self._pipelines: dict[str, ComposedPipelineBase] = {}
        self._components: dict[str, Any] = {}
        self.register_pipelines()
        self.register_components()

        self.prepare_system_environment()
        self.load_pipelines()

    def load_pipelines(self) -> None:
        """
        Create and initialize all registered pipelines.

        This method instantiates pipeline objects from their configurations
        and makes them available as both dictionary entries and instance
        attributes for convenient access.
        """
        for pipeline_name, pipeline_config in self._pipeline_configs.items():
            pipeline_type, server_args = pipeline_config
            pipeline = build_pipeline(server_args, pipeline_type)
            self._pipelines[pipeline_name] = pipeline
            setattr(self, pipeline_name, pipeline)

    def add_pipeline_config(
        self, pipeline_name: str, pipeline_config: tuple[PipelineType, ServerArgs]
    ) -> None:
        """
        Register a pipeline configuration for later instantiation.

        Args:
            pipeline_name: Unique identifier for the pipeline.
            pipeline_config: Tuple containing the pipeline type and
                           configuration arguments.
        """
        self._pipeline_configs[pipeline_name] = pipeline_config

    def add_component(self, component_name: str, component: Any) -> None:
        """
        Register a component instance with the workflow.

        Components are auxiliary objects that may be shared across pipelines
        or used for workflow-level functionality (e.g., databases, caches,
        external services).

        Args:
            component_name: Unique identifier for the component.
            component: The component instance to register.
        """
        self._components[component_name] = component
        setattr(self, component_name, component)

    def get_component(self, component_name: str) -> Any:
        """
        Retrieve a registered component by name.

        Args:
            component_name: The name of the component to retrieve.

        Returns:
            The component instance.
        """
        return self._components[component_name]

    @abstractmethod
    def register_components(self) -> None:
        """
        Register workflow-specific components.

        Subclasses must implement this method to register any components
        needed for their specific workflow (e.g., databases, external APIs,
        shared resources).
        """
        pass

    @abstractmethod
    def register_pipelines(self) -> None:
        """
        Register workflow-specific pipelines.

        Subclasses must implement this method to define which pipelines
        are needed for their specific workflow and how they should be
        configured.
        """
        pass

    @abstractmethod
    def prepare_system_environment(self) -> None:
        """
        Prepare the system environment for workflow execution.

        Subclasses must implement this method to handle any system-level
        setup required before pipeline execution (e.g., GPU initialization,
        temporary directories, resource allocation).
        """
        pass

    @abstractmethod
    def run(self):
        """
        Execute the main workflow logic.

        Subclasses must implement this method to define the specific
        execution flow for their workflow, coordinating the registered
        pipelines and components to accomplish the desired task.
        """
        pass

    @classmethod
    def get_workflow_cls(cls, server_args: ServerArgs) -> Optional["WorkflowBase"]:
        """
        Factory method to get the appropriate workflow class based on execution mode.

        This method acts as a workflow factory, returning the appropriate
        workflow class implementation based on the specified execution mode
        in the configuration arguments.

        Args:
            server_args: Configuration object containing the execution mode
                          and other parameters.

        Returns:
            The appropriate workflow class for the specified execution mode,
            or None if no workflow is available for the given mode.
        """
        if server_args.mode == ExecutionMode.PREPROCESS:
            from sglang.multimodal_gen.runtime.workflow.preprocess.preprocess_workflow import (
                PreprocessWorkflow,
            )

            return PreprocessWorkflow.get_workflow_cls(server_args)
        else:
            raise ValueError(
                f"Execution mode: {server_args.mode} is not supported in workflow."
            )
