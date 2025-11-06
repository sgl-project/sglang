# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import os
from typing import cast

from torch.utils.data import DataLoader

from sglang.multimodal_gen.configs.configs import PreprocessConfig
from sglang.multimodal_gen.dataset.dataloader.record_schema import (
    basic_t2v_record_creator,
    i2v_record_creator,
)
from sglang.multimodal_gen.dataset.dataloader.schema import (
    pyarrow_schema_i2v,
    pyarrow_schema_t2v,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_rank
from sglang.multimodal_gen.runtime.pipelines.pipeline_registry import PipelineType
from sglang.multimodal_gen.runtime.server_args import ServerArgs, WorkloadType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.workflow.preprocess.components import (
    ParquetDatasetSaver,
    PreprocessingDataValidator,
    VideoForwardBatchBuilder,
    build_dataset,
)
from sglang.multimodal_gen.runtime.workflow.workflow_base import WorkflowBase

logger = init_logger(__name__)


class PreprocessWorkflow(WorkflowBase):

    def register_pipelines(self) -> None:
        self.add_pipeline_config(
            "preprocess_pipeline", (PipelineType.PREPROCESS, self.server_args)
        )

    def register_components(self) -> None:
        assert self.server_args.preprocess_config is not None
        preprocess_config: PreprocessConfig = self.server_args.preprocess_config

        # raw data validator
        raw_data_validator = PreprocessingDataValidator(
            max_height=preprocess_config.max_height,
            max_width=preprocess_config.max_width,
            num_frames=preprocess_config.num_frames,
            train_fps=preprocess_config.train_fps,
            speed_factor=preprocess_config.speed_factor,
            video_length_tolerance_range=preprocess_config.video_length_tolerance_range,
            drop_short_ratio=preprocess_config.drop_short_ratio,
        )
        self.add_component("raw_data_validator", raw_data_validator)

        # training dataset
        training_dataset = build_dataset(
            preprocess_config, split="train", validator=raw_data_validator
        )
        # we do not use collate_fn here because we use iterable-style Dataset
        # and want to keep the original type of the dataset
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=preprocess_config.preprocess_video_batch_size,
            num_workers=preprocess_config.dataloader_num_workers,
            collate_fn=lambda x: x,
        )
        self.add_component("training_dataloader", training_dataloader)

        # try to load validation dataset if it exists
        try:
            validation_dataset = build_dataset(
                preprocess_config, split="validation", validator=raw_data_validator
            )
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=preprocess_config.preprocess_video_batch_size,
                num_workers=preprocess_config.dataloader_num_workers,
                collate_fn=lambda x: x,
            )
        except ValueError:
            logger.warning(
                "Validation dataset not found, skipping validation dataset preprocessing."
            )
            validation_dataloader = None

        self.add_component("validation_dataloader", validation_dataloader)

        # forward batch builder
        video_forward_batch_builder = VideoForwardBatchBuilder(
            seed=self.server_args.preprocess_config.seed
        )
        self.add_component("video_forward_batch_builder", video_forward_batch_builder)

        # record creator
        if self.server_args.workload_type == WorkloadType.I2V:
            record_creator = i2v_record_creator
            schema = pyarrow_schema_i2v
        else:
            record_creator = basic_t2v_record_creator
            schema = pyarrow_schema_t2v
        processed_dataset_saver = ParquetDatasetSaver(
            flush_frequency=self.server_args.preprocess_config.flush_frequency,
            samples_per_file=self.server_args.preprocess_config.samples_per_file,
            schema=schema,
            record_creator=record_creator,
        )
        self.add_component("processed_dataset_saver", processed_dataset_saver)

    def prepare_system_environment(self) -> None:
        assert self.server_args.preprocess_config is not None
        dataset_output_dir = self.server_args.preprocess_config.dataset_output_dir
        os.makedirs(dataset_output_dir, exist_ok=True)

        validation_dataset_output_dir = os.path.join(
            dataset_output_dir, "validation_dataset", f"worker_{get_world_rank()}"
        )
        os.makedirs(validation_dataset_output_dir, exist_ok=True)
        self.validation_dataset_output_dir = validation_dataset_output_dir

        training_dataset_output_dir = os.path.join(
            dataset_output_dir, "training_dataset", f"worker_{get_world_rank()}"
        )
        os.makedirs(training_dataset_output_dir, exist_ok=True)
        self.training_dataset_output_dir = training_dataset_output_dir

    @classmethod
    def get_workflow_cls(cls, server_args: ServerArgs) -> "PreprocessWorkflow":
        if server_args.workload_type == WorkloadType.T2V:
            from sglang.multimodal_gen.runtime.workflow.preprocess.preprocess_workflow_t2v import (
                PreprocessWorkflowT2V,
            )

            return cast(PreprocessWorkflow, PreprocessWorkflowT2V)
        elif server_args.workload_type == WorkloadType.I2V:
            from sglang.multimodal_gen.runtime.workflow.preprocess.preprocess_workflow_i2v import (
                PreprocessWorkflowI2V,
            )

            return cast(PreprocessWorkflow, PreprocessWorkflowI2V)
        else:
            raise ValueError(
                f"Workload type: {server_args.workload_type} is not supported in preprocessing workflow."
            )
