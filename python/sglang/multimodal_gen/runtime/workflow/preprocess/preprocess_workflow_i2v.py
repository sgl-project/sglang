# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import TYPE_CHECKING

from tqdm import tqdm

from sglang.multimodal_gen.dataset.preprocessing_datasets import PreprocessBatch
from sglang.multimodal_gen.runtime.workflow.preprocess.components import (
    ParquetDatasetSaver,
)
from sglang.multimodal_gen.runtime.workflow.preprocess.preprocess_workflow import (
    PreprocessWorkflow,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
        ComposedPipelineBase,
    )
    from sglang.multimodal_gen.runtime.workflow.preprocess.components import (
        VideoForwardBatchBuilder,
    )


class PreprocessWorkflowI2V(PreprocessWorkflow):
    training_dataloader: "DataLoader"
    validation_dataloader: "DataLoader"
    preprocess_pipeline: "ComposedPipelineBase"
    processed_dataset_saver: "ParquetDatasetSaver"
    video_forward_batch_builder: "VideoForwardBatchBuilder"

    def run(self) -> None:
        # Training dataset preprocessing
        for batch in tqdm(
            self.training_dataloader,
            desc="Preprocessing training dataset",
            unit="batch",
        ):
            forward_batch: PreprocessBatch = self.video_forward_batch_builder(batch)

            forward_batch = self.preprocess_pipeline.forward(
                forward_batch, self.server_args
            )

            self.processed_dataset_saver.save_and_write_parquet_batch(
                forward_batch, self.training_dataset_output_dir
            )

        self.processed_dataset_saver.flush_tables()
        self.processed_dataset_saver.clean_up()

        # Validation dataset preprocessing
        if self.validation_dataloader is not None:
            for batch in tqdm(
                self.validation_dataloader,
                desc="Preprocessing validation dataset",
                unit="batch",
            ):
                forward_batch = self.video_forward_batch_builder(batch)

                forward_batch = self.preprocess_pipeline.forward(
                    forward_batch, self.server_args
                )

                self.processed_dataset_saver.save_and_write_parquet_batch(
                    forward_batch, self.validation_dataset_output_dir
                )
            self.processed_dataset_saver.flush_tables()
            self.processed_dataset_saver.clean_up()
