# SPDX-License-Identifier: Apache-2.0
"""
Synchronous pipeline executor implementation.
"""
from typing import List

from sgl_diffusion.runtime.pipelines.executors.pipeline_executor import (
    PipelineExecutor,
    Timer,
    logger,
)
from sgl_diffusion.runtime.pipelines.pipeline_batch_info import Req
from sgl_diffusion.runtime.pipelines.stages import PipelineStage
from sgl_diffusion.runtime.server_args import ServerArgs


class SyncExecutor(PipelineExecutor):
    """
    A simple synchronous executor that runs stages sequentially.
    """

    def execute(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Execute the pipeline stages sequentially.
        """
        logger.info("Running pipeline stages sequentially with SyncExecutor.")

        for stage in stages:
            with Timer(stage.__class__.__name__):
                batch = stage(batch, server_args)

        return batch
