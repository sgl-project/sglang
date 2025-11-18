# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base class for all pipeline executors.
"""
import time
from abc import ABC, abstractmethod
from typing import List

from sglang.multimodal_gen.runtime.pipelines.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines.stages import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Timer:
    """
    A very simple timer that doesn't for cuda-stream to be synced
    """

    def __init__(self, name="Stage"):
        self.name = name
        self.start = None
        self.end = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        logger.info(f"[{self.name}] started...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        logger.info(f"[{self.name}] finished in {self.elapsed:.4f} seconds")
        return False


class PipelineExecutor(ABC):
    """
    Abstract base class for all pipeline executors.

    Executors orchestrate the execution of pipeline, with managing the parallel and communications required by stages

    """

    def __init__(self, server_args):
        self.server_args = server_args

    @abstractmethod
    def execute(
        self,
        stages: List[PipelineStage],
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Execute the pipeline stages.

        Args:
            stages: A list of pipeline stages to execute.
            batch: The batch to process.
            server_args: The server arguments.

        Returns:
            The processed batch.
        """
        raise NotImplementedError
