# SPDX-License-Identifier: Apache-2.0
"""Model output transforms, independent of media saving and HTTP formatting."""

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage


class PostprocessStage(PipelineStage):
    @property
    def role_affinity(self):
        return RoleType.DECODER

    def process(self, batch, server_args):
        return server_args.pipeline_config.post_decoding(batch.output, server_args)

    def forward(self, batch, server_args):
        paths = batch.extra.get("decoded_output_paths")
        return OutputBatch(
            output=self.process(batch, server_args)
            if paths is None and batch.output is not None
            else None,
            output_file_paths=paths,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            trajectory_decoded=batch.extra.get("trajectory_decoded"),
            metrics=batch.metrics,
            usage=batch.usage,
        )

    def run_grouped_requests(self, batches, server_args):
        return [self.forward(batch, server_args) for batch in batches]
