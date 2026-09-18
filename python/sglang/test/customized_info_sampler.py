from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.layers.sampler import Sampler, register_sampler_backend
from sglang.srt.managers.scheduler import run_scheduler_process

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


CUSTOMIZED_INFO_FIELD = "sampled_token_ids_copy"
CUSTOMIZED_INFO_ARRAY_FIELD = "sampled_token_ids_array"
CUSTOMIZED_INFO_SAMPLER_BACKEND = "customized_info_probe"


class CustomizedInfoSampler(Sampler):
    """Sampler probe that mirrors every sampled token into customized_info.

    The scheduler already appends sampled token ids to each request's output_ids.
    By copying the same values into customized_info at the sampler boundary, the
    test can assert that customized_info is sliced and accumulated exactly like
    output_ids throughout the scheduler -> tokenizer manager -> Engine path.
    """

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_next_token_ids = super().forward(
            logits_output,
            sampling_info,
            return_logprob,
            top_logprobs_nums,
            token_ids_logprobs,
            positions,
        )

        if logits_output.customized_info is None:
            logits_output.customized_info = {}
        logits_output.customized_info[CUSTOMIZED_INFO_FIELD] = (
            batch_next_token_ids.detach().cpu().tolist()
        )
        logits_output.customized_info[CUSTOMIZED_INFO_ARRAY_FIELD] = (
            batch_next_token_ids.detach().cpu().unsqueeze(-1).numpy()
        )
        return batch_next_token_ids


def install_customized_info_sampler() -> None:
    # Register before ServerArgs validation in the parent and before sampler
    # construction in the scheduler subprocess.
    register_sampler_backend(
        CUSTOMIZED_INFO_SAMPLER_BACKEND,
        CustomizedInfoSampler,
    )


def run_scheduler_process_with_customized_info_sampler(*args, **kwargs):
    # Engine launches the scheduler in a subprocess. Install the sampler there
    # too so create_sampler() can resolve CUSTOMIZED_INFO_SAMPLER_BACKEND.
    install_customized_info_sampler()
    return run_scheduler_process(*args, **kwargs)


class CustomizedInfoEngine(Engine):
    run_scheduler_process_func = staticmethod(
        run_scheduler_process_with_customized_info_sampler
    )


if __name__ == "__main__":
    import os
    import sys

    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    install_customized_info_sampler()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        launch_server(
            server_args,
            run_scheduler_process_func=run_scheduler_process_with_customized_info_sampler,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
