"""Observe actual TP/PP replay outputs independently of the capture exporter."""

import os
import sys
from pathlib import Path

import torch

from sglang.srt.model_executor.model_runner import ModelRunner

_forward = ModelRunner.forward


def observe_forward(self, forward_batch, *args, **kwargs):
    result = _forward(self, forward_batch, *args, **kwargs)
    root = Path(os.environ["TRAINING_CAPTURE_REPLAY_DIRECTORY"])
    rank = self.ps.pp_rank * self.ps.tp_size + self.ps.tp_rank
    root = root / f"TP{self.ps.tp_rank}_PP{self.ps.pp_rank}_Rank{rank}_pid{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    positions = forward_batch.positions.cpu()
    slots = forward_batch.out_cache_loc[: positions.numel()].long()
    observation = {
        "model.forward_batch_info.positions": positions,
        "model.forward_batch_info.input_ids": forward_batch.input_ids.cpu(),
        "cuda_graph": result.can_run_graph,
    }
    pool = self.token_to_kv_pool
    for layer in (0, 14, 27):
        if not pool.start_layer <= layer < pool.start_layer + pool.layer_num:
            continue
        prefix = f"model.layers.{layer}.self_attn.attn.input_"
        observation[prefix + "k"] = pool.get_key_buffer(layer)[slots].cpu()
        observation[prefix + "v"] = pool.get_value_buffer(layer)[slots].cpu()
    if self.pp_group.is_last_rank:
        observation["logits_processor"] = result.logits_output.next_token_logits.cpu()
    torch.save(observation, root / f"Pass{self.forward_pass_id:05d}.pt")
    return result


# Spawned workers import this test entrypoint before constructing ModelRunner.
ModelRunner.forward = observe_forward


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
