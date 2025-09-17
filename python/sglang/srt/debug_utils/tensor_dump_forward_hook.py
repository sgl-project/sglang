"""
This file provides a function `register_forward_hook_for_model` that registers a forward hook on every operator of the model.
After registration, during model inference, all tensors generated throughout the forward pass will be recorded.

Usage:
Specify the output directory for dumping tensors using the argument `--debug-tensor-dump-output-folder`.
A separate directory will be created for each GPU rank, named in the format `f"TP{tp_rank}_PP{pp_rank}_Rank{rank}_pid{pid}"`.
Each complete forward pass of the model generates a `.pt` file named `f"Pass{pass_num}.pt"`, which can be loaded using `torch.load`.
The file contains a series of key-value pairs, where the keys correspond to operator names in the model
(similar to those in model.safetensors.index.json), and the values are the outputs produced by the respective operators.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class TensorDumper:
    def __init__(self, dump_dir: str, tp_size: int, tp_rank: int, pp_rank: int):
        self._forward_pass_id = 0
        self._pid = os.getpid()
        self._current_tensors = {}
        self._base_dir = Path(dump_dir)
        rank = tp_size * pp_rank + tp_rank
        self._process_dir = (
            self._base_dir / f"TP{tp_rank}_PP{pp_rank}_Rank{rank}_pid{self._pid}"
        )
        self._process_dir.mkdir(parents=True, exist_ok=True)

    def get_dump_dir(self):
        return str(self._process_dir)

    def add_tensor(self, name, tensor_item):
        if isinstance(tensor_item, (tuple, list)):
            tensors = [t.cpu() for t in tensor_item if t is not None]
            if len(tensors) == 1:
                self._current_tensors[name] = tensors[0]
            else:
                self._current_tensors[name] = tensors
        elif isinstance(tensor_item, torch.Tensor):
            self._current_tensors[name] = tensor_item.cpu()
        elif isinstance(tensor_item, LogitsProcessorOutput):
            self._current_tensors[name] = tensor_item.next_token_logits.cpu()
        elif isinstance(tensor_item, ForwardBatch):
            self._current_tensors[name + ".forward_batch_info.input_ids"] = (
                tensor_item.input_ids.cpu()
            )
            self._current_tensors[name + ".forward_batch_info.seq_lens"] = (
                tensor_item.seq_lens.cpu()
            )
            self._current_tensors[name + ".forward_batch_info.positions"] = (
                tensor_item.positions.cpu()
            )
        else:
            logger.warning(f"Unsupported type: {type(tensor_item)}: {tensor_item}")

    def dump_current_tensors(self):
        if len(self._current_tensors) == 0:
            return
        tensor_file_for_pass = self._process_dir / f"Pass{self._forward_pass_id:05d}.pt"
        logger.info(
            f"Dump {self._forward_pass_id:05d}th pass to {tensor_file_for_pass}"
        )
        torch.save(self._current_tensors, str(tensor_file_for_pass))
        self._current_tensors = {}
        self._forward_pass_id += 1

    def _add_hook_recursive(self, model, prefix, dump_module_name="model"):
        has_dump_module = False
        for name, module in model._modules.items():
            do_dump = False
            if len(prefix) == 0:
                cur_name = name
                if cur_name == dump_module_name:
                    has_dump_module = True
                    do_dump = True
            else:
                cur_name = prefix + "." + name
            if module is not None:
                temp, sub_count = self._add_hook_recursive(module, cur_name)
                has_dump_module = has_dump_module or temp
                if sub_count == 0 or do_dump:
                    # Avoid duplicated output hooks, e.g. self_attn may contain:
                    # self_attn.qkv_proj, self_attn.attn & self_attn.o_proj.
                    # Therefore, we do not need to add output hooks for self_attn,
                    # since the output of self_attn should be the same to self_attn.o_proj.
                    module.register_forward_hook(self._dump_hook(cur_name, do_dump))
        return has_dump_module, len(model._modules.items())

    def _dump_hook(self, tensor_name, do_dump):
        def inner_dump_hook(module, input, output):
            if do_dump:
                # This is the top-level model, so we will record the input for it.
                for item in input:
                    if isinstance(item, ForwardBatch):
                        self.add_tensor(tensor_name, item)
                self.dump_current_tensors()
            if output is not None:
                self.add_tensor(tensor_name, output)

        return inner_dump_hook


def register_forward_hook_for_model(
    model, dump_dir: str, tp_size: int, tp_rank: int, pp_rank: int
):
    tensor_dumper = TensorDumper(dump_dir, tp_size, tp_rank, pp_rank)
    has_dump_module, _ = tensor_dumper._add_hook_recursive(model, "")
    assert has_dump_module, "model should have a module named 'model'"
    return tensor_dumper
