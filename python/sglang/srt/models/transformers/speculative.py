# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import torch
from torch import nn


class AuxiliaryHiddenStateCapture:
    def __init__(self, model, layer_ids, num_layers):
        ids = tuple(layer_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(
                type(index) is not int or index < 0 or index >= num_layers
                for index in ids
            )
        ):
            raise ValueError(
                f"Capture layers must be distinct indices in [0, {num_layers})"
            )
        candidates = [
            module
            for module in model.modules()
            if isinstance(module, nn.ModuleList)
            and len(module) == num_layers
            and all(
                any(
                    hasattr(child, "_sglang_attention_key") for child in layer.modules()
                )
                for layer in module
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Cannot identify a unique Transformer layer stack for hidden state capture"
            )
        self.layer_ids = ids
        self.values = {}
        self.handles = []
        for index in ids:

            def capture(module, args, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                if (
                    not isinstance(hidden, torch.Tensor)
                    or hidden.ndim != 3
                    or hidden.shape[0] != 1
                ):
                    raise ValueError("Captured layers must return packed hidden states")
                self.values[index] = hidden[0]

            self.handles.append(candidates[0][index].register_forward_hook(capture))

    def reset(self):
        self.values.clear()

    def collect(self):
        if len(self.values) != len(self.layer_ids):
            raise RuntimeError("Requested auxiliary hidden states were not produced")
        return [self.values[index] for index in self.layer_ids]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class SpeculativeMixin:
    def get_embed_and_head(self):
        if self.lm_head is None:
            raise ValueError("Speculative decoding requires a generation model")
        return self.model.get_input_embeddings().weight, self.lm_head.weight

    def _set_capture_layers(self, layer_ids):
        if self.pp_group.world_size != 1:
            raise ValueError("Transformers auxiliary capture currently requires PP=1")
        if self.pooler is not None:
            raise ValueError("Speculative decoding requires a generation model")
        previous = getattr(self, "_aux_capture", None)
        if previous is not None:
            previous.close()
        self._aux_capture = AuxiliaryHiddenStateCapture(
            self.model, layer_ids, self.text_config.num_hidden_layers
        )
        self.capture_aux_hidden_states = True

    def set_eagle3_layers_to_capture(self, layer_ids=None):
        if layer_ids is None:
            count = self.text_config.num_hidden_layers
            layer_ids = [1, count // 2 - 1, count - 4]
        self._set_capture_layers(layer_ids)

    def set_dflash_layers_to_capture(self, layer_ids):
        if layer_ids is None:
            raise ValueError("DFLASH requires explicit capture layer indices")
        self._set_capture_layers(layer_ids)
