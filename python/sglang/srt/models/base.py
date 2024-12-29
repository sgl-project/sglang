from typing import List, Tuple

import torch
from torch import nn


class BaseCausalLM(nn.Module):
    def export_static_params(self) -> List[Tuple[str, torch.Tensor]]:
        static_params = []
        for name, param in self.named_parameters():
            if 'rotary_emb' in name:
                static_params.append((name, param.data.detach().clone()))
        return static_params

    def import_static_params(self, static_params: List[Tuple[str, torch.Tensor]]):
        self_named_parameters = dict(self.named_parameters())
        for name, tensor in static_params:
            self_named_parameters[name].data[...] = tensor
