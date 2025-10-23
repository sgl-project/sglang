# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformers.activations import ACT2FN

from .dconv_fwdbwd import dynamic_conv_triton_autograd
from .dconv_fwd_cache import dynamic_conv_triton_cache
from .dconv_step import causal_conv_step_triton


class DynamicShortConvolution(nn.Module):
    """
    Simple wrapper around `nn.Conv1d` that accepts dimension last.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        generator_input_size: Optional[int] = None,
        generator_reduction: Optional[int] = None,
        generator_activation: str = 'silu',
        activation: Optional[str] = 'silu',
        static_conv_init: Callable = None,
        use_fast_conv1d: bool = True,
        implementation: str = "naive",
    ) -> DynamicShortConvolution:
        super().__init__()

        self.hidden_size = hidden_size
        self.generator_input_size = hidden_size if generator_input_size is None else generator_input_size
        self.generator_hidden_size = hidden_size if generator_reduction is None else (hidden_size // generator_reduction)
        self.kernel_size = kernel_size
        self.activation = None
        self.use_fast_conv1d = use_fast_conv1d
        self.implementation = implementation

        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation
        
        self.static_conv_init = static_conv_init
        
        self.kernel_generator = nn.Sequential(
            OrderedDict([
                ("w1", nn.Linear(self.generator_input_size, self.generator_hidden_size, bias=False)),
                ("act", ACT2FN[generator_activation]),
                ("w2", nn.Linear(self.generator_hidden_size, self.hidden_size * self.kernel_size, bias=True)),
            ])
        )
        self._init_kernel_generator()

    def _init_kernel_generator(self):
        """
        Initialize the kernel generator.
        """
        for layer in self.kernel_generator:
            if isinstance(layer, nn.Linear):
                layer.weight.data.zero_()
                if layer.bias is not None:
                    layer.bias.data.zero_()
        
        if self.static_conv_init is not None:
            # init for static_bias
            self.static_conv_init(self.kernel_generator.w2.bias)

    def get_kernel(self, x: torch.Tensor) -> torch.Tensor:
        flat_kernels = self.kernel_generator(x)
        if flat_kernels.dim() == 3:
            kernels = rearrange(flat_kernels, 'b t (d w) -> b t d w', w=self.kernel_size)
        elif flat_kernels.dim() == 2:
            kernels = rearrange(flat_kernels, 'b (d w) -> b d w', w=self.kernel_size)
        else:
            raise ValueError(f"Invalid kernel shape: {flat_kernels.shape}")
        return kernels

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        generator_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`.
                If `seq_idx` is provided, `B` must be 1.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]

        Returns:
            Tensor of shape `[B, T, D]`.
        """

        """
        x: [B, T, D]
        return: [B, T, D]
        """
        
        assert cu_seqlens is None, "cu_seqlens not supported yet."
        
        B, T, D, W = *x.shape, self.kernel_size
        N = B

        input_dtype = x.dtype

        if mask is not None:
            x = x.mul_(mask.unsqueeze(-1))

        implementation = self.implementation
        if implementation == "triton" and not self.training:
            implementation = "triton_cache"

        # during the decoding phase, we assume the batch is composed of sequences of length 1
        if cache is not None and B * T == N:
            assert T == 1
            if implementation in ["naive", "triton_training"]:
                x, cache = self._step_naive(x, cache, cu_seqlens, generator_input=generator_input)
            elif implementation in ["triton", "triton_cache", "triton_decoding"]:
                x, cache = self._step_triton(x, cache, cu_seqlens, generator_input=generator_input)
            else:
                raise ValueError(f"Unknown implementation: {implementation}")
            return x, cache

        if output_final_state:
            new_cache = rearrange(x[..., -min(W, T):, :], 'n w d -> n d w')
        else:
            new_cache = None
        
        if implementation in ["naive", "triton_decoding"]:
            x = self._forward_naive(x, generator_input=generator_input)  # [B, T, D]
        elif implementation in ["triton", "triton_training"]:
            assert cache is None, "Cache not supported in pure triton mode. Please set model.eval() or use triton_cache mode."
            x = self._forward_triton(x, generator_input=generator_input)
        elif implementation == "triton_cache":
            x = self._forward_triton_cache(x, generator_input=generator_input, cache=cache)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        if self.activation is not None:
            x = ACT2FN[self.activation](x)
        
        x = x.to(input_dtype)
        if output_final_state:
            if cache is None:
                cache = x.new_zeros(N, D, W)
            cache[:, :, -min(W, T):].copy_(new_cache)

        return x, cache

    def _forward_naive(self, x: torch.Tensor, generator_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        W = self.kernel_size
        generator_input = x if generator_input is None else generator_input
        kernels = self.get_kernel(generator_input)
        x = F.pad(x.transpose(1, 2), (W - 1, 0))  # [B, D, T+W-1]
        x = x.unfold(dimension=2, size=W, step=1)  # [B, D, T, W]
        x = x.permute(0, 2, 1, 3)  # [B, T, D, W]
        x = (x * kernels).sum(dim=-1)  # [B, T, D]
        return x

    def _forward_triton(self, x: torch.Tensor, generator_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        generator_input = x if generator_input is None else generator_input
        kernels = self.get_kernel(generator_input)
        output_triton = dynamic_conv_triton_autograd(x, kernels)
        return output_triton

    @torch.no_grad
    def _forward_triton_cache(self, x: torch.Tensor, generator_input: Optional[torch.Tensor] = None, cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        generator_input = x if generator_input is None else generator_input
        assert not self.training, "Triton implementation is only available in eval mode."
        # cache: [B, D, T(W)]
        CHUNK_SIZE = 2048
        n_chunk = (x.shape[1] + CHUNK_SIZE - 1) // CHUNK_SIZE
        output_triton = torch.zeros_like(x)
        if cache is not None:
            cache = rearrange(cache, "b d t -> b t d")  # [B, T(W), D]
        for i in range(n_chunk):
            start = i * CHUNK_SIZE
            end = min((i + 1) * CHUNK_SIZE, x.shape[1])
            kernels = self.get_kernel(generator_input[:, start:end])
            out = dynamic_conv_triton_cache(x[:, start:end], kernels, cache=cache)
            output_triton[:, i*CHUNK_SIZE:end, :] = out
            cache = x[:, end-self.kernel_size:end, :]
        return output_triton

    def _step_naive(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        generator_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[1] == 1, "x must be of shape [B, 1, D]"
        shape = x.shape
        generator_input = x if generator_input is None else generator_input
        x = x.squeeze(1)
        generator_input = generator_input.squeeze(1) # Shape [B, D]
        B, D, W = *x.shape, self.kernel_size

        # we follow the fast mode that updates the cache in-place
        cache.copy_(cache.roll(shifts=-1, dims=-1))
        cache[:, :, -1] = x # [B, D, T(W)]
        
        kernels = self.get_kernel(generator_input) # [B, D, W]
        x = torch.sum(cache * kernels, dim=-1)
        
        if self.activation is not None:
            x = ACT2FN[self.activation](x)
        
        return x.view(shape), cache
    
    def _step_triton(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        generator_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Triton Implementation ---
        assert x.shape[1] == 1, "x must be of shape [B, 1, D]"
        shape = x.shape # Keep original shape [B, 1, D] for return
        generator_input = x if generator_input is None else generator_input

        # 1. Generate kernels
        kernels_triton = self.get_kernel(generator_input.squeeze(1)) # [B, D, W]

        # 2. Call Triton kernel without activation
        x_out_triton = causal_conv_step_triton(
            x,
            cache,
            kernels_triton,
        )
        
        # Apply activation (if any) after kernel execution
        if self.activation is not None:
            x_out_triton = ACT2FN[self.activation](x_out_triton)

        # 3. Return reshaped output and the *same cache tensor* (it was updated in-place)
        return x_out_triton.view(shape), cache
