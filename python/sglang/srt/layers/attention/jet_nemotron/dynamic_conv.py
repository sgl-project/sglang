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
# Modified

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN

from .dconv_fwd_cache_varlen import dynamic_conv_triton_cache_varlen
from .dconv_speculative_step import causal_conv_step_triton_speculative, causal_dynamic_conv1d_update
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
        generator_activation: str = "silu",
        activation: Optional[str] = "silu",
        static_conv_init: Callable = None,
        use_fast_conv1d: bool = True,
        implementation: str = "naive",
    ) -> DynamicShortConvolution:
        super().__init__()

        self.hidden_size = hidden_size
        self.generator_input_size = (
            hidden_size if generator_input_size is None else generator_input_size
        )
        self.generator_hidden_size = (
            hidden_size
            if generator_reduction is None
            else (hidden_size // generator_reduction)
        )
        self.kernel_size = kernel_size
        self.activation = None
        self.use_fast_conv1d = use_fast_conv1d
        self.implementation = implementation

        if activation is not None:
            assert activation in [
                "silu",
                "swish",
            ], f"Activation `{activation}` not supported yet."
            self.activation = activation

        self.static_conv_init = static_conv_init

        self.kernel_generator = nn.Sequential(
            OrderedDict(
                [
                    (
                        "w1",
                        nn.Linear(
                            self.generator_input_size,
                            self.generator_hidden_size,
                            bias=False,
                        ),
                    ),
                    ("act", ACT2FN[generator_activation]),
                    (
                        "w2",
                        nn.Linear(
                            self.generator_hidden_size,
                            self.hidden_size * self.kernel_size,
                            bias=True,
                        ),
                    ),
                ]
            )
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
            kernels = rearrange(
                flat_kernels, "b t (d w) -> b t d w", w=self.kernel_size
            )
        elif flat_kernels.dim() == 2:
            kernels = rearrange(flat_kernels, "b (d w) -> b d w", w=self.kernel_size)
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
        cache_indices: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        has_initial_state: Optional[torch.Tensor] = None,
        intermediate_conv_window: Optional[torch.Tensor] = None,
        retrieve_next_token: Optional[torch.Tensor] = None,
        retrieve_next_sibling: Optional[torch.Tensor] = None,
        retrieve_parent_token: Optional[torch.Tensor] = None,
        is_topk1: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`.
                If `seq_idx` is provided, shape `[T, D]`.
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
            cache_indices (`Optional[torch.Tensor]`):
                Indices of the cache for each sequence. Shape: [B].
            seq_idx (`Optional[torch.Tensor]`):
                Indices of the sequence for each token. Shape: [T].
            has_initial_state (`Optional[torch.Tensor]`):
                Whether the initial state is provided. Shape: [B].
            intermediate_conv_window (`Optional[torch.Tensor]`):
                Shape `[N, W-2+speculative_num_draft_tokens, D]`
                If provided, the intermediate conv window is updated **inplace** and cache_indices must be provided.
                Cache is used but not updated. Used for speculative decoding verification.
            retrieve_next_token (`Optional[torch.Tensor]`):
                Shape `[N, NP2_T]`, retrieve the next token for each token in the sequence.
            retrieve_next_sibling (`Optional[torch.Tensor]`):
                Shape `[N, NP2_T]`, retrieve the next sibling token for each token in the sequence.
            retrieve_parent_token (`Optional[torch.Tensor]`):
                Shape `[N, NP2_T]`, retrieve the parent token for each token in the sequence.
            is_topk1 (`Optional[bool]`):
                Whether to use topk1 mode. Default: `False`.

        Returns:
            Tensor of shape `[B, T, D]`.
        """

        """
        x: [B, T, D]
        return: [B, T, D]
        """
        if cu_seqlens is not None:
            assert cache is not None, "Cache must be provided for varlen mode."
            B = len(cu_seqlens) - 1
            T, _ = x.shape
            W = self.kernel_size
            input_dtype = x.dtype

            out = dynamic_conv_triton_cache_varlen(
                x,
                self.get_kernel(generator_input),
                cu_seqlens,
                cache,
                cache_indices,
                has_initial_state,
                seq_idx,
            )
            if self.activation is not None:
                out = ACT2FN[self.activation](out)
            out = out.to(input_dtype)
            if output_final_state:
                for i in range(B):
                    start_idx = cu_seqlens[i].item()
                    end_idx = cu_seqlens[i + 1].item()
                    cache_idx = cache_indices[i].item()
                    if end_idx - start_idx >= W - 1:
                        cache[cache_idx, :, 1:] = x[
                            end_idx - W + 1 : end_idx
                        ].transpose(0, 1)
                    else:
                        num_beginning = W - 1 - (end_idx - start_idx)
                        if has_initial_state[i].item():
                            cache[cache_idx, :, 1 : num_beginning + 1] = cache[
                                cache_idx, :, -num_beginning:
                            ]
                        cache[cache_idx, :, num_beginning + 1 :] = x[
                            start_idx:end_idx
                        ].transpose(0, 1)

            return out, cache

        B, T, _, W = *x.shape, self.kernel_size

        if intermediate_conv_window is not None:
            if is_topk1:
                assert (
                    W - 2 + T == intermediate_conv_window.shape[1]
                ), f"Shape mismatch between intermediate_conv_window ({intermediate_conv_window.shape}) and x ({x.shape})"
                assert (
                    cache_indices is not None
                ), "cache_indices must be provided for intermediate_conv_window"
                intermediate_conv_window[cache_indices, : W - 2] = intermediate_conv_window[
                    cache_indices, 1 : W - 1
                ]
                out = causal_conv_step_triton_speculative(
                    x,
                    cache,
                    self.get_kernel(generator_input),
                    cache_indices,
                    intermediate_conv_window,
                )
                if self.activation is not None:
                    out = ACT2FN[self.activation](out)
                return out
            else:
                out = causal_dynamic_conv1d_update(
                    x=x.transpose(1, 2).contiguous(),
                    conv_state=cache,
                    weight=self.get_kernel(generator_input),
                    bias=None,
                    # activation=self.activation,
                    cache_seqlens=None,
                    conv_state_indices=cache_indices,
                    num_accepted_tokens=None,
                    intermediate_conv_window=intermediate_conv_window,
                    retrieve_next_token=retrieve_next_token,
                    retrieve_next_sibling=retrieve_next_sibling,
                    retrieve_parent_token=retrieve_parent_token,
                ).transpose(1, 2).contiguous()
                out = ACT2FN[self.activation](out)
                return out

        if mask is not None:
            x = x.mul_(mask.unsqueeze(-1))

        # during the decoding phase, we assume the batch is composed of sequences of length 1
        assert T == 1
        x, cache = self._step_triton(x, cache, generator_input=generator_input)
        return x, cache

    def _step_triton(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        generator_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Triton Implementation ---
        assert x.shape[1] == 1, "x must be of shape [B, 1, D]"
        shape = x.shape  # Keep original shape [B, 1, D] for return
        generator_input = x if generator_input is None else generator_input

        # 1. Generate kernels
        kernels_triton = self.get_kernel(generator_input.squeeze(1))  # [B, D, W]

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
