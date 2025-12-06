from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix

from .dconv_fwd_cache_varlen import dynamic_conv_triton_cache_varlen
from .dconv_speculative_step import (
    causal_conv_step_triton_speculative,
    causal_dynamic_conv1d_update,
)
from .dconv_step import causal_conv_step_triton


class DynamicShortConvolutionKernelGenerator(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.w1 = ColumnParallelLinear(
            input_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w1", prefix),
        )

        self.act = nn.SiLU()

        self.w2 = RowParallelLinear(
            hidden_size,
            output_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("w2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.w1(x)
        x = self.act(x)
        x, _ = self.w2(x)
        return x


class DynamicShortConvolution(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        generator_input_size: int,
        generator_reduction: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        generator_hidden_size = hidden_size // generator_reduction

        self.kernel_generator = DynamicShortConvolutionKernelGenerator(
            input_size=generator_input_size,
            hidden_size=generator_hidden_size,
            output_size=hidden_size * kernel_size,
            quant_config=quant_config,
            prefix=add_prefix("kernel_generator", prefix),
        )

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

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
        if cu_seqlens is not None:
            assert cache is not None, "Cache must be provided for varlen mode."
            B = len(cu_seqlens) - 1
            W = self.kernel_size

            out = dynamic_conv_triton_cache_varlen(
                x,
                self.get_kernel(generator_input),
                cu_seqlens,
                cache,
                cache_indices,
                has_initial_state,
                seq_idx,
            )
            out = nn.functional.silu(out)
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
                        else:
                            cache[cache_idx, :, 1 : num_beginning + 1] = 0
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
                intermediate_conv_window[cache_indices, : W - 2] = (
                    intermediate_conv_window[cache_indices, 1 : W - 1]
                )
                out = causal_conv_step_triton_speculative(
                    x,
                    cache,
                    self.get_kernel(generator_input),
                    cache_indices,
                    intermediate_conv_window,
                )
                return nn.functional.silu(out)
            else:
                return (
                    causal_dynamic_conv1d_update(
                        x=x.transpose(1, 2).contiguous(),
                        conv_state=cache,
                        weight=self.get_kernel(generator_input),
                        conv_state_indices=cache_indices,
                        intermediate_conv_window=intermediate_conv_window,
                        retrieve_next_token=retrieve_next_token,
                        retrieve_next_sibling=retrieve_next_sibling,
                        retrieve_parent_token=retrieve_parent_token,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )

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
        shape = x.shape
        generator_input = x if generator_input is None else generator_input

        kernels_triton = self.get_kernel(generator_input.squeeze(1))  # [B, D, W]

        x_out_triton = causal_conv_step_triton(
            x,
            cache,
            kernels_triton,
        )
        x_out_triton = nn.functional.silu(x_out_triton)
        return x_out_triton.view(shape), cache
