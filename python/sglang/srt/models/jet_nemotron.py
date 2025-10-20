from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import cast

import torch
import torch.nn as nn
from einops import rearrange
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from torch import LongTensor, Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2DecoderLayer, Qwen2Model
from sglang.srt.utils import add_prefix


class DynamicShortConvolution(nn.Module):
    """
    Simple wrapper around `nn.Conv1d` that accepts dimension last.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        generator_input_size: int | None = None,
        generator_reduction: int | None = None,
        generator_activation: str = "silu",
        activation: str | None = "silu",
        static_conv_init: Callable | None = None,
        use_fast_conv1d: bool = True,
        implementation: str = "naive",
    ) -> None:
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
            assert isinstance(self.kernel_generator.w2, nn.Linear)
            self.static_conv_init(self.kernel_generator.w2.bias)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        cache: Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: Tensor | None = None,
        generator_input: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            x (`Tensor`):
                Tensor of shape `[B, T, D]`.
                If `seq_idx` is provided, `B` must be 1.
            mask (`Tensor | None`):
                Attention mask dealing with padded positions.
            cache (`Tensor | None`):
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
                x, cache = self._step_naive(
                    x, cache, cu_seqlens, generator_input=generator_input
                )
            return x, cache

        if output_final_state:
            new_cache = rearrange(x[..., -min(W, T) :, :], "n w d -> n d w")
        else:
            new_cache = None

        if implementation in ["naive", "triton_decoding"]:
            x = self._forward_naive(x, generator_input=generator_input)  # [B, T, D]

        if self.activation is not None:
            x = ACT2FN[self.activation](x)

        x = x.to(input_dtype)
        if output_final_state:
            if cache is None:
                cache = x.new_zeros(N, D, W)
            assert new_cache is not None
            cache[:, :, -min(W, T) :].copy_(new_cache)

        return x, cache

    def get_kernel(self, x: Tensor) -> Tensor:
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

    def _forward_naive(
        self, x: Tensor, generator_input: Tensor | None = None
    ) -> Tensor:
        W = self.kernel_size
        generator_input = x if generator_input is None else generator_input
        kernels = self.get_kernel(generator_input)
        x = F.pad(x.transpose(1, 2), (W - 1, 0))  # [B, D, T+W-1]
        x = x.unfold(dimension=2, size=W, step=1)  # [B, D, T, W]
        x = x.permute(0, 2, 1, 3)  # [B, T, D, W]
        x = (x * kernels).sum(dim=-1)  # [B, T, D]
        return x

    def _step_naive(
        self,
        x: Tensor,
        cache: Tensor,
        cu_seqlens: Tensor | None = None,
        generator_input: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        assert x.shape[1] == 1, "x must be of shape [B, 1, D]"
        shape = x.shape
        generator_input = x if generator_input is None else generator_input
        x = x.squeeze(1)
        generator_input = generator_input.squeeze(1)  # Shape [B, D]
        B, D, W = *x.shape, self.kernel_size

        # we follow the fast mode that updates the cache in-place
        cache.copy_(cache.roll(shifts=-1, dims=-1))
        cache[:, :, -1] = x  # [B, D, T(W)]

        kernels = self.get_kernel(generator_input)  # [B, D, W]
        x = torch.sum(cache * kernels, dim=-1)

        if self.activation is not None:
            x = ACT2FN[self.activation](x)

        return x.view(shape), cache


class JetBlock(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_idx: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        jet_block_config = SimpleNamespace(**config.efficient_attention_config["jet"])

        hidden_size = config.hidden_size
        initializer_range = config.initializer_range

        self.mode = jet_block_config.mode

        self.hidden_size = hidden_size
        self.expand_v = jet_block_config.expand_v

        self.conv_size = jet_block_config.conv_size

        self.head_dim = jet_block_config.head_dim
        self.num_heads = jet_block_config.num_heads

        self.key_dim = int(self.num_heads * self.head_dim)
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_k_dim = jet_block_config.head_dim
        self.head_v_dim = int(jet_block_config.head_dim * self.expand_v)
        self.layer_idx = layer_idx

        self.autotune_interval = (
            32 * 16 * 1024
        )  # 32 batch size * 16 num head * 1024 sequence length

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.key_dim * self.expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={self.expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.key_dim * self.expand_v}, which is invalid for nn.Linear."
            )
        if not math.isclose(
            self.head_dim * self.expand_v, self.head_v_dim, rel_tol=1e-5
        ):
            raise ValueError(
                f"expand_v={self.expand_v} does not produce an integer value when multiplied by head_dim={self.head_dim}. "
                f"Resulting head_v_dim would be {self.head_dim * self.expand_v}, which is invalid for FusedRMSNormGated."
            )
        assert self.mode in [
            "chunk",
            "fused_recurrent",
        ], f"Not suppoerted mode `{jet_block_config.mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        setattr(self.A_log, "_no_weight_decay", True)
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        setattr(self.dt_bias, "_no_weight_decay", True)

        self.dynamic_conv1d = DynamicShortConvolution(
            hidden_size=self.value_dim,
            kernel_size=self.conv_size,
            generator_input_size=self.hidden_size,
            generator_reduction=jet_block_config.dconv_generator_reduction,
            static_conv_init=lambda x: self.init_linear_conv1d(
                x, std=initializer_range
            ),
            # implementation=jet_block_config.dconv_implementation, # DEBUG
        )

        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=float(jet_block_config.norm_eps),
            autotune_interval=self.autotune_interval,
        )
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
    ) -> Tensor:
        hidden_states = hidden_states.unsqueeze(0)

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = "fused_recurrent" if q_len <= 64 else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        cu_seqlens = forward_batch.prefix_chunk_cu_seq_lens  # ???

        q = F.silu(self.q_proj(hidden_states))
        k = F.silu(self.k_proj(hidden_states))

        v, conv_state = self.dynamic_conv1d.forward(
            x=self.v_proj(hidden_states),
            generator_input=hidden_states,
        )

        q, k = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        beta = self.b_proj(hidden_states).sigmoid()

        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias
        )

        if mode == "chunk":
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                output_final_state=False,
                cu_seqlens=cast(LongTensor, cu_seqlens),
                use_qk_l2norm_in_kernel=True,
                autotune_interval=self.autotune_interval,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                output_final_state=False,
                cu_seqlens=cast(LongTensor, cu_seqlens),
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        g = rearrange(
            self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim
        )
        o = self.o_norm(o, g)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        o = o.squeeze(0)

        return o

    @staticmethod
    def init_linear_conv1d(
        weight: Tensor, std: float, bias: Tensor | None = None
    ) -> None:
        weight.data.normal_(mean=0.0, std=std)
        if bias is not None:
            if not getattr(bias, "_no_reinit", False):
                nn.init.zeros_(bias)


class JetNemotronConfig(PretrainedConfig): ...  # Left empty deliberately.


class JetNemotronDecoderLayer(Qwen2DecoderLayer):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            **kwargs,
        )

        # Replace self_attn with JetBlock for "jet" layer types
        layer_type = config.layer_types[layer_id]
        if layer_type == "jet":
            self.self_attn = JetBlock(
                config=config,
                layer_idx=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )


class JetNemotronModel(Qwen2Model):
    def __init__(
        self,
        config: JetNemotronConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=JetNemotronDecoderLayer,
        )


class JetNemotronForCausalLM(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.quant_config = quant_config

        self.model = JetNemotronModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(PoolingType.LAST, normalize=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Tensor | None = None,
        get_embedding: bool = False,
    ) -> EmbeddingPoolerOutput | LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]):
        stacked_params_mapping: list[tuple[str, str, str | int]] = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # Handle stacked parameters first.
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break

            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = JetNemotronForCausalLM
