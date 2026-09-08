from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDecoderLayer, DFlashDraftModel
from sglang.srt.speculative.dflash_utils import can_dflash_slice_qkv_weight
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)

logger = logging.getLogger(__name__)

StepSampler = Callable[[torch.Tensor, int], torch.Tensor]


def _dspark_method_config(config) -> dict:
    """Return SpecForge's nested DFlash/DSpark method configuration."""

    text_config = getattr(config, "text_config", None) or config
    raw = getattr(text_config, "dflash_config", None)
    if raw is None:
        raw = getattr(config, "dflash_config", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        return dict(raw)
    except (TypeError, ValueError):
        return dict(vars(raw))


class DFlashGroupedConv(nn.Module):
    """SpecForge-compatible dynamic grouped convolution for DSpark blocks.

    SGLang flattens the request and proposal dimensions before entering the
    draft model.  Grouping the leading token dimension into consecutive
    ``block_size`` chunks restores the exact per-proposal layout used during
    SpecForge training without mixing requests.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        block_size: int,
        taps: int,
        group_size: int,
        mode: str = "legacy",
        apply_input: bool = True,
        apply_output: bool = True,
        residual_scale: float = 0.1,
        gate_bias: float = 2.0,
        freeze_identity: bool = False,
    ) -> None:
        super().__init__()
        if taps < 1 or taps > block_size:
            raise ValueError(
                "DSpark dynamic conv taps must be in [1, block_size], got "
                f"taps={taps}, block_size={block_size}."
            )
        if group_size < 1 or hidden_size % group_size:
            raise ValueError(
                f"DSpark dynamic conv group_size={group_size} must divide "
                f"hidden_size={hidden_size}."
            )
        if mode not in {"legacy", "survival-gated"}:
            raise ValueError(
                "DSpark dynamic conv mode must be legacy or survival-gated, "
                f"got {mode!r}."
            )
        if not apply_input and not apply_output:
            raise ValueError(
                "DSpark dynamic conv must enable its input or output side."
            )
        if mode == "survival-gated" and taps < 2:
            raise ValueError("DSpark survival-gated dynamic conv requires taps >= 2.")
        if residual_scale < 0:
            raise ValueError("DSpark dynamic conv residual_scale must be non-negative.")

        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size
        self.mode = mode
        self.apply_input = bool(apply_input)
        self.apply_output = bool(apply_output)
        self.residual_scale = float(residual_scale)
        self.gate_bias = float(gate_bias)

        base_kernel = torch.zeros(2, self.taps, int(hidden_size))
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(
            base_kernel, requires_grad=not bool(freeze_identity)
        )
        self.kernel_projection = nn.Linear(
            int(hidden_size),
            2 * self.taps * self.num_groups,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore the identity initialization used by SpecForge."""

        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0].fill_(1.0)
            self.kernel_projection.weight.zero_()

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        *,
        side: int,
        block_size: Optional[int] = None,
    ) -> torch.Tensor:
        if hidden_states.ndim < 2:
            raise ValueError(
                "DSpark dynamic conv expects [..., tokens, hidden] or "
                "[tokens, hidden] input."
            )
        active_block_size = int(block_size or self.block_size)
        if active_block_size < self.taps:
            raise ValueError(
                "DSpark dynamic conv runtime block size must be at least its "
                f"number of taps={self.taps}, got {active_block_size}."
            )
        sequence_length = int(hidden_states.shape[-2])
        if sequence_length % active_block_size:
            raise ValueError(
                "DSpark dynamic conv token count must be divisible by runtime "
                f"block_size={active_block_size}, got {sequence_length}."
            )

        original_shape = hidden_states.shape
        blocks = hidden_states.reshape(
            -1,
            active_block_size,
            self.num_groups,
            self.group_size,
        )
        dynamic = delta.reshape(
            -1,
            active_block_size,
            self.taps,
            self.num_groups,
        )
        base = self.base_kernel[side].reshape(
            1,
            1,
            self.taps,
            self.num_groups,
            self.group_size,
        )

        if self.mode == "survival-gated":
            gate = torch.sigmoid(dynamic[..., 0, :] + self.gate_bias).unsqueeze(-1)
            output = blocks
            for tap in range(1, self.taps):
                shifted = F.pad(
                    blocks[:, : active_block_size - tap],
                    (0, 0, 0, 0, tap, 0),
                )
                lag = torch.tanh(dynamic[..., tap, :]).unsqueeze(-1)
                output = output + self.residual_scale * gate * lag * shifted
            return output.reshape(original_shape)

        coefficients = base + dynamic.unsqueeze(-1)
        output = coefficients[:, :, 0] * blocks
        for tap in range(1, self.taps):
            shifted = F.pad(
                blocks[:, : active_block_size - tap],
                (0, 0, 0, 0, tap, 0),
            )
            output = output + coefficients[:, :, tap] * shifted
        return output.reshape(original_shape)

    def prepare(
        self,
        hidden_states: torch.Tensor,
        *,
        block_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1],
            2,
            self.taps,
            self.num_groups,
        )
        prepared = hidden_states
        if self.apply_input:
            prepared = self._convolve(
                hidden_states,
                coefficients[..., 0, :, :],
                side=0,
                block_size=block_size,
            )
        return prepared, coefficients[..., 1, :, :]

    def finish(
        self,
        hidden_states: torch.Tensor,
        coefficients: torch.Tensor,
        *,
        block_size: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.apply_output:
            return hidden_states
        return self._convolve(
            hidden_states,
            coefficients,
            side=1,
            block_size=block_size,
        )


class DSparkDecoderLayer(DFlashDecoderLayer):
    """DFlash decoder layer with SpecForge-compatible DynamicConv wrappers."""

    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__(config=config, layer_id=layer_id, quant_config=quant_config)
        method_config = _dspark_method_config(config)
        taps = int(method_config.get("conv_kernel_size", 0) or 0)
        group_size = int(method_config.get("conv_group_size", 0) or 0)
        transition_rank = int(method_config.get("local_transition_rank", 0) or 0)
        if transition_rank > 0:
            raise NotImplementedError(
                "This SGLang DSpark implementation supports DynamicConv but not "
                "SpecForge LocalTransitionAttention yet."
            )
        if bool(taps) != bool(group_size):
            raise ValueError(
                "DSpark dynamic conv requires dflash_config.conv_kernel_size and "
                "conv_group_size together."
            )

        text_config = getattr(config, "text_config", None) or config
        num_hidden_layers = int(getattr(text_config, "num_hidden_layers"))
        last_n_layers = int(method_config.get("conv_last_n_layers", 0) or 0)
        if last_n_layers < 0 or last_n_layers > num_hidden_layers:
            raise ValueError(
                "dflash_config.conv_last_n_layers must be in "
                f"[0, {num_hidden_layers}], got {last_n_layers}."
            )
        layer_enabled = taps > 0 and (
            last_n_layers == 0 or layer_id >= num_hidden_layers - last_n_layers
        )
        apply_to = str(method_config.get("conv_apply_to", "attention-mlp"))
        if apply_to not in {"attention-output", "attention", "attention-mlp"}:
            raise ValueError(
                "dflash_config.conv_apply_to must be attention-output, attention, "
                f"or attention-mlp; got {apply_to!r}."
            )
        output_only = apply_to == "attention-output"
        draft_config = parse_dspark_draft_config(draft_hf_config=config)
        block_size = draft_config.resolve_gamma(
            default=method_config.get("block_size")
        )
        if block_size is None:
            raise ValueError(
                "DSpark DynamicConv requires block_size in the checkpoint config."
            )

        def grouped_conv() -> Optional[DFlashGroupedConv]:
            if not layer_enabled:
                return None
            return DFlashGroupedConv(
                hidden_size=int(getattr(text_config, "hidden_size")),
                block_size=int(block_size),
                taps=taps,
                group_size=group_size,
                mode=str(method_config.get("conv_mode", "legacy")),
                apply_input=not output_only,
                apply_output=True,
                residual_scale=float(method_config.get("conv_residual_scale", 0.1)),
                gate_bias=float(method_config.get("conv_gate_bias", 2.0)),
                freeze_identity=bool(
                    method_config.get("conv_freeze_identity", False)
                ),
            )

        self.attention_conv = grouped_conv()
        self.mlp_conv = grouped_conv() if apply_to == "attention-mlp" else None
        dynamic_conv_scale = float(
            method_config.get("local_transition_conv_scale", 1.0)
        )
        self.register_buffer(
            "_dynamic_conv_scale",
            torch.tensor(dynamic_conv_scale, dtype=torch.float32),
            persistent=False,
        )

    def _prepare_conv(
        self,
        hidden_states: torch.Tensor,
        conv: Optional[DFlashGroupedConv],
        *,
        block_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if conv is None:
            return hidden_states, None
        prepared, kernel = conv.prepare(hidden_states, block_size=block_size)
        scale = self._dynamic_conv_scale.to(dtype=hidden_states.dtype)
        prepared = hidden_states + scale * (prepared - hidden_states)
        return prepared, kernel

    def _finish_conv(
        self,
        hidden_states: torch.Tensor,
        kernel: Optional[torch.Tensor],
        conv: Optional[DFlashGroupedConv],
        *,
        block_size: int,
    ) -> torch.Tensor:
        if kernel is None or conv is None:
            return hidden_states
        finished = conv.finish(hidden_states, kernel, block_size=block_size)
        scale = self._dynamic_conv_scale.to(dtype=hidden_states.dtype)
        return hidden_states + scale * (finished - hidden_states)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        dynamic_conv = self.attention_conv or self.mlp_conv
        spec_info = getattr(forward_batch, "spec_info", None)
        runtime_block_size = getattr(spec_info, "draft_token_num", None)
        active_block_size = (
            int(runtime_block_size)
            if runtime_block_size is not None
            else (dynamic_conv.block_size if dynamic_conv is not None else 1)
        )
        if dynamic_conv is not None and active_block_size > dynamic_conv.block_size:
            raise ValueError(
                "DSpark DynamicConv can serve a shorter proposal horizon but cannot "
                "exceed the checkpoint block size: "
                f"checkpoint={dynamic_conv.block_size}, runtime={active_block_size}."
            )

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, attention_kernel = self._prepare_conv(
            hidden_states,
            self.attention_conv,
            block_size=active_block_size,
        )
        attention_output = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        attention_output = self._finish_conv(
            attention_output,
            attention_kernel,
            self.attention_conv,
            block_size=active_block_size,
        )
        hidden_states, residual = self.post_attention_layernorm(
            attention_output, residual
        )
        hidden_states, mlp_kernel = self._prepare_conv(
            hidden_states,
            self.mlp_conv,
            block_size=active_block_size,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self._finish_conv(
            hidden_states,
            mlp_kernel,
            self.mlp_conv,
            block_size=active_block_size,
        )
        return hidden_states, residual


def gather_and_crop_vocab(
    local_logits: torch.Tensor, lm_head: nn.Module
) -> torch.Tensor:
    full_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
    return full_logits[..., : int(lm_head.org_vocab_size)]


def run_markov_block(
    head: nn.Module,
    base_logits: torch.Tensor,
    *,
    first_prev_tokens: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    sampler: StepSampler,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, proposal_len = base_logits.shape[:2]
    if proposal_len == 0:
        empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
        return empty, base_logits

    sampled_tokens = []
    corrected_logits = []
    prev_tokens = first_prev_tokens.long()
    for step_idx in range(proposal_len):
        step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
        step_logits = head.apply_step_logits(
            base_logits[:, step_idx, :],
            token_ids=prev_tokens,
            hidden_states=step_hidden,
        )
        next_tokens = sampler(step_logits, step_idx)
        sampled_tokens.append(next_tokens)
        corrected_logits.append(step_logits.unsqueeze(1))
        prev_tokens = next_tokens
    return (
        torch.stack(sampled_tokens, dim=1),
        torch.cat(corrected_logits, dim=1),
    )


class VanillaMarkov(nn.Module):

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
            )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids, hidden_states)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return run_markov_block(
            self,
            base_logits,
            first_prev_tokens=first_prev_tokens,
            hidden_states=hidden_states,
            sampler=sampler,
        )


class GatedMarkovHead(VanillaMarkov):

    markov_head_type = "gated"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.gate_proj = nn.Linear(int(hidden_size) + markov_rank, markov_rank)

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("GatedMarkovHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(
            dtype=prev_embeddings.dtype
        )
        return self.project_bias(gate * prev_embeddings)


class ContextAwareCausalResidualHead(VanillaMarkov):
    """SpecForge CARH proposal head used by the Tri-Alignment checkpoints."""

    markov_head_type = "carh"

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        block_size: int,
        gate_bias: float = 0.0,
    ) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("CARH requires block_size > 0.")
        self.hidden_proj = nn.Linear(hidden_size, self.markov_rank, bias=False)
        self.depth_embedding = nn.Embedding(self.block_size, self.markov_rank)
        self.fusion_norm = nn.LayerNorm(self.markov_rank)
        self.gate_proj = nn.Linear(self.markov_rank, 1)
        nn.init.constant_(self.gate_proj.bias, float(gate_bias))

    def _causal_latent(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        depth_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("CARH requires current draft hidden_states.")
        fused = (
            self.get_prev_embeddings(token_ids)
            + self.hidden_proj(hidden_states)
            + self.depth_embedding(depth_ids.long())
        )
        latent = F.silu(self.fusion_norm(fused))
        gate = torch.sigmoid(self.gate_proj(latent)).to(dtype=latent.dtype)
        return gate * latent

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        *,
        depth_idx: int = 0,
    ) -> torch.Tensor:
        depth_ids = torch.full_like(token_ids, int(depth_idx), dtype=torch.long)
        latent = self._causal_latent(token_ids, hidden_states, depth_ids)
        return self.project_bias(latent)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        if hidden_states is None:
            raise ValueError("CARH block logits require hidden_states.")
        block_size = token_ids.size(-1)
        if block_size > self.block_size:
            raise ValueError(
                "CARH runtime block size cannot exceed its checkpoint depth "
                f"embedding size: checkpoint={self.block_size}, runtime={block_size}."
            )
        depth_ids = torch.arange(
            block_size,
            device=token_ids.device,
            dtype=torch.long,
        ).view(*((1,) * (token_ids.ndim - 1)), block_size)
        depth_ids = depth_ids.expand_as(token_ids)
        latent = self._causal_latent(token_ids, hidden_states, depth_ids)
        return base_logits + self.project_bias(latent)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            raise ValueError("CARH sampling requires hidden_states.")
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len > self.block_size:
            raise ValueError(
                "CARH runtime proposal length cannot exceed its checkpoint depth "
                f"embedding size: checkpoint={self.block_size}, runtime={proposal_len}."
            )
        if proposal_len == 0:
            empty = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty, base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_tokens = first_prev_tokens.long()
        for step_idx in range(proposal_len):
            step_logits = base_logits[:, step_idx, :] + self.compute_step_bias(
                prev_tokens,
                hidden_states[:, step_idx, :],
                depth_idx=step_idx,
            )
            next_tokens = sampler(step_logits, step_idx)
            sampled_tokens.append(next_tokens)
            corrected_logits.append(step_logits.unsqueeze(1))
            prev_tokens = next_tokens
        return (
            torch.stack(sampled_tokens, dim=1),
            torch.cat(corrected_logits, dim=1),
        )


class RNNHead(VanillaMarkov):

    markov_head_type = "rnn"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.hidden_size = int(hidden_size)
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(2 * markov_rank + self.hidden_size, 3 * markov_rank)

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits
        leading_shape = base_logits.shape[:-2]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            state, bias = self._rnn_step(state, prev_emb, hidden_states[..., k, :])
            output_logits.append(base_logits[..., k, :] + bias)
        return torch.stack(output_logits, dim=-2)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(
                batch_size, 0, dtype=torch.long, device=base_logits.device
            )
            return empty, base_logits

        state = torch.zeros(
            batch_size,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        sampled_tokens = []
        corrected_logits = []
        prev_tokens = first_prev_tokens.long()
        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_tokens)
            state, bias = self._rnn_step(state, prev_emb, hidden_states[:, step_idx, :])
            step_logits = base_logits[:, step_idx, :] + bias
            next_tokens = sampler(step_logits, step_idx)
            sampled_tokens.append(next_tokens)
            corrected_logits.append(step_logits.unsqueeze(1))
            prev_tokens = next_tokens
        return (
            torch.stack(sampled_tokens, dim=1),
            torch.cat(corrected_logits, dim=1),
        )


def build_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0))
    if markov_rank <= 0:
        raise ValueError(
            "DSpark requires markov_rank > 0 (the Markov head is the core of the "
            f"semi-AR draft); got markov_rank={markov_rank}."
        )
    markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    vocab_size = int(config.vocab_size)
    hidden_size = int(config.hidden_size)
    if markov_head_type == "vanilla":
        return VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size
        )
    if markov_head_type == "carh":
        draft_config = parse_dspark_draft_config(draft_hf_config=config)
        block_size = draft_config.resolve_gamma(
            default=getattr(config, "block_size", None)
        )
        if block_size is None:
            raise ValueError("CARH requires block_size in the draft config.")
        return ContextAwareCausalResidualHead(
            vocab_size=int(getattr(config, "draft_vocab_size", vocab_size)),
            markov_rank=markov_rank,
            hidden_size=hidden_size,
            block_size=int(block_size),
            gate_bias=float(getattr(config, "carh_gate_bias", 0.0)),
        )
    if markov_head_type == "rnn":
        return RNNHead(
            vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size
        )
    raise ValueError(f"Unsupported DSpark markov_head_type={markov_head_type!r}.")


class DSparkConfidenceHead(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        markov_rank: int,
        with_markov: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.with_markov = bool(with_markov)
        input_dim = int(hidden_size) + (int(markov_rank) if self.with_markov else 0)
        self.proj = nn.Linear(input_dim, 1, bias=bias, dtype=dtype)
        self.register_buffer(
            "sts_temperatures", torch.ones((), dtype=torch.float32), persistent=False
        )
        self._last_confidence_raw: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embed_stack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.with_markov:
            if markov_embed_stack is None:
                raise ValueError(
                    "DSparkConfidenceHead(with_markov=True) requires markov_embed_stack."
                )
            features = torch.cat(
                [hidden_states, markov_embed_stack.to(dtype=hidden_states.dtype)],
                dim=-1,
            )
        else:
            features = hidden_states
        features = features.to(dtype=self.proj.weight.dtype)
        return self.proj(features).squeeze(-1)

    def apply_sts(self, confidence_raw: torch.Tensor) -> torch.Tensor:
        self._last_confidence_raw = confidence_raw
        return torch.sigmoid(confidence_raw.float() / self.sts_temperatures)


def build_confidence_head(config) -> Optional[nn.Module]:
    if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
        return None
    if not hasattr(config, "enable_confidence_head"):
        logger.warning(
            "DSpark draft config has no enable_confidence_head field; treating the "
            "confidence head as enabled."
        )
    hidden_size = int(config.hidden_size)
    markov_rank = int(getattr(config, "markov_rank", 0))
    with_markov = bool(getattr(config, "confidence_head_with_markov", markov_rank > 0))
    if with_markov and markov_rank <= 0:
        raise ValueError(
            "DSpark confidence_head_with_markov requires markov_rank > 0, "
            f"got markov_rank={markov_rank}."
        )
    return DSparkConfidenceHead(
        hidden_size=hidden_size,
        markov_rank=markov_rank,
        with_markov=with_markov,
    )


_DSPARK_SKIPPED_WEIGHT_PREFIXES = (
    "embed_tokens.",
    "lm_head.",
    "rotary_emb.",
)


class DSparkDraftMixin:

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self._fused_kv_write_cache = None
        self.logits_mup_width_multiplier = None
        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark draft requires markov_rank > 0, "
                f"got markov_rank={dspark_config.markov_rank}."
            )
        self.gamma = int(dspark_config.resolve_gamma(default=self.block_size))
        self.markov_head = build_markov_head(config)
        self.confidence_head = build_confidence_head(config)
        self.lm_head: Optional[nn.Module] = None

        dynamic_convs = [
            conv
            for layer in self.layers
            for conv in (
                getattr(layer, "attention_conv", None),
                getattr(layer, "mlp_conv", None),
            )
            if conv is not None
        ]
        self.dynamic_conv_enabled = bool(dynamic_convs)
        self.dynamic_conv_module_count = len(dynamic_convs)
        self.dynamic_conv_block_size = (
            dynamic_convs[0].block_size if dynamic_convs else None
        )
        self.dynamic_conv_mode = dynamic_convs[0].mode if dynamic_convs else None

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embeds with the shared target embedding INSIDE the draft graph
        # (the runner skips the eager input_embeds staging when the draft
        # model exposes forward_embed).
        return self.embed_tokens(input_ids)

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project the draft's raw final hidden through the target lm_head.

        muP targets (Inkling) train the draft against a FOLDED head (weights
        pre-divided by logits_mup_width_multiplier) while serving attaches the
        target's unfolded head, so the division happens here — exactly once,
        keeping base logits in the scale the markov bias and confidence head
        were trained against. DSparkWorkerV2 wires the multiplier from the
        target config; it stays None for non-muP targets.
        """
        if self.lm_head is None:
            raise ValueError(
                "DSpark dense draft requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        if self.logits_mup_width_multiplier:
            hidden = hidden / self.logits_mup_width_multiplier
        weight = self.lm_head.weight
        if hidden.dtype != weight.dtype:
            hidden = hidden.to(weight.dtype)
        local_logits = torch.matmul(hidden, weight.T)
        base_logits = gather_and_crop_vocab(local_logits, self.lm_head)
        return base_logits, None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        markov_weights = []
        confidence_weights = []
        backbone_weights = []
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if any(name.startswith(p) for p in _DSPARK_SKIPPED_WEIGHT_PREFIXES):
                continue
            if name.startswith("confidence_head."):
                if self.confidence_head is None:
                    continue
                confidence_weights.append((name, loaded_weight))
            elif name.startswith("markov_head."):
                markov_weights.append((name, loaded_weight))
            else:
                backbone_weights.append((name, loaded_weight))

        self._validate_dynamic_conv_weights(
            weights=backbone_weights, params_dict=params_dict
        )
        super().load_weights(backbone_weights)

        for name, loaded_weight in markov_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DSpark unexpected markov weight {name!r} not found in model "
                    f"parameters (known markov params require a {type(self.markov_head).__name__} head)."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        self._load_confidence_weights(
            confidence_weights=confidence_weights, params_dict=params_dict
        )

    @staticmethod
    def _validate_dynamic_conv_weights(
        *,
        weights: list[Tuple[str, torch.Tensor]],
        params_dict: dict[str, nn.Parameter],
    ) -> None:
        """Reject train/serve DynamicConv mismatches instead of skipping them."""

        markers = (".attention_conv.", ".mlp_conv.")
        expected = {name for name in params_dict if any(m in name for m in markers)}
        provided = set()
        unexpected = []
        for raw_name, _ in weights:
            if not any(marker in raw_name for marker in markers):
                continue
            candidates = [raw_name]
            if raw_name.startswith("model."):
                candidates.append(raw_name[len("model.") :])
            else:
                candidates.append(f"model.{raw_name}")
            resolved = next((name for name in candidates if name in params_dict), None)
            if resolved is None:
                unexpected.append(raw_name)
            else:
                provided.add(resolved)

        if unexpected:
            raise ValueError(
                "DSpark checkpoint contains DynamicConv weights but the runtime "
                "model did not construct matching modules. Check dflash_config: "
                f"{sorted(unexpected)}"
            )
        missing = expected - provided
        if missing:
            raise ValueError(
                "DSpark DynamicConv is enabled but the checkpoint is missing "
                f"weights: {sorted(missing)}"
            )
        if expected:
            logger.info(
                "Validated DSpark DynamicConv checkpoint contract with %d parameters.",
                len(expected),
            )

    def _load_confidence_weights(
        self,
        *,
        confidence_weights: list,
        params_dict: dict,
    ) -> None:
        if self.confidence_head is None:
            return
        loaded_names = set()
        for name, loaded_weight in confidence_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DSpark unexpected confidence weight {name!r} not found in "
                    "model parameters."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_names.add(name)

        confidence_param_names = {
            name for name in params_dict if name.startswith("confidence_head.")
        }
        missing = confidence_param_names - loaded_names
        if missing:
            raise ValueError(
                f"DSpark confidence head is enabled but the checkpoint is missing "
                f"{sorted(missing)}. Provide a checkpoint with trained confidence weights, "
                f"or disable the confidence head (enable_confidence_head=False)."
            )

    def _fused_kv_write_bundle(self, pool):
        cached = self._fused_kv_write_cache
        if cached is not None and cached[0] == id(pool):
            return cached[1]
        bundle = self._build_fused_kv_write_bundle(pool)
        self._fused_kv_write_cache = (id(pool), bundle)
        return bundle

    def _build_fused_kv_write_bundle(self, pool):
        layers = list(self.layers)
        if not layers:
            return None
        if not (hasattr(pool, "get_key_buffer") and hasattr(pool, "get_value_buffer")):
            return None
        attn0 = layers[0].self_attn
        head_dim = attn0.head_dim
        kv_size = attn0.kv_size
        rotary = attn0.rotary_emb
        if type(rotary).__name__ != "RotaryEmbedding":
            return None
        if not getattr(rotary, "is_neox_style", False):
            return None
        if getattr(rotary, "rotary_dim", None) != head_dim:
            return None
        eps = attn0.k_norm.variance_epsilon
        weights, knws, meta_rows = [], [], []
        for layer in layers:
            attn = layer.self_attn
            ok, _ = can_dflash_slice_qkv_weight(attn.qkv_proj)
            if not ok:
                return None
            if attn.qkv_proj.bias is not None:
                return None
            if attn.attn.k_scale is not None or attn.attn.v_scale is not None:
                return None
            if attn.head_dim != head_dim or attn.kv_size != kv_size:
                return None
            if attn.rotary_emb is not rotary and not torch.equal(
                attn.rotary_emb.cos_sin_cache, rotary.cos_sin_cache
            ):
                return None
            if attn.k_norm.variance_epsilon != eps:
                return None
            k_buf = pool.get_key_buffer(attn.attn.layer_id)
            v_buf = pool.get_value_buffer(attn.attn.layer_id)
            nh = kv_size // head_dim
            for buf in (k_buf, v_buf):
                if buf.dtype != torch.bfloat16:
                    return None
                if buf.shape[1:] != (nh, head_dim):
                    return None
                if buf.stride(1) != head_dim or buf.stride(2) != 1:
                    return None
            kv_slice = slice(attn.q_size, attn.q_size + 2 * attn.kv_size)
            w = attn.qkv_proj.weight[kv_slice]
            if w.dtype != torch.bfloat16:
                return None
            weights.append(w)
            knws.append(attn.k_norm.weight.data)
            meta_rows.append(
                [k_buf.data_ptr(), v_buf.data_ptr(), k_buf.stride(0), v_buf.stride(0)]
            )
        device = weights[0].device
        w_all = torch.cat(weights, dim=0).contiguous()
        knw = torch.stack(knws).to(device)
        meta = torch.tensor(meta_rows, dtype=torch.int64, device=device)
        cos_sin = rotary.cos_sin_cache.to(device)
        return (w_all, meta, knw, cos_sin, eps, len(layers), kv_size, head_dim)

    def _stacked_ctx_kv_params(self) -> Optional[dict]:
        """Stack every layer's KV projection into one weight (exact: the input
        hidden is shared, so concatenating output columns is equivalent).
        Cached; None (per-layer fallback) when a QKV weight cannot be sliced
        (quantized) or layers disagree on norm epsilon / bias presence.
        """
        cached = getattr(self, "_stacked_ctx_kv_cache", False)
        if cached is not False:
            return cached
        weights, biases, k_norm_weights = [], [], []
        eps = None
        for layer in self.layers:
            attn = layer.self_attn
            can_slice, _ = can_dflash_slice_qkv_weight(attn.qkv_proj)
            if not can_slice or eps not in (None, attn.k_norm.variance_epsilon):
                self._stacked_ctx_kv_cache = None
                return None
            eps = attn.k_norm.variance_epsilon
            kv_slice = slice(attn.q_size, attn.q_size + 2 * attn.kv_size)
            weights.append(attn.qkv_proj.weight[kv_slice])
            biases.append(
                attn.qkv_proj.bias[kv_slice] if attn.qkv_proj.bias is not None else None
            )
            k_norm_weights.append(attn.k_norm.weight)
        has_bias = [b is not None for b in biases]
        if any(has_bias) and not all(has_bias):
            self._stacked_ctx_kv_cache = None
            return None
        self._stacked_ctx_kv_cache = {
            "weight": torch.cat(weights, dim=0),
            "bias": torch.cat(biases, dim=0) if all(has_bias) else None,
            "k_norm_weight": torch.stack(k_norm_weights, dim=0).float(),
            "eps": eps,
        }
        return self._stacked_ctx_kv_cache

    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        ctx_hidden = self.project_target_hidden(target_hidden)

        bundle = self._fused_kv_write_bundle(pool)
        if bundle is not None:
            from sglang.kernels.ops.speculative.dspark.fused_kv_write import (
                fused_kv_norm_rope_write,
            )

            w_all, meta, knw, cos_sin, eps, num_layers, kv_size, head_dim = bundle
            kv_all = F.linear(ctx_hidden, w_all)
            if cache_loc_2d is not None and commit_lens is not None:
                locs = cache_loc_2d.reshape(-1)
                write_commit_lens = commit_lens
                locs_row_width = cache_loc_2d.shape[1]
            else:
                locs = cache_loc
                write_commit_lens = None
                locs_row_width = None
            fused_kv_norm_rope_write(
                kv_all,
                meta,
                knw,
                cos_sin,
                positions,
                locs,
                num_layers,
                kv_size,
                head_dim,
                eps,
                commit_lens=write_commit_lens,
                locs_row_width=locs_row_width,
            )
            return

        stacked = self._stacked_ctx_kv_params()
        if stacked is not None:
            k_all, v_all = self._project_ctx_kv_stacked(
                ctx_hidden=ctx_hidden, positions=positions, stacked=stacked
            )
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            if stacked is not None:
                k = k_all[i]
                v = v_all[i]
            else:
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(positions, k)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            if cache_loc_2d is not None and commit_lens is not None:
                pool.set_kv_buffer_prefix_valid(
                    attn.attn,
                    cache_loc_2d,
                    commit_lens,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )
            else:
                pool.set_kv_buffer(
                    attn.attn,
                    cache_loc,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )

    def _project_ctx_kv_stacked(
        self,
        *,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        stacked: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn0 = self.layers[0].self_attn
        num_layers = len(self.layers)
        kv_size = attn0.kv_size
        head_dim = attn0.head_dim
        num_kv_heads = attn0.num_kv_heads
        tokens = ctx_hidden.shape[0]

        kv_all = F.linear(ctx_hidden, stacked["weight"], stacked["bias"])
        kv_all = kv_all.view(tokens, num_layers, 2, kv_size)
        # Batched per-head k-norm across layers (fp32 variance + weight, cast back).
        k32 = (
            kv_all[:, :, 0, :]
            .reshape(tokens, num_layers, num_kv_heads, head_dim)
            .to(torch.float32)
        )
        variance = k32.pow(2).mean(dim=-1, keepdim=True)
        k32 = k32 * torch.rsqrt(variance + stacked["eps"])
        k32 = k32 * stacked["k_norm_weight"].view(1, num_layers, 1, head_dim)
        k_all = k32.to(ctx_hidden.dtype)
        # One RoPE over all layers' heads (shared rotary params + positions).
        k_flat = k_all.reshape(tokens, num_layers * kv_size)
        dummy_q = k_flat.new_empty(k_flat.shape)
        _, k_flat = attn0.rotary_emb(positions, dummy_q, k_flat)
        # [layers, tokens, heads, dim]: per-layer slices are contiguous views.
        k_all = (
            k_flat.view(tokens, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        v_all = (
            kv_all[:, :, 1, :]
            .view(tokens, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        return k_all, v_all


class DSparkDraftModel(DSparkDraftMixin, DFlashDraftModel):

    decoder_layer_cls = DSparkDecoderLayer

    def prune_to_ctx_kv_injection(self) -> None:
        self.markov_head = None
        self.confidence_head = None
        for layer in self.layers:
            layer.mlp = None
            layer.self_attn.o_proj = None
        torch.cuda.empty_cache()


class Qwen3DSparkModel(DSparkDraftModel):
    pass


EntryClass = [Qwen3DSparkModel, DSparkDraftModel]
