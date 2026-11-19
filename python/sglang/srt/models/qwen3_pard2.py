"""PARD-2 parallel draft model on a Qwen3 stack (https://arxiv.org/abs/2605.08632).

The draft proposes every draft token of a step in one forward. In the
target-dependent mode each row's input embedding adds
``pard2_scale * target_proj(target features of the previous position)``; the
speculative worker builds those embeddings and applies the draft's own
``lm_head``, so ``forward`` returns post-norm hidden states without logits.
"""

import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen3 import Qwen3ForCausalLM

logger = logging.getLogger(__name__)

# pard2_target_layers entry for the target's post-final-norm hidden state.
PARD2_FINAL_LAYER = -1


def pard2_capture_block_order(target_layers: List[int]) -> List[int]:
    """Config indices of ``target_layers`` in the order sglang delivers them.

    The target packs its captured layers in ascending depth and returns the
    post-norm final separately, so the final comes last.
    """
    aux = sorted(
        (i for i, layer in enumerate(target_layers) if layer != PARD2_FINAL_LAYER),
        key=lambda i: target_layers[i],
    )
    final = [i for i, layer in enumerate(target_layers) if layer == PARD2_FINAL_LAYER]
    return aux + final


def permute_target_proj_columns(
    weight: torch.Tensor, *, target_layers: List[int]
) -> torch.Tensor:
    """Reorder ``weight``'s per-layer column blocks from config order to capture order."""
    blocks = weight.chunk(len(target_layers), dim=1)
    order = pard2_capture_block_order(target_layers)
    return torch.cat([blocks[i] for i in order], dim=1).contiguous()


def load_pard2_target_proj(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


class Pard2Qwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.pard2_scale = float(config.pard2_scale)
        self.target_dependent = bool(config.pard2_target_dependent)
        self.target_layers = list(config.pard2_target_layers)
        self.num_target_aux_layers = sum(
            1 for layer in self.target_layers if layer != PARD2_FINAL_LAYER
        )
        self.uses_target_final = PARD2_FINAL_LAYER in self.target_layers
        self.target_proj: Optional[nn.Linear] = None
        if self.target_dependent:
            target_dim = int(config.pard2_target_dim)
            self.target_hidden_size = target_dim // len(self.target_layers)
            self.target_proj = nn.Linear(
                target_dim,
                config.hidden_size,
                bias=bool(getattr(config, "pard2_proj_bias", False)),
            )

    def project(
        self,
        aux_hidden_states: Optional[torch.Tensor],
        final_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """``pard2_scale * target_proj(features)`` for target rows.

        ``aux_hidden_states``: [num_tokens, num_aux * target_hidden], the captured
        layers packed in ascending depth. ``final_hidden_states``:
        [num_tokens, target_hidden], the post-norm final.
        """
        weight = self.target_proj.weight
        aux_width = self.num_target_aux_layers * self.target_hidden_size
        projected = None
        if self.num_target_aux_layers:
            projected = F.linear(aux_hidden_states, weight[:, :aux_width])
        if self.uses_target_final:
            final = F.linear(final_hidden_states, weight[:, aux_width:])
            projected = final if projected is None else projected + final
        if self.target_proj.bias is not None:
            projected = projected + self.target_proj.bias
        return projected * self.pard2_scale

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The checkpoint stores lm_head.weight although it is tied to embed_tokens.
        tied = self.config.tie_word_embeddings
        super().load_weights(
            (name, tensor)
            for name, tensor in weights
            if not (tied and name == "lm_head.weight")
        )
        if self.target_dependent:
            self._load_target_proj(self.config.pard2_warp_model_path)
        logger.info(
            "PARD2 draft: mode=%s, target_proj loaded=%s, dtype=%s",
            "target-dependent" if self.target_dependent else "target-independent",
            self.target_dependent,
            self.model.embed_tokens.weight.dtype,
        )

    def _load_target_proj(self, path: str) -> None:
        state = load_pard2_target_proj(path)
        weight = state["target_proj.weight"]
        expected = tuple(self.target_proj.weight.shape)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"PARD2 target_proj.weight in {path} has shape "
                f"{tuple(weight.shape)}, expected {expected}."
            )
        self.target_proj.weight.data.copy_(
            permute_target_proj_columns(weight, target_layers=self.target_layers)
        )
        if self.target_proj.bias is not None:
            self.target_proj.bias.data.copy_(state["target_proj.bias"])


EntryClass = Pard2Qwen3ForCausalLM
