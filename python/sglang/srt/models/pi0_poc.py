# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""A minimal pi0-style continuous-action POC for SGLang.

This model is intentionally tiny: it proves the serving contract for a VLA
action head without depending on OpenPI weights. A real pi0 port should replace
``_sample_actions`` with PaliGemma prefix encoding plus the action expert flow
matching loop.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class Pi0ForActionPrediction(nn.Module):
    """Toy pi0-style action predictor.

    Request-side robot state is passed through ``sampling_params.custom_params``:

    - ``state`` or ``proprio_state``: 1-D numeric vector
    - ``num_inference_steps`` / ``num_steps``: denoising loop length
    - ``seed`` / ``action_seed``: deterministic action seed

    The model emits one dummy token and attaches a continuous ``actions`` chunk
    to ``LogitsProcessorOutput.customized_info``.
    """

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        del quant_config, prefix

        self.config = config
        self.vocab_size = int(getattr(config, "vocab_size", 8))
        self.hidden_size = int(getattr(config, "hidden_size", 16))
        self.action_dim = int(
            getattr(config, "action_dim", getattr(config, "pi0_action_dim", 14))
        )
        self.action_horizon = int(
            getattr(
                config, "action_horizon", getattr(config, "pi0_action_horizon", 50)
            )
        )
        self.default_num_inference_steps = int(
            getattr(
                config,
                "num_inference_steps",
                getattr(config, "pi0_num_inference_steps", 10),
            )
        )
        dummy_token_id = getattr(
            config, "pi0_dummy_token_id", getattr(config, "eos_token_id", 0)
        )
        self.dummy_token_id = max(0, min(int(dummy_token_id or 0), self.vocab_size - 1))

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        del positions, input_embeds, kwargs

        batch_size = self._infer_batch_size(input_ids, forward_batch)
        device = input_ids.device
        token_slices = self._request_token_slices(input_ids, forward_batch, batch_size)
        custom_params = self._custom_params(forward_batch, batch_size)

        actions: List[List[List[float]]] = []
        num_steps_for_request: List[int] = []
        for i in range(batch_size):
            params = custom_params[i]
            action, num_steps = self._sample_actions(
                token_slices[i],
                params if isinstance(params, dict) else {},
                request_index=i,
                device=device,
            )
            actions.append(action.detach().cpu().tolist())
            num_steps_for_request.append(num_steps)

        logits = torch.full(
            (batch_size, self.vocab_size),
            torch.finfo(torch.float32).min,
            device=device,
            dtype=torch.float32,
        )
        logits[:, self.dummy_token_id] = 0.0

        return LogitsProcessorOutput(
            next_token_logits=logits,
            customized_info={
                "actions": actions,
                "action_horizon": [self.action_horizon] * batch_size,
                "action_dim": [self.action_dim] * batch_size,
                "num_inference_steps": num_steps_for_request,
            },
        )

    def _sample_actions(
        self,
        tokens: torch.Tensor,
        params: Dict[str, Any],
        request_index: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        num_steps_value = params.get("num_inference_steps", params.get("num_steps"))
        if num_steps_value is None:
            num_steps_value = self.default_num_inference_steps
        num_steps = max(1, int(num_steps_value))
        seed_value = params.get("action_seed", params.get("seed"))
        if seed_value is None:
            seed_value = 0
        seed = int(seed_value)
        token_sum = int(tokens.to(torch.int64).sum().item()) if tokens.numel() else 0
        state = self._state_tensor(params, device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed((seed + token_sum + request_index * 9973) % (2**63))
        action = torch.randn(
            (self.action_horizon, self.action_dim),
            generator=generator,
            dtype=torch.float32,
        ).to(device)

        time_axis = torch.linspace(0.0, 1.0, self.action_horizon, device=device)[
            :, None
        ]
        dim_axis = torch.linspace(-1.0, 1.0, self.action_dim, device=device)[None, :]
        prompt_bias = 0.001 * float(token_sum % 1009)
        target = torch.tanh(0.1 * state[None, :] + time_axis + 0.05 * dim_axis)
        target = target + prompt_bias

        dt = 1.0 / float(num_steps)
        for _ in range(num_steps):
            action = action + dt * (target - action)
        return action, num_steps

    def _state_tensor(self, params: Dict[str, Any], device: torch.device) -> torch.Tensor:
        state_value = params.get("state", params.get("proprio_state", None))
        if state_value is None:
            return torch.zeros(self.action_dim, dtype=torch.float32, device=device)

        if isinstance(state_value, torch.Tensor):
            state = state_value.to(device=device, dtype=torch.float32).reshape(-1)
        else:
            state = torch.tensor(state_value, dtype=torch.float32, device=device).reshape(
                -1
            )

        if state.numel() == self.action_dim:
            return state
        if state.numel() > self.action_dim:
            return state[: self.action_dim]
        return F.pad(state, (0, self.action_dim - state.numel()))

    @staticmethod
    def _custom_params(
        forward_batch: ForwardBatch, batch_size: int
    ) -> List[Optional[Dict[str, Any]]]:
        sampling_info = getattr(forward_batch, "sampling_info", None)
        custom_params = getattr(sampling_info, "custom_params", None)
        if custom_params is None:
            return [None] * batch_size
        custom_params = list(custom_params)
        if len(custom_params) < batch_size:
            custom_params.extend([None] * (batch_size - len(custom_params)))
        return custom_params[:batch_size]

    @staticmethod
    def _infer_batch_size(input_ids: torch.Tensor, forward_batch: ForwardBatch) -> int:
        batch_size = getattr(forward_batch, "batch_size", None)
        if isinstance(batch_size, int) and batch_size > 0:
            return batch_size

        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None:
            return len(seq_lens)

        return max(1, int(input_ids.numel()))

    @staticmethod
    def _request_token_slices(
        input_ids: torch.Tensor, forward_batch: ForwardBatch, batch_size: int
    ) -> List[torch.Tensor]:
        if input_ids.dim() == 2:
            return [input_ids[i].reshape(-1) for i in range(batch_size)]

        flat_ids = input_ids.reshape(-1)
        extend_start_loc = getattr(forward_batch, "extend_start_loc", None)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_start_loc is not None and extend_seq_lens is not None:
            slices = []
            for i in range(batch_size):
                start = int(extend_start_loc[i])
                length = int(extend_seq_lens[i])
                slices.append(flat_ids[start : start + length])
            return slices

        if flat_ids.numel() == batch_size:
            return [flat_ids[i : i + 1] for i in range(batch_size)]

        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None and sum(int(x) for x in seq_lens) <= flat_ids.numel():
            slices = []
            offset = 0
            for length in seq_lens:
                length = int(length)
                slices.append(flat_ids[offset : offset + length])
                offset += length
            return slices

        chunks = torch.tensor_split(flat_ids, batch_size)
        return [chunk.reshape(-1) for chunk in chunks]

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            param = params.get(name)
            if param is None or tuple(param.shape) != tuple(weight.shape):
                continue
            param.data.copy_(weight)
            loaded.add(name)
        return loaded


EntryClass = Pi0ForActionPrediction
