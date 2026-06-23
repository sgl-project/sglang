from __future__ import annotations

import torch

from sglang.srt.speculative.dflash_utils import is_dflash_domino_projector


class DFlashDominoHelper:
    """Helper for Domino projector rollout state and GRU utilities.

    The draft model owns ``prefix_gru`` and ``embed_proj`` directly so checkpoint
    parameter names remain unchanged. This helper keeps Domino-specific rollout
    logic out of the base DFLASH model file.
    """

    def __init__(self, draft_model) -> None:
        self.draft_model = draft_model
        self._rollout_state_cache = None
        self._gru_input_table_cache = None

    def _check_projector(self) -> None:
        if not is_dflash_domino_projector(
            getattr(self.draft_model, "projector_type", None)
        ):
            raise RuntimeError("Domino helper called on a non-Domino draft model.")
        if not hasattr(self.draft_model, "prefix_gru") or not hasattr(
            self.draft_model, "embed_proj"
        ):
            raise RuntimeError("Domino draft model is missing prefix_gru/embed_proj.")

    def init_gru_hidden(self, prefix_embeds: torch.Tensor) -> torch.Tensor:
        """Run the Domino prefix GRU over ``prefix_embeds``."""

        self._check_projector()
        prefix_embeds = prefix_embeds.contiguous()
        gru = self.draft_model.prefix_gru
        hidden_size = int(gru.hidden_size)
        h = torch.zeros(
            (prefix_embeds.shape[0], hidden_size),
            dtype=prefix_embeds.dtype,
            device=prefix_embeds.device,
        )
        w_ih = gru.weight_ih_l0.to(dtype=prefix_embeds.dtype)
        w_hh = gru.weight_hh_l0.to(dtype=prefix_embeds.dtype)
        b_ih = gru.bias_ih_l0.to(dtype=prefix_embeds.dtype) if gru.bias else None
        b_hh = gru.bias_hh_l0.to(dtype=prefix_embeds.dtype) if gru.bias else None

        for step in range(prefix_embeds.shape[1]):
            gi = torch.nn.functional.linear(prefix_embeds[:, step, :], w_ih, b_ih)
            gh = torch.nn.functional.linear(h, w_hh, b_hh)
            r = torch.sigmoid(gi[:, :hidden_size] + gh[:, :hidden_size])
            z = torch.sigmoid(
                gi[:, hidden_size : 2 * hidden_size]
                + gh[:, hidden_size : 2 * hidden_size]
            )
            n = torch.tanh(gi[:, 2 * hidden_size :] + r * gh[:, 2 * hidden_size :])
            h = (1.0 - z) * n + z * h
        return h

    def get_rollout_state(self) -> dict:
        """Return cached weight views used by the optimized Domino rollout loop."""

        if self._rollout_state_cache is not None:
            return self._rollout_state_cache

        self._check_projector()
        fc1 = self.draft_model.embed_proj[0]
        fc2 = self.draft_model.embed_proj[2]
        hidden_dim = int(self.draft_model.config.hidden_size)

        w_z = fc1.weight[:, :hidden_dim].detach()
        w_s = fc1.weight[:, hidden_dim:].detach()
        b1 = fc1.bias.detach() if fc1.bias is not None else None

        gru = self.draft_model.prefix_gru
        w_ih = gru.weight_ih_l0.detach()
        w_hh = gru.weight_hh_l0.detach()
        b_ih = gru.bias_ih_l0.detach() if gru.bias else None
        b_hh = gru.bias_hh_l0.detach() if gru.bias else None

        self._rollout_state_cache = {
            "w_z": w_z,
            "w_s": w_s,
            "w_s_hh_T": torch.cat([w_s.T, w_hh.T], dim=1).contiguous(),
            "b1": b1,
            "fc2_weight": fc2.weight.detach(),
            "fc2_bias": fc2.bias.detach() if fc2.bias is not None else None,
            "w_ih": w_ih,
            "w_hh": w_hh,
            "b_ih": b_ih,
            "b_hh": b_hh,
            "gru_hidden_size": int(gru.hidden_size),
        }
        return self._rollout_state_cache

    def get_gru_input_proj_table(self, embed_weight: torch.Tensor) -> torch.Tensor:
        """Return [vocab, 3*gru_hidden] table: embed_weight @ W_ih.T + b_ih."""

        cached = self._gru_input_table_cache
        if cached is not None and cached["weight_ptr"] == embed_weight.data_ptr():
            return cached["table"]

        state = self.get_rollout_state()
        table = torch.nn.functional.linear(
            embed_weight, state["w_ih"], state["b_ih"]
        ).contiguous()
        self._gru_input_table_cache = {
            "weight_ptr": embed_weight.data_ptr(),
            "table": table,
        }
        return table
