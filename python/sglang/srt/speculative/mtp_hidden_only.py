"""Skip unused native-MTP logits only when the caller requests all hidden rows."""

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput


class HiddenOnlyLogitsProcessor(nn.Module):
    def __init__(self, original, enabled=False):
        super().__init__()
        self.original = original
        self.enabled = enabled

    def forward(
        self, input_ids, hidden_states, lm_head, forward_batch, *args, **kwargs
    ):
        if (
            self.enabled
            and not args
            and not kwargs
            and torch.is_tensor(hidden_states)
            and forward_batch.capture_hidden_mode.is_full()
        ):
            # Qwen3.5 native MTP supplies normalized hidden with no aux/pre-norm
            # override. Its normal FULL capture returns these exact tensor rows.
            return LogitsProcessorOutput(
                next_token_logits=None, hidden_states=hidden_states
            )
        return self.original(
            input_ids, hidden_states, lm_head, forward_batch, *args, **kwargs
        )
