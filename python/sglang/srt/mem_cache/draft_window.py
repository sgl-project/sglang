"""Request-private paged draft KV; target slots and absolute positions stay intact."""

from dataclasses import dataclass

import numpy as np
import torch


def draft_window_size(config) -> int:
    """Require one native window on every draft layer, rather than a model name."""
    text = getattr(config, "hf_text_config", None)
    if text is None:
        text = getattr(config, "text_config", config)

    if isinstance(text, dict):
        text = text.get("text_config", text)

    def field(name, default=None):
        return (
            text.get(name, default)
            if isinstance(text, dict)
            else getattr(text, name, default)
        )

    # DFlash extras belong to the root HF config, not hf_text_config.
    root = getattr(config, "hf_config", config)
    extras = (
        root.get("dflash_config")
        if isinstance(root, dict)
        else getattr(root, "dflash_config", None)
    )
    if extras is None:
        extras = {}
    else:
        try:
            extras = dict(extras)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid DFLASH dflash_config") from exc

    layers = field("layer_types")
    count = field("num_hidden_layers")
    window = field("sliding_window")
    if (
        not layers
        or len(layers) != count
        or any(kind != "sliding_attention" for kind in layers)
        or not isinstance(window, int)
        or isinstance(window, bool)
        or window <= 0
        or field("use_sliding_window", True) is False
        or field("attention_sink_bias", False)
        or extras.get("attention_sink_bias", False) is not False
    ):
        raise ValueError(
            "Draft window KV needs uniform, pure sliding attention without sinks"
        )
    return window


def uses_draft_window_kv() -> bool:
    from sglang.srt.runtime_context import get_disagg, get_spec

    return (
        get_spec().speculative_draft_window_kv
        and get_disagg().disaggregation_mode == "decode"
    )


@dataclass(frozen=True)
class DraftWindowLayout:
    window: int
    page_size: int
    num_slots: int
    block_size: int

    def __post_init__(self):
        if min(self.window, self.page_size, self.block_size) <= 0 or self.num_slots < 2:
            raise ValueError("Invalid draft window geometry")

    @property
    def capacity(self):
        # The first query needs history aligned down to a page boundary. Keep
        # two blocks beyond it, matching the ordinary DFlash allocation reserve.
        rows = self.window + self.page_size - 1 + 2 * self.block_size
        return (rows + self.page_size - 1) // self.page_size * self.page_size

    @property
    def usable_tokens(self):
        return (self.num_slots - 1) * self.capacity

    @property
    def storage_tokens(self):
        # MHATokenToKVPool adds one physical padding page to size.
        return self.usable_tokens + self.page_size

    def locations(self, slots, positions):
        valid = slots != 0
        return torch.where(
            valid,
            self.page_size
            + (slots.to(torch.int64) - 1) * self.capacity
            + positions.to(torch.int64).remainder(self.capacity),
            0,
        )

    def prepare(self, slots, prefix_lens):
        """Build compact reads + candidate writes without changing RoPE positions."""
        start = (prefix_lens.to(torch.int64) - self.window).clamp_min(0)
        start = start // self.page_size * self.page_size
        offsets = torch.arange(self.capacity, device=slots.device)
        positions = start[:, None] + offsets
        lengths = prefix_lens - start
        live = (slots[:, None] != 0) & (offsets < lengths[:, None] + self.block_size)
        reads = torch.where(live, self.locations(slots[:, None], positions), 0)
        writes = self.locations(
            slots[:, None], prefix_lens[:, None] + offsets[: self.block_size]
        )
        return reads.to(torch.int32), lengths.to(torch.int32), writes

    def transfer_rows(self, slot, prefix_len):
        if not 0 < slot < self.num_slots or prefix_len < 0:
            raise ValueError("Invalid draft window transfer owner/length")
        start = max(0, prefix_len - self.window) // self.page_size * self.page_size
        positions = np.arange(start, prefix_len, dtype=np.int64)
        return (
            self.page_size + (slot - 1) * self.capacity + positions % self.capacity
        ).astype(np.int32)


def resolve_draft_window_layout(kvc):
    """Use the same resolved running/transfer slot counts as the request pool."""
    from sglang.srt.runtime_context import (
        get_disagg,
        get_schedule,
        max_speculative_num_draft_tokens,
    )

    config = kvc.spec_aux_config.draft_model_config
    if config is None:
        raise ValueError(
            "Draft window KV requires a resolved draft model configuration"
        )
    running = kvc.resolve_max_num_reqs(1 << 60)
    return DraftWindowLayout(
        draft_window_size(config),
        get_schedule().page_size,
        running + (get_disagg().disaggregation_decode_extra_slots or 0) + 1,
        max_speculative_num_draft_tokens(),
    )


def draft_window_transfer_rows(pool, req_to_token, slot, prefix_len):
    """The full aligned suffix, including when target KV is a complete hit."""
    layout = getattr(pool, "draft_window_layout", None)
    if layout is not None:
        return layout.transfer_rows(slot, prefix_len)
    start = (
        max(0, prefix_len - pool.draft_window_size) // pool.page_size * pool.page_size
    )
    return req_to_token[slot, start:prefix_len].to(torch.int32).cpu().numpy()
