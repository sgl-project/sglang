# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 duration head.

Predicts the shot length a caption implies from the text connector outputs.
Used only when the caller omits `num_frames`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.adapter.ltx_2_duration_head import (
    LTX2DurationHeadConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LTX2DurationAttentionPooler(nn.Module):
    """Cross-attends `num_queries` learnable tokens against the caption tokens.

    Produces a fixed `(batch, num_queries, hidden_dim)` output regardless of
    input length. No attention mask: the connectors already replaced padded
    positions with learnable registers and marked everything attendable.
    """

    def __init__(
        self, hidden_dim: int = 256, num_queries: int = 1, num_heads: int = 4
    ) -> None:
        super().__init__()
        self.heads = num_heads
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        queries = self.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)

        query = self.to_q(queries).unflatten(2, (self.heads, -1)).transpose(1, 2)
        key = self.to_k(tokens).unflatten(2, (self.heads, -1)).transpose(1, 2)
        value = self.to_v(tokens).unflatten(2, (self.heads, -1)).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        return self.to_out(hidden_states)


class LTX2DurationHead(nn.Module):
    """Modality-agnostic duration regressor over the connector outputs.

    Per-modality input projections map each stream into a shared pooler width,
    learnable modality embeddings tag the streams, and a small MLP turns the
    pooled vector into a log-duration. The target is trained in log-seconds, so
    `forward` exponentiates and callers always get seconds.
    """

    def __init__(self, config: LTX2DurationHeadConfig) -> None:
        super().__init__()
        arch = config.arch_config
        pooler_hidden_dim = arch.pooler_hidden_dim

        self.video_input_proj = nn.Linear(
            arch.video_cross_attention_dim, pooler_hidden_dim
        )
        self.video_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)

        self.audio_input_proj = nn.Linear(
            arch.audio_cross_attention_dim, pooler_hidden_dim
        )
        self.audio_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)

        self.attention_pooler = LTX2DurationAttentionPooler(
            hidden_dim=pooler_hidden_dim,
            num_queries=arch.num_queries,
            num_heads=arch.num_pooler_heads,
        )
        self.mlp_hidden = nn.Linear(
            pooler_hidden_dim * arch.num_queries, arch.mlp_hidden_dim
        )
        self.mlp_out = nn.Linear(arch.mlp_hidden_dim, 1)

    def forward(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns predicted duration in seconds, shape `(batch,)`."""
        if video_tokens is None and audio_tokens is None:
            raise ValueError(
                "LTX2DurationHead requires at least one of video_tokens / audio_tokens."
            )

        # The connector output can arrive in a different dtype than the head.
        head_dtype = self.mlp_out.weight.dtype

        token_groups = []
        if video_tokens is not None:
            token_groups.append(
                self.video_input_proj(video_tokens.to(head_dtype))
                + self.video_modality_emb
            )
        if audio_tokens is not None:
            token_groups.append(
                self.audio_input_proj(audio_tokens.to(head_dtype))
                + self.audio_modality_emb
            )

        tokens = torch.cat(token_groups, dim=1)
        pooled = self.attention_pooler(tokens).flatten(1)

        # tanh-approximated GELU matches the JAX-trained head; exact GELU does not.
        hidden_states = F.gelu(self.mlp_hidden(pooled), approximate="tanh")
        log_duration = self.mlp_out(hidden_states).squeeze(-1)
        return log_duration.exp()

    def predict_num_frames(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
        *,
        frame_rate: float,
        temporal_compression_ratio: int,
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
    ) -> int:
        """Predict a frame count on the VAE's causal temporal grid.

        Clamp first, then snap: a clamped count is not necessarily grid-aligned,
        so snapping first would give a different answer.
        """
        predicted_seconds = self(video_tokens, audio_tokens)
        if predicted_seconds.numel() != 1:
            raise ValueError(
                "predict_num_frames supports a single prediction only, got shape "
                f"{tuple(predicted_seconds.shape)}. One frame count cannot serve "
                "prompts with different natural durations."
            )
        seconds = predicted_seconds.item()

        # Floor at 1 so the grid arithmetic cannot go negative.
        min_frames = max(1, round(min_seconds * frame_rate))
        max_frames = round(max_seconds * frame_rate)
        clamped_frames = max(min_frames, min(round(seconds * frame_rate), max_frames))

        num_frames = (
            (clamped_frames - 1) // temporal_compression_ratio
        ) * temporal_compression_ratio + 1

        if num_frames < min_frames:
            # Flooring undershot the lower bound; take the next grid point up.
            snapped_up = num_frames + temporal_compression_ratio
            if snapped_up <= max_frames:
                num_frames = snapped_up
            else:
                # No grid point fits the bounds; overshooting by under a step
                # beats refusing to generate.
                if abs(snapped_up - clamped_frames) < abs(num_frames - clamped_frames):
                    num_frames = snapped_up
                logger.warning(
                    "Duration bounds [%.2fs, %.2fs] at %.2f fps admit no frame count "
                    "on the VAE temporal grid (k * %d + 1); using nearest: %d frames",
                    min_seconds,
                    max_seconds,
                    frame_rate,
                    temporal_compression_ratio,
                    num_frames,
                )

        if seconds < min_seconds or seconds > max_seconds:
            logger.warning(
                "Duration prediction clamped: raw %.2fs outside [%.2fs, %.2fs], "
                "using %.2fs (%d frames) @ %.2f fps",
                seconds,
                min_seconds,
                max_seconds,
                num_frames / frame_rate,
                num_frames,
                frame_rate,
            )
        else:
            logger.info(
                "Predicted duration %.2fs (%d frames @ %.2f fps)",
                seconds,
                num_frames,
                frame_rate,
            )
        return num_frames


EntryClass = LTX2DurationHead
