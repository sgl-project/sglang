# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Direct-encode text embeddings: {"positive":
#   {"hidden_states": Tensor[text_len, 5120] bf16 cpu, "text_len": int}}
MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY = "minimax_h3_text_embeddings"
# Direct keyframe encode: {"rows": Tensor[n_rows, 96] fp32 cpu,
#   "latent_h": int, "latent_w": int, "canvas_height": int,
#   "canvas_width": int, "keyframes": [...],
#   "semantic_frame_indices": [...], "pixel_frame_indices": [...]}
MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY = "minimax_h3_keyframe_cond_rows"
# Direct sigma schedules: {"video": [float], "audio": [float]}
MINIMAX_H3_SIGMAS_EXTRA_KEY = "minimax_h3_sigmas"
# Direct denoise state: {"initial_video_rows", "initial_audio_rows",
#   "latent_t", "latent_h", "latent_w", "audio_t"}
MINIMAX_H3_DENOISE_STATE_EXTRA_KEY = "minimax_h3_denoise_state"
# ref2va direct reference encodes.
MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY = "minimax_h3_reference_image_rows"
MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY = "minimax_h3_reference_audio_rows"
MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY = "minimax_h3_reference_video_rows"
MINIMAX_H3_PREPARED_REFERENCE_VIDEO_EXTRA_KEY = "minimax_h3_prepared_reference_video"

MINIMAX_H3_SUPPORTED_FPS = 24
MINIMAX_H3_MIN_DURATION_SECONDS = 4.0
MINIMAX_H3_MAX_DURATION_SECONDS = 15.0

# The distilled checkpoint has exactly one positive denoise branch.
MINIMAX_H3_DEFAULT_BRANCHES: tuple = ({"name": "cond_1"},)

# Audited 4xH200 T2VA Cache-DiT parameters for quality="high":
# (warmup steps, residual-difference threshold, max consecutive cached steps).
# Measured SSIM 0.931 / PSNR 28.16 dB against quality="lossless" on the
# validated workload. quality="lossless" (the default) uses no Cache-DiT
# configuration at all. Process-wide SGLANG_CACHE_DIT_* environment controls
# remain available for manual experiments and are independent of this field.
MINIMAX_H3_HIGH_QUALITY_CACHE_DIT_CONFIG: tuple[int, float, int] = (4, 0.04, 1)
