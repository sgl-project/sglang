# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CUDA graphs for the Nemotron ASR audio encoder over padded shape buckets.

The eager encoder is bound by kernel launches: on an A6000 a 10 s clip costs
~70 ms of wall time for ~9 ms of GPU work. Each prefill instead replays a graph
captured for its (batch, mel frames) bucket, padded up; the padding is masked
out, so the frames match the eager encoder.
"""

from typing import Callable, Dict, Optional, Tuple

import torch

# encode(features [B, F, mel], attention_mask [B, F], prompt_ids [B])
#   -> (projected frames [B, F', hidden], valid frames [B, F'])
EncodeFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
]

_BATCH_BUCKETS = (1, 2, 4, 8, 16)
# Mel frames are 10 ms; 3072 covers the transcription adapter's 30 s clips.
_FRAME_BUCKET = 512
_MAX_FRAMES = 3072


class NemotronAsrEncoderCudaGraphRunner:
    """Replays a captured encoder graph per bucket, capturing on first use."""

    max_batch_size = _BATCH_BUCKETS[-1]

    def __init__(self, encode: EncodeFn):
        self._encode = encode
        self._graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._inputs: Dict[Tuple[int, int], Tuple[torch.Tensor, ...]] = {}
        self._outputs: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._pool: Optional[Tuple[int, int]] = None

    def __call__(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, num_frames = attention_mask.shape
        if batch > self.max_batch_size or num_frames > _MAX_FRAMES:
            return self._encode(features, attention_mask, prompt_ids)
        key = (
            next(b for b in _BATCH_BUCKETS if b >= batch),
            -(-num_frames // _FRAME_BUCKET) * _FRAME_BUCKET,
        )
        if key not in self._graphs:
            self._capture(key, like=features)
        static_features, static_mask, static_prompt_ids = self._inputs[key]
        static_features.zero_()
        static_features[:batch, :num_frames].copy_(features)
        static_mask.fill_(False)
        static_mask[:batch, :num_frames].copy_(attention_mask)
        # Padding rows attend over zeros; an all-masked row would give NaNs.
        static_mask[batch:] = True
        static_prompt_ids.zero_()
        static_prompt_ids[:batch].copy_(prompt_ids)
        self._graphs[key].replay()
        # All graphs share one memory pool; callers copy these out before the
        # next replay.
        projected, valid = self._outputs[key]
        return projected[:batch], valid[:batch]

    def _capture(self, key: Tuple[int, int], *, like: torch.Tensor) -> None:
        batch, num_frames = key
        features = like.new_zeros((batch, num_frames, like.shape[-1]))
        mask = torch.ones((batch, num_frames), dtype=torch.bool, device=like.device)
        prompt_ids = torch.zeros(batch, dtype=torch.long, device=like.device)
        # Warm up outside capture so lazy initialization is not recorded.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._encode(features, mask, prompt_ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=self._pool, capture_error_mode="thread_local"
        ):
            outputs = self._encode(features, mask, prompt_ids)
        self._pool = graph.pool()
        self._graphs[key] = graph
        self._inputs[key] = (features, mask, prompt_ids)
        self._outputs[key] = outputs
