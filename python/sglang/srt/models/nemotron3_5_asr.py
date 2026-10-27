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

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron3_5_asr/modeling_nemotron3_5_asr.py
"""NVIDIA Nemotron 3.5 ASR: FastConformer encoder with greedy RNN-T decoding.

RNN-T emits zero or more tokens per encoder frame, so one engine decode step
does not map to one model step. Each forward instead advances every request to
its next non-blank token and returns logits that allow only that token. The
per-request decoder state (LSTM state, frame pointer, encoder frames) lives in
`RNNTDecodeState`, indexed by the request's req_to_token_pool slot; the KV
cache is unused.

Prefill runs the encoder on the whole clip. Streaming (cache-aware chunked)
encoding is not supported.
"""

import functools
from array import array
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.models.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrPromptProjector,
    Nemotron3_5AsrRNNTDecoder,
    Nemotron3_5AsrRNNTJointNetwork,
)
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingEncoder,
)

from sglang.srt.configs.nemotron3_5_asr import Nemotron3_5AsrConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.multimodal.nemotron_asr_encoder_cuda_graph_runner import (
    NemotronAsrEncoderCudaGraphRunner,
)
from sglang.srt.utils import is_cuda

# Frames scored by the first joint call while skipping blanks; each further call
# doubles it, since rows still scanning are usually in a long silence. Measured
# on A6000 with 5-15 s clips; past the first token the extra frames are wasted.
_FRAMES_PER_SCAN = 8
_MAX_FRAMES_PER_SCAN = 256


class RNNTDecodeState:
    """Greedy RNN-T decoding state for every req_to_token_pool slot.

    A request's step is split in two so the engine's sampled token drives the
    LSTM: `predict` scans frames to the next non-blank token without consuming
    it, and `commit` feeds that token back on the following step.
    """

    def __init__(
        self,
        *,
        decoder: Nemotron3_5AsrRNNTDecoder,
        joint: Nemotron3_5AsrRNNTJointNetwork,
        blank_token_id: int,
        max_symbols_per_step: int,
    ):
        self._decoder = decoder
        self._joint = joint
        self._blank_token_id = blank_token_id
        self._max_symbols_per_step = max_symbols_per_step
        self._num_slots = 0
        # [num_lstm_layers, num_slots, hidden], like nn.LSTM's (h, c).
        self._hidden: Optional[torch.Tensor] = None
        self._cell: Optional[torch.Tensor] = None
        # Projected LSTM output for the last committed token: [num_slots, hidden].
        self._decoder_out: Optional[torch.Tensor] = None
        self._frame_idx: Optional[torch.Tensor] = None
        self._symbols_at_frame: Optional[torch.Tensor] = None
        self._num_frames: Optional[torch.Tensor] = None
        # Projected encoder frames per slot, [num_frames, hidden]; dropped once
        # the request reaches the end of its audio.
        self._frames: dict[int, torch.Tensor] = {}

    def start(self, *, slots: List[int], frames: List[torch.Tensor]) -> None:
        """Reset `slots` for new requests and seed their LSTMs with blank."""
        like = frames[0]
        self._ensure_capacity(max(slots) + 1, like=like)
        for slot, request_frames in zip(slots, frames):
            self._frames[slot] = request_frames
        index = torch.tensor(slots, device=like.device)
        self._num_frames[index] = torch.tensor(
            [f.shape[0] for f in frames], device=like.device
        )
        self._frame_idx[index] = 0
        self._symbols_at_frame[index] = 0
        self._hidden[:, index] = 0
        self._cell[:, index] = 0
        blank = torch.full((len(slots),), self._blank_token_id, device=like.device)
        self._run_lstm(slots=index, token_ids=blank, update=None)

    def replay(self, *, slot: int, token_ids: List[int]) -> None:
        """Re-consume tokens a request already emitted (retraction or chunking)."""
        device = self._frame_idx.device
        slots = torch.tensor([slot], device=device)
        for token_id in token_ids:
            self.predict(slots=slots, slots_cpu=[slot])
            self.commit(slots=slots, token_ids=torch.tensor([token_id], device=device))

    def commit(self, *, slots: torch.Tensor, token_ids: torch.Tensor) -> None:
        """Consume the token each request emitted on its previous step."""
        # Blank means the request already ended; its state stays put.
        emitted = token_ids != self._blank_token_id
        self._run_lstm(slots=slots, token_ids=token_ids, update=emitted)
        symbols = self._symbols_at_frame[slots] + emitted.long()
        force_advance = symbols >= self._max_symbols_per_step
        self._frame_idx[slots] = self._frame_idx[slots] + force_advance.long()
        self._symbols_at_frame[slots] = torch.where(
            force_advance, torch.zeros_like(symbols), symbols
        )

    def predict(self, *, slots: torch.Tensor, slots_cpu: List[int]) -> torch.Tensor:
        """Next non-blank token per slot, or blank once its audio is exhausted."""
        frame_idx = self._frame_idx[slots]
        num_frames = self._num_frames[slots]
        symbols = self._symbols_at_frame[slots]
        decoder_out = self._decoder_out[slots]
        next_ids = torch.full_like(frame_idx, self._blank_token_id)
        flat_frames, frame_offsets = self._flatten_frames(slots_cpu, like=decoder_out)
        # Rows still scanning for a token; shrinks as rows find one.
        rows = (frame_idx < num_frames).nonzero().squeeze(1)
        window = _FRAMES_PER_SCAN
        while rows.numel():
            window_ids, scanned, found = self._scan_window(
                window=window,
                decoder_out=decoder_out[rows],
                frame_idx=frame_idx[rows],
                num_frames=num_frames[rows],
                frames=flat_frames,
                frame_offsets=frame_offsets[rows],
            )
            next_ids[rows] = torch.where(found, window_ids, next_ids[rows])
            frame_idx[rows] = frame_idx[rows] + scanned
            symbols[rows] = torch.where(scanned > 0, 0, symbols[rows])
            rows = rows[~found & (frame_idx[rows] < num_frames[rows])]
            window = min(2 * window, _MAX_FRAMES_PER_SCAN)
        self._frame_idx[slots] = frame_idx
        self._symbols_at_frame[slots] = symbols
        self._release_finished(slots_cpu=slots_cpu, next_ids=next_ids)
        return next_ids

    def _scan_window(
        self,
        *,
        window: int,
        decoder_out: torch.Tensor,
        frame_idx: torch.Tensor,
        num_frames: torch.Tensor,
        frames: torch.Tensor,
        frame_offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint over the next frames of each row, at once.

        The decoder state is fixed while blanks are skipped, so the frames up
        to the first non-blank are independent. Returns that token, the number
        of blank frames before it, and whether one was found.
        """
        window_idx = frame_idx[:, None] + torch.arange(window, device=frame_idx.device)
        in_audio = window_idx < num_frames[:, None]
        frame_rows = (frame_offsets[:, None] + window_idx).clamp(0, frames.shape[0] - 1)
        logits = self._joint(
            decoder_hidden_states=decoder_out[:, None, :],
            encoder_hidden_states=frames[frame_rows],
        )
        token_ids = logits.argmax(dim=-1)
        is_token = (token_ids != self._blank_token_id) & in_audio
        found = is_token.any(dim=-1)
        # argmax returns the first maximal position, i.e. the first token.
        first = is_token.int().argmax(dim=-1)
        scanned = torch.where(found, first, in_audio.sum(dim=-1))
        window_ids = token_ids.gather(1, first[:, None]).squeeze(1)
        return window_ids, scanned, found

    def _run_lstm(
        self,
        *,
        slots: torch.Tensor,
        token_ids: torch.Tensor,
        update: Optional[torch.Tensor],
    ) -> None:
        hidden = self._hidden[:, slots]
        cell = self._cell[:, slots]
        embeddings = self._decoder.embedding(token_ids)[:, None, :]
        lstm_out, (new_hidden, new_cell) = self._decoder.lstm(
            embeddings, (hidden, cell)
        )
        decoder_out = self._decoder.decoder_projector(lstm_out[:, 0])
        if update is not None:
            keep = update[:, None]
            decoder_out = torch.where(keep, decoder_out, self._decoder_out[slots])
            new_hidden = torch.where(keep[None], new_hidden, hidden)
            new_cell = torch.where(keep[None], new_cell, cell)
        self._decoder_out[slots] = decoder_out
        self._hidden[:, slots] = new_hidden
        self._cell[:, slots] = new_cell

    def _flatten_frames(
        self, slots_cpu: List[int], *, like: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        empty = like.new_empty((0, like.shape[-1]))
        frames = [self._frames.get(slot, empty) for slot in slots_cpu]
        lengths = torch.tensor([0] + [f.shape[0] for f in frames[:-1]])
        offsets = lengths.cumsum(0).to(like.device)
        flat = torch.cat(frames) if frames else empty
        if flat.shape[0] == 0:
            flat = like.new_zeros((1, like.shape[-1]))
        return flat, offsets

    def _release_finished(
        self, *, slots_cpu: List[int], next_ids: torch.Tensor
    ) -> None:
        finished = (next_ids == self._blank_token_id).tolist()
        for slot, done in zip(slots_cpu, finished):
            if done:
                self._frames.pop(slot, None)

    def _ensure_capacity(self, num_slots: int, *, like: torch.Tensor) -> None:
        if num_slots <= self._num_slots:
            return
        # Grow geometrically; the pool size is not visible to the model.
        new_size = max(num_slots, 2 * self._num_slots, 64)
        lstm = self._decoder.lstm
        hidden_size = lstm.hidden_size
        state_shape = (lstm.num_layers, new_size, hidden_size)
        hidden = like.new_zeros(state_shape)
        cell = like.new_zeros(state_shape)
        decoder_out = like.new_zeros((new_size, hidden_size))
        counters = [
            torch.zeros(new_size, dtype=torch.long, device=like.device)
            for _ in range(3)
        ]
        if self._num_slots:
            old = self._num_slots
            hidden[:, :old] = self._hidden
            cell[:, :old] = self._cell
            decoder_out[:old] = self._decoder_out
            for new, cur in zip(
                counters, (self._frame_idx, self._symbols_at_frame, self._num_frames)
            ):
                new[:old] = cur
        self._hidden, self._cell, self._decoder_out = hidden, cell, decoder_out
        self._frame_idx, self._symbols_at_frame, self._num_frames = counters
        self._num_slots = new_size


class Nemotron3_5AsrForRNNT(nn.Module):
    def __init__(
        self,
        config: Nemotron3_5AsrConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        encoder_config = config.encoder_config
        # The relative position bias is a float attention mask, which SDPA
        # accepts and flash attention does not.
        encoder_config._attn_implementation = "sdpa"
        self.encoder = NemotronAsrStreamingEncoder(encoder_config)
        self.encoder_projector = nn.Linear(
            encoder_config.hidden_size, config.decoder_hidden_size
        )
        self.prompt_projector = Nemotron3_5AsrPromptProjector(config)
        self.decoder = Nemotron3_5AsrRNNTDecoder(config)
        self.joint = Nemotron3_5AsrRNNTJointNetwork(config)
        # Built on the first prefill; needs the processor's lookahead.
        self._encoder_graphs = None
        self.decode_state = RNNTDecodeState(
            decoder=self.decoder,
            joint=self.joint,
            blank_token_id=config.blank_token_id,
            max_symbols_per_step=config.max_symbols_per_step,
        )

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        # The audio never enters the token stream; the encoder runs in forward.
        return input_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        slots = forward_batch.req_pool_indices
        slots_cpu = slots.tolist()
        if forward_batch.forward_mode.is_decode():
            self.decode_state.commit(slots=slots, token_ids=input_ids)
        else:
            self._consume_extend(
                input_ids=input_ids, forward_batch=forward_batch, slots_cpu=slots_cpu
            )
        next_ids = self.decode_state.predict(slots=slots, slots_cpu=slots_cpu)
        return LogitsProcessorOutput(next_token_logits=self._forced_logits(next_ids))

    def _consume_extend(
        self,
        *,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        slots_cpu: List[int],
    ) -> None:
        mm_inputs = forward_batch.mm_inputs or [None] * len(slots_cpu)
        fresh = [
            i
            for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
            if prefix_len == 0
        ]
        if fresh:
            self.decode_state.start(
                slots=[slots_cpu[i] for i in fresh],
                frames=self._encode([mm_inputs[i] for i in fresh]),
            )

        token_ids = input_ids.tolist()
        start = 0
        for i, extend_len in enumerate(forward_batch.extend_seq_lens_cpu):
            extend_ids = token_ids[start : start + extend_len]
            start += extend_len
            if forward_batch.extend_prefix_lens_cpu[i] == 0:
                # The prompt is one placeholder token; the rest was emitted
                # before a retraction.
                extend_ids = extend_ids[1:]
            if extend_ids:
                self.decode_state.replay(slot=slots_cpu[i], token_ids=extend_ids)

    def _encode(
        self, mm_inputs: List[Optional[MultimodalInputs]]
    ) -> List[torch.Tensor]:
        """Projected encoder frames per request, each [num_frames, decoder_hidden].

        Requests are encoded together in padded batches: the eager encoder is
        bound by kernel launches, so per-request calls cost nearly a full call
        each.
        """
        dtype = self.encoder_projector.weight.dtype
        device = self.encoder_projector.weight.device
        # A request without audio (e.g. the server warmup) ends immediately.
        frames = [
            torch.empty(
                (0, self.config.decoder_hidden_size), dtype=dtype, device=device
            )
        ] * len(mm_inputs)
        audio = [
            (i, m.mm_items[0]) for i, m in enumerate(mm_inputs) if m and m.mm_items
        ]
        encode = self._padded_encoder(audio)
        batch_size = NemotronAsrEncoderCudaGraphRunner.max_batch_size
        for start in range(0, len(audio), batch_size):
            chunk = audio[start : start + batch_size]
            features = [
                item.feature.to(device=device, dtype=dtype).reshape(
                    -1, item.feature.shape[-1]
                )
                for _, item in chunk
            ]
            lengths = torch.tensor([f.shape[0] for f in features], device=device)
            padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
            mask = torch.arange(padded.shape[1], device=device) < lengths[:, None]
            prompt_ids = torch.cat([item.prompt_ids.reshape(1) for _, item in chunk])
            projected, valid = encode(
                padded, mask, prompt_ids.to(device=device, dtype=torch.long)
            )
            # Valid frames are a prefix; the clone detaches each request from
            # the graph's output buffer, which the next replay overwrites.
            for row, ((i, _), num_frames) in enumerate(
                zip(chunk, valid.sum(dim=1).tolist())
            ):
                frames[i] = projected[row, :num_frames].clone()
        return frames

    def _padded_encoder(self, audio: List[Tuple[int, MultimodalDataItem]]):
        if not audio:
            return None
        if self._encoder_graphs is None:
            # The processor's lookahead is a checkpoint constant, the same for
            # every request, so it is fixed once.
            num_lookahead_tokens = int(audio[0][1].num_lookahead_tokens)
            encode = functools.partial(
                self._encode_padded, num_lookahead_tokens=num_lookahead_tokens
            )
            self._encoder_graphs = (
                NemotronAsrEncoderCudaGraphRunner(encode) if is_cuda() else encode
            )
        return self._encoder_graphs

    def _encode_padded(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        *,
        num_lookahead_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out = self.encoder(
            input_features=features,
            attention_mask=attention_mask,
            num_lookahead_tokens=num_lookahead_tokens,
        )
        projected = self._project(encoder_out.last_hidden_state, prompt_ids=prompt_ids)
        return projected, encoder_out.attention_mask.bool()

    def _project(
        self, encoder_out: torch.Tensor, *, prompt_ids: torch.Tensor
    ) -> torch.Tensor:
        one_hot = nn.functional.one_hot(prompt_ids, num_classes=self.config.num_prompts)
        one_hot = one_hot.to(encoder_out.dtype)[:, None, :]
        one_hot = one_hot.expand(-1, encoder_out.shape[1], -1)
        fused = self.prompt_projector(torch.cat([encoder_out, one_hot], dim=-1))
        return self.encoder_projector(fused)

    def _forced_logits(self, next_ids: torch.Tensor) -> torch.Tensor:
        # Greedy RNN-T already picked the token; the sampler must keep it
        # whatever the sampling settings, so these are not probabilities.
        logits = torch.full(
            (next_ids.shape[0], self.config.vocab_size),
            float("-inf"),
            dtype=torch.float32,
            device=next_ids.device,
        )
        return logits.scatter_(1, next_ids[:, None], 0.0)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            default_weight_loader(params_dict[name], loaded_weight)


EntryClass = [Nemotron3_5AsrForRNNT]
