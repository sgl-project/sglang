# SPDX-License-Identifier: Apache-2.0
"""Return Valen candidate logits through SGLang's prefill-only pooling path."""

import torch
from torch import nn

from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.valen_layout import parse_valen_layout
from sglang.srt.layers.valen_mixer import ValenMixerHead


class ValenMixerPooler(nn.Module):
    def __init__(self, head: ValenMixerHead):
        super().__init__()
        self.head = head

    def forward(
        self, hidden_states: torch.Tensor, forward_batch
    ) -> EmbeddingPoolerOutput:
        lengths = forward_batch.extend_seq_lens_cpu
        outputs = [
            hidden_states.new_empty(0, dtype=torch.float32)
            for _ in range(forward_batch.batch_size)
        ]
        if lengths is None or forward_batch.valen_layouts is None:
            return EmbeddingPoolerOutput(embeddings=outputs)
        lengths = list(lengths)
        prefixes = forward_batch.extend_prefix_lens_cpu or [0] * len(lengths)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        runs, layouts = [], {}
        for i, (raw, prefix, length) in enumerate(
            zip(forward_batch.valen_layouts, prefixes, lengths)
        ):
            if raw is None or prefix != 0:
                continue
            try:
                layouts[i] = parse_valen_layout(raw, length)
            except ValueError:
                # A malformed/truncated request must not kill its batch neighbors.
                continue
            if runs and runs[-1][-1] == i - 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        for run in runs:
            scores = self.head.forward_batch(
                hidden_states[offsets[run[0]] : offsets[run[-1] + 1]],
                [lengths[i] for i in run],
                [layouts[i] for i in run],
            )
            for i, values in zip(run, scores):
                outputs[i] = values
        return EmbeddingPoolerOutput(embeddings=outputs)
