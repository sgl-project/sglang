"""SUFFIX speculative decoding under PD disaggregation.

SUFFIX is model-free: its draft source is a CPU suffix tree over each
request's prompt + generated tokens, both of which are already part of the
request metadata that rides the PD protocol. Unlike EAGLE, nothing extra
(topk_p / topk_index / hidden states) has to be captured on the prefill
instance or shipped through the KV-transfer metadata buffers.

The decode instance therefore only needs two things:

1. ``build_suffix_disagg_draft_input`` — reconstruct the first decode batch's
   cross-iteration spec_info. On the ngram-v2 stack that relay object is a
   ``NgramVerifyInput`` carrying the previous round's ``accept_tokens`` /
   ``accept_lens`` (which ``SuffixWorker._prepare_draft_tokens`` stages). On
   the very first decode step ``_missing_tail`` returns ``[]`` (the
   prefill-produced bonus token is already in ``req.output_ids``), so those
   values are never consumed; we still provide well-shaped tensors with the
   bonus at slot 0. The cross-step seq_lens relay goes through
   ``future_map.publish`` with SUFFIX's ``seq_lens + 1`` convention (the
   generated text is always ``seq_lens + 1`` tokens long; EAGLE publishes
   ``seq_lens`` as-is). ``FutureMap.stash`` is a no-op for ``is_ngram()`` —
   the NgramVerifyInput object itself is not relayed through the FutureMap
   (the worker rebuilds it each step), so nothing else has to be shipped.

2. Suffix-tree state, which is rebuilt decode-side from
   ``req.origin_input_ids`` — by the prewarm hook while the KV transfer is
   still in flight (see ``SuffixWorker.prewarm_disagg_requests``), or lazily
   at the first decode step for requests that outran the prewarm (the
   ``SuffixCacheAdapter`` builds the tree on first ``batch_get``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.speculative.ngram_info import NgramVerifyInput

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


def build_suffix_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> NgramVerifyInput:
    bs = int(last_tokens_tensor.numel())
    draft_token_num = int(server_args.speculative_num_draft_tokens)
    device = batch.device

    # SUFFIX publish convention: seq_lens + 1 (the pending bonus token is
    # counted into the relayed length).
    new_seq_lens = (batch.seq_lens + 1).to(torch.int64)

    # accept_tokens / accept_lens are the "previous round's accepts" that
    # _prepare_draft_tokens stages. Unused on the first decode step
    # (_missing_tail returns [] because the bonus token is already in
    # req.output_ids), but must be present and well-shaped: (bs * K,) flat
    # tokens with the bonus at each request's slot 0, and (bs,) lens of 1.
    accept_tokens = torch.zeros((bs * draft_token_num,), dtype=torch.int32, device=device)
    accept_tokens[::draft_token_num] = last_tokens_tensor.to(torch.int32)
    accept_lens = torch.ones((bs,), dtype=torch.int64, device=device)

    spec_info = NgramVerifyInput(
        draft_token_num=draft_token_num,
        new_seq_lens=new_seq_lens,
        accept_tokens=accept_tokens,
        accept_lens=accept_lens,
    )

    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        # Only the seq_lens relay matters for SUFFIX; FutureMap.stash is a
        # no-op for is_ngram() so the NgramVerifyInput is not shipped through
        # it (the worker rebuilds each step's verify input).
        future_map.publish(spec_info.future_indices, new_seq_lens)

    return spec_info
