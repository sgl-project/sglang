# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import time

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from sglang.srt.models.yue2.nar import synthesize as nar_synthesize
from sglang.srt.models.yue2.protocol import CODEC_OFFSET, CONTEXT, SongRequest


logger = init_logger(__name__)


class _NarBatchUnsupported(Exception):
    """Raised when a request mix cannot use the batched NAR path."""


class Yue2NARStage(PipelineStage):
    """Solve the NAR flow-matching ODE for each original acoustic chunk."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run_grouped_requests(self, batches, server_args):
        """Flow-match a whole group in one ragged NAR forward."""
        if len(batches) < 2:
            return [self(batch, server_args) for batch in batches]
        try:
            return self._forward_batched(batches)
        except _NarBatchUnsupported as exc:
            logger.info("Batched NAR not applicable (%s); running per request", exc)
        except Exception:
            logger.exception("Batched NAR failed; falling back to per request")
        return [self(batch, server_args) for batch in batches]

    def _forward_batched(self, batches):
        from sglang.srt.models.yue2.nar_fast import _sgl_kernel, synthesize_batched

        if os.environ.get("SGLANG_YUE2_FAST_NAR", "1") != "1":
            raise _NarBatchUnsupported("FAST_NAR disabled")
        if _sgl_kernel() is None:
            raise _NarBatchUnsupported("sgl_kernel unavailable")

        session = batches[0].extra.get("yue2_session")
        if session is None:
            raise _NarBatchUnsupported("no AR session")
        steps = int(getattr(batches[0].sampling_params, "ode_steps", 32))

        branches, prefix_lens, codecs, seeds = [], [], [], []
        for batch in batches:
            if batch.extra.get("yue2_session") is not session:
                raise _NarBatchUnsupported("rows do not share one AR session")
            if not batch.extra.get("yue2_nar_skip_feed") or batch.extra.get(
                    "yue2_batched_truncated"):
                raise _NarBatchUnsupported("row needs a tail feed or is truncated")
            if int(getattr(batch.sampling_params, "ode_steps", 32)) != steps:
                raise _NarBatchUnsupported("ode_steps differ")
            request: SongRequest = batch.extra["yue2_request"]
            branches.append(int(batch.extra.get("yue2_session_branch", 0)))
            prefix_lens.append(len(batch.extra["yue2_semantic_prefix"]))
            codecs.append([int(token) - CODEC_OFFSET
                           for token in batch.extra["yue2_semantic_ids"]])
            seeds.append(request.seed)

        started = time.perf_counter()
        try:
            latents = synthesize_batched(
                self.model, session, branches, prefix_lens, codecs, seeds, steps)
        finally:
            for batch in batches:
                if batch.extra.get("yue2_session") is session:
                    self._release_session(
                        batch, session, batch.extra.get("yue2_session_lease"))
        logger.info("Batched NAR: %d rows in %.2fs", len(batches),
                    time.perf_counter() - started)

        for batch, latent in zip(batches, latents):
            batch.extra.update(
                {
                    "yue2_latents": latent,
                    "yue2_nar_seconds": time.perf_counter() - started,
                    "yue2_nar_fast": True,
                }
            )
        return batches

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        model = self.model
        request: SongRequest = batch.extra["yue2_request"]

        # codec tokens sampled by AR
        codec = [
            int(token) - CODEC_OFFSET for token in batch.extra["yue2_semantic_ids"]
        ]

        # AR prefix
        prefix = list(batch.extra["yue2_semantic_prefix"])
        steps = int(getattr(batch.sampling_params, "ode_steps", 32))

        session = batch.extra.get("yue2_session")

        # counter
        lease = batch.extra.get("yue2_session_lease")

        started = time.perf_counter()
        latents, fast_path = None, False
        try:
            # The fast path borrows the AR session KV as the NAR prefix KV.
            if (
                session is not None
                and not batch.extra.get("yue2_batched_truncated", False)
                and os.environ.get("SGLANG_YUE2_FAST_NAR", "1") == "1"
            ):
                from sglang.srt.models.yue2.nar_fast import synthesize_from_session

                try:
                    latents = synthesize_from_session(
                        model,
                        session,
                        prefix_len=len(prefix),
                        codec=codec,
                        seed=request.seed,
                        fed_codec=int(batch.extra.get("yue2_fed_codec", 0)),
                        steps=steps,
                        context=CONTEXT,
                        branch=int(batch.extra.get("yue2_session_branch", 0)),
                        feed_tail=not batch.extra.get("yue2_nar_skip_feed", False),
                    )
                    fast_path = True
                except ValueError:
                    latents = None
            if latents is None:
                latents = nar_synthesize(
                    model,
                    prefix=prefix,
                    codec=codec,
                    seed=request.seed,
                    steps=steps,
                    context=CONTEXT,
                    attention="sdpa",
                    offload_ar=False,
                    query_chunk_size=None,
                )
        finally:
            # The AR stage borrowed the session; return it exactly once even if
            # synthesis raised.
            if session is not None:
                self._release_session(batch, session, lease)

        batch.extra.update(
            {
                "yue2_latents": latents,
                "yue2_nar_seconds": time.perf_counter() - started,
                "yue2_nar_fast": fast_path,
            }
        )
        return batch

    @staticmethod
    def _release_session(batch, session, lease) -> None:
        """Return a borrowed session to the right pool exactly once per request."""
        if lease is not None:
            lease.release()
        else:
            from sglang.srt.models.yue2.cuda_graph import default_session_pool

            default_session_pool.release(session)
        batch.extra.pop("yue2_session", None)
        batch.extra.pop("yue2_session_lease", None)
