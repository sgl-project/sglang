"""Graph the DFlash draft step's eager prep.

Per decode step the DFlash worker used to run, eagerly and host-latency bound:

  prepare_block (block ids / positions / verify cache locs)       1 launch
  noise embedding (vocab-parallel lookup + TP all-reduce)          3 launches
  compact window lens + draft req->token rebuild                   2 launches
  ForwardBatch + model_runner.forward + graph runner load_batch:
    buffer fill (small copy) + input-embeds copy + attn metadata   3 launches
  draft graph replay                                               1 launch

plus the generic ``model_runner.forward`` / ``load_batch`` Python. Every op
above reads only device tensors whose addresses are stable for a given batch
size (the worker's static draft-block buffers, the draft runner's static graph
buffers, the request->token tables), so the whole prep is captured once per
draft batch size into one CUDA graph fed by three static input rows (bonus
token, committed seq len, req pool index). A step becomes: one small copy into
the static inputs, the prep graph, the draft graph.

The draft runner's Python-side replay state (``raw_bs`` / ``bs`` /
``_replay_graph_key``, the attention backends' ``forward_metadata``) is
snapshotted at capture and re-installed per replay (same contract as
``runner/metadata_glue_graph.py``). Only batch sizes the draft runner captured
exactly (no padding) take the graph; everything else runs the eager path.

Correctness: identical kernels on identical inputs (bit-identical drafts); a
stale/incorrect draft could only lower acceptance, never change verified
tokens.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


class K3DFlashPrepGraph:
    def __init__(self, worker) -> None:
        self.w = worker
        self.runner = worker.draft_model_runner.decode_cuda_graph_runner
        self.states: Dict[int, dict] = {}
        self.max_bs = 0
        self._pool = None

    # ------------------------------------------------------------------ build
    @classmethod
    def maybe_build(cls, worker) -> Optional[K3DFlashPrepGraph]:
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils import is_hip

        reasons = []
        if not is_hip():
            reasons.append("not ROCm")
        if not worker.use_compact_draft_cache:
            reasons.append("no draft window (compact cache)")
        if not (
            worker._use_triton_prepare_block and worker._use_triton_compact_rebuild
        ):
            reasons.append("triton prep disabled")
        if worker._draft_sampler is None:
            reasons.append("no in-graph draft sampler")
        if (
            worker.selector is not None
            or worker.lilicorr is not None
            or worker._is_domino
        ):
            reasons.append("selector/lilicorr/domino draft")
        if get_parallel().attn_dp_enabled:
            reasons.append("dp attention")
        from sglang.srt.model_executor.forward_batch_info import (
            enable_num_token_non_padded,
        )

        if enable_num_token_non_padded():
            reasons.append("num_token_non_padded (EP / DP gather)")
        runner = getattr(worker.draft_model_runner, "decode_cuda_graph_runner", None)
        if runner is None:
            reasons.append("no draft decode graph runner")
        else:
            backend_name = type(getattr(runner, "backend", None)).__name__
            if backend_name not in (
                "FullCudaGraphBackend",
                "BreakableCudaGraphBackend",
            ):
                reasons.append(f"draft graph backend {backend_name}")
            if (
                getattr(runner, "ragged_verify_mode", False)
                or getattr(runner, "enable_two_batch_overlap", False)
                or getattr(runner, "enable_pdmux", False)
                or getattr(runner, "require_mlp_tp_gather", False)
                or getattr(runner, "dllm_uses_input_embeds", False)
                or worker.draft_model_runner.lora_manager is not None
            ):
                reasons.append("unsupported draft runner mode")
            if int(getattr(runner, "captured_req_width", -1)) != int(worker.block_size):
                reasons.append("captured_req_width != block_size")
        if reasons:
            logger.warning("K3 DFlash prep graph off: %s", "; ".join(reasons))
            return None
        pg = cls(worker)
        pg.capture_all()
        if not pg.states:
            return None
        return pg

    def _bucket_sizes(self):
        cap = 16
        return sorted({int(b) for b in self.runner.capture_bs if 0 < int(b) <= cap})

    def capture_all(self) -> None:
        from sglang.srt.distributed.parallel_state import graph_capture
        from sglang.srt.model_executor.runner_utils.pool import (
            get_or_create_global_graph_capture_stream,
        )
        from sglang.srt.runtime_context import get_parallel

        w = self.w
        bss = self._bucket_sizes()
        if not bss:
            return
        self.max_bs = max(bss)
        dev = w.device
        # Own references: the worker may later grow (reallocate) its buffers
        # for an eager batch size; the graphs keep pointing at these.
        w._ensure_draft_block_buffers(self.max_bs)
        self.block_ids_buf = w._draft_block_ids_buf
        self.positions_buf = w._draft_block_positions_buf
        self.cache_loc_buf = w._draft_verify_out_cache_loc_buf
        self.seq_lens_cpu_buf = torch.empty((self.max_bs,), dtype=torch.int32)
        self.in_bonus = torch.zeros((self.max_bs,), dtype=torch.int64, device=dev)
        self.in_seq = torch.ones((self.max_bs,), dtype=torch.int64, device=dev)
        self.in_rpi = torch.zeros((self.max_bs,), dtype=torch.int64, device=dev)
        self._pool = torch.cuda.graph_pool_handle()

        stream = get_or_create_global_graph_capture_stream()
        with graph_capture(stream=stream) as ctx:
            for bs in reversed(bss):
                for _ in range(2):
                    torch.cuda.synchronize()
                    get_parallel().tp_group.barrier()
                    self._body(bs)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self._pool, stream=ctx.stream):
                    self._body(bs)
                r = self.runner
                attn = r._replay_attn_backend()
                leaves = [attn] + list(getattr(attn, "attn_backend_list", None) or [])
                self.states[bs] = dict(
                    graph=g,
                    key=r._replay_graph_key,
                    raw_bs=r.raw_bs,
                    bs=r.bs,
                    raw_num_token=r.raw_num_token,
                    meta=[(b, getattr(b, "forward_metadata", None)) for b in leaves],
                )
        torch.cuda.synchronize()
        if get_parallel().tp_rank == 0:
            logger.info(
                "Captured draft prep graphs for bs=%s",
                sorted(self.states),
            )

    def _body(self, bs: int):
        """Draft prep for ``bs`` rows of the static inputs, then the draft
        runner's load_batch. Mirrors DFlashWorkerV2._draft_block_eager."""
        from sglang.kernels.ops.speculative.dflash import (
            _prepare_dflash_draft_block_unchecked,
        )
        from sglang.kernels.ops.speculative.k3_dflash_post import compact_draft_lens
        from sglang.srt.lora.layers import unwrap_lora_layer
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
            has_forward_context,
        )
        from sglang.srt.speculative.dflash_worker_v2 import (
            _resolve_dflash_embedding_module,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.speculative.spec_utils import draft_tp_context

        w = self.w
        block_size = int(w.block_size)
        block_ids = self.block_ids_buf[:bs]
        positions_2d = self.positions_buf[:bs]
        cache_loc_2d = self.cache_loc_buf[:bs]
        bonus = self.in_bonus[:bs]
        prefix_lens = self.in_seq[:bs]
        rpi = self.in_rpi[:bs]

        _prepare_dflash_draft_block_unchecked(
            bonus_tokens=bonus,
            prefix_lens=prefix_lens,
            req_pool_indices=rpi,
            req_to_token=w.model_runner.req_to_token_pool.req_to_token,
            block_ids_out=block_ids,
            positions_out=positions_2d,
            cache_loc_out=cache_loc_2d,
            mask_token_id=int(w._mask_token_id),
        )
        with draft_tp_context(w.draft_parallel_layout):
            if w._full_embed_gpu is not None:
                noise = torch.nn.functional.embedding(block_ids, w._full_embed_gpu)
            else:
                target_model = w.target_worker.model_runner.model
                embed_module = unwrap_lora_layer(
                    _resolve_dflash_embedding_module(w.draft_model, target_model)
                )
                noise = embed_module(block_ids)
        if w._noise_embed_scale != 1.0:
            noise = noise * w._noise_embed_scale
        input_embeds = noise.view(-1, noise.shape[-1])

        draft_prefix_lens, suffix_start = compact_draft_lens(
            prefix_lens,
            int(w.draft_window_size),
            w.page_size if w.page_size > 1 else 1,
        )
        w._rebuild_compact_draft_cache(
            req_pool_indices=rpi,
            prefix_lens=prefix_lens,
            draft_prefix_lens=draft_prefix_lens,
            verify_out_cache_loc_2d=cache_loc_2d,
            bs=bs,
            block_size=block_size,
            suffix_start=suffix_start,
        )
        # Host planning bound only (read on the host at capture; the graph
        # replay path never consults it).
        seq_lens_cpu = self.seq_lens_cpu_buf[:bs]
        seq_lens_cpu.fill_(
            int(w.draft_window_size) + (w.page_size if w.page_size > 1 else 0)
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            out_cache_loc_is_physical=True,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=rpi,
            seq_lens=draft_prefix_lens,
            out_cache_loc=cache_loc_2d.reshape(-1),
            seq_lens_sum=int(seq_lens_cpu.sum().item()),
            seq_lens_cpu=seq_lens_cpu,
            positions=positions_2d.reshape(-1),
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=w._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=None,
            global_num_token_non_padded_cpu=bs * block_size,
        )
        mr = w.draft_model_runner
        ctx = (
            contextlib.nullcontext()
            if has_forward_context()
            else forward_context(ForwardContext(attn_backend=mr.attn_backend))
        )
        with ctx, draft_tp_context(w.draft_parallel_layout):
            if not self.runner.can_run_graph(fb):
                raise RuntimeError(f"draft graph cannot run bs={bs}")
            self.runner.load_batch(fb)
        return fb

    # ----------------------------------------------------------------- replay
    def run(self, *, batch, draft_input, bs: int):
        st = self.states.get(bs)
        if st is None:
            return None
        from sglang.kernels.ops.memory.small_copy import try_small_copy

        w = self.w
        bonus = draft_input.bonus_tokens.reshape(-1)
        seq = batch.seq_lens.reshape(-1)
        rpi = batch.req_pool_indices.reshape(-1)
        dsts = [self.in_bonus[:bs], self.in_seq[:bs], self.in_rpi[:bs]]
        srcs = [bonus, seq, rpi]
        if not try_small_copy(dsts, srcs):
            for d, s in zip(dsts, srcs):
                d.copy_(s)
        st["graph"].replay()
        r = self.runner
        for b, md in st["meta"]:
            if md is not None:
                b.forward_metadata = md
        r.raw_bs = st["raw_bs"]
        r.bs = st["bs"]
        r.raw_num_token = st["raw_num_token"]
        r._replay_graph_key = st["key"]
        with r.backend.replay_session():
            r.backend.replay(st["key"], None)

        block_size = int(w.block_size)
        draft_next = w._draft_sampler.out[: bs * (block_size - 1)].view(
            bs, block_size - 1
        )
        block_ids = self.block_ids_buf[:bs]
        positions = self.positions_buf[:bs].reshape(-1)
        cache_2d = self.cache_loc_buf[:bs]
        out = (
            block_ids,
            positions,
            cache_2d.reshape(-1),
            cache_2d,
            draft_next,
            batch.seq_lens,
        )
        return out
