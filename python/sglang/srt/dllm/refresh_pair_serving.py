"""Opt-in refresh pairing for single-GPU SDAR FDFO serving.

The scheduler owns the completed block. The adapter borrows one scratch block
per request during a paired forward. It carries IDs, never scratch KV, forward.
"""

import json
import logging
from array import array
from contextlib import contextmanager
from copy import copy
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashinfer_block_pair_backend import (
    FlashInferBlockPairBackend,
    bind_owned_cache_view,
    pair_part,
    set_pair_token_extent,
    split_pair_batch,
)
from sglang.srt.utils.device_timer import device_timer_ctx
from sglang.srt.utils.nvtx_utils import profile_range
from sglang.srt.utils.profile_utils import build_step_span_name

logger = logging.getLogger(__name__)


class RefreshPairNumericalError(RuntimeError):
    """The paired forward failed its comparison with native graph execution."""


def eligible_request(req, block, mask_id):
    """Require a complete generated block and another full output block."""
    params = req.sampling_params
    if (
        params.stop
        or params.stop_token_ids
        or params.stop_regex
        or getattr(params, "stop_strs", None)
        or getattr(params, "stop_regex_strs", None)
    ):
        return False
    # These outputs or adapters need request-specific graph state.
    if (
        req.return_logprob
        or req.dllm_algo_state is not None
        or getattr(req, "return_hidden_states", False)
        or getattr(req, "lora_id", None) is not None
        or getattr(req, "input_embeds", None) is not None
    ):
        return False
    ids = req.dllm_incomplete_ids
    if getattr(req, "dllm_pair_applied_offset", None) == req.extend_end - block:
        ids = req.full_untruncated_fill_ids[req.extend_end - block : req.extend_end]
    if len(ids) != block or mask_id in ids:
        return False
    end = req.extend_end
    if not params.ignore_eos:
        stop_ids = set(getattr(req, "eos_token_ids", None) or ())
        tokenizer = getattr(req, "tokenizer", None)
        if tokenizer is not None:
            if tokenizer.eos_token_id is not None:
                stop_ids.add(tokenizer.eos_token_id)
            stop_ids.update(tokenizer.additional_stop_token_ids or ())
        # Only generated positions can finish a request; prompt EOS is retained context.
        generated_start = max(0, len(req.origin_input_ids) - (end - block))
        if not stop_ids or any(token in stop_ids for token in ids[generated_start:]):
            return False
    generated_after = end - len(req.origin_input_ids)
    return (
        generated_after > 0
        and generated_after + block <= params.max_new_tokens
        and end + block <= req.dllm_pair_context_limit
    )


def apply_prefetch(req):
    """Materialize predictions only after the previous block was committed."""
    pending = getattr(req, "dllm_pair_pending", None)
    if pending is None:
        return False
    offset, ids = pending
    current = len(req.origin_input_ids) + len(req.output_ids)
    if current > offset:
        raise RuntimeError("Refresh-pair predictions passed their block boundary")
    if current != offset:
        return False
    block = req.dllm_config.block_size
    if len(ids) != block or len(req.full_untruncated_fill_ids) != offset + block:
        raise RuntimeError("Refresh-pair prediction extent mismatch")
    req.full_untruncated_fill_ids[offset:] = array("q", ids)
    req.dllm_pair_applied_offset = offset
    return True


def discard_incompatible_prefetch(source, requests, block, mask_id, capture_sizes):
    """Restore the ordinary masked step when the numerical row context changed."""
    width = source.batch_size
    discarded = 0
    for index, req in enumerate(requests):
        offset = getattr(req, "dllm_pair_applied_offset", None)
        if offset is None or offset != req.extend_end - block:
            continue
        if (
            width in capture_sizes
            and getattr(req, "dllm_pair_future_width", None) == width
        ):
            continue
        if req.dllm_incomplete_ids:
            raise RuntimeError(
                "Cannot discard prefetch after ordinary denoising started"
            )
        source.input_ids.view(width, block)[index].fill_(mask_id)
        req.full_untruncated_fill_ids[offset : offset + block] = array(
            "q", [mask_id] * block
        )
        req.dllm_pair_pending = None
        req.dllm_pair_applied_offset = None
        req.dllm_pair_future_width = None
        discarded += 1
    return discarded


def finish_paired_step(algorithm, batch, output, states):
    """Retain native completion after the completed block's refresh."""
    done = algorithm.step(batch, output.full_logits, states)
    if done != [True]:
        raise RuntimeError("Refresh pairing requires one complete block")
    ids = batch.input_ids.view(1, algorithm.block_size).tolist()
    return output, ids, [algorithm.block_size], [None], True


class PairGraph:
    _REPLAY_INPUTS = (
        "input_ids",
        "positions",
        "out_cache_loc",
        "req_pool_indices",
        "seq_lens",
        "seq_lens_cpu",
        "extend_seq_lens",
        "extend_prefix_lens",
        "extend_start_loc",
    )

    def __init__(self, runner, source):
        self.runner = runner
        self.backend = FlashInferBlockPairBackend(runner)
        self.batch = copy(source)
        # Own all replay inputs. Do not retain mutable scheduler tensor storage.
        for name in self._REPLAY_INPUTS:
            setattr(self.batch, name, getattr(source, name).clone())
        if source.dllm_prompt_mask is not None:
            self.batch.dllm_prompt_mask = source.dllm_prompt_mask.clone()
        bind_owned_cache_view(self.batch, runner.kv_index_translator)
        self.backend.init_cuda_graph_state(
            source.batch_size,
            source.extend_num_tokens,
        )
        self.backend.init_forward_metadata_out_graph(self.batch, in_capture=True)
        for _ in range(3):
            self.model()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self.model()

    def model(self):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        with forward_context(ForwardContext(attn_backend=self.backend)):
            return self.runner.model(
                self.batch.input_ids, self.batch.positions, self.batch
            )

    def __call__(self, source):
        for name in self._REPLAY_INPUTS:
            getattr(self.batch, name).copy_(getattr(source, name))
        if self.batch.dllm_prompt_mask is not None:
            self.batch.dllm_prompt_mask.copy_(source.dllm_prompt_mask)
        for name in (
            "seq_lens_sum",
            "extend_seq_lens_cpu",
            "extend_prefix_lens_cpu",
            "extend_logprob_start_lens_cpu",
        ):
            setattr(self.batch, name, getattr(source, name))
        bind_owned_cache_view(self.batch, self.runner.kv_index_translator)
        self.backend.init_forward_metadata_out_graph(self.batch)
        self.graph.replay()
        return self.output


class RefreshPairServing:
    def __init__(self, runner, algorithm):
        if algorithm is None or not algorithm.fdfo:
            raise ValueError("Serving refresh pairing requires FDFO")
        self.runner, self.algorithm = runner, algorithm
        self.block = algorithm.block_size
        self.graph = None
        self.scratch = None
        self.disabled = False
        self.qualified = False
        self.native_graph = getattr(runner, "decode_cuda_graph_runner", None)

    def initialize(self):
        """Validate and capture before the server accepts requests."""
        from sglang.srt.debug_utils.dumper import dumper
        from sglang.srt.runtime_context import get_exec

        FlashInferBlockPairBackend.validate_config(self.runner)
        if not self.runner.spec_algorithm.is_none():
            raise ValueError("Refresh sharing does not support speculative decoding")
        execution = get_exec()
        if (
            execution.features.enable_memory_saver
            or execution.offload.cpu_offload_gb > 0
            or execution.offload.offload_group_size > 0
        ):
            raise ValueError("Refresh sharing requires resident model weights")
        if (
            self.runner.canary_manager is not None
            or self.runner.msprobe_debugger is not None
            or dumper.may_enable
            or execution.graph.debug_cuda_graph
            or self.runner.capture_head_hooks
            or self.runner.capture_tail_hooks
        ):
            raise ValueError(
                "Refresh sharing does not support model debug or capture hooks"
            )
        if 1 not in self.runner.decode_cuda_graph_capture_bs:
            raise ValueError("Refresh sharing requires a native N1 CUDA graph")
        self.capture()

    @torch.no_grad()
    def capture(self):
        """Capture before readiness, using the reserved padding request and KV slot."""
        from sglang.srt.layers.dp_attention import DpPaddingMode
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        block, device = self.block, self.runner.device
        tokens = 2 * block
        ids = torch.zeros(tokens, dtype=torch.int64, device=device)
        ids[block:] = self.algorithm.mask_id
        positions = torch.arange(tokens, dtype=torch.int64, device=device)
        # Row and slot zero are reserved for graph padding. No live request
        # mapping or allocator state is changed by these dummy forwards.
        batch = ForwardBatch(
            forward_mode=ForwardMode.DLLM_EXTEND,
            batch_size=1,
            input_ids=ids,
            positions=positions,
            req_pool_indices=torch.zeros(1, dtype=torch.int64, device=device),
            out_cache_loc=torch.zeros_like(positions),
            seq_lens=torch.full((1,), tokens, dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([tokens], dtype=torch.int64),
            seq_lens_sum=tokens,
            extend_num_tokens=tokens,
            extend_seq_lens=torch.full((1,), tokens, dtype=torch.int32, device=device),
            extend_seq_lens_cpu=[tokens],
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device=device),
            extend_prefix_lens_cpu=[0],
            extend_start_loc=torch.zeros(1, dtype=torch.int32, device=device),
            extend_logprob_start_lens_cpu=[0],
            dllm_prompt_mask=(positions < block).view(1, tokens),
            dllm_block_pair_size=block,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            dp_padding_mode=DpPaddingMode.SUM_LEN,
            spec_algorithm=self.runner.spec_algorithm,
        )
        set_pair_token_extent(batch)
        self.runner.prepare_dummy_forward_batch(batch)
        self.graph = PairGraph(self.runner, batch)

    @contextmanager
    def scratch_batch(self, source):
        if self.scratch is not None:
            raise RuntimeError("A refresh-pair scratch allocation is already active")
        try:
            yield self.paired_batch(source)
        finally:
            if self.scratch is not None:
                self.runner.token_to_kv_pool_allocator.free(self.scratch.reshape(-1))
                self.scratch = None

    def paired_batch(self, source):
        block, n = self.block, source.batch_size
        if self.scratch is None:
            # Scratch slots never enter scheduler request tables.
            self.scratch = self.runner.token_to_kv_pool_allocator.alloc(n * block)
            if self.scratch is None:
                return None
            self.scratch = self.scratch.to(dtype=torch.int64).reshape(n, block)
        b = copy(source)
        device = source.input_ids.device
        ids = source.input_ids.view(n, block)
        masks = torch.full_like(ids, self.algorithm.mask_id)
        b.input_ids = torch.cat((ids, masks), dim=1).reshape(-1)
        # Use one graph input layout for implicit and explicit prompt masks.
        prompt = (
            source.dllm_prompt_mask.view(n, block)
            if source.dllm_prompt_mask is not None
            else ids != self.algorithm.mask_id
        )
        b.dllm_prompt_mask = torch.cat((prompt, torch.zeros_like(prompt)), dim=1)
        b.out_cache_loc = (
            torch.cat((source.out_cache_loc.view(n, block), self.scratch[:n]), dim=1)
            .reshape(-1)
            .to(torch.int64)
        )
        positions = source.positions.view(n, block)
        b.positions = (
            torch.cat((positions, positions + block), dim=1).reshape(-1).to(torch.int64)
        )
        if b.out_cache_loc.dtype != b.positions.dtype:
            raise TypeError(
                "RoPE positions and KV write indices must have the same dtype"
            )
        b.seq_lens = source.seq_lens + block
        b.seq_lens_cpu = source.seq_lens_cpu + block
        b.seq_lens_sum = source.seq_lens_sum + n * block
        b.extend_seq_lens = torch.full_like(source.extend_seq_lens, 2 * block)
        b.extend_seq_lens_cpu = [2 * block] * n
        b.extend_start_loc = (
            torch.arange(n, device=device, dtype=torch.int32) * 2 * block
        )
        b.extend_num_tokens = n * 2 * block
        b.dllm_block_pair_size = block
        set_pair_token_extent(b)
        bind_owned_cache_view(b, self.runner.kv_index_translator)
        self.runner.prepare_dummy_forward_batch(b)
        return b

    def qualify(self, combined, pair_logits):
        """Compare against the production CUDA graph path for both stock forwards."""
        details = []
        for index, part in enumerate(split_pair_batch(combined, self.block)):
            part.dllm_block_pair_size = None
            # Use the same wrapper planning and graph replay as ordinary FDFO.
            stock = self.runner.forward(part, pp_proxy_tensors=None)
            if not stock.can_run_graph:
                raise RefreshPairNumericalError(
                    "Refresh-pair qualification requires the stock CUDA graph"
                )
            ref = stock.logits_output.full_logits
            candidate = pair_part(pair_logits, combined, self.block, index)
            # Chunk the vocabulary comparison to bound temporary FP32 storage.
            bad, maximum, finite = 0, 0.0, True
            for a, b in zip(candidate.split(16), ref.split(16)):
                a, b = a.float(), b.float()
                finite = finite and bool(
                    torch.isfinite(a).all() and torch.isfinite(b).all()
                )
                difference = (a - b).abs()
                maximum = max(maximum, difference.max().item())
                bad += (difference > 0.01 + 0.01 * b.abs()).sum().item()
            greedy_changes = (candidate.argmax(-1) != ref.argmax(-1)).sum().item()
            item = dict(
                block_index=index,
                finite=finite,
                out_of_tolerance=bad,
                max_abs_error=maximum,
                greedy_changes=greedy_changes,
                stock_cuda_graph=stock.can_run_graph,
            )
            details.append(item)
            if not finite or bad or greedy_changes:
                raise RefreshPairNumericalError(
                    f"Refresh-pair stock-graph qualification failed: {item}"
                )
            if index == 1:
                from sglang.srt.dllm.algorithm.low_confidence import LowConfidence

                left, right = copy(part), copy(part)
                left.input_ids, right.input_ids = (
                    part.input_ids.clone(),
                    part.input_ids.clone(),
                )
                left_done = LowConfidence.step(self.algorithm, left, candidate, None)
                right_done = LowConfidence.step(self.algorithm, right, ref, None)
                if left_done != right_done or not torch.equal(
                    left.input_ids, right.input_ids
                ):
                    raise RefreshPairNumericalError(
                        "Refresh-pair qualification changed confidence decisions"
                    )
        logger.info(
            "SDAR_REFRESH_PAIR numerical_gate_passed %s",
            json.dumps(details, sort_keys=True),
        )

    def run(self, source, requests):
        n, block = source.batch_size, self.block
        if len(requests) != n:
            raise ValueError("Request metadata does not match the forward batch")
        if (
            getattr(self.runner, "tp_size", 1) != 1
            or getattr(source, "attn_tp_sequence_sharded", False)
            or any(
                len(getattr(source, name, None) or []) > 1
                for name in ("original_global_num_tokens_cpu", "global_num_tokens_cpu")
            )
        ):
            return None
        if (
            not self.disabled
            and self.runner.decode_cuda_graph_runner is not self.native_graph
        ):
            self.disabled = True
            self.graph = None
            logger.info("Disable SDAR refresh sharing after native graph replacement")
        captures = self.runner.decode_cuda_graph_capture_bs
        supported = (1,) if not self.disabled and 1 in captures else ()
        # An N1 prediction must not enter a larger or padded native batch.
        discard_incompatible_prefetch(
            source, requests, block, self.algorithm.mask_id, supported
        )
        if self.disabled or n != 1 or not supported:
            return None
        req = requests[0]
        req.dllm_pair_context_limit = self.runner.model_config.context_len
        eligible = eligible_request(req, block, self.algorithm.mask_id)
        if getattr(req, "dllm_pair_applied_offset", None) == req.extend_end - block:
            req.dllm_pair_pending = None
            req.dllm_pair_applied_offset = None
            req.dllm_pair_future_width = None
        if not eligible:
            return None
        with self.scratch_batch(source) as combined:
            if combined is None:
                return None
            graph = self.graph
            self.runner.forward_pass_id += 1
            with (
                profile_range(
                    build_step_span_name(
                        combined,
                        is_draft_worker=False,
                        forward_id=self.runner.forward_pass_id,
                    )
                ),
                device_timer_ctx(
                    self.runner.device_timer, combined.forward_mode.name.lower()
                ),
            ):
                output = graph(combined)
            logits = output.full_logits
            if not self.qualified:
                try:
                    self.qualify(combined, logits)
                except RefreshPairNumericalError as error:
                    # Native qualification restores current-block KV before fallback.
                    self.disabled = True
                    self.graph = None
                    logger.warning("Disable SDAR refresh sharing: %s", error)
                    return None
                self.qualified = True
            next_ids = pair_part(combined.input_ids, combined, block, 1).clone()
            next_batch = SimpleNamespace(
                batch_size=1, input_ids=next_ids, dllm_prompt_mask=None
            )
            self.algorithm.step(next_batch, pair_part(logits, combined, block, 1), None)
            req.dllm_pair_pending = (req.extend_end, next_ids.tolist())
            req.dllm_pair_future_width = 1
            result = copy(output)
            result.full_logits = pair_part(logits, combined, block, 0)
            return result
