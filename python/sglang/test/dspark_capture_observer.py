"""Test-only target output snapshots before serving mutations or slot reuse."""

import hashlib
import itertools
from pathlib import Path

import torch

from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.runtime_context import get_spec
from sglang.srt.training_capture.coordinator import CaptureCoordinator

_sequence = itertools.count()
_token_prefixes = {}
_verify_frames = {}
_after_forward = CaptureCoordinator.after_forward
_after_verify_forward = CaptureCoordinator.after_verify_forward
_after_verify_accept = CaptureCoordinator.after_verify_accept


def observe(
    coordinator,
    batch,
    forward_batch,
    logits_output,
    width=None,
    can_run_cuda_graph=False,
    verify_lens=None,
):
    draft_path = get_spec().speculative_draft_model_path
    root = (
        Path(draft_path)
        if draft_path
        else Path(coordinator.config.journal_directory).parent
    ) / "capture-reference"
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    pp_rank = get_pipeline_model_parallel_rank()
    if get_pipeline_model_parallel_world_size() > 1:
        root = root / f"pp{pp_rank}"
    if tp_size > 1:
        root = root / f"tp{tp_rank}"
    root.mkdir(parents=True, exist_ok=True)
    if width is not None:
        _verify_frames[id(coordinator)] = []
    offset = 0
    lengths = verify_lens.cpu().tolist() if verify_lens is not None else None
    for row, req in enumerate(batch.reqs):
        count = (lengths[row] if lengths is not None else width) or (
            forward_batch.extend_seq_lens_cpu[row]
            if batch.forward_mode.is_extend()
            else 1
        )
        region = slice(offset, offset + count)
        offset += count
        if req.training_capture_context is None:
            continue
        end = int(batch.seq_lens_cpu[row])
        tokens = list(req.origin_input_ids) + list(req.output_ids)
        if width is None:
            start = 0
            slots = coordinator.req_to_token.req_to_token[req.req_pool_idx, :end]
            history = _token_prefixes.setdefault(req.rid, list(req.origin_input_ids))
            for position, token in zip(
                forward_batch.positions[region].tolist(),
                forward_batch.input_ids[region].tolist(),
                strict=True,
            ):
                if position == len(history):
                    history.append(token)
                else:
                    assert history[position] == token
            tokens = history[:end]
            assert len(tokens) == end
            predictions = (
                [end]
                if end >= len(req.origin_input_ids) and logits_output is not None
                else []
            )
            logits = (
                logits_output.next_token_logits[row : row + len(predictions)]
                if logits_output is not None
                else None
            )
        else:
            start = end
            slots = forward_batch.out_cache_loc[region]
            history = _token_prefixes[req.rid]
            assert len(history) >= start
            verify_tokens = forward_batch.input_ids[region].tolist()
            tokens = history[:start] + verify_tokens
            predictions = list(range(start + 1, start + count + 1))
            logits = logits_output.next_token_logits[region]
        # Read all raw vocabulary scores. The test computes its own top-k and
        # row alignment from final output tokens, without using capture tickets.
        reference = {
            "trace_id": hashlib.sha256(req.rid.encode()).hexdigest(),
            "result_lag": end - len(req.origin_input_ids) - len(req.output_ids),
            "batch_size": len(batch.reqs),
            "cuda_graph": can_run_cuda_graph,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "pp_rank": pp_rank,
            "verify_count": count if width is not None else None,
            "verify_width": width,
            "verify_padding": (
                forward_batch.input_ids.numel() - sum(lengths)
                if lengths is not None
                else 0
            ),
            "tokens": tokens,
            "kv_start": start,
            "kv_slots": slots.long().cpu(),
            "kv": (
                {
                    name: buffer[slots.long()].cpu()
                    for name, buffer in coordinator.exporter.buffers.items()
                }
                if coordinator.exporter is not None
                else {}
            ),
            "predictions": predictions,
            "logits": (
                logits[:, : coordinator.teacher.vocab_size].float().cpu()
                if logits is not None
                else torch.empty(0, coordinator.teacher.vocab_size)
            ),
        }
        path = root / f"{next(_sequence):06d}.pt"
        if width is None:
            torch.save(reference, path)
        else:
            _verify_frames[id(coordinator)].append(
                (req.rid, row, start, verify_tokens[0], reference, path)
            )


def after_forward(self, batch, forward_batch, logits_output, **kwargs):
    observe(
        self,
        batch,
        forward_batch,
        logits_output,
        can_run_cuda_graph=kwargs.get("can_run_cuda_graph", False),
    )
    return _after_forward(self, batch, forward_batch, logits_output, **kwargs)


def after_verify_forward(self, batch, forward_batch, logits_output, **kwargs):
    observe(
        self,
        batch,
        forward_batch,
        logits_output,
        width=kwargs["width"],
        can_run_cuda_graph=kwargs["can_run_cuda_graph"],
        verify_lens=kwargs.get("verify_lens"),
    )
    return _after_verify_forward(self, batch, forward_batch, logits_output, **kwargs)


def after_verify_accept(self, ticket, *, commit_lens, out_tokens):
    result = _after_verify_accept(
        self, ticket, commit_lens=commit_lens, out_tokens=out_tokens
    )
    counts, outputs = commit_lens.tolist(), out_tokens.tolist()
    for rid, row, start, anchor, reference, path in _verify_frames.pop(id(self), []):
        history = _token_prefixes[rid]
        if start < len(history):
            assert history[start] == anchor
        _token_prefixes[rid] = history[:start] + [anchor] + outputs[row][: counts[row]]
        reference["num_commit"] = counts[row]
        torch.save(reference, path)
    return result


def install_capture_observer():
    CaptureCoordinator.after_forward = after_forward
    CaptureCoordinator.after_verify_forward = after_verify_forward
    CaptureCoordinator.after_verify_accept = after_verify_accept


def check_capture_snapshot(
    test,
    manifest,
    tensors,
    references,
    *,
    capture_mode="speculative_accepted_target_path",
):
    tokens = tensors["token_ids"].tolist()
    teacher, kv = {}, {}
    for item in references:
        if item["trace_id"] != manifest.provenance.trace_id:
            continue
        # A budget-trimmed token can match a later output without committed KV.
        num_commit = item.get("num_commit")
        for row, prediction in enumerate(item["predictions"][:num_commit]):
            if (
                prediction < len(tokens)
                and item["tokens"][:prediction] == tokens[:prediction]
            ):
                teacher[prediction] = item["logits"][row]
        if not item["kv"]:
            continue
        num_rows = next(iter(item["kv"].values())).shape[0]
        if num_commit is not None:
            num_rows = min(num_rows, num_commit)
        for row in range(num_rows):
            position = item["kv_start"] + row
            if (
                position < len(tokens)
                and item["tokens"][: position + 1] == tokens[: position + 1]
            ):
                # Prefix dedup may remap a row after it has been copied. The
                # sample retains its first observation, just like draft context.
                by_name = kv.setdefault(position, {})
                for name, value in item["kv"].items():
                    layer = next(
                        layer
                        for layer in manifest.kv.layers
                        if layer.layer_id == int(name.split(".")[1])
                    )
                    rank, size = item.get("tp_rank", 0), item.get("tp_size", 1)
                    first = rank * layer.num_kv_heads // size
                    by_head = by_name.setdefault(name, {})
                    for head in range(value.shape[1]):
                        by_head.setdefault(first + head, value[row, head])
    test.assertEqual(manifest.provenance.capture_mode, capture_mode)
    test.assertEqual(tensors["position_ids"].tolist(), list(range(len(tokens))))
    test.assertEqual(
        tensors["loss_mask"].tolist(),
        [0] * manifest.sequence.prompt_length + [1] * manifest.sequence.response_length,
    )
    test.assertEqual(
        tensors["logits_positions"].tolist(),
        list(range(manifest.sequence.prompt_length, len(tokens))),
    )
    for row, prediction in enumerate(tensors["logits_positions"].tolist()):
        raw = teacher[prediction]
        values = tensors["teacher_topk_logits"][row]
        ids = tensors["teacher_topk_ids"][row].long()
        torch.testing.assert_close(values, raw[ids], rtol=0, atol=0)
        torch.testing.assert_close(values, raw.topk(128).values, rtol=0, atol=0)
        torch.testing.assert_close(
            tensors["teacher_logsumexp"][row], raw.logsumexp(0), rtol=1e-6, atol=1e-6
        )
    test.assertEqual(
        tensors["kv_valid"].tolist(),
        [int(position in kv) for position in range(len(tokens))],
    )
    for name in (name for name in tensors if name.startswith("target_")):
        expected = torch.stack(
            [
                torch.stack(
                    [kv[position][name][head] for head in range(tensors[name].shape[1])]
                )
                for position in sorted(kv)
            ]
        )
        torch.testing.assert_close(tensors[name], expected, rtol=0, atol=0)
