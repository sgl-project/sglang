"""Test-only target output snapshots before serving mutations or slot reuse."""

import hashlib
import itertools
from pathlib import Path

import torch

from sglang.srt.runtime_context import get_spec
from sglang.srt.training_capture.coordinator import CaptureCoordinator

_sequence = itertools.count()
_after_forward = CaptureCoordinator.after_forward
_after_verify_forward = CaptureCoordinator.after_verify_forward


def observe(coordinator, batch, forward_batch, logits_output, width=None):
    root = Path(get_spec().speculative_draft_model_path) / "capture-reference"
    root.mkdir(exist_ok=True)
    for row, req in enumerate(batch.reqs):
        if req.training_capture_context is None:
            continue
        end = int(batch.seq_lens_cpu[row])
        tokens = list(req.origin_input_ids) + list(req.output_ids)
        if width is None:
            start = 0
            slots = coordinator.req_to_token.req_to_token[req.req_pool_idx, :end]
            tokens = tokens[:end]
            predictions = [end] if end >= len(req.origin_input_ids) else []
            logits = logits_output.next_token_logits[row : row + len(predictions)]
        else:
            start = end
            region = slice(row * width, (row + 1) * width)
            slots = forward_batch.out_cache_loc[region]
            tokens = tokens[:start] + forward_batch.input_ids[region].tolist()
            predictions = list(range(start + 1, start + width + 1))
            logits = logits_output.next_token_logits[region]
        # Read all raw vocabulary scores. The test computes its own top-k and
        # row alignment from final output tokens, without using capture tickets.
        torch.save(
            {
                "trace_id": hashlib.sha256(req.rid.encode()).hexdigest(),
                "tokens": tokens,
                "kv_start": start,
                "kv": {
                    name: buffer[slots.long()].cpu()
                    for name, buffer in coordinator.exporter.buffers.items()
                },
                "predictions": predictions,
                "logits": logits[:, : coordinator.teacher.vocab_size].float().cpu(),
            },
            root / f"{next(_sequence):06d}.pt",
        )


def after_forward(self, batch, forward_batch, logits_output, **kwargs):
    observe(self, batch, forward_batch, logits_output)
    return _after_forward(self, batch, forward_batch, logits_output, **kwargs)


def after_verify_forward(self, batch, forward_batch, logits_output, **kwargs):
    observe(self, batch, forward_batch, logits_output, width=kwargs["width"])
    return _after_verify_forward(self, batch, forward_batch, logits_output, **kwargs)


def install_capture_observer():
    CaptureCoordinator.after_forward = after_forward
    CaptureCoordinator.after_verify_forward = after_verify_forward


def check_speculative_snapshot(test, manifest, tensors, references):
    tokens = tensors["token_ids"].tolist()
    teacher, kv = {}, {}
    for item in references:
        if item["trace_id"] != manifest.provenance.trace_id:
            continue
        for row, prediction in enumerate(item["predictions"]):
            if (
                prediction < len(tokens)
                and item["tokens"][:prediction] == tokens[:prediction]
            ):
                teacher[prediction] = item["logits"][row]
        for row in range(next(iter(item["kv"].values())).shape[0]):
            position = item["kv_start"] + row
            if (
                position < len(tokens)
                and item["tokens"][: position + 1] == tokens[: position + 1]
            ):
                kv[position] = {name: value[row] for name, value in item["kv"].items()}
    test.assertEqual(
        manifest.provenance.capture_mode, "speculative_accepted_target_path"
    )
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
        expected = torch.stack([kv[position][name] for position in sorted(kv)])
        torch.testing.assert_close(tensors[name], expected, rtol=0, atol=0)
