"""Latest sampled tokens remain penalized when prefill joins ongoing decode."""

import asyncio
import functools
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

MODEL_PATH = os.environ.get("TEST_MODEL_PATH", "Qwen/Qwen3-8B")


def _observed_scheduler(trace_path, *args, **kwargs):
    # Install read-only probes inside the spawned scheduler process.
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    forward = TpModelWorker.forward_batch_generation
    transform = SamplingBatchInfo._apply_pre_grammar_logits_transforms
    history = {}
    context = None
    with open(trace_path, "w", buffering=1) as trace:

        def observe_transform(info, logits):
            before = logits.detach().clone()
            transform(info, logits)
            assert context is not None
            batch = context
            decode_ids = {req.rid for req in (batch.decoding_reqs or [])}
            for row, req in enumerate(batch.reqs):
                if req is batch.chunked_req or batch.is_prefill_only:
                    continue
                # Do not derive the reference from Req.output_ids or penalty state.
                seen = sorted(set(history.get(req.rid, [])))
                expected = before[row].clone()
                factor = req.sampling_params.repetition_penalty
                if seen:
                    values = expected[seen]
                    expected[seen] = torch.where(
                        values < 0, values * factor, values / factor
                    )
                # Check unseen tokens too: penalties must not leak between rows.
                bad = ~torch.isclose(logits[row], expected, rtol=1e-6, atol=1e-6)
                bad_ids = bad.nonzero().flatten()[:4].tolist()
                record = {
                    "kind": "check",
                    "rid": req.rid,
                    "mode": batch.forward_mode.name,
                    "decode": batch.forward_mode.is_decode() or req.rid in decode_ids,
                    "step": len(history.get(req.rid, [])),
                    "factor": factor,
                    "bad_count": bad.sum().item(),
                    "examples": [
                        {
                            "token": token,
                            "raw": before[row, token].item(),
                            "expected": expected[token].item(),
                            "actual": logits[row, token].item(),
                        }
                        for token in bad_ids
                    ],
                }
                trace.write(json.dumps(record) + "\n")

        def observe_forward(worker, batch=None, *args, **kwargs):
            nonlocal context
            assert batch is not None
            context = batch
            try:
                result = forward(worker, batch, *args, **kwargs)
                assert result.delay_sample_func is None
                if result.next_token_ids is not None and not batch.is_prefill_only:
                    ids = result.next_token_ids.detach().cpu().tolist()
                    for req, token in zip(batch.reqs, ids):
                        if req is not batch.chunked_req:
                            # Record the actual sample before the next forward.
                            history.setdefault(req.rid, []).append(token)
                            trace.write(
                                json.dumps(
                                    {"kind": "sample", "rid": req.rid, "token": token}
                                )
                                + "\n"
                            )
                return result
            finally:
                context = None

        with (
            patch.object(TpModelWorker, "forward_batch_generation", observe_forward),
            patch.object(
                SamplingBatchInfo,
                "_apply_pre_grammar_logits_transforms",
                observe_transform,
            ),
        ):
            run_scheduler_process(*args, **kwargs)


class _ObservedEngine(Engine):
    def __init__(self, trace_path, **kwargs):
        self.run_scheduler_process_func = functools.partial(
            _observed_scheduler, str(trace_path)
        )
        super().__init__(**kwargs)


async def _mixed_workload(engine):
    started = asyncio.Event()
    outputs = {}

    async def long_request():
        stream = await engine.async_generate(
            prompt="Write the word hello many times, separated by spaces.",
            sampling_params={
                "temperature": 0,
                "repetition_penalty": 2.0,
                "max_new_tokens": 192,
                "ignore_eos": True,
            },
            stream=True,
            rid="long",
        )
        async for item in stream:
            if item.get("output_ids"):
                started.set()
            outputs["long"] = item

    task = asyncio.create_task(long_request())
    await asyncio.wait_for(started.wait(), 120)
    # Introduce prefill only after the long request has produced a real token.
    for wave in range(3):
        jobs = []
        for j, (factor, length) in enumerate([(1.0, 12), (1.2, 20), (2.0, 28)]):
            jobs.append(
                engine.async_generate(
                    prompt=f"Round {wave}. Continue counting: 1, 2, 3,",
                    sampling_params={
                        "temperature": 0,
                        "repetition_penalty": factor,
                        "max_new_tokens": length,
                        "ignore_eos": True,
                    },
                    rid=f"w{wave}_{j}",
                )
            )
        for j, row in enumerate(await asyncio.gather(*jobs)):
            outputs[f"w{wave}_{j}"] = row
    await task
    return outputs


class TestRepetitionPenaltyMixedOverlap(CustomTestCase):
    def test_mixed_logits_match_generated_history(self):
        """Real MIXED rows must include the latest sample, without cross-row leakage."""
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"SGLANG_ENABLE_DELAY_SAMPLE": "0"}),
        ):
            trace = Path(directory) / "trace.jsonl"
            engine = _ObservedEngine(
                trace_path=trace,
                model_path=MODEL_PATH,
                tp_size=1,
                context_length=1024,
                mem_fraction_static=0.7,
                max_total_tokens=4096,
                max_running_requests=16,
                attention_backend="triton",
                sampling_backend="pytorch",
                cuda_graph_backend_decode="disabled",
                cuda_graph_backend_prefill="disabled",
                disable_overlap_schedule=False,
                enable_mixed_chunk=True,
                chunked_prefill_size=128,
                stream_interval=1,
            )
            try:
                outputs = engine.loop.run_until_complete(
                    asyncio.wait_for(_mixed_workload(engine), 240)
                )
            finally:
                engine.shutdown()
            records = [json.loads(line) for line in trace.read_text().splitlines()]

        checks = [record for record in records if record["kind"] == "check"]
        mixed = [record for record in checks if record["mode"] == "MIXED"]
        self.assertTrue(
            any(record["decode"] and record["step"] > 0 for record in mixed),
            "No MIXED decode row was checked; mixed-chunk configuration alone is insufficient.",
        )
        self.assertTrue(any(not record["decode"] for record in mixed))
        failures = [record for record in checks if record["bad_count"]]
        self.assertFalse(bool(failures), json.dumps(failures[:3], indent=2))

        sampled = {}
        for record in records:
            if record["kind"] == "sample":
                sampled.setdefault(record["rid"], []).append(record["token"])
        self.assertEqual(len(outputs), 10)
        for rid, output in outputs.items():
            ids = output["output_ids"]
            expected_length = 192 if rid == "long" else [12, 20, 28][int(rid[-1])]
            self.assertEqual(len(ids), expected_length, rid)
            # Overlap may sample an extra in-flight token after the length limit.
            self.assertEqual(ids, sampled[rid][: len(ids)], rid)


if __name__ == "__main__":
    unittest.main()
