"""Exercise the embedded scheduler with a real model in both loop modes."""

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from array import array

import torch

from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
    TokenizedGenerateReqInput,
    row_output_ids,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import SpawnRanks, publish
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-small")

_RESULT_PREFIX = "SGLANG_SINGLE_ITERATION_RESULT="


class _Collector:
    def __init__(self):
        self.outputs = []

    def send_output(self, output, recv_obj=None):
        self.outputs.append(output)


def _run_scheduler(mode):
    # Isolate each mode's CUDA and process-group lifetime. The output collector
    # stands in for the separate in-process output API requested in #48473.
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory(prefix="sglang-single-iteration-") as temp_dir:
        tempfile.tempdir = temp_dir
        args = ServerArgs(
            model_path="Qwen/Qwen3-0.6B",
            dtype="float16",
            device="cuda",
            tp_size=1,
            skip_tokenizer_init=True,
            skip_server_warmup=True,
            disable_overlap_schedule=mode == "normal",
            disable_cuda_graph=True,
            attention_backend="triton",
            sampling_backend="pytorch",
            mem_fraction_static=0.4,
            max_total_tokens=512,
            max_running_requests=4,
            context_length=256,
            chunked_prefill_size=-1,
            log_level="warning",
        )
        publish(args, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
        scheduler = Scheduler(args, PortArgs.init_new(args))
        collector = _Collector()
        scheduler.output_streamer.send_to_detokenizer = collector
        scheduler.ipc_channels = dataclasses.replace(
            scheduler.ipc_channels, send_to_tokenizer=collector
        )

        sampling = SamplingParams(temperature=0.0, max_new_tokens=2, ignore_eos=True)
        sampling.normalize(None)
        sampling.verify(scheduler.model_config.vocab_size)
        request = TokenizedGenerateReqInput(
            rid="single-step-test",
            input_text=None,
            input_ids=array("q", [9707, 11, 1917]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=sampling,
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
        )
        with scheduler.device_module.StreamContext(scheduler.schedule_stream):
            scheduler.process_input_requests([request])

        tokens = []
        finish_reason = None
        consumed = 0
        for step in range(1, 17):
            scheduler.run_one_iteration()
            for output in collector.outputs[consumed:]:
                if not isinstance(output, BatchTokenIDOutput):
                    continue
                for index, rid in enumerate(output.rids or []):
                    if rid == request.rid:
                        tokens.extend(row_output_ids(output, index))
                        finish_reason = output.finished_reasons[index] or finish_reason
            consumed = len(collector.outputs)
            if finish_reason is not None and scheduler.is_fully_idle():
                break

        torch.cuda.synchronize()
        assert finish_reason is not None, "request did not finish"
        assert len(tokens) == 2, f"expected two tokens, got {tokens}"
        assert scheduler.is_fully_idle(), "scheduler did not drain"
        print(
            _RESULT_PREFIX + json.dumps({"tokens": tokens, "steps": step}), flush=True
        )
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class TestSchedulerSingleIterationGpu(CustomTestCase):
    def test_normal_and_overlap_generate_the_same_tokens(self):
        results = []
        for mode in ("normal", "overlap"):
            process = subprocess.run(
                [sys.executable, __file__, "--single-step-worker", mode],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stdout)
            result_lines = [
                line.split(_RESULT_PREFIX, 1)[1]
                for line in process.stdout.splitlines()
                if _RESULT_PREFIX in line
            ]
            self.assertEqual(len(result_lines), 1, process.stdout)
            results.append(json.loads(result_lines[0]))
        self.assertEqual(results[0]["tokens"], results[1]["tokens"])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--single-step-worker":
        _run_scheduler(sys.argv[2])
    else:
        unittest.main()
