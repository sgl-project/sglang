# SPDX-License-Identifier: Apache-2.0
"""Manual live smoke for UG requests executed by a real SRT Scheduler.

Usage:
CUDA_VISIBLE_DEVICES=0 \
SGLANG_TEST_UG_SRT_SCHEDULER_MODEL=/path/to/text-model \
python3 test/registered/scheduler/test_ug_srt_scheduler_executor_live.py
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=300,
    suite="stage-b-test-1-gpu-large",
    disabled=(
        "Manual UG/SRT scheduler live smoke; requires "
        "SGLANG_TEST_UG_SRT_SCHEDULER_MODEL"
    ),
)

_MODEL_ENV = "SGLANG_TEST_UG_SRT_SCHEDULER_MODEL"


class _NoopSender:
    def send_output(self, *args, **kwargs):
        del args, kwargs


def _replace_sender_with_noop(scheduler, name: str) -> None:
    sender = getattr(scheduler, name, None)
    socket = getattr(sender, "socket", None)
    if socket is not None:
        socket.close(linger=0)
    setattr(scheduler, name, _NoopSender())


@unittest.skipUnless(os.getenv(_MODEL_ENV), f"Set {_MODEL_ENV} for live smoke")
class TestUGSRTSchedulerExecutorLive(CustomTestCase):
    def test_real_scheduler_runs_prefill_only_ug_session_requests(self):
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the live SRT scheduler smoke")

        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.server_args import (
            PortArgs,
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
        from sglang.srt.ug.runtime import (
            FakeUGModelRunner,
            UGInterleavedMessage,
            UGSessionRuntime,
        )
        from sglang.srt.ug.srt_executor import UGSRTSchedulerExecutor

        model_path = os.environ[_MODEL_ENV]
        server_args = ServerArgs(
            model_path=model_path,
            tokenizer_path=os.getenv(
                "SGLANG_TEST_UG_SRT_SCHEDULER_TOKENIZER", model_path
            ),
            trust_remote_code=True,
            tp_size=1,
            pp_size=1,
            dp_size=1,
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
            disable_overlap_schedule=True,
            skip_server_warmup=True,
            mem_fraction_static=float(
                os.getenv("SGLANG_TEST_UG_SRT_SCHEDULER_MEM_FRACTION", "0.35")
            ),
            chunked_prefill_size=int(
                os.getenv("SGLANG_TEST_UG_SRT_SCHEDULER_CHUNKED_PREFILL", "256")
            ),
            log_level="error",
        )
        server_args.check_server_args()
        set_global_server_args_for_scheduler(server_args)

        scheduler = Scheduler(
            server_args,
            PortArgs.init_new(server_args),
            gpu_id=int(os.getenv("SGLANG_TEST_UG_SRT_SCHEDULER_GPU_ID", "0")),
            tp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
            attn_cp_rank=0,
            moe_dp_rank=0,
            dp_rank=None,
        )
        _replace_sender_with_noop(scheduler, "send_to_tokenizer")
        _replace_sender_with_noop(scheduler, "send_to_detokenizer")

        try:
            executor = UGSRTSchedulerExecutor(scheduler, max_sync_steps=16)
            runtime = UGSessionRuntime(
                model_runner=FakeUGModelRunner(),
                session_controller=scheduler.session_controller,
                srt_request_executor=executor,
                tokenizer=scheduler.tokenizer,
                vocab_size=scheduler.model_config.vocab_size,
            )

            handle = runtime.prefill_interleaved(
                [UGInterleavedMessage(type="text", content="hello from ug")],
                session_id="ug-live-scheduler-prefill",
            )
            self.assertTrue(scheduler.is_fully_idle())

            handle = runtime.prefill_interleaved(
                [UGInterleavedMessage(type="text", content="continue the session")],
                session_id=handle.session_id,
            )
            counters = runtime.get_debug_counters(handle)

            self.assertEqual(counters["session_id"], "ug-live-scheduler-prefill")
            self.assertEqual(counters["state"], "u_decode")
            self.assertEqual(counters["prefill_count"], 2)
            self.assertEqual(counters["srt_request_count"], 2)
            self.assertEqual(counters["srt_executed_request_count"], 2)
            self.assertEqual(executor.sync_step_count, 2)
            self.assertTrue(scheduler.is_fully_idle())

            runtime.close_session(handle)
            self.assertNotIn(handle.session_id, scheduler.session_controller.sessions)
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
