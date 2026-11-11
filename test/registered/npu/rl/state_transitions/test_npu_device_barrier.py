"""P0 NPU tests: the device-idle contract around pause/resume.

``release_memory_occupation`` must not return before the device is idle: an RL
caller starts copying weights as soon as it returns. The unit cases assert the
ordering directly; the e2e case repeats pause -> update -> continue to catch a
race that only shows up over rounds.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.scheduler_components import (
    weight_updater as weight_updater_module,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_rl_utils import (
    begin_weight_update,
    checkpoint_tensor_index,
    continue_generation,
    end_weight_update,
    generate,
    load_checkpoint_tensor,
    npu_available,
    npu_device_count,
    npu_rl_env,
    npu_server,
    pause_generation,
    start_stream,
    update_weights_from_tensor,
    wait_for_first_token,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=700, suite="base-b-test-2-npu-a3")
register_npu_ci(est_time=700, suite="nightly-2-npu-a3", nightly=True)

MODEL = QWEN3_0_6B_WEIGHTS_PATH
PROMPT = "The capital of France is"
SERVER_ARGS = (
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--disable-overlap-schedule",
    "--mem-fraction-static",
    "0.45",
    "--context-length",
    "512",
    "--max-running-requests",
    "4",
)
DP_ARGS = SERVER_ARGS + (
    "--dp-size",
    "2",
    "--tp-size",
    "1",
    "--load-balance-method",
    "round_robin",
)
NORM_WEIGHT = "model.norm.weight"


def _enqueue_npu_work(iterations=256, size=4096):
    """Queue real NPU work without waiting, like torch.cuda._sleep on CUDA."""
    lhs = torch.randn(size, size, dtype=torch.float16, device="npu")
    rhs = torch.randn(size, size, dtype=torch.float16, device="npu")
    out = torch.empty_like(lhs)
    for _ in range(iterations):
        torch.mm(lhs, rhs, out=out)


def _make_manager(events):
    manager = object.__new__(SchedulerWeightUpdaterManager)
    manager.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(model=object()))
    manager.draft_worker = None
    manager.memory_saver_adapter = SimpleNamespace(
        pause=lambda tag: events.append(("pause", tag)),
        resume=lambda tag: events.append(("resume", tag)),
    )
    manager.flush_cache = lambda **kwargs: events.append(("flush", None)) or True
    manager.is_fully_idle = lambda **kwargs: True
    manager.scheduler = None
    manager.offload_tags = set()
    manager.stashed_model_static_state = None
    manager.tp_cpu_group = object()
    return manager


class TestNpuDeviceBarrier(CustomTestCase):
    @unittest.skipUnless(npu_available(), "requires an Ascend NPU")
    def test_release_synchronizes_before_return(self):
        torch.npu.set_device(0)
        torch.npu.synchronize()

        events = []
        manager = _make_manager(events)

        _enqueue_npu_work()
        self.assertFalse(
            torch.npu.current_stream().query(),
            "helper failed to leave NPU work in flight",
        )

        with (
            patch.object(
                weight_updater_module,
                "_export_static_state",
                lambda model: events.append(("export", None)),
            ),
            patch.object(
                weight_updater_module,
                "get_model",
                lambda: SimpleNamespace(weight_cache_mode="off"),
            ),
            patch(
                "torch.distributed.barrier",
                lambda *args, **kwargs: events.append(("barrier", None)),
            ),
        ):
            manager.release_memory_occupation(SimpleNamespace(tags=["weights"]))

        self.assertTrue(
            torch.npu.current_stream().query(),
            "release returned while NPU work was still in flight",
        )
        self.assertIn(
            "pause", [kind for kind, _ in events], "weights were never released"
        )

    def test_resume_orders_restore_after_barrier(self):
        events = []
        manager = _make_manager(events)
        manager.offload_tags = {"weights"}

        with (
            patch.object(
                weight_updater_module,
                "_import_static_state",
                lambda model, state: events.append(("import", None)),
            ),
            patch.object(
                weight_updater_module,
                "get_model",
                lambda: SimpleNamespace(weight_cache_mode="off"),
            ),
            patch(
                "torch.distributed.barrier",
                lambda *args, **kwargs: events.append(("barrier", None)),
            ),
        ):
            manager.resume_memory_occupation(SimpleNamespace(tags=["weights"]))

        kinds = [kind for kind, _ in events]
        self.assertIn("barrier", kinds, "resume did not join the TP ranks")
        self.assertIn("import", kinds, "resume did not restore static state")
        self.assertLess(
            kinds.index("resume"),
            kinds.index("barrier"),
            f"state restored before the replica barrier: {events}",
        )
        self.assertLess(
            kinds.index("barrier"),
            kinds.index("import"),
            f"state restored before all ranks finished: {events}",
        )
        # resume_memory_occupation has no device-wide synchronize today; this
        # test locks the barrier/restore ordering that the current code provides.

    @unittest.skipUnless(
        npu_available() and npu_device_count() >= 2,
        "requires at least 2 Ascend NPUs",
    )
    @unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
    def test_02_dp_pause_returns_after_all_ranks_drain(self):
        with npu_server(MODEL, extra_args=DP_ARGS, env=npu_rl_env()) as base_url:
            requests_ = [
                start_stream(base_url, PROMPT, max_new_tokens=384) for _ in range(2)
            ]
            for idx, request in enumerate(requests_):
                wait_for_first_token(request, label=f"DP request {idx}")

            dp_ranks = {
                request.events[0].get("meta_info", {}).get("dp_rank")
                for request in requests_
            }
            self.assertEqual(
                dp_ranks,
                {0, 1},
                f"requests did not cover both DP ranks: {dp_ranks}",
            )

            pause_generation(base_url, mode="abort")
            for request in requests_:
                request.thread.join(timeout=60)

            for idx, request in enumerate(requests_):
                self.assertFalse(
                    request.errors, f"DP stream {idx} failed: {request.errors}"
                )
                self.assertFalse(
                    request.thread.is_alive(), f"DP stream {idx} did not end"
                )
                last = request.events[-1]
                finish_type = (
                    last.get("meta_info", {}).get("finish_reason", {}).get("type")
                )
                self.assertEqual(
                    finish_type,
                    "abort",
                    f"DP rank {idx} was not aborted by pause: {last}",
                )
                self.assertLess(
                    len(last.get("output_ids", [])),
                    384,
                    f"DP rank {idx} produced the full completion",
                )

            # The per-rank scheduler stream is not observable over HTTP, so
            # this case proves both DP ranks were decoding and were aborted by
            # pause. The physical idle contract is covered by the single-rank
            # unit cases above; a true DP idle assertion needs an in-process
            # worker/RPC hook rather than a production readback field.

            continue_generation(base_url)
            self.assertTrue(generate(base_url, PROMPT, max_new_tokens=4))

    @unittest.skipUnless(npu_available(), "requires an Ascend NPU")
    @unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
    def test_pause_update_continue_is_stable_over_rounds(self):
        tensor_index = checkpoint_tensor_index(MODEL)
        original = load_checkpoint_tensor(tensor_index, NORM_WEIGHT)

        with npu_server(MODEL, extra_args=SERVER_ARGS, env=npu_rl_env()) as base_url:
            for round_idx in range(3):
                version = f"round-{round_idx}"
                updated = original.new_full(original.shape, 0.5)

                pause_generation(base_url, mode="in_place")
                begin_weight_update(base_url)
                update_weights_from_tensor(
                    base_url,
                    [(NORM_WEIGHT, updated.to("npu"))],
                    weight_version=version,
                )
                end_weight_update(base_url)
                continue_generation(base_url)

                response = generate(base_url, PROMPT, max_new_tokens=4)
                self.assertEqual(
                    response["meta_info"]["weight_version"],
                    version,
                    f"round {round_idx} resumed on an older weight version",
                )

            begin_weight_update(base_url)
            update_weights_from_tensor(
                base_url,
                [(NORM_WEIGHT, original.to("npu"))],
                weight_version="restored",
            )
            end_weight_update(base_url)


if __name__ == "__main__":
    unittest.main()
