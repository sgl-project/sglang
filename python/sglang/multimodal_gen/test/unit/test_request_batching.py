import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.flux import FluxPipelineConfig
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.orchestrator import DiffusionServer
from sglang.multimodal_gen.runtime.managers.dynamic_batch_admission import (
    BatchAdmissionController,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


@pytest.mark.parametrize("disaggregated_ar", [False, True])
@pytest.mark.parametrize(
    "mode,cap,dispatches",
    [
        ("dynamic", 1, [["a"], ["b"], ["c"], ["d"]]),
        ("opt-in", 1, [["a"], ["b"], ["c"], ["d"]]),
        ("dynamic", 2, [["a", "b"], ["c"], []]),  # d still waits for a partner
        ("opt-in", 2, [["a", "d"], ["b"], ["c"]]),
    ],
)
def test_mixed_queue_respects_request_and_server_policy(
    disaggregated_ar, mode, cap, dispatches
):
    args = SimpleNamespace(
        pipeline_config=FluxPipelineConfig(),
        model_path="flux",
        batching_mode=mode,
        batching_max_size=cap,
        batching_config=None,
        batching_delay_ms=60000,
        has_layerwise_offload_components=lambda: False,
    )
    requests = [
        Req(request_id=key, sampling_params=SamplingParams(prompt=key, **options))
        for key, options in [
            ("a", {"allow_dynamic_batching": True}),
            ("b", {}),
            ("c", {"allow_dynamic_batching": False}),
            ("d", {"allow_dynamic_batching": True}),
        ]
    ]
    now = time.monotonic()
    if disaggregated_ar:
        server = object.__new__(DiffusionServer)
        server._tracker = Mock()
        state = server._glm_distributed_state = SimpleNamespace(
            server_args=args,
            active_ar_batch=None,
            ar_stage=Mock(),
            executor=Mock(),
            pending_ar_requests=[
                SimpleNamespace(
                    req=req, client_request_id=req.request_id, enqueue_time=now
                )
                for req in requests
            ],
        )

        def dispatch():
            state.active_ar_batch = None
            server._dispatch_glm_ar_batch_if_ready()
            if state.active_ar_batch is None:
                return []
            return [item.req.request_id for item in state.active_ar_batch[1]]

    else:
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = args
        scheduler._batch_admission = BatchAdmissionController(args, gpu_id=0)
        scheduler._batch_metrics_enabled = False
        scheduler._batching_max_size = cap
        scheduler._batching_delay_s = 60
        scheduler.waiting_queue = deque((None, req, now) for req in requests)

        def dispatch():
            return [
                req.request_id for _, req in scheduler.get_next_batch_to_run() or []
            ]

    for expected in dispatches:
        assert dispatch() == expected
