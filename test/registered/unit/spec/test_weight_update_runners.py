"""Every spec family must name exactly the draft runners that own weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.speculative import eagle_worker_v2
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
    FrozenKVMTPDraftWorker,
    FrozenKVMTPWorkerV2,
)
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleDraftWorker,
    MultiLayerEagleWorkerV2,
)
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.standalone_worker_v2 import StandaloneWorkerV2
from sglang.srt.speculative.uno_worker_v2 import UnoWorkerV2


def _bare(cls, **attrs):
    # skip __init__: only the attributes the enumeration reads are set
    obj = object.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def _eagle_draft(runners):
    return SimpleNamespace(draft_runners=runners[:1])


CASES = {
    "eagle": (
        EAGLEWorkerV2,
        lambda rs: {"_draft_worker": _eagle_draft(rs)},
        [("draft", 0)],
    ),
    "standalone": (
        StandaloneWorkerV2,
        lambda rs: {"_draft_worker": _eagle_draft(rs)},
        [("draft", 0)],
    ),
    # reuses the target KV but owns its weights, unlike _draft_model_runners()
    "frozen_kv_mtp": (
        FrozenKVMTPWorkerV2,
        lambda rs: {
            "_draft_worker": _bare(FrozenKVMTPDraftWorker, _model_runner=rs[0])
        },
        [("draft", 0)],
    ),
    "multi_layer_eagle": (
        MultiLayerEagleWorkerV2,
        lambda rs: {
            "_draft_worker": _bare(MultiLayerEagleDraftWorker, draft_runner_list=rs)
        },
        [("draft_step_0", 0), ("draft_step_1", 1)],
    ),
    "dflash": (
        DFlashWorkerV2,
        lambda rs: {"draft_model_runner": rs[0]},
        [("draft", 0)],
    ),
    "dspark_last_pp_stage": (
        DSparkWorkerV2,
        lambda rs: {"_hosts_draft": True, "draft_model_runner": rs[0]},
        [("draft", 0)],
    ),
    "dspark_non_last_pp_stage": (
        DSparkWorkerV2,
        lambda rs: {"_hosts_draft": False},
        [],
    ),
    "ngram": (NGRAMWorker, lambda rs: {}, []),
    "uno": (UnoWorkerV2, lambda rs: {}, []),
}


class TestWeightUpdateRunners(CustomTestCase):
    def test_weight_update_runners_cover_each_spec_family(self):
        for case, (cls, attrs, expected) in CASES.items():
            with self.subTest(case):
                runners = [object(), object()]
                worker = _bare(cls, **attrs(runners))

                got = worker.weight_update_runners()

                self.assertEqual(
                    [(role, runners.index(r)) for role, r in got], expected
                )


class TestEagleAdaptiveGraphRelease(CustomTestCase):
    def test_every_adaptive_state_is_released_once(self):
        """Each adaptive step state captures its own draft, verify and
        draft-extend graphs, and only the active verify runner is the target's.
        Releasing just the target's runners left the other states' graphs alive
        past the distributed teardown."""
        released = []
        runners = {}

        def state(steps):
            names = [f"draft{steps}", f"verify{steps}", f"extend{steps}"]
            for name in names:
                runners[name] = SimpleNamespace(name=name)
            return SpecRuntimeState(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                draft_attn_backend=None,
                cuda_graph_runner=runners[names[0]],
                target_attn_backend=None,
                target_graph_runner=runners[names[1]],
                draft_extend_attn_backend=None,
                cuda_graph_runner_for_draft_extend=runners[names[2]],
            )

        controller = AdaptiveController(worker=None, policy=None)
        controller.register(state(1))
        controller.register(state(3))
        target = SimpleNamespace(
            model_runner=SimpleNamespace(decode_cuda_graph_runner=runners["verify1"])
        )
        target.release_cuda_graphs = lambda: released.append("verify1")
        worker = _bare(
            EAGLEWorkerV2, _target_worker=target, adaptive_controller=controller
        )

        with patch.object(
            eagle_worker_v2,
            "release_graph_runners",
            lambda runners: released.extend(r.name for r in runners),
        ):
            worker.release_cuda_graphs()

        self.assertCountEqual(released, list(runners))


if __name__ == "__main__":
    unittest.main()
