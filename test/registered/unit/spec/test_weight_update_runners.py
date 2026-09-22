"""Every spec family must name exactly the draft runners that own weights."""

import sys
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

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


CASES = [
    (EAGLEWorkerV2, lambda rs: {"_draft_worker": _eagle_draft(rs)}, [("draft", 0)]),
    (
        StandaloneWorkerV2,
        lambda rs: {"_draft_worker": _eagle_draft(rs)},
        [("draft", 0)],
    ),
    # reuses the target KV but owns its weights, unlike _draft_model_runners()
    (
        FrozenKVMTPWorkerV2,
        lambda rs: {
            "_draft_worker": _bare(FrozenKVMTPDraftWorker, _model_runner=rs[0])
        },
        [("draft", 0)],
    ),
    (
        MultiLayerEagleWorkerV2,
        lambda rs: {
            "_draft_worker": _bare(MultiLayerEagleDraftWorker, draft_runner_list=rs)
        },
        [("draft_step_0", 0), ("draft_step_1", 1)],
    ),
    (DFlashWorkerV2, lambda rs: {"draft_model_runner": rs[0]}, [("draft", 0)]),
    (
        DSparkWorkerV2,
        lambda rs: {"_hosts_draft": True, "draft_model_runner": rs[0]},
        [("draft", 0)],
    ),
    (DSparkWorkerV2, lambda rs: {"_hosts_draft": False}, []),
    (NGRAMWorker, lambda rs: {}, []),
    (UnoWorkerV2, lambda rs: {}, []),
]


@pytest.mark.parametrize(
    "cls, attrs, expected",
    CASES,
    ids=[
        "eagle",
        "standalone",
        "frozen_kv_mtp",
        "multi_layer_eagle",
        "dflash",
        "dspark_last_pp_stage",
        "dspark_non_last_pp_stage",
        "ngram",
        "uno",
    ],
)
def test_weight_update_runners_cover_each_spec_family(cls, attrs, expected):
    runners = [object(), object()]
    worker = _bare(cls, **attrs(runners))

    got = worker.weight_update_runners()

    assert [(role, runners.index(r)) for role, r in got] == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
