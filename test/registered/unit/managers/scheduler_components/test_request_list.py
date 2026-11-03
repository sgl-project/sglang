import ast
import inspect
import re
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.dllm.mixin.scheduler import DllmManager
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.deferred_responses import (
    DeferredResponses,
)
from sglang.srt.models.unlimited_ocr_v2_support.scheduler_runtime import (
    UocrSchedulerRuntime,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

# The classes that own the request lists, with their mixins.
_OWNERS = (
    Scheduler,
    GrammarManager,
    PrefillBootstrapQueue,
    DecodePreallocQueue,
    DecodeTransferQueue,
    DllmManager,
    HiSparseCoordinator,
    UocrSchedulerRuntime,
    DeferredResponses,
)

# Fields annotated as holding requests that are not a `RequestList`. Only
# shrinks: a list that migrates leaves it.
_NOT_YET_LISTS = {
    "Scheduler.waiting_queue",
    "DecodePreallocQueue.queue",
    # Shares its entries with `queue`.
    "DecodePreallocQueue.pending_reqs",
    "DecodeTransferQueue.queue",
    "DecodeTransferQueue._deferred_releases",
    # A subset of `waiting_queue`.
    "DllmManager.staging_queue",
    "DeferredResponses.held",
}

_HOLDS_REQUESTS = re.compile(r"\b(Req|DecodeRequest)\b")


def _fields_holding_requests(owner) -> set:
    fields = set()
    for klass in owner.__mro__:
        if not klass.__module__.startswith("sglang."):
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "self"
                and _HOLDS_REQUESTS.search(ast.unparse(node.annotation))
            ):
                fields.add(f"{owner.__name__}.{node.target.attr}")
    return fields


class TestRequestHoldersAreLists(CustomTestCase):
    def test_every_field_holding_requests_is_a_request_list(self):
        fields = set().union(*map(_fields_holding_requests, _OWNERS))
        self.assertEqual(fields, _NOT_YET_LISTS)


if __name__ == "__main__":
    unittest.main()
