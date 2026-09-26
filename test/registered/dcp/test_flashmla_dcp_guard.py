"""``flashmla`` must refuse decode context parallelism at construction time.

The backend builds its decode block table straight off ``req_to_token`` with
the batch's GLOBAL ``seq_lens`` (``create_flashmla_kv_indices_triton``, and
``cache_seqlens=forward_batch.seq_lens`` into the kernel). Under DCP the pool
is widened -- ``kv_cache_configurator`` allocates
``max_total_num_tokens * dcp_size`` at ``page_size * dcp_size`` -- so those
rows name widened pages rather than this rank's physical ones, and the kernel
is told to read the whole sequence out of one shard. bf16 carried no guard at
all and computed silently wrong output; the fp8 branch carried an assert whose
message blamed the dtype rather than the missing page table.

Prefill is untouched: ``ForwardMode.EXTEND`` delegates to the
FlashInferMLA parent, which does have a DCP path. So the refusal keys on
flashmla being the DECODE backend, mirroring
``create_trtllm_mla_backend``'s DCP-with-speculative-decoding check.

Usage:
    python -m pytest test_flashmla_dcp_guard.py -v
    python test_flashmla_dcp_guard.py
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention import attention_registry as registry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BACKEND_MODULE = "sglang.srt.layers.attention.flashmla_backend"
SENTINEL = object()


def _call(*, dcp_size, decode_backend, prefill_backend="flashmla"):
    """Run the factory with the topology and backend split faked out.

    ``resolved_view`` needs a real ServerArgs to resolve overrides and the
    happy path needs a real ModelRunner, so both are stubbed; the factory's
    own branch is what is under test. The backend module is injected into
    ``sys.modules`` rather than patched in place so the test does not depend
    on something else having imported it (it pulls in sgl_kernel, which a
    CPU runner has no reason to load).
    """
    parallel = SimpleNamespace(dcp_enabled=dcp_size > 1, dcp_size=dcp_size, dcp_rank=0)
    runner = SimpleNamespace(server_args=SimpleNamespace())
    stub = types.ModuleType(BACKEND_MODULE)
    stub.FlashMLABackend = lambda _runner: SENTINEL
    with (
        patch.object(registry, "get_parallel", return_value=parallel),
        patch.object(registry, "resolved_view", side_effect=lambda sa: sa),
        patch.object(
            registry,
            "attention_backends_of",
            return_value=(prefill_backend, decode_backend),
        ),
        patch.dict(sys.modules, {BACKEND_MODULE: stub}),
    ):
        return registry.create_flashmla_backend(runner)


class TestFlashMLARejectsDcpDecode(CustomTestCase):
    def test_dcp_decode_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _call(dcp_size=4, decode_backend="flashmla")
        # The message has to name the page table, not the KV dtype: the fp8
        # assert this replaces sent readers after a dtype problem instead.
        self.assertIn("decode context parallelism", str(ctx.exception))
        self.assertIn("block table", str(ctx.exception))

    def test_dcp_disabled_is_allowed(self):
        self.assertIs(_call(dcp_size=1, decode_backend="flashmla"), SENTINEL)

    def test_prefill_only_under_dcp_is_allowed(self):
        # EXTEND runs through the FlashInferMLA parent, which has a DCP path,
        # so a split config that only prefills on flashmla must still build.
        self.assertIs(_call(dcp_size=4, decode_backend="cutedsl_mla"), SENTINEL)

    def test_dcp_size_two_also_raises(self):
        # dcp_size 2 and 4 take different owner-rule paths; neither is served.
        with self.assertRaises(ValueError):
            _call(dcp_size=2, decode_backend="flashmla")


if __name__ == "__main__":
    unittest.main()
