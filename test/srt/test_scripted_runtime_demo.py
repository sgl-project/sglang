"""ScriptedRuntime end-to-end demo.

This file is a *teaching demo*: a single script that exercises every
public piece of the ScriptedRuntime API with inline commentary, so a
new user can read it top-to-bottom and learn the full surface.

For focused tests of individual behaviors (assertion surfacing,
non-generator script rejection, lambda-as-script rejection, etc.) see
``test_scripted_runtime_functional.py``. For the minimal "does it
launch?" smoke see ``test_scripted_runtime_smoke.py``.

What the demo covers:
1. Engine launch via :func:`execute_scripted_runtime` (one line).
2. Submitting requests with :meth:`ScriptedRuntime.start_req`.
3. Observing request status via :class:`ReqHandle`.
4. The bare-``yield`` primitive as the single "step the scheduler"
   action.
5. The ``while not <condition>: yield`` idiom for waiting on a state.
6. Multiple concurrent requests.
7. Constructing a :class:`ReqHandle` directly for an arbitrary rid.
8. Clean shutdown via generator return.
"""

import unittest

from sglang.test.scripted_runtime import (
    ReqHandle,
    ScriptedRuntime,
    execute_scripted_runtime,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase


def _demo_script(t: ScriptedRuntime):
    """Walk through every ScriptedRuntime feature in one script.

    Read top-to-bottom to learn the API. The script generator is the
    unit of test: ``yield`` returns control to the scheduler for one
    event-loop iteration; everything between yields is pure Python
    that can read scheduler state via :class:`ReqHandle` properties.
    """

    # ----------------------------------------------------------------
    # Part 1 — Submit a request.
    #
    # ``start_req`` builds a TokenizedGenerateReqInput with placeholder
    # token ids of length ``prompt_len`` and injects it directly into
    # the scheduler's input queue (bypassing HTTP server and tokenizer
    # manager — fine-grained tests don't care about those layers). It
    # returns a :class:`ReqHandle` we use later to ask questions about
    # the request.
    # ----------------------------------------------------------------
    r1 = t.start_req(prompt_len=12, max_new_tokens=4)
    assert isinstance(r1, ReqHandle)
    assert r1.rid == "scripted-0"  # rids auto-assigned in submission order

    # Before any ``yield``, the scheduler has not run another event-loop
    # iteration, so it has not pulled r1 from the in-process queue yet.
    # ``ReqHandle.status`` reports ``"unknown"`` — "the scheduler does
    # not currently see this rid in any of its structures".
    assert r1.status == "unknown"

    # ----------------------------------------------------------------
    # Part 2 — ``yield`` advances the scheduler by one iteration.
    #
    # Bare ``yield`` is the single step primitive. It returns control
    # from this generator to the scheduler, which executes exactly one
    # event-loop iteration (one ``recv_requests`` call plus whatever
    # else fits in that iteration). On the next iteration the
    # ``_yield_to_script`` hook re-enters this generator at the
    # statement after the yield. Test logic between yields is plain
    # Python and runs synchronously inside the scheduler subprocess on
    # the driver rank (pp_rank == 0 and tp_rank == 0 and
    # attn_cp_rank == 0).
    # ----------------------------------------------------------------
    yield

    # After one yield, the scheduler has pulled r1 from the queue and
    # placed it in its waiting or running structures.
    assert r1.status in ("waiting", "running"), (
        f"after one yield the scheduler should know about r1; got {r1.status!r}"
    )

    # ----------------------------------------------------------------
    # Part 3 — Multiple concurrent requests.
    #
    # Submit two more, with different shapes. They become visible to
    # the scheduler on subsequent yields.
    # ----------------------------------------------------------------
    r2 = t.start_req(prompt_len=6, max_new_tokens=2)
    r3 = t.start_req(prompt_len=20, max_new_tokens=8)
    assert r2.rid == "scripted-1"
    assert r3.rid == "scripted-2"

    yield  # scheduler picks up r2 and r3
    yield  # one more step gives them a chance to enter running

    for r in (r1, r2, r3):
        # In v0 ``status`` is one of {"waiting", "running", "unknown"}.
        # "unknown" covers both "not yet seen" and "already finished
        # and dropped from running_batch" — v0 deliberately does not
        # distinguish those two states.
        assert r.status in ("waiting", "running", "unknown")

    # ----------------------------------------------------------------
    # Part 4 — The ``while not <cond>: yield`` idiom.
    #
    # ScriptedRuntime intentionally does not expose a ``run_until``
    # helper; idiomatic scripts use plain Python loops with ``yield``
    # in the body. This makes step semantics explicit at the call
    # site and avoids hiding the iteration count.
    #
    # The pattern is: cap the iteration count so a stuck scheduler
    # doesn't hang the test forever.
    # ----------------------------------------------------------------
    max_steps = 200
    for _ in range(max_steps):
        if r1.status in ("running", "unknown"):
            break
        yield
    else:
        raise AssertionError(
            f"r1 never reached running/finished after {max_steps} steps"
        )

    # ----------------------------------------------------------------
    # Part 5 — Construct a :class:`ReqHandle` for an arbitrary rid.
    #
    # ReqHandle is a public frozen dataclass; you can build one for
    # any rid (e.g. to assert "the scheduler must not know about
    # this"). Status lookup is just a query against the scheduler's
    # waiting/running structures, no side effects. Normal tests just
    # hold the handles returned by ``start_req``; this path is for
    # advanced scenarios.
    # ----------------------------------------------------------------
    bogus = ReqHandle(rid="never-submitted-rid", runtime=t)
    assert bogus.status == "unknown"

    # ----------------------------------------------------------------
    # Part 6 — Clean shutdown.
    #
    # When this generator returns (here implicitly), every rank in
    # the scheduler subprocess raises ScriptedRuntimeFinished after a
    # cross-rank cpu broadcast of (done=True, exc=None). The scheduler
    # subprocesses exit with code 0, the test process unblocks from
    # ``wait_for_completion``, and ``execute_scripted_runtime``
    # returns normally. No SIGQUIT, no leftover Engine state.
    #
    # If you raise from this generator (e.g. an ``assert`` fails),
    # the same machinery captures the traceback, writes it to a temp
    # file, and ``execute_scripted_runtime`` re-raises an
    # ``AssertionError`` on the caller side carrying that text.
    # ----------------------------------------------------------------


class TestScriptedRuntimeDemo(CustomTestCase):
    """Single test that walks through every feature for didactic value."""

    def test_demo(self):
        execute_scripted_runtime(
            _demo_script,
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            tp_size=1,
            dp_size=1,
            pp_size=1,
            disable_overlap_schedule=True,
            disable_cuda_graph=True,
        )


if __name__ == "__main__":
    unittest.main()
