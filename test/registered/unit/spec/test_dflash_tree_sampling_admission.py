"""Test DFLASH tree sampling admission."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# `dflash_tree_verify_active` imports this lazily inside the function body, so patching
# the definition site is what takes effect.
_GET_SPEC = "sglang.srt.runtime_context.get_spec"

# `SamplingParams.normalize` maps `temperature < eps` to `top_k = 1` and leaves an
# unset top_k at TOP_K_ALL (1 << 30), which is why the admission check reads top_k
# rather than temperature: it is the one field that is already normalized.
_GREEDY_TOP_K = 1
_SAMPLING_TOP_K = 1 << 30


def _req(*, top_k: int, aborted: bool = False):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(top_k=top_k),
        return_hidden_states=False,
        # What `set_finish_with_abort` leaves behind: a rejected request still carries its
        # original sampling_params into one forward pass.
        to_finish=object() if aborted else None,
        finished=lambda: False,
    )


def _batch(*reqs):
    return SimpleNamespace(
        reqs=list(reqs),
        sampling_info=SimpleNamespace(
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs)
        ),
    )


def _spec(*, tree_width, algorithm="DFLASH"):
    return mock.patch(
        _GET_SPEC,
        return_value=SimpleNamespace(
            speculative_algorithm=algorithm,
            speculative_dflash_tree_width=tree_width,
        ),
    )


class TestDflashTreeRejectsSampling(CustomTestCase):
    def test_sampling_request_rejected_on_a_tree(self):
        with _spec(tree_width=4):
            error = validate_dflash_request(
                _req(top_k=_SAMPLING_TOP_K), enable_overlap=False
            )

        self.assertIsNotNone(error)
        self.assertIn("temperature 0", error)
        self.assertIn("--speculative-dflash-tree-width 1", error)

    def test_batch_entry_backstops_a_non_greedy_batch(self):
        worker = SimpleNamespace(_use_tree_verify=True)
        batch = _batch(_req(top_k=_SAMPLING_TOP_K))

        with self.assertRaises(ValueError) as caught:
            DFlashWorkerV2._validate_phase1_sampling_support(worker, batch)

        self.assertIn("greedy", str(caught.exception))

    def test_an_aborted_sampling_request_does_not_kill_the_worker(self):
        """A rejected request must not take the scheduler down with it.

        `validate_dflash_request` rejects a sampling request, but `set_finish_with_abort`
        keeps it runnable -- it shortens origin_input_ids to one token instead of dropping
        the request, and its sampling_params stay non-greedy. Raising here runs inside
        `run_batch`, so it would turn a request-level 400 into a scheduler SIGQUIT.
        """
        # A tree worker always has a selector (tree width > 1 requires a DFlash 2
        # checkpoint). The selector-enabled path handles the batch after the aborted
        # request is exempted from the tree admission backstop.
        worker = SimpleNamespace(
            _use_tree_verify=True, selector=object(), _selector_sampling_enabled=True
        )
        batch = _batch(_req(top_k=_SAMPLING_TOP_K, aborted=True))

        DFlashWorkerV2._validate_phase1_sampling_support(worker, batch)

    def test_a_live_sampling_request_beside_an_aborted_one_still_raises(self):
        # The abort exemption is per request, not "any abort disarms the check".
        worker = SimpleNamespace(_use_tree_verify=True)
        batch = _batch(
            _req(top_k=_SAMPLING_TOP_K, aborted=True),
            _req(top_k=_SAMPLING_TOP_K),
        )

        with self.assertRaises(ValueError):
            DFlashWorkerV2._validate_phase1_sampling_support(worker, batch)


class TestDflashTreeAdmitsWhatItMust(CustomTestCase):
    """Cases that tree admission must not reject."""

    def test_greedy_request_admitted_on_a_tree(self):
        with _spec(tree_width=4):
            self.assertIsNone(
                validate_dflash_request(_req(top_k=_GREEDY_TOP_K), enable_overlap=False)
            )

    def test_sampling_request_admitted_on_the_chain(self):
        with _spec(tree_width=1):
            self.assertIsNone(
                validate_dflash_request(
                    _req(top_k=_SAMPLING_TOP_K), enable_overlap=False
                )
            )

    def test_forced_tree_verify_rejects_sampling_at_width_one(self):
        with _spec(tree_width=1):
            with envs.SGLANG_DFLASH_FORCE_TREE_VERIFY.override(True):
                error = validate_dflash_request(
                    _req(top_k=_SAMPLING_TOP_K), enable_overlap=False
                )

        self.assertIsNotNone(error)

    def test_force_tree_verify_is_ignored_by_dspark(self):
        with _spec(tree_width=1, algorithm="DSPARK"):
            with envs.SGLANG_DFLASH_FORCE_TREE_VERIFY.override(True):
                self.assertIsNone(
                    validate_dflash_request(
                        _req(top_k=_SAMPLING_TOP_K), enable_overlap=False
                    )
                )


if __name__ == "__main__":
    unittest.main()
