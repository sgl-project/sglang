"""Who gets turned away when DFLASH verifies a tree instead of a chain.

The tree is built by a beam walk over the candidate selector's transition lattice, and
that walk emits no per-edge `q`. Every rejection-sampling accept in the repo needs one,
so a tree run can only serve greedy requests. Two independent checks enforce that, and
this file covers both plus the cases they must *not* catch:

- `validate_dflash_request` (request admission, `dflash_utils.py`) -- the user-facing
  arm. Rejecting here aborts one request with a readable message and leaves the server
  serving.
- `DFlashWorkerV2._validate_phase1_sampling_support` (batch entry) -- the backstop for
  callers that build batches directly (unit tests, bench scripts) and never pass through
  admission. Raising here is loud by design: the accept dispatch downstream derives gamma
  from `block_size` and reads chain-shaped retrieve links, so on a tree it would
  miscompute silently rather than fail.

Both read one predicate, `dflash_tree_verify_active()`, so a request cannot be admitted
on the chain's terms and then verified as a tree.
"""

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


def _req(*, top_k: int):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(top_k=top_k),
        return_hidden_states=False,
    )


def _spec(*, tree_width):
    return mock.patch(
        _GET_SPEC,
        return_value=SimpleNamespace(speculative_dflash_tree_width=tree_width),
    )


class TestDflashTreeRejectsSampling(CustomTestCase):
    def test_sampling_request_rejected_on_a_tree(self):
        with _spec(tree_width=4):
            error = validate_dflash_request(
                _req(top_k=_SAMPLING_TOP_K), enable_overlap=False
            )

        self.assertIsNotNone(error)
        # Both escapes must be stated: one for the caller who wants this request served,
        # one for the operator who wants the whole server to sample.
        self.assertIn("temperature 0", error)
        self.assertIn("--speculative-dflash-tree-width 1", error)

    def test_batch_entry_backstops_a_non_greedy_batch(self):
        # Reached only by callers that bypass request admission. Without this the batch
        # would flow into an accept path that reads chain-shaped links off a tree.
        worker = SimpleNamespace(_use_tree_verify=True)
        batch = SimpleNamespace(sampling_info=SimpleNamespace(is_all_greedy=False))

        with self.assertRaises(ValueError) as caught:
            DFlashWorkerV2._validate_phase1_sampling_support(worker, batch)

        self.assertIn("greedy", str(caught.exception))


class TestDflashTreeAdmitsWhatItMust(CustomTestCase):
    """The three shapes the rejection must not swallow."""

    def test_greedy_request_admitted_on_a_tree(self):
        # Guards the choice of `top_k` as the judgement: keyed on temperature instead,
        # a temperature-0 request (normalized to temperature 1.0, top_k 1) reads as
        # sampling and every greedy request on a tree run would be rejected.
        with _spec(tree_width=4):
            self.assertIsNone(
                validate_dflash_request(_req(top_k=_GREEDY_TOP_K), enable_overlap=False)
            )

    def test_sampling_request_admitted_on_the_chain(self):
        # DFLASH has always served sampling requests through `_selector_sampling_accept`.
        # Only the tree lacks the per-edge q, so a predicate broadened to "is this
        # DFLASH" would silently drop a capability that exists today.
        with _spec(tree_width=1):
            self.assertIsNone(
                validate_dflash_request(
                    _req(top_k=_SAMPLING_TOP_K), enable_overlap=False
                )
            )

    def test_forced_tree_verify_rejects_sampling_at_width_one(self):
        # SGLANG_DFLASH_FORCE_TREE_VERIFY is the one way the verify *shape* decouples
        # from the resolved widths: tree_width stays 1 while the run verifies a tree. If
        # the env drops out of the predicate, this configuration serves sampling requests
        # through the greedy tree accept and silently ignores temperature and top_p.
        with _spec(tree_width=1):
            with envs.SGLANG_DFLASH_FORCE_TREE_VERIFY.override(True):
                error = validate_dflash_request(
                    _req(top_k=_SAMPLING_TOP_K), enable_overlap=False
                )

        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
