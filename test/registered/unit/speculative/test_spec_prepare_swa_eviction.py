"""spec_prepare_for_decode must run SWA eviction on the dflash-family path.

The dflash-family branch (DFLASH, DSPARK) historically skipped both
``maybe_evict_swa`` and the ``decode_batch_idx`` tick that
``eagle_prepare_for_decode`` performs. On hybrid-SWA models that retained
SWA KV for every generated token, so a single request exhausted the SWA pool
after ``swa_full_tokens_ratio * max_total_num_tokens`` generated tokens and
was retracted by the scheduler. Pool sizing assumes the eviction runs
(``pool_configurator``: ``trailing_tokens = sliding_window +
eviction_interval * draft_tokens + page_size``).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_batch(spec_is_dflash_family: bool):
    batch = MagicMock()
    batch.spec_algorithm.is_dflash_family.return_value = spec_is_dflash_family
    req = SimpleNamespace(decode_batch_idx=0)
    batch.reqs = [req]
    return batch, req


class TestSpecPrepareSwaEviction(CustomTestCase):
    def _run(self, batch):
        from sglang.srt.speculative import spec_utils

        with patch.object(
            spec_utils, "mamba_extra_buffer_lazy_enabled", return_value=False
        ):
            spec_utils.spec_prepare_for_decode(batch)

    def test_dflash_family_evicts_and_ticks(self):
        batch, req = _make_batch(spec_is_dflash_family=True)
        self._run(batch)
        batch.maybe_evict_swa.assert_called_once_with()
        self.assertEqual(req.decode_batch_idx, 1)
        batch.spec_info.prepare_for_decode.assert_called_once_with(batch)

    def test_tick_advances_every_iteration(self):
        """decode_batch_idx is a clock, not a flag: it must keep advancing so
        the SWA leaf-lock release gate (decode_batch_idx >= sliding_window_size)
        can fire."""
        batch, req = _make_batch(spec_is_dflash_family=True)
        self._run(batch)
        self._run(batch)
        self.assertEqual(req.decode_batch_idx, 2)
        self.assertEqual(batch.maybe_evict_swa.call_count, 2)

    def test_dflash_family_evicts_before_tick(self):
        """The overlap-scheduler gate (decode_batch_idx >= 1) must see the
        pre-tick value, exactly as in eagle_prepare_for_decode."""
        batch, req = _make_batch(spec_is_dflash_family=True)
        seen_idx = []
        batch.maybe_evict_swa.side_effect = lambda: seen_idx.append(
            req.decode_batch_idx
        )
        self._run(batch)
        self.assertEqual(seen_idx, [0])
        self.assertEqual(req.decode_batch_idx, 1)

    def test_eagle_path_unchanged(self):
        batch, req = _make_batch(spec_is_dflash_family=False)
        with patch(
            "sglang.srt.speculative.eagle_utils.eagle_prepare_for_decode"
        ) as eagle_prep:
            self._run(batch)
        eagle_prep.assert_called_once_with(batch)
        # The dflash-branch tick must not run on the eagle path.
        self.assertEqual(req.decode_batch_idx, 0)
        batch.spec_info.prepare_for_decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
