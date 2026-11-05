"""The unified pool's page-safety stress harness over a fixed sweep of seeds
(weekly, CPU): 512 generated sequences of 400 operations and 32 of 3000, each
checked after every operation on two ranks with different completion timing
(`sglang.test.unified_page_safety_harness`). A failure names its seed and
carries its operation sequence as a case: save it to a file and replay it with
`python test/manual/mem_cache/unified_page_safety_stress.py --replay case.json`,
or shrink the seed with `--seeds <seed> --ops <ops> --minimize`.
"""

import json
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_page_safety_harness import case_to_json, run_seed

register_cpu_ci(est_time=500, stage="weekly", runner_config="cpu")

# (seeds, operations per seed): many short sequences, and a few long ones that
# reach the states only long histories build.
_SWEEP = ((range(512), 400), (range(32), 3000))


class TestPageSafetySweep(CustomTestCase):
    def test_sweep_keeps_pages_safe(self):
        failures = []
        for seeds, n_ops in _SWEEP:
            for seed in seeds:
                result = run_seed(seed, n_ops)
                if result.failure is not None:
                    case = case_to_json(result.cfg, result.ops)
                    failures.append(
                        f"seed {seed} ({n_ops} ops) failed at operation "
                        f"{result.failed_at}: {result.failure}\n"
                        f"case: {json.dumps(case)}"
                    )
        if failures:
            self.fail("\n\n".join(failures))


if __name__ == "__main__":
    unittest.main()
