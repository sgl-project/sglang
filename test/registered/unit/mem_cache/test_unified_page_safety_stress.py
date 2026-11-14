"""A fast, seeded slice of the unified pool's page-safety stress harness
(`sglang.test.unified_page_safety_harness`), checked after every operation.
The long sweeps are `test_unified_page_safety_sweep*.py`."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_page_safety_harness import replay_case, run_seed

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Seeds of generated chains (mixed page sizes, one or two floating windows,
# decode context parallelism on some) whose sequences run in a few seconds.
_SEEDS = (0, 1, 2, 3, 4, 5)
_OPS = 250


class TestPageSafetyStress(CustomTestCase):
    def test_seeded_sequences_keep_pages_safe(self):
        for seed in _SEEDS:
            with self.subTest(seed=seed):
                result = run_seed(seed, _OPS)
                self.assertIsNone(
                    result.failure,
                    f"seed {seed} failed at operation {result.failed_at}: "
                    f"{result.failure}",
                )
                self.assertGreater(result.counts.get("forwards", 0), 0)


# Operation sequences, shrunk from failing seeds, that failed before their fix:
# each name says which defect it replays.
_RECORDED = {
    "a_send_moved_out_of_the_owner_end": {
        "cfg": {
            "page_size": 2,
            "dcp": 2,
            "owner_grows": "up",
            "full_layers": 2,
            "window_layers": [2, 1],
            "window_lens": [8, 16],
            "full_pages": 27,
            "window_pages": [12, 16],
            "state_slots": 7,
            "max_reqs": 5,
            "max_context_pages": 5,
        },
        "ops": [
            ["step", [[[11, 8], [23, 7]], 0, [], []]],
            ["step", [[[11, 4], [26, 1], [27, 4]], 1, [], []]],
            ["retire", [[], 0.49687760993169083]],
            ["retire", [[], 0.866761456642786]],
            ["send", [27]],
            ["relief", [0, 0.5341272740121392]],
            ["state_take", [11]],
        ],
    },
    "a_send_moved_out_of_the_owner_end_under_a_receive": {
        "cfg": {
            "page_size": 1,
            "dcp": 1,
            "owner_grows": "down",
            "full_layers": 3,
            "window_layers": [1, 1],
            "window_lens": [1, 2],
            "full_pages": 21,
            "window_pages": [8, 11],
            "state_slots": 5,
            "max_reqs": 7,
            "max_context_pages": 6,
        },
        "ops": [
            ["load", [2, 1]],
            ["step", [[[1, 2]], 0, [], []]],
            ["quiesce", [False]],
            ["step", [[[3, 3]], 0, [], [0.7162303010832193]]],
            ["step", [[[1, 1], [3, 1], [4, 1], [5, 2]], 0, [], []]],
            ["retire", [[], 0.9018586352193212]],
            ["retire", [[], 0.1515059929401147]],
            ["state_take", [5]],
            ["send", [4]],
            ["receive", [8, 1, True]],
            ["free", [3]],
            ["relief", [2, 0.24422031032544633]],
            ["step", [[[1, 1], [2, 1], [5, 1]], 0, [], [0.2567713206416967]]],
        ],
    },
    "a_send_moved_out_of_the_owner_end_beside_a_host_load": {
        "cfg": {
            "page_size": 4,
            "dcp": 2,
            "owner_grows": "down",
            "full_layers": 2,
            "window_layers": [2],
            "window_lens": [32],
            "full_pages": 26,
            "window_pages": [12],
            "state_slots": 9,
            "max_reqs": 5,
            "max_context_pages": 6,
        },
        "ops": [
            ["step", [[[1, 10]], 1, [], []]],
            ["step", [[[1, 1], [2, 1], [3, 15]], 0, [], []]],
            ["load", [7, 3]],
            ["retire", [[2], 0.06163377208217069]],
            ["load", [15, 2]],
            ["retire", [[11], 0.31208773243857435]],
            ["load_done", [15]],
            ["step", [[[15, 1], [20, 13]], 0, [], []]],
            ["free", [15]],
            ["load", [23, 2]],
            ["retire", [[], 0.18313110325674598]],
            ["send", [20]],
            ["relief", [3, 0.34973915688825163]],
            ["load_done", [7]],
            ["load_done", [23]],
            ["step", [[[7, 10], [18, 1], [23, 8]], 0, [], []]],
        ],
    },
    "held_receive_rows_a_relief_moves_under_dcp": {
        "cfg": {
            "page_size": 1,
            "dcp": 2,
            "owner_grows": "up",
            "full_layers": 3,
            "window_layers": [2],
            "window_lens": [4],
            "full_pages": 23,
            "window_pages": [15],
            "state_slots": 5,
            "max_reqs": 5,
            "max_context_pages": 4,
        },
        "ops": [
            ["receive", [5, 3, False]],
            ["step", [[[8, 1]], 0, [], []]],
            ["publish", [5]],
        ],
    },
    "held_receive_rows_a_step_moves_under_dcp": {
        "cfg": {
            "page_size": 2,
            "dcp": 2,
            "owner_grows": "down",
            "full_layers": 3,
            "window_layers": [2, 2],
            "window_lens": [4, 16],
            "full_pages": 18,
            "window_pages": [14, 16],
            "state_slots": 8,
            "max_reqs": 7,
            "max_context_pages": 5,
        },
        "ops": [
            ["step", [[[1, 8]], 0, [], []]],
            ["receive", [2, 3, False]],
            ["step", [[[4, 12]], 0, [], []]],
            ["step", [[[1, 1], [5, 12]], 0, [], []]],
            ["step", [[[1, 7], [4, 1], [5, 1]], 0, [], []]],
        ],
    },
    "a_partly_waited_fence_split_by_a_finished_copy": {
        "cfg": {
            "page_size": 1,
            "dcp": 2,
            "owner_grows": "up",
            "full_layers": 3,
            "window_layers": [1, 1],
            "window_lens": [2, 6],
            "full_pages": 21,
            "window_pages": [14, 10],
            "state_slots": 5,
            "max_reqs": 7,
            "max_context_pages": 4,
        },
        "ops": [
            ["load", [12, 1]],
            ["load_done", [12]],
            ["backup", [12]],
            ["receive", [13, 1, False]],
            ["publish", [13]],
            ["step", [[[11, 1], [12, 2], [17, 4]], 0, [], []]],
            ["step", [[[11, 4], [12, 4]], 0, [], []]],
            ["retire", [[17], 0.8763871127701541]],
            ["retire", [[11], 0.8573478823888919]],
            ["step", [[[20, 3]], 0, [], []]],
            ["receive_commit", [13]],
            ["step", [[[12, 1], [21, 4]], 0, [], []]],
            [
                "step",
                [[[12, 1], [13, 1], [20, 1], [21, 1]], 2, [], [0.7336311838444295]],
            ],
            ["retire", [[], 0.9057646743834589]],
            ["retire", [[], 0.10334339660361136]],
            ["retire", [[12], 0.9177787525360096]],
            ["backup_done", [12]],
            ["load", [23, 3]],
            ["step", [[[12, 1], [13, 1], [20, 3], [21, 1], [24, 6]], 0, [], []]],
            ["backup", [20]],
            ["free", [13]],
            ["receive", [29, 1, False]],
            ["load_done", [23]],
            ["state_take", [20]],
            ["send", [21]],
            ["step", [[[12, 1], [20, 3]], 0, [], [0.022378141473617497]]],
            ["publish", [29]],
            ["send_release", [21, True]],
            ["retire", [[20], 0.9728390607199402]],
            ["step", [[[12, 1], [18, 1], [20, 1], [23, 1]], 0, [], []]],
            ["load", [35, 1]],
            ["retire", [[], 0.39433003234058395]],
            ["free", [23]],
            ["step", [[[12, 1], [39, 4]], 0, [], []]],
            ["retire", [[39], 0.3787138866800478]],
            ["load", [40, 3]],
            ["state_take", [18]],
            ["step", [[[12, 1], [18, 1], [20, 1], [42, 5]], 2, [], []]],
            ["backup", [20]],
            ["tick", [1, 0.84258061044427]],
            ["receive_commit", [29]],
            ["load_done", [35]],
            ["free", [29]],
            ["step", [[[20, 1]], 0, [], []]],
            ["step", [[[12, 1], [18, 2], [35, 3]], 0, [], [0.8636255982848404]]],
        ],
    },
    "a_float_placed_afresh_over_its_settled_holes": {
        "cfg": {
            "page_size": 4,
            "dcp": 1,
            "owner_grows": "up",
            "full_layers": 2,
            "window_layers": [2],
            "window_lens": [16],
            "full_pages": 22,
            "window_pages": [14],
            "state_slots": 7,
            "max_reqs": 5,
            "max_context_pages": 7,
        },
        "ops": [
            ["step", [[[1, 4]], 0, [], [0.9333325694828937]]],
            ["step", [[[1, 1]], 0, [], [0.6773393106768724]]],
            ["retire", [[1], 0.5362071565370583]],
            ["load", [2, 3]],
        ],
    },
}


class TestRecordedSequences(CustomTestCase):
    def test_recorded_sequences_replay_clean(self):
        for name, case in _RECORDED.items():
            with self.subTest(name):
                failure = replay_case(case)
                self.assertIsNone(failure, f"{name}: {failure}")


if __name__ == "__main__":
    unittest.main()
