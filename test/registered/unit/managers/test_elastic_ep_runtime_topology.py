import unittest
from datetime import timedelta
from unittest.mock import patch

from sglang.srt.elastic_ep.runtime_topology import (
    ElasticEPRecoveryRequiredError,
    RuntimeTopology,
    commit_runtime_topology,
    get_runtime_topology,
    probe_runtime_topology,
    publish_runtime_topology,
    validate_append_candidate,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Store:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value

    def check(self, keys):
        return all(key in self.values for key in keys)

    def get(self, key):
        return self.values[key]


def _topology(*, effective: int = 8, maximum: int = 8) -> RuntimeTopology:
    return RuntimeTopology(
        runtime_instance_id="runtime-1",
        initial_ep_size=8,
        allocation_width=4,
        effective_ep_size=effective,
        max_committed_ep_size=maximum,
    )


class TestElasticEPRuntimeTopology(unittest.TestCase):
    def setUp(self):
        self.store = _Store()
        self.store_patch = patch(
            "sglang.srt.elastic_ep.runtime_topology.get_global_tcp_store",
            return_value=self.store,
        )
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)

    def test_publish_and_read_runtime_topology(self):
        publish_runtime_topology(_topology())

        self.assertEqual(get_runtime_topology(), _topology())

    def test_commit_preserves_maximum_after_shrink(self):
        publish_runtime_topology(_topology(effective=12, maximum=12))

        committed = commit_runtime_topology(8)

        self.assertEqual(committed.effective_ep_size, 8)
        self.assertEqual(committed.max_committed_ep_size, 12)

    def test_commit_advances_maximum_after_growth(self):
        publish_runtime_topology(_topology())

        committed = commit_runtime_topology(12)

        self.assertEqual(committed.effective_ep_size, 12)
        self.assertEqual(committed.max_committed_ep_size, 12)

    def test_next_unused_offset_is_appendable(self):
        validate_append_candidate(
            _topology(),
            rank_offset=8,
            allocation_width=4,
            initial_ep_size=8,
        )

    def test_previously_occupied_offset_requires_recovery(self):
        with self.assertRaisesRegex(
            ElasticEPRecoveryRequiredError,
            "recovery mode is required",
        ):
            validate_append_candidate(
                _topology(effective=8, maximum=12),
                rank_offset=8,
                allocation_width=4,
                initial_ep_size=8,
            )

    def test_probe_prefers_dist_init_address_and_retries_connection(self):
        publish_runtime_topology(_topology())

        with (
            patch.dict("os.environ", {"MASTER_ADDR": "192.0.2.10"}),
            patch(
                "torch.distributed.TCPStore",
                side_effect=[OSError("not ready"), self.store],
            ) as tcp_store,
            patch("sglang.srt.elastic_ep.runtime_topology.time.sleep") as sleep,
        ):
            topology = probe_runtime_topology(
                "tcp://10.0.0.1:20000",
                attempts=2,
            )

        self.assertEqual(topology, _topology())
        self.assertEqual(tcp_store.call_count, 2)
        self.assertEqual(tcp_store.call_args.kwargs["host_name"], "10.0.0.1")
        self.assertEqual(tcp_store.call_args.kwargs["timeout"], timedelta(seconds=1))
        sleep.assert_called_once_with(0.25)


if __name__ == "__main__":
    unittest.main()
