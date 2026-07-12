import time
import unittest


class FakeClient:
    def __init__(self):
        self.status_idle = False

    def get_json(self, base_url, path):
        if path == "/pd_flip/runtime_role/status":
            return {
                "success": True,
                "role": "decode",
                "status": {
                    "role": "decode",
                    "is_idle": self.status_idle,
                    "is_idle_for_flip": self.status_idle,
                },
            }
        if path == "/v1/loads?include=all":
            return {
                "loads": [
                    {
                        "num_running_reqs": 1,
                        "num_waiting_reqs": 2,
                        "disaggregation": {
                            "decode_prealloc_queue_reqs": 3,
                            "decode_transfer_queue_reqs": 4,
                            "decode_retracted_queue_reqs": 5,
                        },
                    }
                ]
            }
        if path == "/server_info":
            return {"internal_states": [{"pd_flip": {"state": "safe"}}]}
        raise AssertionError((base_url, path))


class FailedMigrationClient(FakeClient):
    def get_json(self, base_url, path):
        if path == "/pd_flip/migration/status":
            if base_url == "http://source":
                return {
                    "success": True,
                    "status": {
                        "state": "source_failed",
                        "pending_reqs": 0,
                        "failed_reqs": 1,
                        "last_error": "source transfer failed",
                    },
                }
            return {
                "success": True,
                "status": {
                    "state": "target_failed",
                    "pending_reqs": 0,
                    "failed_reqs": 4,
                    "last_error": "target req_pool full",
                },
            }
        return super().get_json(base_url, path)


class CompleteMigrationClient(FakeClient):
    def get_json(self, base_url, path):
        if path == "/pd_flip/migration/status":
            return {
                "success": True,
                "status": {
                    "state": "transferred",
                    "pending_reqs": 0,
                    "failed_reqs": 0,
                },
            }
        return super().get_json(base_url, path)


class UnrecoveredSLOMonitor:
    def collect_cluster(self, monitor_nodes):
        return type(
            "Snapshot",
            (),
            {"prefill_slo_attainment": 0.0, "decode_slo_attainment": 1.0},
        )()


class PDFlipControllerQuiesceTest(unittest.TestCase):
    def test_observe_source_quiesce_records_residual_queue_counts(self):
        from scripts.playground.disaggregation.pd_flip_controller import (
            ActionRecord,
            NodeMetrics,
            PDClusterConfig,
            PDFlipController,
            PDNode,
        )

        controller = PDFlipController(
            PDClusterConfig(
                router_url="http://router",
                nodes=[
                    PDNode(
                        name="node2",
                        worker_url="http://node2",
                        router_worker_id="http://node2",
                    )
                ],
                observation_quiesce_seconds=0.0,
            ),
            FakeClient(),
        )
        source = NodeMetrics(
            name="node2",
            worker_url="http://node2",
            router_worker_id="http://node2",
            worker_role="decode",
        )
        records = []

        response = controller._observe_source_quiesce(records, source)

        self.assertEqual(records[-1].step, "observe_source_quiesce")
        self.assertEqual(response["source_running_reqs"], 1)
        self.assertEqual(response["source_waiting_queue_reqs"], 2)
        self.assertEqual(response["source_decode_prealloc_queue_reqs"], 3)
        self.assertEqual(response["source_decode_transfer_queue_reqs"], 4)
        self.assertEqual(response["source_decode_retracted_queue_reqs"], 5)
        self.assertEqual(response["source_total_residual_reqs"], 15)

    def test_post_migration_idle_assertion_uses_distinct_bounded_stage_name(self):
        from scripts.playground.disaggregation.pd_flip_controller import (
            NodeMetrics,
            PDClusterConfig,
            PDFlipController,
            PDNode,
        )

        client = FakeClient()
        client.status_idle = True
        controller = PDFlipController(
            PDClusterConfig(
                router_url="http://router",
                nodes=[
                    PDNode(
                        name="node2",
                        worker_url="http://node2",
                        router_worker_id="http://node2",
                    )
                ],
                post_migration_idle_timeout_seconds=0.01,
                migration_poll_interval_seconds=0.001,
            ),
            client,
        )
        source = NodeMetrics(
            name="node2",
            worker_url="http://node2",
            router_worker_id="http://node2",
            worker_role="decode",
        )
        records = []

        controller._assert_source_idle_after_migration(records, source)

        self.assertEqual(records[-1].step, "post_migration_idle_assertion")

    def test_decode_migration_target_override_can_be_draining(self):
        from scripts.playground.disaggregation.pd_flip_controller import (
            NodeMetrics,
            PDClusterConfig,
            PDFlipController,
            PDNode,
        )

        controller = PDFlipController(
            PDClusterConfig(
                router_url="http://router",
                nodes=[
                    PDNode(
                        name="node2",
                        worker_url="http://node2",
                        router_worker_id="node2",
                    )
                ],
            ),
            FakeClient(),
        )
        source = NodeMetrics(
            name="node2",
            worker_url="http://node2",
            router_worker_id="node2",
            worker_role="decode",
        )
        target = NodeMetrics(
            name="node3",
            worker_url="http://node3",
            router_worker_id="node3",
            worker_role="decode",
            draining=True,
        )

        selected = controller._select_decode_migration_target(
            [source, target], source, target_name="node3"
        )

        self.assertEqual(selected.name, "node3")

    def test_two_phase_wait_raises_on_failed_migration_status(self):
        from scripts.playground.disaggregation.pd_flip_controller import (
            NodeMetrics,
            PDClusterConfig,
            PDFlipController,
            PDNode,
        )

        controller = PDFlipController(
            PDClusterConfig(
                router_url="http://router",
                nodes=[
                    PDNode(
                        name="node2",
                        worker_url="http://source",
                        router_worker_id="node2",
                    ),
                    PDNode(
                        name="node3",
                        worker_url="http://target",
                        router_worker_id="node3",
                    ),
                ],
                migration_timeout_seconds=0.01,
                migration_poll_interval_seconds=0.001,
            ),
            FailedMigrationClient(),
        )
        source = NodeMetrics(
            name="node2",
            worker_url="http://source",
            router_worker_id="node2",
            worker_role="decode",
        )
        target = NodeMetrics(
            name="node3",
            worker_url="http://target",
            router_worker_id="node3",
            worker_role="decode",
        )

        with self.assertRaisesRegex(RuntimeError, "source transfer failed"):
            controller._wait_two_phase_migration_or_recovery(
                records=[],
                source=source,
                target=target,
                slo_monitor=UnrecoveredSLOMonitor(),
                monitor_nodes=[],
                exit_threshold=0.9,
            )

    def test_two_phase_wait_holds_observation_window_after_transfer_complete(self):
        from scripts.playground.disaggregation.pd_flip_controller import (
            NodeMetrics,
            PDClusterConfig,
            PDFlipController,
            PDNode,
        )

        controller = PDFlipController(
            PDClusterConfig(
                router_url="http://router",
                nodes=[
                    PDNode(
                        name="node2",
                        worker_url="http://source",
                        router_worker_id="node2",
                    ),
                    PDNode(
                        name="node3",
                        worker_url="http://target",
                        router_worker_id="node3",
                    ),
                ],
                migration_timeout_seconds=1.0,
                migration_poll_interval_seconds=0.001,
                observation_quiesce_seconds=0.02,
            ),
            CompleteMigrationClient(),
        )
        source = NodeMetrics(
            name="node2",
            worker_url="http://source",
            router_worker_id="node2",
            worker_role="decode",
        )
        target = NodeMetrics(
            name="node3",
            worker_url="http://target",
            router_worker_id="node3",
            worker_role="decode",
        )

        started = time.monotonic()
        result = controller._wait_two_phase_migration_or_recovery(
            records=[],
            source=source,
            target=target,
            slo_monitor=UnrecoveredSLOMonitor(),
            monitor_nodes=[],
            exit_threshold=0.9,
        )

        self.assertEqual(result, "transferred")
        self.assertGreaterEqual(time.monotonic() - started, 0.015)

    def test_two_phase_commits_target_before_source_finish_then_activates(self):
        from scripts.playground.disaggregation.pd_flip_controller import (
            ActionRecord,
            NodeMetrics,
            PDClusterConfig,
            PDFlipController,
            PDNode,
        )

        controller = PDFlipController(
            PDClusterConfig(
                router_url="http://router",
                nodes=[
                    PDNode(
                        name="node2",
                        worker_url="http://source",
                        router_worker_id="node2",
                    ),
                    PDNode(
                        name="node3",
                        worker_url="http://target",
                        router_worker_id="node3",
                    ),
                ],
            ),
            FakeClient(),
        )
        source = NodeMetrics(
            name="node2",
            worker_url="http://source",
            router_worker_id="node2",
            worker_role="decode",
        )
        target = NodeMetrics(
            name="node3",
            worker_url="http://target",
            router_worker_id="node3",
            worker_role="decode",
        )

        def record_worker(records, step, node, path, payload):
            response = {"success": True, "status": {"pending_reqs": 0}}
            if step == "start_decode_migration_source":
                response["manifests"] = [{"rid": "rid-1"}]
            records.append(
                ActionRecord(
                    step=step,
                    target=node.name,
                    method="POST",
                    url=node.worker_url + path,
                    payload=payload,
                    response=response,
                    elapsed_seconds=0.0,
                )
            )
            return response

        def record_router(records, step, node, path, payload):
            records.append(
                ActionRecord(
                    step=step,
                    target="router:" + node.router_worker_id,
                    method="POST",
                    url="http://router" + path,
                    payload=payload,
                    response={"success": True},
                    elapsed_seconds=0.0,
                )
            )
            return {"success": True}

        controller._post_worker = record_worker
        controller._post_router = record_router
        controller._observe_source_quiesce = lambda records, source: {}
        controller._wait_two_phase_migration_or_recovery = lambda **kwargs: "complete"
        controller._poll_source_delta_manifests = (
            lambda records, source, session_id, rids: [{"rid": rid} for rid in rids]
        )
        controller._wait_two_phase_delta = lambda **kwargs: None
        controller._assert_source_idle_after_migration = lambda records, source: None
        controller.collect_metrics = lambda: [source, target]

        result = controller._execute_d_to_p_two_phase(
            source=source,
            target=target,
            slo_monitor=UnrecoveredSLOMonitor(),
            enter_threshold=0.9,
            exit_threshold=0.9,
            commit_threshold=0.9,
        )

        self.assertTrue(result.success)
        steps = [record.step for record in result.actions]
        self.assertLess(
            steps.index("commit_decode_migration_target"),
            steps.index("finish_decode_migration_source"),
        )
        self.assertLess(
            steps.index("finish_decode_migration_source"),
            steps.index("activate_decode_migration_target"),
        )


if __name__ == "__main__":
    unittest.main()
