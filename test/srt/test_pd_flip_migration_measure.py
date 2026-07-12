import unittest


class PDFlipMigrationMeasureTest(unittest.TestCase):
    def test_build_timeline_detects_migration_link_stages(self):
        from scripts.playground.disaggregation.pd_flip_migration_measure import (
            build_timeline,
        )

        events = [
            {
                "event_type": "router_workers",
                "ts_mono": 10.0,
                "workers": [
                    {
                        "name": "node2",
                        "url": "http://node2",
                        "role": "decode",
                        "draining": True,
                        "active_load": 2,
                    }
                ],
            },
            {
                "event_type": "worker_status",
                "ts_mono": 10.5,
                "node": "node2",
                "pd_flip": {
                    "current_role": "decode",
                    "admission_paused": True,
                },
            },
            {
                "event_type": "migration_status",
                "ts_mono": 11.0,
                "node": "node2",
                "status": {
                    "role": "source",
                    "state": "source_started",
                    "session_id": "pd-flip-node2-to-node3",
                    "pending_reqs": 1,
                    "transferred_reqs": 0,
                    "failed_reqs": 0,
                },
            },
            {
                "event_type": "migration_status",
                "ts_mono": 11.2,
                "node": "node3",
                "status": {
                    "role": "target",
                    "state": "target_prepared",
                    "session_id": "pd-flip-node2-to-node3",
                    "pending_reqs": 1,
                    "transferred_reqs": 0,
                    "failed_reqs": 0,
                },
            },
            {
                "event_type": "migration_status",
                "ts_mono": 12.0,
                "node": "node2",
                "status": {
                    "role": "source",
                    "state": "source_transferred",
                    "session_id": "pd-flip-node2-to-node3",
                    "pending_reqs": 0,
                    "transferred_reqs": 1,
                    "failed_reqs": 0,
                },
            },
            {
                "event_type": "router_workers",
                "ts_mono": 13.0,
                "workers": [
                    {
                        "name": "node2",
                        "url": "http://node2",
                        "role": "prefill",
                        "draining": False,
                        "active_load": 0,
                    }
                ],
            },
        ]

        timeline = build_timeline(events)
        stages = [item["stage"] for item in timeline]

        self.assertEqual(
            stages,
            [
                "router_source_drained",
                "source_admission_paused",
                "source_migration_started",
                "target_migration_prepared",
                "kv_transfer_first_progress",
                "kv_transfer_complete",
                "source_role_committed",
                "cleanup_router_undrain",
            ],
        )
        self.assertEqual(timeline[0]["node"], "node2")
        self.assertEqual(timeline[4]["session_id"], "pd-flip-node2-to-node3")

    def test_label_request_impact_uses_migration_window(self):
        from scripts.playground.disaggregation.pd_flip_migration_measure import (
            label_request_impact,
        )

        timeline = [
            {"stage": "source_migration_started", "ts_mono": 20.0},
            {"stage": "migration_abort_or_failed", "ts_mono": 30.0},
        ]
        metrics = [
            {"request_id": "before", "start_monotonic": 19.0, "end_monotonic": 19.5},
            {"request_id": "overlap", "start_monotonic": 19.5, "end_monotonic": 20.5},
            {"request_id": "during", "start_monotonic": 21.0, "end_monotonic": 22.0},
            {"request_id": "after", "start_monotonic": 31.0, "end_monotonic": 32.0},
        ]

        rows = label_request_impact(metrics, timeline)

        self.assertEqual(
            [(row["request_id"], row["migration_phase"]) for row in rows],
            [
                ("before", "before_migration"),
                ("overlap", "overlaps_migration"),
                ("during", "during_migration"),
                ("after", "after_migration"),
            ],
        )
        overlap = [row for row in rows if row["request_id"] == "overlap"][0]
        self.assertTrue(overlap["active_during_migration"])


if __name__ == "__main__":
    unittest.main()
