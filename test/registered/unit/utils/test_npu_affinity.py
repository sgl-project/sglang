import errno
import os
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

from sglang.srt.environ import envs
from sglang.srt.utils import npu_affinity, npu_pd_affinity
from sglang.srt.utils.npu_affinity import (
    NpuAffinityApplyResult,
    NpuAffinityAssignment,
    NpuAffinityError,
    NpuAffinityThreadResult,
    NpuTopology,
    NpuTopologyEntry,
    PhysicalCore,
    _build_affinity_assignment,
    _build_npu_topology,
    _infer_unique_numa_node,
    _parse_npu_smi_topology_output,
    apply_npu_cpu_affinity,
    build_npu_affinity_report,
    build_npu_pd_affinity_assignment,
    format_cpu_list,
    log_npu_affinity_summary,
    parse_cpu_list,
    resolve_physical_npu_id,
)
from sglang.srt.utils.npu_pd_affinity import (
    PythonPdThread,
    get_pd_affinity_budget,
    make_pd_thread_binder,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


NPU_SMI_TOPOLOGY = """\
NPU0 NPU1 NPU2 NPU3 NPU4 NPU5 NPU6 NPU7 CPU Affinity
NPU0 X UB UB UB SYS SYS SYS SYS 0-95,192-287
NPU1 UB X UB UB SYS SYS SYS SYS 0-95,192-287
NPU2 UB UB X UB SYS SYS SYS SYS 0-95,192-287
NPU3 UB UB UB X SYS SYS SYS SYS 0-95,192-287
NPU4 SYS SYS SYS SYS X UB UB UB 96-191,288-383
NPU5 SYS SYS SYS SYS UB X UB UB 96-191,288-383
NPU6 SYS SYS SYS SYS UB UB X UB 96-191,288-383
NPU7 SYS SYS SYS SYS UB UB UB X 96-191,288-383

Legend:
X = Self
SYS = Path traversing PCIe and NUMA nodes.
UB = Connection traversing UB.
"""


def _make_test_topology(cores_per_node=8) -> NpuTopology:
    entries = {}
    online_cpus = set()
    for node in range(2):
        first_core = node * cores_per_node
        local_cpus = set()
        physical_cores = []
        for core_id in range(first_core, first_core + cores_per_node):
            siblings = (core_id, core_id + 2 * cores_per_node)
            local_cpus.update(siblings)
            physical_cores.append(
                PhysicalCore(
                    socket_id=node,
                    core_id=core_id,
                    logical_cpu_ids=siblings,
                )
            )
        online_cpus.update(local_cpus)
        for physical_npu_id in range(node * 4, node * 4 + 4):
            entries[physical_npu_id] = NpuTopologyEntry(
                physical_npu_id=physical_npu_id,
                raw_cpu_affinity=format_cpu_list(local_cpus),
                local_cpu_ids=frozenset(local_cpus),
                numa_node=node,
                physical_cores=tuple(physical_cores),
            )
    return NpuTopology(
        source="test",
        entries=entries,
        online_cpu_ids=frozenset(online_cpus),
    )


def _make_machine_topology() -> NpuTopology:
    return _build_npu_topology(
        _parse_npu_smi_topology_output(NPU_SMI_TOPOLOGY),
        online_cpu_ids=frozenset(range(384)),
        numa_cpu_sets={
            0: parse_cpu_list("0-95,192-287"),
            1: parse_cpu_list("96-191,288-383"),
        },
        lscpu_topology={
            cpu_id: (cpu_id % 192, (cpu_id % 192) // 96, (cpu_id % 192) // 96)
            for cpu_id in range(384)
        },
    )


class TestCpuList(unittest.TestCase):
    def test_parse_and_format(self):
        self.assertEqual(parse_cpu_list("0,2,4-7"), frozenset({0, 2, 4, 5, 6, 7}))
        self.assertEqual(format_cpu_list({0, 1, 2, 5, 7, 8}), "0-2,5,7-8")

    def test_invalid_cpu_lists(self):
        for value in ("", "-1", "4-2", "a", "1,,2", "1-2-3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_cpu_list(value)


class TestNpuSmiParser(unittest.TestCase):
    def test_parse_current_machine_shape(self):
        parsed = _parse_npu_smi_topology_output(NPU_SMI_TOPOLOGY)
        self.assertEqual(sorted(parsed), list(range(8)))
        self.assertEqual(parsed[0][1], parse_cpu_list("0-95,192-287"))
        self.assertEqual(parsed[7][1], parse_cpu_list("96-191,288-383"))

    def test_reject_unknown_topology_token(self):
        text = "NPU0 CPU Affinity\nNPU0 BAD 0-3\n"
        with self.assertRaisesRegex(NpuAffinityError, "Unsupported topology"):
            _parse_npu_smi_topology_output(text)

    def test_reject_duplicate_or_missing_npu_rows(self):
        duplicate = "NPU0 CPU Affinity\nNPU0 X 0-3\nNPU0 X 0-3\n"
        with self.assertRaisesRegex(NpuAffinityError, "Duplicate NPU"):
            _parse_npu_smi_topology_output(duplicate)

        missing = "NPU0 NPU1 CPU Affinity\nNPU0 X UB 0-3\n"
        with self.assertRaisesRegex(NpuAffinityError, "do not match"):
            _parse_npu_smi_topology_output(missing)


class TestTopologyConstruction(unittest.TestCase):
    def test_current_machine_numa_and_core_mapping(self):
        topology = _make_machine_topology()
        assignment = _build_affinity_assignment(
            topology,
            logical_npu_id=4,
            physical_npu_id=4,
            allowed_cpu_ids=frozenset(range(384)),
            requested_pcores=8,
        )
        self.assertEqual(topology.entries[0].numa_node, 0)
        self.assertEqual(topology.entries[7].numa_node, 1)
        self.assertEqual(
            assignment.physical_core_keys,
            tuple((1, i) for i in range(96, 104)),
        )
        self.assertEqual(
            assignment.logical_cpu_ids,
            tuple(range(96, 104)) + tuple(range(288, 296)),
        )

    def test_raw_affinity_missing_an_online_sibling_excludes_the_whole_core(self):
        topology = _build_npu_topology(
            {0: ("0-1,3", frozenset({0, 1, 3}))},
            online_cpu_ids=frozenset(range(4)),
            numa_cpu_sets={0: frozenset(range(4))},
            lscpu_topology={
                0: (0, 0, 0),
                1: (1, 0, 0),
                2: (0, 0, 0),
                3: (1, 0, 0),
            },
        )
        assignment = _build_affinity_assignment(
            topology,
            logical_npu_id=0,
            physical_npu_id=0,
            allowed_cpu_ids=topology.online_cpu_ids,
            requested_pcores=0,
        )
        self.assertEqual(assignment.physical_core_keys, ((0, 1),))
        self.assertEqual(assignment.logical_cpu_ids, (1, 3))

    def test_offline_sibling_is_not_required_for_a_complete_online_core(self):
        topology = _build_npu_topology(
            {0: ("0-1,3", frozenset({0, 1, 3}))},
            online_cpu_ids=frozenset({0, 1, 3}),
            numa_cpu_sets={0: frozenset(range(4))},
            lscpu_topology={
                0: (0, 0, 0),
                1: (1, 0, 0),
                2: (0, 0, 0),
                3: (1, 0, 0),
            },
        )
        assignment = _build_affinity_assignment(
            topology,
            logical_npu_id=0,
            physical_npu_id=0,
            allowed_cpu_ids=topology.online_cpu_ids,
            requested_pcores=0,
        )
        self.assertEqual(assignment.physical_core_keys, ((0, 0), (0, 1)))
        self.assertEqual(assignment.logical_cpu_ids, (0, 1, 3))

    def test_npu_cpu_set_must_belong_to_exactly_one_numa_node(self):
        with self.assertRaisesRegex(NpuAffinityError, "exactly one"):
            _infer_unique_numa_node(
                frozenset({0, 4}),
                {0: frozenset(range(4)), 1: frozenset(range(4, 8))},
            )


class TestVisibleDeviceMapping(unittest.TestCase):
    def test_visible_device_reordering(self):
        with patch.dict(
            os.environ,
            {"ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7"},
            clear=False,
        ):
            self.assertEqual(resolve_physical_npu_id(0), 4)
            self.assertEqual(resolve_physical_npu_id(3), 7)
            with self.assertRaisesRegex(NpuAffinityError, "outside"):
                resolve_physical_npu_id(4)

    def test_no_visible_device_filter(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_physical_npu_id(4), 4)

    def test_base_gpu_id_and_visible_devices_keep_physical_ownership(self):
        topology = _make_machine_topology()
        for visible_devices, runtime_ids in (
            (None, range(4, 8)),
            ("4,5,6,7", range(4)),
        ):
            environment = (
                {}
                if visible_devices is None
                else {"ASCEND_RT_VISIBLE_DEVICES": visible_devices}
            )
            with (
                self.subTest(visible_devices=visible_devices),
                patch.dict(os.environ, environment, clear=True),
                envs.SGLANG_NPU_AFFINITY_PCORES_PER_PROC.override(0),
                patch.object(
                    npu_affinity, "query_npu_smi_topology", return_value=topology
                ),
                patch.object(
                    npu_affinity,
                    "_get_allowed_cpu_ids",
                    return_value=topology.online_cpu_ids,
                ),
            ):
                for runtime_id, physical_id in zip(runtime_ids, range(4, 8)):
                    assignment = npu_affinity.resolve_npu_affinity_assignment(
                        logical_npu_id=runtime_id, emit_topology_log=False
                    )
                    self.assertEqual(assignment.logical_npu_id, runtime_id)
                    self.assertEqual(assignment.physical_npu_id, physical_id)
                    self.assertEqual(assignment.numa_node, 1)
                    self.assertEqual(assignment.slots_on_node, 4)
                    self.assertEqual(assignment.slot_index, physical_id - 4)
                    self.assertEqual(
                        assignment.physical_core_keys,
                        tuple(
                            (1, core)
                            for core in range(physical_id * 24, (physical_id + 1) * 24)
                        ),
                    )


class TestAssignment(unittest.TestCase):
    def test_384_logical_cpu_machine_assigns_24_complete_cores_per_npu(self):
        topology = _make_machine_topology()
        assignments = [
            _build_affinity_assignment(
                topology,
                logical_npu_id=npu_id,
                physical_npu_id=npu_id,
                allowed_cpu_ids=topology.online_cpu_ids,
                requested_pcores=0,
            )
            for npu_id in range(8)
        ]
        allocated = set()
        for npu_id, assignment in enumerate(assignments):
            with self.subTest(npu_id=npu_id):
                cores = range(npu_id * 24, (npu_id + 1) * 24)
                self.assertEqual(assignment.effective_pcores, 24)
                self.assertEqual(
                    assignment.physical_core_keys,
                    tuple((npu_id // 4, core) for core in cores),
                )
                self.assertEqual(
                    assignment.logical_cpu_ids,
                    tuple(cores) + tuple(core + 192 for core in cores),
                )
                self.assertEqual(len(assignment.logical_cpu_ids), 48)
                self.assertTrue(
                    set(assignment.logical_cpu_ids)
                    <= topology.entries[npu_id].local_cpu_ids
                )
                self.assertFalse(allocated & set(assignment.logical_cpu_ids))
                allocated.update(assignment.logical_cpu_ids)
        self.assertEqual(allocated, set(range(384)))

    def test_assignments_are_local_and_non_overlapping(self):
        topology = _make_test_topology()
        assignments = [
            _build_affinity_assignment(
                topology,
                logical_npu_id=npu_id,
                physical_npu_id=npu_id,
                allowed_cpu_ids=topology.online_cpu_ids,
                requested_pcores=2,
            )
            for npu_id in range(8)
        ]
        for assignment in assignments:
            entry = topology.entries[assignment.physical_npu_id]
            self.assertEqual(assignment.effective_pcores, 2)
            self.assertTrue(set(assignment.logical_cpu_ids) <= entry.local_cpu_ids)
        for index, lhs in enumerate(assignments):
            for rhs in assignments[index + 1 :]:
                self.assertFalse(
                    set(lhs.logical_cpu_ids) & set(rhs.logical_cpu_ids),
                    (lhs, rhs),
                )

    def test_zero_means_even_division_and_large_request_is_clipped(self):
        topology = _make_test_topology()
        automatic = _build_affinity_assignment(
            topology,
            logical_npu_id=0,
            physical_npu_id=0,
            allowed_cpu_ids=topology.online_cpu_ids,
            requested_pcores=0,
        )
        clipped = _build_affinity_assignment(
            topology,
            logical_npu_id=0,
            physical_npu_id=0,
            allowed_cpu_ids=topology.online_cpu_ids,
            requested_pcores=99,
        )
        self.assertEqual(automatic.effective_pcores, 2)
        self.assertEqual(clipped.effective_pcores, 2)

    def test_cpuset_missing_one_sibling_excludes_the_whole_core(self):
        topology = _make_test_topology()
        assignment = _build_affinity_assignment(
            topology,
            logical_npu_id=0,
            physical_npu_id=0,
            allowed_cpu_ids=topology.online_cpu_ids - {16},
            requested_pcores=2,
        )
        self.assertEqual(assignment.physical_core_keys, ((0, 1),))
        self.assertEqual(assignment.logical_cpu_ids, (1, 17))

    def test_cpuset_with_only_single_siblings_cannot_allocate_cores(self):
        topology = _make_test_topology()
        with self.assertRaises(NpuAffinityError):
            _build_affinity_assignment(
                topology,
                logical_npu_id=0,
                physical_npu_id=0,
                allowed_cpu_ids=frozenset(range(8)),
                requested_pcores=0,
            )

    def test_core_requests_do_not_move_the_start_of_other_npu_shares(self):
        topology = _make_test_topology()
        for requested_pcores in (0, 1, 2, 99):
            with self.subTest(requested_pcores=requested_pcores):
                assignment = _build_affinity_assignment(
                    topology,
                    logical_npu_id=1,
                    physical_npu_id=1,
                    allowed_cpu_ids=topology.online_cpu_ids,
                    requested_pcores=requested_pcores,
                )
                expected = ((0, 2),) if requested_pcores == 1 else ((0, 2), (0, 3))
                self.assertEqual(assignment.physical_core_keys, expected)

    def test_different_process_cpusets_do_not_repartition_or_overlap(self):
        topology = _make_test_topology()
        assigned_cpus = set()
        for npu_id, excluded, expected in (
            (0, {16}, ((0, 1),)),
            (1, {0, 16, 3}, ((0, 2),)),
            (2, {0, 1, 16, 17}, ((0, 4), (0, 5))),
            (3, set(), ((0, 6), (0, 7))),
        ):
            with self.subTest(npu_id=npu_id):
                assignment = _build_affinity_assignment(
                    topology,
                    logical_npu_id=npu_id,
                    physical_npu_id=npu_id,
                    allowed_cpu_ids=topology.online_cpu_ids - excluded,
                    requested_pcores=0,
                )
                self.assertEqual(assignment.physical_core_keys, expected)
                self.assertFalse(assigned_cpus & set(assignment.logical_cpu_ids))
                assigned_cpus.update(assignment.logical_cpu_ids)

    def test_exhausted_share_does_not_borrow_from_another_npu(self):
        topology = _make_test_topology()
        with self.assertRaises(NpuAffinityError):
            _build_affinity_assignment(
                topology,
                logical_npu_id=0,
                physical_npu_id=0,
                allowed_cpu_ids=topology.online_cpu_ids - {0, 1, 16, 17},
                requested_pcores=0,
            )

    def test_nondivisible_core_count_uses_disjoint_fixed_shares(self):
        topology = _make_test_topology(cores_per_node=10)
        allocated = set()
        for npu_id, expected_cores in enumerate(((0, 1), (2, 3, 4), (5, 6), (7, 8, 9))):
            assignment = _build_affinity_assignment(
                topology,
                logical_npu_id=npu_id,
                physical_npu_id=npu_id,
                allowed_cpu_ids=topology.online_cpu_ids,
                requested_pcores=0,
            )
            self.assertEqual(
                assignment.physical_core_keys,
                tuple((0, core) for core in expected_cores),
            )
            self.assertFalse(allocated & set(assignment.logical_cpu_ids))
            allocated.update(assignment.logical_cpu_ids)
        self.assertEqual(allocated, set(topology.entries[0].local_cpu_ids))

    def test_negative_pcore_setting_is_rejected(self):
        from sglang.srt.utils import npu_affinity

        with envs.SGLANG_NPU_AFFINITY_PCORES_PER_PROC.override(-1):
            with self.assertRaisesRegex(ValueError, "must be >= 0"):
                npu_affinity.resolve_npu_affinity_assignment(
                    logical_npu_id=0,
                    emit_topology_log=False,
                )


class TestApplyAffinity(unittest.TestCase):
    def setUp(self):
        self.assignment = NpuAffinityAssignment(
            logical_npu_id=0,
            physical_npu_id=0,
            numa_node=0,
            slot_index=0,
            slots_on_node=1,
            requested_pcores=1,
            effective_pcores=1,
            physical_core_keys=((0, 0),),
            logical_cpu_ids=(0, 16),
            topology_source="test",
        )

    def test_final_bind_updates_all_threads_and_reads_back(self):
        affinity_by_tid = {0: {0, 16}, 101: {0, 16}, 102: {0, 16}}

        def get_affinity(tid):
            return affinity_by_tid[tid]

        def set_affinity(tid, cpus):
            affinity_by_tid[tid] = set(cpus)

        process = SimpleNamespace(
            threads=lambda: [SimpleNamespace(id=101), SimpleNamespace(id=102)]
        )
        with (
            patch.object(
                os, "sched_getaffinity", side_effect=get_affinity, create=True
            ),
            patch.object(
                os, "sched_setaffinity", side_effect=set_affinity, create=True
            ),
            patch(
                "sglang.srt.utils.npu_affinity.psutil.Process",
                return_value=process,
            ),
        ):
            result = apply_npu_cpu_affinity(
                self.assignment,
                phase="final",
                bind_all_threads=True,
            )

        self.assertTrue(result.main_matched)
        self.assertEqual(result.threads_total, 2)
        self.assertEqual(result.threads_bound, 2)
        self.assertEqual(result.threads_failed, 0)
        self.assertEqual(result.threads_mismatched, 0)

    def test_final_restores_early_assignment_after_runtime_narrows_main_mask(self):
        affinity_by_tid = {0: set(range(32)), 101: {0}, 102: {16}}

        def set_affinity(tid, cpus):
            affinity_by_tid[tid] = set(cpus)

        process = SimpleNamespace(
            threads=lambda: [SimpleNamespace(id=101), SimpleNamespace(id=102)]
        )
        with (
            patch.object(
                os,
                "sched_getaffinity",
                side_effect=lambda tid: affinity_by_tid[tid],
                create=True,
            ),
            patch.object(
                os, "sched_setaffinity", side_effect=set_affinity, create=True
            ) as bind,
            patch.object(npu_affinity.psutil, "Process", return_value=process),
            patch.object(npu_affinity, "resolve_npu_affinity_assignment") as resolve,
        ):
            early = apply_npu_cpu_affinity(
                self.assignment, phase="early", bind_all_threads=False
            )
            affinity_by_tid[0] = {0}
            final = apply_npu_cpu_affinity(
                self.assignment, phase="final", bind_all_threads=True
            )

        resolve.assert_not_called()
        self.assertEqual(early.requested_cpu_ids, (0, 16))
        self.assertEqual(early.threads_total, 0)
        self.assertEqual(final.requested_cpu_ids, early.requested_cpu_ids)
        self.assertEqual(final.main_actual_cpu_ids, (0, 16))
        self.assertTrue(final.main_matched)
        self.assertEqual(final.threads_bound, 2)
        self.assertTrue(all(cpus == {0, 16} for cpus in affinity_by_tid.values()))
        self.assertEqual(
            bind.call_args_list,
            [
                call(0, {0, 16}),
                call(0, {0, 16}),
                call(101, {0, 16}),
                call(102, {0, 16}),
            ],
        )

    def test_thread_exit_permission_failure_and_readback_mismatch_are_reported(self):
        affinity_by_tid = {0: {0, 16}, 101: {0, 16}, 104: {0}}

        def set_affinity(tid, cpus):
            if tid == 102:
                raise ProcessLookupError(errno.ESRCH, "thread exited before binding")
            if tid == 103:
                raise PermissionError(errno.EPERM, "binding denied")
            if tid != 104:
                affinity_by_tid[tid] = set(cpus)

        def get_affinity(tid):
            if tid == 105:
                raise OSError(errno.ESRCH, "thread exited before readback")
            return affinity_by_tid[tid]

        process = SimpleNamespace(
            threads=lambda: [SimpleNamespace(id=tid) for tid in range(101, 106)]
        )
        with (
            patch.object(
                os, "sched_getaffinity", side_effect=get_affinity, create=True
            ),
            patch.object(
                os, "sched_setaffinity", side_effect=set_affinity, create=True
            ),
            patch.object(npu_affinity.psutil, "Process", return_value=process),
            self.assertLogs(npu_affinity.logger, level="WARNING") as logs,
        ):
            result = apply_npu_cpu_affinity(
                self.assignment, phase="final", bind_all_threads=True
            )
        self.assertTrue(result.main_matched)
        self.assertEqual(result.threads_total, 5)
        self.assertEqual(result.threads_bound, 2)
        self.assertEqual(result.threads_exited, 2)
        self.assertEqual(result.threads_failed, 1)
        self.assertEqual(result.threads_mismatched, 1)
        output = logs.records[0].getMessage().strip().splitlines()
        self.assertEqual(len(output), 4)  # Two content lines and two separators.
        self.assertIn("status=INCOMPLETE", output[1])
        self.assertIn("threads_matched=1/5", output[2])
        self.assertIn("threads_failed=1", output[2])
        self.assertIn("threads_mismatched=1", output[2])
        self.assertIn("threads_exited=2", output[2])

    def test_thread_enumeration_failure_is_reported(self):
        def denied_threads():
            raise PermissionError("denied")

        process = SimpleNamespace(threads=denied_threads)
        with (
            patch.object(os, "sched_getaffinity", return_value={0, 16}, create=True),
            patch.object(os, "sched_setaffinity", create=True),
            patch.object(npu_affinity.psutil, "Process", return_value=process),
            self.assertLogs(npu_affinity.logger, level="WARNING"),
        ):
            result = apply_npu_cpu_affinity(
                self.assignment, phase="final", bind_all_threads=True
            )
        self.assertEqual(result.threads_total, 0)
        self.assertEqual(result.threads_bound, 0)
        self.assertEqual(result.threads_failed, 1)

    def test_main_bind_failure_retains_npu_error_context(self):
        with (
            patch.object(os, "sched_getaffinity", return_value={0, 16}, create=True),
            patch.object(
                os,
                "sched_setaffinity",
                side_effect=PermissionError("denied"),
                create=True,
            ),
            self.assertRaises(NpuAffinityError) as raised,
        ):
            apply_npu_cpu_affinity(
                self.assignment, phase="final", bind_all_threads=False
            )
        self.assertEqual(raised.exception.stage, "apply_cpu_affinity")
        self.assertEqual(raised.exception.logical_npu_id, 0)
        self.assertEqual(raised.exception.physical_npu_id, 0)

    def test_main_readback_mismatch_is_not_reported_as_success(self):
        with (
            patch.object(os, "sched_getaffinity", return_value={0}, create=True),
            patch.object(os, "sched_setaffinity", create=True),
            self.assertLogs(npu_affinity.logger, level="INFO") as logs,
        ):
            result = apply_npu_cpu_affinity(
                self.assignment, phase="final", bind_all_threads=False
            )
        self.assertFalse(result.main_matched)
        self.assertEqual(result.requested_cpu_ids, (0, 16))
        self.assertEqual(result.main_actual_cpu_ids, (0,))
        banner = "\n".join(logs.output)
        self.assertIn("main_matched=False", banner)
        self.assertNotIn("SUCCESS", banner)

    def test_emit_log_false_suppresses_success_banner(self):
        with (
            patch.object(os, "sched_getaffinity", return_value={0, 16}, create=True),
            patch.object(os, "sched_setaffinity", create=True),
            patch.object(npu_affinity.logger, "info") as info,
        ):
            result = apply_npu_cpu_affinity(
                self.assignment, phase="early", bind_all_threads=False, emit_log=False
            )
        self.assertTrue(result.main_matched)
        info.assert_not_called()

    def test_topology_and_result_logs_include_raw_affinity_and_multiline_banner(self):
        topology = _make_test_topology()
        affinity = set(topology.online_cpu_ids)

        def set_affinity(_tid, cpus):
            affinity.clear()
            affinity.update(cpus)

        with (
            patch.dict(os.environ, {}, clear=True),
            envs.SGLANG_NPU_AFFINITY_PCORES_PER_PROC.override(0),
            patch.object(npu_affinity, "query_npu_smi_topology", return_value=topology),
            patch.object(
                npu_affinity,
                "_get_allowed_cpu_ids",
                return_value=topology.online_cpu_ids,
            ),
            patch.object(npu_affinity, "_LOGGED_TOPOLOGY_SCOPES", set()),
            patch.object(
                os,
                "sched_getaffinity",
                side_effect=lambda _tid: set(affinity),
                create=True,
            ),
            patch.object(
                os, "sched_setaffinity", side_effect=set_affinity, create=True
            ),
            self.assertLogs(npu_affinity.logger, level="INFO") as logs,
        ):
            assignment = npu_affinity.resolve_npu_affinity_assignment(
                logical_npu_id=0, emit_topology_log=True
            )
            apply_npu_cpu_affinity(assignment, phase="early", bind_all_threads=False)
            with patch.object(
                npu_affinity.psutil,
                "Process",
                return_value=SimpleNamespace(threads=lambda: [SimpleNamespace(id=101)]),
            ):
                apply_npu_cpu_affinity(assignment, phase="final", bind_all_threads=True)
        output = "\n".join(logs.output)
        self.assertIn("raw_cpu_affinity=0-7,16-23", output)
        self.assertIn("===============", output)
        self.assertIn("NPU CPU AFFINITY", output)
        self.assertIn("phase=early", output)
        self.assertIn("phase=final", output)
        self.assertIn("status=SUCCESS", output)
        self.assertIn("threads=main-only", output)
        self.assertIn("threads_matched=1/1", output)
        self.assertIn("threads_failed=0", output)
        result_blocks = [
            record.getMessage().strip().splitlines()
            for record in logs.records
            if "NPU CPU AFFINITY RESULT" in record.getMessage()
        ]
        self.assertEqual(len(result_blocks), 2)
        for lines in result_blocks:
            self.assertEqual(len(lines), 4)  # Two content lines and two separators.
            self.assertIn("raw_cpu_affinity=0-7,16-23", lines[1])
            self.assertIn("main_cpu_mask=0-1,16-17 assigned_cores=2P/4L", lines[2])


class TestNpuAffinitySummary(unittest.TestCase):
    def setUp(self):
        self.topology = _make_machine_topology()

    def _make_report(self, physical_id, runtime_id=0, **result_overrides):
        assignment = _build_affinity_assignment(
            self.topology,
            logical_npu_id=runtime_id,
            physical_npu_id=physical_id,
            allowed_cpu_ids=self.topology.online_cpu_ids,
            requested_pcores=0,
        )
        result_args = dict(
            requested_cpu_ids=assignment.logical_cpu_ids,
            main_actual_cpu_ids=assignment.logical_cpu_ids,
            main_matched=True,
            threads_total=3,
            threads_bound=3,
            threads_exited=0,
            threads_failed=0,
            threads_mismatched=0,
            bind_all_threads=True,
        )
        result_args.update(result_overrides)
        with patch.object(npu_affinity.os, "getpid", return_value=1000 + physical_id):
            return build_npu_affinity_report(
                assignment,
                NpuAffinityApplyResult(**result_args),
                runtime_npu_id=runtime_id,
                tp_rank=physical_id % 4,
                pp_rank=0,
                dp_rank=None,
            )

    def test_each_four_card_instance_reports_only_its_own_sorted_cards(self):
        for base_gpu_id in (0, 4):
            with self.subTest(base_gpu_id=base_gpu_id):
                physical_ids = list(range(base_gpu_id, base_gpu_id + 4))
                reports = [
                    self._make_report(physical_ids[rank], runtime_id=rank)
                    for rank in (3, 1, 0, 2)
                ]
                self.assertEqual(pickle.loads(pickle.dumps(reports)), reports)
                infos = [{"npu_cpu_affinity": report} for report in reports]
                with self.assertLogs(npu_affinity.logger, level="INFO") as logs:
                    log_npu_affinity_summary(
                        infos, base_gpu_id=base_gpu_id, tp_size=4, port=30000
                    )

                self.assertEqual(len(logs.records), 1)
                self.assertEqual(logs.records[0].levelname, "INFO")
                output = logs.records[0].getMessage()
                self.assertIn("NPU CPU AFFINITY SUMMARY", output)
                self.assertIn(
                    f"instance: base_gpu_id={base_gpu_id} tp_size=4 port=30000",
                    output,
                )
                self.assertIn("local_schedulers=4", output)
                headings = [f"Physical NPU {npu_id}" for npu_id in physical_ids]
                positions = [output.index(heading) for heading in headings]
                self.assertEqual(positions, sorted(positions))
                self.assertEqual(output.count("Physical NPU "), 4)
                for npu_id in set(range(8)) - set(physical_ids):
                    self.assertNotIn(f"Physical NPU {npu_id}", output)
                for report in reports:
                    npu_id = report["physical_npu_id"]
                    block = output.split(f"Physical NPU {npu_id}", 1)[1].split(
                        "\n\n", 1
                    )[0]
                    mask = format_cpu_list(report["expected_cpu_ids"])
                    self.assertIn(f"runtime_npu={report['runtime_npu_id']}", block)
                    self.assertIn(f"pid={1000 + npu_id}", block)
                    self.assertIn(f"tp_rank={npu_id % 4} pp_rank=0 dp_rank=None", block)
                    self.assertIn(
                        "npu-smi raw_cpu_affinity=" + report["raw_cpu_affinity"],
                        block,
                    )
                    self.assertIn(f"expected_cpu_mask={mask}\n", block)
                    self.assertIn(f"actual_main_cpu_mask={mask}\n", block)
                    self.assertIn("status=SUCCESS", block)
                    self.assertIn("threads_total=3 threads_bound=3", block)

    def test_mismatched_readback_and_thread_failures_remain_visible(self):
        report = self._make_report(
            4,
            main_actual_cpu_ids=(96,),
            main_matched=False,
            threads_total=4,
            threads_bound=1,
            threads_failed=1,
            threads_mismatched=1,
            threads_exited=1,
            thread_results=(
                NpuAffinityThreadResult(101, "bound", (96, 288)),
                NpuAffinityThreadResult(102, "failed", (), "binding denied"),
                NpuAffinityThreadResult(103, "mismatched", (96,)),
                NpuAffinityThreadResult(104, "exited"),
            ),
        )
        self.assertEqual(report["actual_cpu_ids"], [96])
        self.assertNotEqual(report["actual_cpu_ids"], report["expected_cpu_ids"])
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(pickle.loads(pickle.dumps(report)), report)
        with self.assertLogs(npu_affinity.logger, level="WARNING") as logs:
            log_npu_affinity_summary(
                [
                    {"npu_cpu_affinity": report},
                    {"npu_cpu_affinity": self._make_report(5, runtime_id=1)},
                ],
                base_gpu_id=4,
                tp_size=4,
                port=30001,
            )
        output = logs.records[0].getMessage()
        self.assertEqual(logs.records[0].levelname, "WARNING")
        self.assertIn("expected_cpu_mask=96-119,288-311\n", output)
        self.assertIn("actual_main_cpu_mask=96\n", output)
        self.assertIn("threads_total=4 threads_bound=1", output)
        self.assertIn("threads_failed=1 threads_mismatched=1 threads_exited=1", output)
        self.assertIn("tid=102 status=failed", output)
        self.assertIn("error=binding denied", output)
        self.assertIn("tid=103 status=mismatched\n  actual_cpu_mask=96\n", output)
        failed_block = output.split("Physical NPU 4", 1)[1].split("\n\n", 1)[0]
        self.assertIn("status=INCOMPLETE", failed_block)
        self.assertNotIn("status=SUCCESS", failed_block)

    def test_main_only_binding_is_incomplete_even_when_its_mask_matches(self):
        report = self._make_report(
            0, bind_all_threads=False, threads_total=0, threads_bound=0
        )
        self.assertEqual(report["actual_cpu_ids"], report["expected_cpu_ids"])
        self.assertEqual(report["status"], "INCOMPLETE")
        with self.assertLogs(npu_affinity.logger, level="WARNING") as logs:
            log_npu_affinity_summary(
                [{"npu_cpu_affinity": report}], base_gpu_id=0, tp_size=1, port=30000
            )
        self.assertIn("status=INCOMPLETE", logs.records[0].getMessage())
        self.assertEqual(logs.records[0].levelname, "WARNING")

    def test_missing_assignment_never_invents_masks_and_unknown_card_sorts_last(self):
        reports = []
        for physical_id in (None, 5):
            error = NpuAffinityError(
                "topology unavailable",
                stage="query_npu_smi_topology",
                logical_npu_id=1,
                physical_npu_id=physical_id,
            )
            report = build_npu_affinity_report(
                None,
                None,
                runtime_npu_id=1,
                tp_rank=1,
                pp_rank=0,
                dp_rank=2,
                error=error,
            )
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["physical_npu_id"], physical_id)
            self.assertIsNone(report["raw_cpu_affinity"])
            self.assertIsNone(report["expected_cpu_ids"])
            self.assertIsNone(report["actual_cpu_ids"])
            self.assertEqual(pickle.loads(pickle.dumps(report)), report)
            reports.append(report)
        with self.assertLogs(npu_affinity.logger, level="WARNING") as logs:
            log_npu_affinity_summary(
                [{"npu_cpu_affinity": report} for report in reports],
                base_gpu_id=4,
                tp_size=4,
                port=30000,
            )
        output = logs.records[0].getMessage()
        self.assertLess(
            output.index("Physical NPU 5"), output.index("Physical NPU unknown")
        )
        self.assertEqual(output.count("expected_cpu_mask=unavailable"), 2)
        self.assertEqual(output.count("actual_main_cpu_mask=unavailable"), 2)
        self.assertEqual(output.count("npu-smi raw_cpu_affinity=unavailable"), 2)
        self.assertIn("error=topology unavailable", output)
        self.assertNotIn("status=SUCCESS", output)

    def test_disabled_or_empty_reports_produce_no_summary(self):
        for infos in ([], [{"status": "ready"}], [{"npu_cpu_affinity": None}]):
            with self.subTest(infos=infos), self.assertNoLogs(npu_affinity.logger):
                log_npu_affinity_summary(infos, base_gpu_id=0, tp_size=4, port=30000)

    def test_consumes_only_metadata_and_never_invents_missing_cards(self):
        nested_ready_field = {"max_total_num_tokens": 4096}
        infos = [
            {
                "status": "ready",
                "capacity": nested_ready_field,
                "npu_cpu_affinity": self._make_report(4),
            },
            {"status": "ready", "max_req_input_len": 2048},
        ]
        with (
            patch.object(npu_affinity, "query_npu_smi_topology") as query,
            patch.object(npu_affinity, "resolve_physical_npu_id") as resolve,
            patch.object(npu_affinity, "_get_allowed_cpu_ids") as allowed,
            patch.object(npu_affinity, "_build_affinity_assignment") as allocate,
            self.assertLogs(npu_affinity.logger, level="INFO") as logs,
        ):
            log_npu_affinity_summary(infos, base_gpu_id=4, tp_size=4, port=30000)
        query.assert_not_called()
        resolve.assert_not_called()
        allowed.assert_not_called()
        allocate.assert_not_called()
        self.assertEqual(
            infos,
            [
                {"status": "ready", "capacity": nested_ready_field},
                {"status": "ready", "max_req_input_len": 2048},
            ],
        )
        self.assertIs(infos[0]["capacity"], nested_ready_field)
        output = logs.records[0].getMessage()
        self.assertIn("local_schedulers=1", output)
        self.assertEqual(output.count("Physical NPU "), 1)
        self.assertIn("Physical NPU 4", output)
        with self.assertNoLogs(npu_affinity.logger):
            log_npu_affinity_summary(infos, base_gpu_id=4, tp_size=4, port=30000)


class TestPdAffinity(unittest.TestCase):
    """PD budgets, unused-core allocation, and per-thread classification."""

    def _server_args(self, **overrides):
        values = {
            "disaggregation_mode": "decode",
            "disaggregation_transfer_backend": "ascend",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _assignments(self, requested_pcores=1):
        topology = _make_test_topology()
        compute = _build_affinity_assignment(
            topology,
            logical_npu_id=0,
            physical_npu_id=0,
            allowed_cpu_ids=topology.online_cpu_ids,
            requested_pcores=requested_pcores,
        )
        return topology, compute

    def test_budget_requires_npu_ascend_host_rdma(self):
        with patch.dict(os.environ, {"ASCEND_MF_TRANSFER_PROTOCOL": "host_rdma"}):
            self.assertEqual(
                get_pd_affinity_budget(self._server_args(), is_npu=False), 0
            )
            self.assertEqual(
                get_pd_affinity_budget(
                    self._server_args(disaggregation_mode="null"), is_npu=True
                ),
                0,
            )
            self.assertEqual(
                get_pd_affinity_budget(
                    self._server_args(disaggregation_transfer_backend="mooncake"),
                    is_npu=True,
                ),
                0,
            )
        with patch.dict(os.environ, {"ASCEND_MF_TRANSFER_PROTOCOL": "device_rdma"}):
            self.assertEqual(
                get_pd_affinity_budget(self._server_args(), is_npu=True), 0
            )

    def test_budget_rejects_negative_and_requires_affinity_switch(self):
        with (
            patch.dict(os.environ, {"ASCEND_MF_TRANSFER_PROTOCOL": "host_rdma"}),
            envs.SGLANG_NPU_PD_AFFINITY_PCORES_PER_PROC.override(-1),
        ):
            with self.assertRaises(NpuAffinityError):
                get_pd_affinity_budget(self._server_args(), is_npu=True)
        with (
            patch.dict(os.environ, {"ASCEND_MF_TRANSFER_PROTOCOL": "host_rdma"}),
            envs.SGLANG_NPU_PD_AFFINITY_PCORES_PER_PROC.override(2),
            envs.SGLANG_SET_CPU_AFFINITY.override(False),
        ):
            with self.assertRaises(NpuAffinityError):
                get_pd_affinity_budget(self._server_args(), is_npu=True)
        with (
            patch.dict(os.environ, {"ASCEND_MF_TRANSFER_PROTOCOL": "host_rdma"}),
            envs.SGLANG_NPU_PD_AFFINITY_PCORES_PER_PROC.override(2),
            envs.SGLANG_SET_CPU_AFFINITY.override(True),
        ):
            self.assertEqual(
                get_pd_affinity_budget(self._server_args(), is_npu=True), 2
            )

    def test_pd_assignment_uses_only_unused_owned_cores(self):
        topology, compute = self._assignments(requested_pcores=1)
        pd = build_npu_pd_affinity_assignment(topology, compute, 1)
        self.assertEqual(pd.physical_npu_id, compute.physical_npu_id)
        self.assertEqual(pd.physical_core_keys, ((0, 1),))
        self.assertEqual(pd.logical_cpu_ids, (1, 17))
        self.assertFalse(
            set(pd.logical_cpu_ids) & set(compute.logical_cpu_ids),
        )

    def test_pd_assignment_never_reduces_the_compute_slice(self):
        topology, compute = self._assignments(requested_pcores=2)
        with self.assertRaises(NpuAffinityError) as raised:
            build_npu_pd_affinity_assignment(topology, compute, 1)
        self.assertEqual(raised.exception.stage, "plan_pd_affinity")
        self.assertEqual(compute.effective_pcores, 2)

    def test_binder_classifies_python_native_and_compute_threads(self):
        topology, compute = self._assignments(requested_pcores=1)
        pd = build_npu_pd_affinity_assignment(topology, compute, 1)
        python_thread = PythonPdThread(native_id=1001, start_time=11, role="pool")
        identities = {
            1001: ("Thread-1", 11),
            1002: ("adpt_lkdwn", 12),
            1003: ("worker", 13),
        }
        masks = {}

        def set_affinity(tid, cpus):
            masks[tid] = set(cpus)

        def get_affinity(tid):
            return masks[tid]

        binder = make_pd_thread_binder(compute, pd, (python_thread,))
        with (
            patch.object(
                npu_pd_affinity,
                "read_thread_identity",
                side_effect=lambda pid, tid: identities[tid],
            ),
            patch.object(
                npu_pd_affinity.os,
                "sched_setaffinity",
                side_effect=set_affinity,
                create=True,
            ),
            patch.object(
                npu_pd_affinity.os,
                "sched_getaffinity",
                side_effect=get_affinity,
                create=True,
            ),
        ):
            python_result = binder(1000, 1001)
            native_result = binder(1000, 1002)
            compute_result = binder(1000, 1003)

        self.assertEqual(
            (python_result.status, python_result.source, python_result.role),
            ("bound", "python_pd", "pool"),
        )
        self.assertEqual(python_result.actual_cpu_ids, pd.logical_cpu_ids)
        self.assertEqual(
            (native_result.status, native_result.source, native_result.role),
            ("bound", "native_pd", "adpt_lkdwn"),
        )
        self.assertEqual(native_result.actual_cpu_ids, pd.logical_cpu_ids)
        self.assertEqual(
            (compute_result.status, compute_result.source, compute_result.role),
            ("bound", "default", "compute"),
        )
        self.assertEqual(compute_result.actual_cpu_ids, compute.logical_cpu_ids)

    def test_deferred_plan_survives_pickling(self):
        # The launcher receives this plan through multiprocessing.Pipe.
        topology, compute = self._assignments(requested_pcores=1)
        pd = build_npu_pd_affinity_assignment(topology, compute, 1)
        plan = {
            "assignment": compute,
            "pd_assignment": pd,
            "python_pd_threads": (PythonPdThread(3, 4, "pool"),),
        }
        self.assertEqual(pickle.loads(pickle.dumps(plan)), plan)

    def test_finalize_rejects_a_reused_scheduler_pid(self):
        _, compute = self._assignments(requested_pcores=1)
        plan = {
            "assignment": compute,
            "pid": os.getpid(),
            "create_time": -1.0,
            "tp_rank": 0,
            "pp_rank": 0,
            "dp_rank": 0,
        }
        with patch.object(npu_affinity, "apply_npu_cpu_affinity") as apply:
            report = npu_affinity.finalize_npu_cpu_affinity(plan)
        apply.assert_not_called()
        self.assertEqual(report["status"], "FAILED")
        self.assertIn("was reused", report["error"])


if __name__ == "__main__":
    unittest.main()
