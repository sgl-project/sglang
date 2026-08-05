"""Unit tests for srt/utils/phase_checker.py — SimplePhaseChecker."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from enum import IntEnum

import torch

from sglang.srt.utils.phase_checker import SimplePhaseChecker
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=120, stage="stage-b", runner_config="1-gpu-small-amd")


class _Phase(IntEnum):
    IDLE = 0
    A = 1
    B = 2
    C = 3


def _phase_value(checker: SimplePhaseChecker) -> int:
    """Read the device-side phase tensor as a host int."""
    return int(checker._phase.item())


def _assert_flag(checker: SimplePhaseChecker) -> int:
    """Read the device-side assert-enable flag as a host int."""
    return int(checker._enable_assert_device.item())


class TestConstruction(CustomTestCase):
    """Constructor sets phase and leaves assert OFF so init-time work is tolerated."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")

    def test_init_stores_initial_phase_int(self) -> None:
        checker = SimplePhaseChecker(initial_phase=7, device=self.device)
        self.assertEqual(_phase_value(checker), 7)

    def test_init_stores_initial_phase_intenum(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.B, device=self.device)
        self.assertEqual(_phase_value(checker), int(_Phase.B))

    def test_init_assert_flag_is_off(self) -> None:
        """assert flag must default to OFF so warmup / cuda-graph-capture don't trip."""
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        self.assertEqual(_assert_flag(checker), 0)

    def test_init_caller_registry_is_empty(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        self.assertEqual(checker._caller_tag_registry, {})

    def test_init_phase_tensor_is_on_requested_device(self) -> None:
        checker = SimplePhaseChecker(initial_phase=0, device=self.device)
        self.assertEqual(checker._phase.device, self.device)
        self.assertEqual(checker._enable_assert_device.device, self.device)
        self.assertEqual(checker._phase.dtype, torch.int32)
        self.assertEqual(checker._enable_assert_device.dtype, torch.int32)


class TestUpdateAssertDisabled(CustomTestCase):
    """With the assert flag OFF, update always advances the phase and never raises."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")

    def test_update_advances_phase_on_match(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        checker.update(expect_phase=_Phase.IDLE, next_phase=_Phase.A, caller_name="t")
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), int(_Phase.A))

    def test_update_advances_phase_on_mismatch(self) -> None:
        """assert OFF tolerates mismatches — store still happens unconditionally."""
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        checker.update(expect_phase=_Phase.C, next_phase=_Phase.B, caller_name="t")
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), int(_Phase.B))

    def test_init_time_lifecycle_violations_tolerated(self) -> None:
        """Documented use case: pre-enable_assert work may freely violate the lifecycle."""
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        # Bogus sequence — would all mismatch if assert were on.
        checker.update(expect_phase=_Phase.B, next_phase=_Phase.C, caller_name="warmup")
        checker.update(
            expect_phase=_Phase.IDLE, next_phase=_Phase.A, caller_name="warmup"
        )
        checker.update(
            expect_phase=_Phase.B, next_phase=_Phase.IDLE, caller_name="warmup"
        )
        torch.cuda.synchronize()  # no raise
        self.assertEqual(_phase_value(checker), int(_Phase.IDLE))


class TestUpdateAssertEnabled(CustomTestCase):
    """With assert ON, matching updates pass and walk a multi-state lifecycle."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")

    def test_update_advances_phase_on_match(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        checker.enable_assert()
        checker.update(expect_phase=_Phase.IDLE, next_phase=_Phase.A, caller_name="t")
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), int(_Phase.A))

    def test_full_4_state_lifecycle_round_trip(self) -> None:
        """IDLE -> A -> B -> C -> IDLE under assert ON, no synchronization mid-cycle."""
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        checker.enable_assert()
        for _ in range(3):  # multiple cycles to exercise stable behavior
            checker.update(
                expect_phase=_Phase.IDLE, next_phase=_Phase.A, caller_name="p1"
            )
            checker.update(expect_phase=_Phase.A, next_phase=_Phase.B, caller_name="p2")
            checker.update(expect_phase=_Phase.B, next_phase=_Phase.C, caller_name="p3")
            checker.update(
                expect_phase=_Phase.C, next_phase=_Phase.IDLE, caller_name="p4"
            )
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), int(_Phase.IDLE))

    def test_update_mismatch_after_enable_raises_in_subprocess(self) -> None:
        """A mismatched update with assert ON must fire device_assert at the next sync.

        Run in a subprocess because device-side asserts poison the CUDA context.
        """
        script = textwrap.dedent("""
            import sys

            import torch

            from sglang.srt.utils.phase_checker import SimplePhaseChecker

            device = torch.device("cuda:0")
            checker = SimplePhaseChecker(initial_phase=0, device=device)
            checker.enable_assert()
            # phase=0 but we claim expect=99 — kernel must fire device_assert.
            checker.update(expect_phase=99, next_phase=1, caller_name="bad")
            try:
                torch.cuda.synchronize()
            except RuntimeError as e:
                msg = str(e).lower()
                if "device-side assert" in msg or "phase mismatch" in msg:
                    sys.exit(0)
                print(f"Unexpected RuntimeError: {e}", file=sys.stderr)
                sys.exit(2)
            print("expected RuntimeError but none was raised", file=sys.stderr)
            sys.exit(1)
            """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
        )
        # Accept both exit-0 (RuntimeError caught) and SIGABRT (-6, the kernel's
        # device_assert killed the process directly before sync could raise).
        # Either way, the kernel-side check fired — which is what we're verifying.
        # The presence of the SimplePhaseChecker FAIL line in stdout confirms it.
        self.assertIn(
            "SimplePhaseChecker FAIL",
            result.stdout,
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )
        self.assertIn(
            result.returncode,
            (0, -6),
            f"unexpected returncode {result.returncode}; "
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )


class TestEnableAssert(CustomTestCase):
    """enable_assert flips the device flag and resets phase to the initial value."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")

    def test_enable_assert_sets_flag_to_one(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        self.assertEqual(_assert_flag(checker), 0)
        checker.enable_assert()
        self.assertEqual(_assert_flag(checker), 1)

    def test_enable_assert_resets_phase_to_initial(self) -> None:
        """Phase advanced during init must be wiped by enable_assert.

        Otherwise post-init lifecycle would start from whatever captured kernels
        left in the phase tensor — the explicit purpose of enable_assert.
        """
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        # Mess with the phase as init-time work might.
        checker.update(
            expect_phase=_Phase.IDLE, next_phase=_Phase.C, caller_name="warmup"
        )
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), int(_Phase.C))

        checker.enable_assert()
        self.assertEqual(_phase_value(checker), int(_Phase.IDLE))

    def test_enable_assert_with_nonzero_initial_phase(self) -> None:
        """Reset target tracks the original initial_phase, not 0."""
        checker = SimplePhaseChecker(initial_phase=42, device=self.device)
        checker.update(expect_phase=42, next_phase=7, caller_name="t")
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), 7)

        checker.enable_assert()
        self.assertEqual(_phase_value(checker), 42)

    def test_enable_assert_is_idempotent(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        checker.enable_assert()
        checker.enable_assert()
        self.assertEqual(_assert_flag(checker), 1)
        self.assertEqual(_phase_value(checker), int(_Phase.IDLE))


class TestResetToIdle(CustomTestCase):
    """_reset_to_idle only touches the phase tensor; the assert flag is untouched."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")

    def test_reset_after_update_restores_initial_phase(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        checker.update(expect_phase=_Phase.IDLE, next_phase=_Phase.B, caller_name="t")
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), int(_Phase.B))

        checker._reset_to_idle()
        self.assertEqual(_phase_value(checker), int(_Phase.IDLE))

    def test_reset_does_not_toggle_assert_flag_when_off(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        self.assertEqual(_assert_flag(checker), 0)
        checker._reset_to_idle()
        self.assertEqual(_assert_flag(checker), 0)

    def test_reset_does_not_toggle_assert_flag_when_on(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        checker.enable_assert()
        self.assertEqual(_assert_flag(checker), 1)
        checker._reset_to_idle()
        self.assertEqual(_assert_flag(checker), 1)

    def test_reset_with_nonzero_initial_phase(self) -> None:
        checker = SimplePhaseChecker(initial_phase=5, device=self.device)
        checker.update(expect_phase=5, next_phase=9, caller_name="t")
        torch.cuda.synchronize()
        checker._reset_to_idle()
        self.assertEqual(_phase_value(checker), 5)


class TestCallerTagRegistry(CustomTestCase):
    """Caller tags are interned 1-indexed integers, assigned in first-seen order."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")

    def test_first_caller_gets_tag_one(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        self.assertEqual(checker._resolve_caller_tag("first"), 1)

    def test_distinct_callers_get_distinct_increasing_tags(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        self.assertEqual(checker._resolve_caller_tag("a"), 1)
        self.assertEqual(checker._resolve_caller_tag("b"), 2)
        self.assertEqual(checker._resolve_caller_tag("c"), 3)

    def test_repeated_caller_returns_same_tag(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        first = checker._resolve_caller_tag("repeat")
        # Mix other callers in between.
        checker._resolve_caller_tag("other")
        second = checker._resolve_caller_tag("repeat")
        self.assertEqual(first, second)

    def test_empty_caller_name_is_interned_like_any_other(self) -> None:
        """default caller_name="" must be a legal key, not crash on use."""
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        checker.enable_assert()
        checker.update(expect_phase=_Phase.IDLE, next_phase=_Phase.A)  # caller_name=""
        torch.cuda.synchronize()
        self.assertIn("", checker._caller_tag_registry)
        self.assertEqual(_phase_value(checker), int(_Phase.A))

    def test_update_populates_registry(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        checker.update(
            expect_phase=_Phase.IDLE, next_phase=_Phase.A, caller_name="alpha"
        )
        checker.update(
            expect_phase=_Phase.A, next_phase=_Phase.IDLE, caller_name="beta"
        )
        torch.cuda.synchronize()
        self.assertEqual(checker._caller_tag_registry, {"alpha": 1, "beta": 2})


class TestMultipleInstances(CustomTestCase):
    """Independent checkers must not share phase, assert flag, or caller registry."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")

    def test_phase_tensors_are_independent(self) -> None:
        a = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        b = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        a.update(expect_phase=_Phase.IDLE, next_phase=_Phase.B, caller_name="a")
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(a), int(_Phase.B))
        self.assertEqual(_phase_value(b), int(_Phase.IDLE))

    def test_assert_flags_are_independent(self) -> None:
        a = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        b = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        a.enable_assert()
        self.assertEqual(_assert_flag(a), 1)
        self.assertEqual(_assert_flag(b), 0)

    def test_caller_registries_are_independent(self) -> None:
        a = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        b = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        a._resolve_caller_tag("shared_name")
        # Same string, different checker — should still get tag=1, not 2.
        self.assertEqual(b._resolve_caller_tag("shared_name"), 1)


class TestCudaGraphCapture(CustomTestCase):
    """The kernel is launched unconditionally so it is capture-safe; the device flag
    decides at replay time whether the assert fires.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")

    def _capture_one_update(
        self,
        checker: SimplePhaseChecker,
        *,
        expect_phase: int,
        next_phase: int,
        caller_name: str,
    ) -> torch.cuda.CUDAGraph:
        """Warm up + capture a single update() call into a fresh CUDAGraph.

        Both warmup and capture launches actually execute on the GPU and advance
        the phase tensor. Caller must keep assert OFF here — otherwise the second
        launch (capture) would see phase=next_phase and mismatch the expected.
        """
        assert _assert_flag(checker) == 0, "capture helper requires assert OFF"
        stream = torch.cuda.Stream(self.device)
        # Warmup launch on a side stream — required before torch.cuda.graph.
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            checker.update(
                expect_phase=expect_phase,
                next_phase=next_phase,
                caller_name=caller_name,
            )
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            checker.update(
                expect_phase=expect_phase,
                next_phase=next_phase,
                caller_name=caller_name,
            )
        return graph

    def test_captured_update_advances_phase_on_replay(self) -> None:
        """Replay reissues the captured kernel; phase tensor is updated each replay."""
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        # Capture under assert OFF (the documented init-time pattern).
        graph = self._capture_one_update(
            checker,
            expect_phase=int(_Phase.IDLE),
            next_phase=int(_Phase.B),
            caller_name="captured",
        )

        # Enable assert (resets phase -> IDLE) and replay — captured expect=IDLE matches.
        checker.enable_assert()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), int(_Phase.B))

        # Reset + replay again — same result, no raise.
        checker._reset_to_idle()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), int(_Phase.B))

    def test_assert_flag_toggle_visible_to_replayed_graph(self) -> None:
        """Key contract: capture with assert OFF, enable assert later, replay sees the new flag.

        We don't try to make the replay raise (that would poison the context); we
        verify the *positive* side — capture under assert OFF (with deliberate
        phase mismatches tolerated) and confirm the same captured kernel still
        works correctly when assert is later turned on.
        """
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        graph = self._capture_one_update(
            checker,
            expect_phase=int(_Phase.IDLE),
            next_phase=int(_Phase.A),
            caller_name="captured",
        )
        self.assertEqual(_assert_flag(checker), 0)

        # Replay with assert OFF tolerates a deliberately diverged phase.
        checker._phase.fill_(999)
        graph.replay()
        torch.cuda.synchronize()  # no raise — flag is OFF
        self.assertEqual(_phase_value(checker), int(_Phase.A))

        # Now turn on asserts (also resets phase -> IDLE) and replay.
        checker.enable_assert()
        self.assertEqual(_assert_flag(checker), 1)
        self.assertEqual(_phase_value(checker), int(_Phase.IDLE))

        graph.replay()
        torch.cuda.synchronize()  # no raise — phase matched expect
        self.assertEqual(_phase_value(checker), int(_Phase.A))


class TestPhaseReprNoCrash(CustomTestCase):
    """Debug logging path (host_debug) tolerates both int and IntEnum phase args."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda:0")

    def test_update_with_int_phases_does_not_crash(self) -> None:
        checker = SimplePhaseChecker(initial_phase=0, device=self.device)
        checker.enable_assert()
        checker.update(expect_phase=0, next_phase=1, caller_name="ints")
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), 1)

    def test_update_with_intenum_phases_does_not_crash(self) -> None:
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        checker.enable_assert()
        checker.update(
            expect_phase=_Phase.IDLE, next_phase=_Phase.A, caller_name="enums"
        )
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), int(_Phase.A))

    def test_update_mixing_int_and_intenum_phases(self) -> None:
        """expect_phase is an IntEnum, next_phase is a plain int (and vice versa)."""
        checker = SimplePhaseChecker(initial_phase=_Phase.IDLE, device=self.device)
        checker.enable_assert()
        checker.update(expect_phase=_Phase.IDLE, next_phase=5, caller_name="mix1")
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), 5)
        checker.update(expect_phase=5, next_phase=_Phase.IDLE, caller_name="mix2")
        torch.cuda.synchronize()
        self.assertEqual(_phase_value(checker), int(_Phase.IDLE))


if __name__ == "__main__":
    unittest.main()
