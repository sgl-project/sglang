import os
import queue
import threading
import traceback
import unittest
from multiprocessing import Process
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.distributed.parallel_state import (
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.utils.rank_consensus_checker import (
    assert_same,
    configure,
    rank_consensus,
    shutdown,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cpu_ci(est_time=193, suite="base-b-test-cpu")


def run_distributed_test(
    rank: int,
    world_size: int,
    pp_size: int,
    tp_size: int,
    master_port: int,
    fn,
) -> None:
    """Child-process entry point: set up gloo, then run fn.

    Exit codes:
      * 0 -> fn finished cleanly
      * 1 -> rdc detected divergence and called os._exit(1) from its worker
      * 2 -> fn raised (test setup/scenario bug)
    """
    # CUDA_VISIBLE_DEVICES is set to "99" (a non-existent device) by the parent
    # in _spawn() before this process starts, so by the time the test module
    # (and torch) is re-imported here, is_cuda_alike() returns False and
    # GroupCoordinator picks device="cpu". That keeps this test CPU-only and
    # lets world_size exceed the host's physical GPU count.

    # The CUDA-only communicators (pynccl, custom allreduce) cannot be built
    # without a GPU -- PyNcclCommunicator calls torch.cuda.device(device).
    # initialize_model_parallel has no flag to disable pynccl, so patch
    # init_model_parallel_group to force use_pynccl=False (and clear the
    # module-level custom-allreduce default via its public setter). patch.object
    # auto-restores on exit, including the os._exit(2) path below.
    ps.set_custom_all_reduce(False)

    def _cpu_init_model_parallel_group(
        *args, _orig=ps.init_model_parallel_group, **kwargs
    ):
        kwargs.setdefault("use_pynccl", False)
        kwargs.setdefault("use_custom_allreduce", False)
        return _orig(*args, **kwargs)

    with patch.object(ps, "init_model_parallel_group", _cpu_init_model_parallel_group):
        try:
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(master_port)
            os.environ["LOCAL_SIZE"] = str(world_size)

            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method="env://",
                local_rank=rank,
                backend="gloo",
            )

            initialize_model_parallel(
                tensor_model_parallel_size=tp_size,
                pipeline_model_parallel_size=pp_size,
                backend="gloo",
            )

            fn()
        except Exception as e:
            print(f"subprocess[{rank=}] has error: {e}", flush=True)
            traceback.print_exc()
            os._exit(2)
        finally:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass


class _DummyClass:
    def __init__(self, a: int = None, b: int = None):
        self.a = a
        self.b = b

    def __repr__(self) -> str:
        return f"DummyClass(a={self.a}, b={self.b})"


class _MethodHost:
    @rank_consensus(same_params=True)
    def instance_method(obj, a, b):
        return a + b

    @rank_consensus(same_params=True)
    @classmethod
    def class_method(klass, a):
        return a + 1

    @rank_consensus(same_params=True)
    @staticmethod
    def static_method(a, b):
        return a * b


class RankConsensusCheckerTestCase(CustomTestCase):
    def _spawn(self, fn, pp_size: int = 1, tp_size: int = 1, enable_env: bool = True):
        """Run fn in world_size spawned gloo children. Returns True iff every
        child exited with code 0. A detected divergence makes rdc call
        os._exit(1) from its worker thread; an exception inside fn makes
        run_distributed_test call os._exit(2). Either way _spawn returns
        False for that child."""
        mp.set_start_method("spawn", force=True)
        master_port = find_available_port(23456)

        old_env = os.getenv("SGLANG_ENABLE_RANK_CONSENSUS_CHECKER")
        os.environ["SGLANG_ENABLE_RANK_CONSENSUS_CHECKER"] = str(enable_env)

        world_size = pp_size * tp_size
        processes = []
        for rank in range(world_size):
            p = Process(
                target=run_distributed_test,
                kwargs=dict(
                    rank=rank,
                    world_size=world_size,
                    pp_size=pp_size,
                    tp_size=tp_size,
                    master_port=master_port,
                    fn=fn,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if old_env is None:
            os.environ.pop("SGLANG_ENABLE_RANK_CONSENSUS_CHECKER")
        else:
            os.environ["SGLANG_ENABLE_RANK_CONSENSUS_CHECKER"] = old_env

        return all(p.exitcode == 0 for p in processes)


class TestAssertSame(RankConsensusCheckerTestCase):
    @staticmethod
    def same_fn():
        configure([get_tp_group()])
        assert_same("same %d", 10)
        shutdown()

    def test_same(self):
        """Same args on every rank -> no divergence, clean exit."""
        self.assertTrue(self._spawn(TestAssertSame.same_fn, tp_size=2))

    @staticmethod
    def divergence_fn():
        tp_group = get_tp_group()
        configure([tp_group])
        assert_same("diverge %d", tp_group.rank_in_group)
        shutdown()

    def test_divergence(self):
        """Different args on different ranks -> rdc calls os._exit(1) -> child
        exit code is 1 -> _spawn returns False."""
        self.assertFalse(self._spawn(TestAssertSame.divergence_fn, tp_size=2))

    @staticmethod
    def divergent_multi_group_fn():
        tp_group = get_tp_group()
        pp_group = get_pp_group()
        configure([tp_group, pp_group])
        assert_same("diverge %d", tp_group.rank_in_group)
        shutdown()

    def test_divergence_detected_multi_group(self):
        """Passing the same group twice must still surface the divergence."""
        self.assertFalse(
            self._spawn(
                TestAssertSame.divergent_multi_group_fn,
                pp_size=2,
                tp_size=2,
            )
        )

    @staticmethod
    def wrong_thread_fn():
        tp_group = get_tp_group()
        configure([tp_group])

        err_box: queue.Queue = queue.Queue()

        def _other_thread():
            try:
                assert_same("from other thread")
                err_box.put(None)
            except Exception as e:  # noqa: BLE001
                err_box.put(e)

        t = threading.Thread(target=_other_thread)
        t.start()
        t.join()

        err = err_box.get()
        shutdown()
        assert isinstance(
            err, RuntimeError
        ), f"Expected RuntimeError from stray-thread assert_same, got {err!r}"

    def test_assert_same_rejects_non_scheduler_thread(self):
        """Check that assert_same() must be called in the scheduler thread.  Otherwise report error."""
        self.assertTrue(self._spawn(TestAssertSame.wrong_thread_fn, tp_size=2))

    @staticmethod
    def disabled_fn():
        tp_group = get_tp_group()
        configure([tp_group])
        assert_same("diverge %d", tp_group.rank_in_group)

    def test_disabled_is_noop(self):
        """Test that when SGLANG_ENABLE_RANK_CONSENSUS_CHECKER=false, assert_same is no-op."""
        self.assertTrue(
            self._spawn(TestAssertSame.disabled_fn, tp_size=2, enable_env=False)
        )


class TestRankConsensusDecorator(RankConsensusCheckerTestCase):
    @staticmethod
    def consensus_bare_diverge_fn():
        @rank_consensus
        def foo(a: int) -> int:
            return a

        # Bare decorator only checks "was called", not args; even with diverging
        # args this must exit clean (no rank divergence).
        tp_group = get_tp_group()
        configure([tp_group])
        foo(tp_group.rank_in_group)
        shutdown()

    def test_bare_decorator_clean_with_diverging_args(self):
        """Bare decorator only checks that every rank calls the function;
        diverging args must NOT be flagged."""
        self.assertTrue(
            self._spawn(TestRankConsensusDecorator.consensus_bare_diverge_fn, tp_size=2)
        )

    @staticmethod
    def consensus_all_params_same_fn():
        @rank_consensus(same_params=True)
        def foo(a: int, b: int) -> int:
            return a + b

        configure([get_tp_group()])
        foo(1, 2)
        shutdown()

    def test_all_params_same(self):
        self.assertTrue(
            self._spawn(
                TestRankConsensusDecorator.consensus_all_params_same_fn, tp_size=2
            )
        )

    @staticmethod
    def consensus_all_params_diverge_fn():
        @rank_consensus(same_params=True)
        def foo(a, b):
            return a + b

        tp_group = get_tp_group()
        configure([tp_group])
        # The second argument differs on rank.  Expect divergence.
        foo(1, tp_group.rank_in_group)
        shutdown()

    def test_all_params_diverge(self):
        self.assertFalse(
            self._spawn(
                TestRankConsensusDecorator.consensus_all_params_diverge_fn, tp_size=2
            )
        )

    @staticmethod
    def consensus_named_params_same_fn():
        @rank_consensus(same_params=["a", "c"])
        def foo(a: int, b: int, c: int) -> int:
            return a + b + c

        tp_group = get_tp_group()
        configure([tp_group])
        # b diverges but is NOT in the selector list.  Expect good.
        foo(1, tp_group.rank_in_group, 3)
        shutdown()

    def test_named_params_ignores_unselected_divergence(self):
        self.assertTrue(
            self._spawn(
                TestRankConsensusDecorator.consensus_named_params_same_fn, tp_size=2
            )
        )

    @staticmethod
    def consensus_named_params_diverge_fn():
        @rank_consensus(same_params=["a", "c"])
        def foo(a: int, b: int, c: int) -> int:
            return a + b + c

        # c diverges and IS in the selector list.  Expect divergence.
        tp_group = get_tp_group()
        configure([tp_group])
        foo(1, 2, tp_group.rank_in_group)
        shutdown()

    def test_named_params_flags_selected_divergence(self):
        self.assertFalse(
            self._spawn(
                TestRankConsensusDecorator.consensus_named_params_diverge_fn,
                tp_size=2,
            )
        )

    @staticmethod
    def consensus_dotted_param_same_fn():
        @rank_consensus(same_params=["_a.a"])
        def foo(_a: _DummyClass) -> None:
            pass

        tp_group = get_tp_group()
        configure([tp_group])
        dummy = _DummyClass(a=10, b=tp_group.rank_in_group)
        foo(dummy)
        shutdown()

    def test_dotted_param_same(self):
        self.assertTrue(
            self._spawn(
                TestRankConsensusDecorator.consensus_dotted_param_same_fn, tp_size=2
            )
        )

    @staticmethod
    def consensus_dotted_param_diverge_fn():
        @rank_consensus(same_params=["_a.a"])
        def foo(_a: _DummyClass) -> None:
            pass

        tp_group = get_tp_group()
        configure([tp_group])
        dummy = _DummyClass(a=tp_group.rank_in_group, b=10)
        foo(dummy)
        shutdown()

    def test_dotted_param_diverge(self):
        self.assertFalse(
            self._spawn(
                TestRankConsensusDecorator.consensus_dotted_param_diverge_fn,
                tp_size=2,
            )
        )

    @staticmethod
    def consensus_full_result_same_fn():
        @rank_consensus(same_results=True)
        def foo(value: int) -> _DummyClass:
            return _DummyClass(a=value, b=value * 2)

        configure([get_tp_group()])
        foo(5)
        shutdown()

    def test_full_result_same(self):
        self.assertTrue(
            self._spawn(
                TestRankConsensusDecorator.consensus_full_result_same_fn, tp_size=2
            )
        )

    @staticmethod
    def consensus_full_result_diverge_fn():
        @rank_consensus(same_results=True)
        def foo(value: int) -> _DummyClass:
            return _DummyClass(a=value, b=value * 2)

        tp_group = get_tp_group()
        configure([tp_group])
        foo(tp_group.rank_in_group)
        shutdown()

    def test_full_result_diverge(self):
        self.assertFalse(
            self._spawn(
                TestRankConsensusDecorator.consensus_full_result_diverge_fn,
                tp_size=2,
            )
        )

    @staticmethod
    def consensus_partial_result_same_fn():
        @rank_consensus(same_results=["result.x", "len(result.y)"])
        def foo(x, y_list):
            class _R:
                pass

            r = _R()
            r.x = x
            r.y = y_list
            return r

        tp_group = get_tp_group()
        configure([tp_group])
        # x and len(y) both equal across ranks; y contents differ but are not selected.  Expect good.
        foo(x=3, y_list=[tp_group.rank_in_group] * 4)
        shutdown()

    def test_partial_result_same(self):
        self.assertTrue(
            self._spawn(
                TestRankConsensusDecorator.consensus_partial_result_same_fn,
                tp_size=2,
            )
        )

    @staticmethod
    def consensus_partial_result_diverge_fn():
        @rank_consensus(same_results=["result.x", "len(result.y)"])
        def foo(x, y_list):
            class _R:
                pass

            r = _R()
            r.x = x
            r.y = y_list
            return r

        tp_group = get_tp_group()
        configure([tp_group])
        # x diverges and IS selected.  Expect divergence.
        foo(x=tp_group.rank_in_group, y_list=[1, 2, 3])
        shutdown()

    def test_partial_result_diverge(self):
        self.assertFalse(
            self._spawn(
                TestRankConsensusDecorator.consensus_partial_result_diverge_fn,
                tp_size=2,
            )
        )

    @staticmethod
    def consensus_both_same_fn():
        @rank_consensus(same_params=True, same_results=True)
        def foo(a: int) -> int:
            return a * 2

        configure([get_tp_group()])
        foo(7)
        shutdown()

    def test_both_same(self):
        self.assertTrue(
            self._spawn(TestRankConsensusDecorator.consensus_both_same_fn, tp_size=2)
        )

    @staticmethod
    def consensus_both_diverge_fn():
        @rank_consensus(same_params=True, same_results=True)
        def foo(a: int) -> int:
            return a * 2

        tp_group = get_tp_group()
        configure([tp_group])
        foo(tp_group.rank_in_group)
        shutdown()

    def test_both_diverge(self):
        self.assertFalse(
            self._spawn(TestRankConsensusDecorator.consensus_both_diverge_fn, tp_size=2)
        )

    @staticmethod
    def consensus_instance_method_same_fn():
        configure([get_tp_group()])
        _MethodHost().instance_method(1, 2)
        shutdown()

    def test_instance_method_receiver_dropped(self):
        # Two ranks build two different _MethodHost instances; without the
        # receiver-skip the per-rank address would diverge. Clean exit
        # confirms the receiver is dropped.
        self.assertTrue(
            self._spawn(
                TestRankConsensusDecorator.consensus_instance_method_same_fn,
                tp_size=2,
            )
        )

    @staticmethod
    def consensus_class_method_same_fn():
        configure([get_tp_group()])
        _MethodHost.class_method(5)
        shutdown()

    def test_class_method_receiver_dropped(self):
        # First param is named ``klass`` (not cls); detection must still work.
        self.assertTrue(
            self._spawn(
                TestRankConsensusDecorator.consensus_class_method_same_fn, tp_size=2
            )
        )

    @staticmethod
    def consensus_class_method_via_instance_same_fn():
        configure([get_tp_group()])
        _MethodHost().class_method(5)
        shutdown()

    def test_class_method_via_instance_receiver_dropped(self):
        # Accessing the classmethod through an instance still binds the class
        # as the receiver; verify it is still dropped.
        self.assertTrue(
            self._spawn(
                TestRankConsensusDecorator.consensus_class_method_via_instance_same_fn,
                tp_size=2,
            )
        )

    @staticmethod
    def consensus_static_method_same_fn():
        configure([get_tp_group()])
        _MethodHost.static_method(3, 4)
        shutdown()

    def test_static_method_no_receiver(self):
        # Static method: no receiver, equal args -> clean.
        self.assertTrue(
            self._spawn(
                TestRankConsensusDecorator.consensus_static_method_same_fn,
                tp_size=2,
            )
        )

    @staticmethod
    def consensus_static_method_diverge_fn():
        tp_group = get_tp_group()
        configure([tp_group])
        # Static method: no receiver to drop, so a rank-dependent arg diverges.
        _MethodHost.static_method(tp_group.rank_in_group, 4)
        shutdown()

    def test_static_method_flags_diverging_arg(self):
        # Static method: no receiver to drop, so a rank-dependent arg must
        # still be flagged. Confirms we did not over-skip for static methods.
        self.assertFalse(
            self._spawn(
                TestRankConsensusDecorator.consensus_static_method_diverge_fn,
                tp_size=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
