import inspect
import unittest

from sglang.srt.layers.moe.shared_ep.epoch import GpuEpoch
from sglang.srt.layers.moe.shared_ep.state import SharedEpState
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


class _Closable:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def close(self):
        self.calls.append(self.name)


class TestSharedEpEpoch(unittest.TestCase):
    def test_protocol_uses_system_scope_release_and_acquire(self):
        """Weakening either PTX scope would permit stale peer activations."""
        source = inspect.getsource(
            __import__(
                "sglang.srt.layers.moe.shared_ep.epoch",
                fromlist=["GpuEpoch"],
            )
        )
        self.assertIn("atom.global.release.sys.exch.b32", source)
        self.assertIn("ld.acquire.sys.global", source)

    def test_forward_methods_contain_no_cpu_synchronization(self):
        """A host collective/readback in publish or wait breaks graph replay."""
        forbidden = (
            "dist.barrier",
            "dist.all_reduce",
            ".cpu(",
            ".item(",
            "torch.cuda.synchronize",
        )
        source = inspect.getsource(GpuEpoch.publish) + inspect.getsource(
            GpuEpoch.wait_all
        )
        for expression in forbidden:
            self.assertNotIn(expression, source)

    def test_state_close_releases_reverse_construction_order_once(self):
        """Partial-state cleanup must not leak or double-release VMM mappings."""
        calls = []
        state = SharedEpState(
            layout=object(),
            input_allocation=_Closable("input", calls),
            output_allocation=_Closable("output", calls),
            input_epoch=_Closable("input_epoch", calls),
            output_epoch=_Closable("output_epoch", calls),
            global_input=object(),
            local_input=object(),
            global_output=object(),
            local_output=object(),
        )

        state.close()
        state.close()

        self.assertEqual(
            calls,
            ["output_epoch", "input_epoch", "output", "input"],
        )


if __name__ == "__main__":
    unittest.main()
