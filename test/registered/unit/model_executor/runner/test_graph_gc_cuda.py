"""Exercise real graph recapture/replay after serving requests a GC freeze."""

import subprocess
import sys
import textwrap
import unittest

import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGraphGCReplay(unittest.TestCase):
    def test_recapture_preserves_freeze_and_collects_new_cycles(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""
                    import gc
                    import weakref
                    import torch
                    from sglang.srt.utils.common import freeze_gc as freeze_serving_gc
                    from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc

                    class Cycle:
                        def __init__(self):
                            self.link = self

                    source = torch.ones(1024, device='cuda')
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            output = source.square() + 3
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    old = Cycle()
                    reference = weakref.ref(old)
                    freeze_serving_gc('CUDA regression')
                    del old
                    for enabled in (False, True, False):
                        graph = torch.cuda.CUDAGraph()
                        with freeze_gc(enabled):
                            with torch.cuda.graph(graph):
                                output = source.square() + 3
                            new = Cycle()
                            recent = weakref.ref(new)
                            del new
                        for value in (2, 7):
                            source.fill_(value)
                            graph.replay()
                            torch.testing.assert_close(output, torch.full_like(output, value ** 2 + 3))
                        gc.collect()
                        assert reference() is not None, 'graph capture revoked serving freeze'
                        assert recent() is None, 'new cyclic request objects became permanent'
                        del graph, output
                    gc.unfreeze()
                    gc.collect()
                    assert reference() is None
                    """),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
