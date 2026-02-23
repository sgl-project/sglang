import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from sglang.srt.debug_utils.comparator.entrypoint import run
from sglang.srt.debug_utils.dumper import DumperConfig, _Dumper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default", nightly=True)


class TestEntrypoint:
    def test_run(self, tmp_path):
        d1 = tmp_path / "baseline"
        d2 = tmp_path / "target"
        d1.mkdir()
        d2.mkdir()

        baseline_tensor = torch.randn(10, 10)
        target_tensor = baseline_tensor + torch.randn(10, 10) * 0.01

        dump_dirs = []
        for d, tensor in [(str(d1), baseline_tensor), (str(d2), target_tensor)]:
            dumper = _Dumper(
                config=DumperConfig(
                    enable=True,
                    dir=d,
                    enable_http_server=False,
                )
            )
            dumper.dump("tensor_a", tensor)
            dumper.step()
            dumper.dump("tensor_b", tensor * 2)
            dumper.step()
            dump_dirs.append(Path(d) / dumper._config.exp_name)

        args = Namespace(
            baseline_path=str(dump_dirs[0]),
            target_path=str(dump_dirs[1]),
            start_id=0,
            end_id=1,
            baseline_start_id=0,
            diff_threshold=1e-3,
            filter=None,
        )
        run(args)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
