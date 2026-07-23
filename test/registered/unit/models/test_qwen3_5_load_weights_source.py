import pathlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


class TestQwen35LoadWeightsSource(unittest.TestCase):
    def test_shared_expert_fusion_skips_missing_mapped_params(self):
        source_path = (
            pathlib.Path(__file__).resolve().parents[4]
            / "python"
            / "sglang"
            / "srt"
            / "models"
            / "qwen3_5.py"
        )
        lines = source_path.read_text().splitlines()

        branch_line = next(
            i
            for i, line in enumerate(lines)
            if "elif self.enable_shared_expert_fusion:" in line
        )
        first_param_access = next(
            i
            for i in range(branch_line + 1, len(lines))
            if "param = params_dict[name_mapped]" in lines[i]
        )
        guard_block = "\n".join(lines[branch_line:first_param_access])

        self.assertIn("if name_mapped not in params_dict:", guard_block)
        self.assertIn("continue", guard_block)


if __name__ == "__main__":
    unittest.main()
