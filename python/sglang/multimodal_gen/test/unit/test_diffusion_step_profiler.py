import unittest
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.utils.profiler import (
    DiffusionStepProfiler,
    is_primary_rank,
    parse_step_range,
)


class TestParseStepRange(unittest.TestCase):
    def test_valid_range(self):
        self.assertEqual(parse_step_range("100-150"), (100, 150))

    def test_single_step_range(self):
        self.assertEqual(parse_step_range("5-6"), (5, 6))

    def test_none_returns_none(self):
        self.assertIsNone(parse_step_range(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_step_range(""))

    def test_start_equals_end_returns_none(self):
        self.assertIsNone(parse_step_range("100-100"))

    def test_start_greater_than_end_returns_none(self):
        self.assertIsNone(parse_step_range("150-100"))

    def test_missing_dash_returns_none(self):
        self.assertIsNone(parse_step_range("100150"))

    def test_non_integer_returns_none(self):
        self.assertIsNone(parse_step_range("abc-def"))


class TestIsPrimaryRank(unittest.TestCase):
    def test_single_gpu_is_primary(self):
        # No distributed setup — should return True
        with patch(
            "sglang.multimodal_gen.runtime.utils.profiler._get_world_group", None
        ):
            self.assertTrue(is_primary_rank())

    def test_rank_0_is_primary(self):
        mock_group = MagicMock()
        mock_group.rank = 0
        mock_get = MagicMock(return_value=mock_group)
        with patch(
            "sglang.multimodal_gen.runtime.utils.profiler._get_world_group", mock_get
        ):
            self.assertTrue(is_primary_rank())

    def test_rank_1_is_not_primary(self):
        mock_group = MagicMock()
        mock_group.rank = 1
        mock_get = MagicMock(return_value=mock_group)
        with patch(
            "sglang.multimodal_gen.runtime.utils.profiler._get_world_group", mock_get
        ):
            self.assertFalse(is_primary_rank())

    def test_runtime_error_falls_back_to_primary(self):
        mock_get = MagicMock(side_effect=RuntimeError("not initialized"))
        with patch(
            "sglang.multimodal_gen.runtime.utils.profiler._get_world_group", mock_get
        ):
            self.assertTrue(is_primary_rank())


class TestDiffusionStepProfilerSingleton(unittest.TestCase):
    def setUp(self):
        DiffusionStepProfiler.reset()

    def tearDown(self):
        DiffusionStepProfiler.reset()

    def test_get_instance_returns_same_object(self):
        a = DiffusionStepProfiler.get_instance()
        b = DiffusionStepProfiler.get_instance()
        self.assertIs(a, b)

    def test_reset_creates_fresh_instance(self):
        a = DiffusionStepProfiler.get_instance()
        DiffusionStepProfiler.reset()
        b = DiffusionStepProfiler.get_instance()
        self.assertIsNot(a, b)

    def test_no_profile_range_when_env_unset(self):
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = None
            p = DiffusionStepProfiler()
        self.assertIsNone(p.profile_range)
        self.assertFalse(p.should_profile())

    def test_profile_range_set_from_env(self):
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = "100-150"
            p = DiffusionStepProfiler()
        self.assertEqual(p.profile_range, (100, 150))
        self.assertTrue(p.should_profile())


class TestStepCounting(unittest.TestCase):
    def setUp(self):
        DiffusionStepProfiler.reset()

    def tearDown(self):
        DiffusionStepProfiler.reset()

    def _make_profiler(self, step_range="10-13"):
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = step_range
            return DiffusionStepProfiler()

    def test_step_increments_counter(self):
        p = self._make_profiler()
        self.assertEqual(p.global_step_count, 0)
        p.step()
        self.assertEqual(p.global_step_count, 1)
        p.step()
        self.assertEqual(p.global_step_count, 2)

    def test_step_returns_false_before_range(self):
        p = self._make_profiler("10-13")
        for _ in range(10):
            result = p.step()
        self.assertFalse(result)

    def test_step_returns_true_in_range(self):
        p = self._make_profiler("10-13")
        for _ in range(10):
            p.step()
        # steps 10, 11, 12 are in range
        self.assertTrue(p.step())  # step 10
        self.assertTrue(p.step())  # step 11
        self.assertTrue(p.step())  # step 12

    def test_step_returns_false_at_range_end(self):
        p = self._make_profiler("10-13")
        for _ in range(13):
            p.step()
        self.assertFalse(p.step())  # step 13 — range is exclusive-end

    def test_step_returns_false_when_no_range(self):
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = None
            p = DiffusionStepProfiler()
        self.assertFalse(p.step())


class TestCudaProfilerControl(unittest.TestCase):
    def setUp(self):
        DiffusionStepProfiler.reset()

    def tearDown(self):
        DiffusionStepProfiler.reset()

    def _make_profiler(self, step_range="5-8"):
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = step_range
            return DiffusionStepProfiler()

    @patch(
        "sglang.multimodal_gen.runtime.utils.profiler.torch.cuda.is_available",
        return_value=True,
    )
    @patch("sglang.multimodal_gen.runtime.utils.profiler.torch.cuda.cudart")
    def test_profiler_starts_on_range_entry(self, mock_cudart, _):
        mock_rt = MagicMock()
        mock_cudart.return_value = mock_rt
        p = self._make_profiler("5-8")
        for _ in range(5):
            p.step()
        p.step()  # step 5 — enters range
        mock_rt.cudaProfilerStart.assert_called_once()

    @patch(
        "sglang.multimodal_gen.runtime.utils.profiler.torch.cuda.is_available",
        return_value=True,
    )
    @patch("sglang.multimodal_gen.runtime.utils.profiler.torch.cuda.cudart")
    def test_profiler_stops_on_range_exit(self, mock_cudart, _):
        mock_rt = MagicMock()
        mock_cudart.return_value = mock_rt
        p = self._make_profiler("5-8")
        for _ in range(8):
            p.step()
        p.step()  # step 8 — exits range
        mock_rt.cudaProfilerStop.assert_called_once()

    @patch(
        "sglang.multimodal_gen.runtime.utils.profiler.is_primary_rank",
        return_value=False,
    )
    @patch(
        "sglang.multimodal_gen.runtime.utils.profiler.torch.cuda.is_available",
        return_value=True,
    )
    @patch("sglang.multimodal_gen.runtime.utils.profiler.torch.cuda.cudart")
    def test_non_primary_rank_skips_cuda_calls(self, mock_cudart, _, __):
        p = self._make_profiler("5-8")
        for _ in range(9):
            p.step()
        mock_cudart.return_value.cudaProfilerStart.assert_not_called()
        mock_cudart.return_value.cudaProfilerStop.assert_not_called()


class TestEnsureStopped(unittest.TestCase):
    def setUp(self):
        DiffusionStepProfiler.reset()

    def tearDown(self):
        DiffusionStepProfiler.reset()

    @patch(
        "sglang.multimodal_gen.runtime.utils.profiler.torch.cuda.is_available",
        return_value=True,
    )
    @patch("sglang.multimodal_gen.runtime.utils.profiler.torch.cuda.cudart")
    def test_ensure_stopped_stops_active_profiling(self, mock_cudart, _):
        mock_rt = MagicMock()
        mock_cudart.return_value = mock_rt
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = "2-5"
            p = DiffusionStepProfiler()
        for _ in range(3):
            p.step()  # step 0, 1, 2 — enters range at step 2
        p.ensure_stopped()
        mock_rt.cudaProfilerStop.assert_called_once()

    @patch(
        "sglang.multimodal_gen.runtime.utils.profiler.torch.cuda.is_available",
        return_value=True,
    )
    @patch("sglang.multimodal_gen.runtime.utils.profiler.torch.cuda.cudart")
    def test_ensure_stopped_is_idempotent(self, mock_cudart, _):
        mock_rt = MagicMock()
        mock_cudart.return_value = mock_rt
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = "2-5"
            p = DiffusionStepProfiler()
        for _ in range(3):
            p.step()
        p.ensure_stopped()
        p.ensure_stopped()  # second call must be a no-op
        self.assertEqual(mock_rt.cudaProfilerStop.call_count, 1)

    def test_ensure_stopped_noop_when_not_started(self):
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = "100-150"
            p = DiffusionStepProfiler()
        # Never reached range — ensure_stopped should be safe to call
        p.ensure_stopped()
        self.assertFalse(p.profiling_active)


class TestLogRequestStart(unittest.TestCase):
    def setUp(self):
        DiffusionStepProfiler.reset()

    def tearDown(self):
        DiffusionStepProfiler.reset()

    def test_noop_when_no_profile_range(self):
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = None
            p = DiffusionStepProfiler()
        with patch("sglang.multimodal_gen.runtime.utils.profiler.logger") as mock_log:
            p.log_request_start(num_steps=50)
            mock_log.info.assert_not_called()

    @patch(
        "sglang.multimodal_gen.runtime.utils.profiler.is_primary_rank",
        return_value=False,
    )
    def test_noop_on_non_primary_rank(self, _):
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = "100-150"
            p = DiffusionStepProfiler()
        with patch("sglang.multimodal_gen.runtime.utils.profiler.logger") as mock_log:
            p.log_request_start(num_steps=50)
            mock_log.info.assert_not_called()

    def test_logs_will_be_profiled_when_overlapping(self):
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = "100-150"
            p = DiffusionStepProfiler()
        p.global_step_count = 90
        with patch("sglang.multimodal_gen.runtime.utils.profiler.logger") as mock_log:
            p.log_request_start(num_steps=50)  # global steps 90-139 — overlaps 100-150
        logged = mock_log.info.call_args[0][0]
        self.assertIn("WILL BE PROFILED", logged)

    def test_logs_not_in_range_when_no_overlap(self):
        with patch("sglang.multimodal_gen.runtime.utils.profiler.envs") as mock_envs:
            mock_envs.SGLANG_DIFFUSION_PROFILE_STEP_RANGE = "100-150"
            p = DiffusionStepProfiler()
        p.global_step_count = 0
        with patch("sglang.multimodal_gen.runtime.utils.profiler.logger") as mock_log:
            p.log_request_start(num_steps=50)  # global steps 0-49 — no overlap
        logged = mock_log.info.call_args[0][0]
        self.assertIn("not in profile range", logged)


if __name__ == "__main__":
    unittest.main()
