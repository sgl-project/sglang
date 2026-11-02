"""
Unit tests for sglang.srt.hardware_backend.npu.utils
"""

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import torch

from sglang.srt.hardware_backend.npu.utils import (
    NPUACLFormat,
    _call_once,
    _is_nz_aligned,
    get_indexer_weight_stream,
    get_routed_stream,
    get_share_stream,
    npu_format_cast,
    set_default_server_args,
    set_routed_stream,
    set_share_stream,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3, suite="stage-a-unit-test-npu")


class TestNPUACLFormat(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(int(NPUACLFormat.ACL_FORMAT_UNDEFINED), -1)
        self.assertEqual(int(NPUACLFormat.ACL_FORMAT_ND), 2)
        self.assertEqual(int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ), 29)

    def test_is_intenum(self):
        self.assertIsInstance(int(NPUACLFormat.ACL_FORMAT_ND), int)
        self.assertIsInstance(NPUACLFormat.ACL_FORMAT_ND, int)


class TestIsNzAligned(unittest.TestCase):
    # ---- bf16 / fp16: both dims divisible by 16 ----

    def test_bf16_aligned(self):
        t = torch.empty(32, 48, dtype=torch.bfloat16)
        self.assertTrue(_is_nz_aligned(t))

    def test_bf16_unaligned_k(self):
        t = torch.empty(17, 32, dtype=torch.bfloat16)
        self.assertFalse(_is_nz_aligned(t))

    def test_bf16_unaligned_n(self):
        t = torch.empty(32, 17, dtype=torch.bfloat16)
        self.assertFalse(_is_nz_aligned(t))

    def test_fp16_aligned(self):
        t = torch.empty(16, 16, dtype=torch.float16)
        self.assertTrue(_is_nz_aligned(t))

    def test_fp16_unaligned(self):
        t = torch.empty(8, 16, dtype=torch.float16)
        self.assertFalse(_is_nz_aligned(t))

    # ---- int8: k % 16 == 0, n % 32 == 0 ----

    def test_int8_aligned(self):
        t = torch.empty(16, 32, dtype=torch.int8)
        self.assertTrue(_is_nz_aligned(t))

    def test_int8_n_not_div32(self):
        t = torch.empty(16, 16, dtype=torch.int8)
        self.assertFalse(_is_nz_aligned(t))

    def test_int8_k_not_div16(self):
        t = torch.empty(8, 32, dtype=torch.int8)
        self.assertFalse(_is_nz_aligned(t))

    # ---- uint8 / int32 (INT4 packed): k % 16 == 0, n % 64 == 0 ----

    def test_uint8_aligned(self):
        t = torch.empty(16, 64, dtype=torch.uint8)
        self.assertTrue(_is_nz_aligned(t))

    def test_uint8_unaligned_n(self):
        t = torch.empty(16, 32, dtype=torch.uint8)
        self.assertFalse(_is_nz_aligned(t))

    def test_int32_aligned(self):
        t = torch.empty(32, 64, dtype=torch.int32)
        self.assertTrue(_is_nz_aligned(t))

    # ---- other dtypes: always True ----

    def test_float32_always_true(self):
        t = torch.empty(3, 5, dtype=torch.float32)
        self.assertTrue(_is_nz_aligned(t))

    def test_float64_always_true(self):
        t = torch.empty(1, 1, dtype=torch.float64)
        self.assertTrue(_is_nz_aligned(t))

    # ---- dimension checks ----

    def test_1d_tensor_returns_false(self):
        t = torch.empty(32, dtype=torch.bfloat16)
        self.assertFalse(_is_nz_aligned(t))

    def test_3d_tensor_checks_last_two(self):
        t = torch.empty(4, 16, 16, dtype=torch.bfloat16)
        self.assertTrue(_is_nz_aligned(t))
        t2 = torch.empty(4, 8, 16, dtype=torch.bfloat16)
        self.assertFalse(_is_nz_aligned(t2))


class TestCallOnce(unittest.TestCase):
    def test_runs_exactly_once(self):
        call_count = 0

        @_call_once
        def fn():
            nonlocal call_count
            call_count += 1
            return "first"

        self.assertEqual(fn(), "first")
        self.assertEqual(call_count, 1)
        # Second call is a no-op (returns None).
        self.assertIsNone(fn())
        self.assertEqual(call_count, 1)

    def test_second_call_returns_none(self):
        @_call_once
        def fn(x):
            return x * 2

        self.assertEqual(fn(5), 10)
        self.assertIsNone(fn(5))

    def test_with_args(self):
        captured = []

        @_call_once
        def fn(a, b, c=3):
            captured.append((a, b, c))
            return a + b + c

        self.assertEqual(fn(1, 2, c=3), 6)
        self.assertEqual(captured, [(1, 2, 3)])
        # Second call with different args is still a no-op.
        self.assertIsNone(fn(100, 200, c=300))
        self.assertEqual(len(captured), 1)

    def test_multiple_independent_decorators(self):
        call_count1 = 0
        call_count2 = 0

        @_call_once
        def fn1():
            nonlocal call_count1
            call_count1 += 1

        @_call_once
        def fn2():
            nonlocal call_count2
            call_count2 += 1

        fn1()
        fn1()
        fn2()
        self.assertEqual(call_count1, 1)
        self.assertEqual(call_count2, 1)


class TestNpuFormatCast(unittest.TestCase):
    def test_returns_tensor_unchanged(self):
        t = torch.randn(4, 4)
        result = npu_format_cast(t)
        self.assertIs(result, t)

    def test_explicit_fractal_nz_on_cpu(self):
        t = torch.randn(32, 32)
        result = npu_format_cast(t, acl_format=NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)
        self.assertIs(result, t)

    def test_nd_format_on_cpu(self):
        t = torch.randn(8, 8)
        result = npu_format_cast(t, acl_format=NPUACLFormat.ACL_FORMAT_ND)
        self.assertIs(result, t)


class TestNpuFormatCastExtended(unittest.TestCase):
    @patch("sglang.srt.hardware_backend.npu.utils._is_npu", True)
    @patch(
        "sglang.srt.hardware_backend.npu.utils.envs"
        ".SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get"
    )
    def test_disable_acl_format_env(self, mock_env_get):
        mock_env_get.return_value = True
        t = torch.randn(4, 4)
        result = npu_format_cast(t)
        self.assertIs(result, t)
        mock_env_get.assert_called_once()

    @patch("sglang.srt.hardware_backend.npu.utils._is_npu", True)
    @patch("sglang.srt.hardware_backend.npu.utils.logger.warning_once")
    def test_cpu_tensor_warning_and_return(self, mock_warning_once):
        t = torch.randn(4, 4)
        result = npu_format_cast(t)
        self.assertIs(result, t)
        mock_warning_once.assert_called_once()
        expected_msg = "The conversion from 'ND' to 'NZ' does not work on the CPU"
        self.assertIn(expected_msg, mock_warning_once.call_args[0][0])

    @patch("sglang.srt.hardware_backend.npu.utils._is_npu", True)
    def test_meta_tensor_skipped(self):
        t = torch.empty(4, 4, device="meta")
        result = npu_format_cast(t)
        self.assertIs(result, t)

    @patch("sglang.srt.hardware_backend.npu.utils._is_npu", True)
    def test_unaligned_tensor_falls_back_to_nd(self):
        t = torch.randn(3, 5, dtype=torch.bfloat16)
        with patch(
            "torch.Tensor.device",
            new_callable=PropertyMock,
            return_value=torch.device("npu:0"),
        ):
            result = npu_format_cast(t, acl_format=NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)
        self.assertIs(result, t)

    @patch("sglang.srt.hardware_backend.npu.utils._is_npu", True)
    @patch("sglang.srt.hardware_backend.npu.utils.torch.ops.npu.npu_format_cast")
    def test_fp4_kwargs_bypasses_alignment_check(self, mock_npu_format_cast):
        mock_npu_format_cast.return_value = "mocked"
        t = torch.randn(3, 5)  # unaligned — would be caught by alignment
        with patch(
            "torch.Tensor.device",
            new_callable=PropertyMock,
            return_value=torch.device("npu:0"),
        ):
            result = npu_format_cast(
                t,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch.float8_e4m3fn,
            )
        mock_npu_format_cast.assert_called_once_with(
            t,
            int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ),
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch.float8_e4m3fn,
        )
        self.assertEqual(result, "mocked")

    @patch("sglang.srt.hardware_backend.npu.utils._is_npu", True)
    @patch("sglang.srt.hardware_backend.npu.utils.torch.ops.npu.npu_format_cast")
    def test_fp4_input_dtype_only(self, mock_npu_format_cast):
        mock_npu_format_cast.return_value = "mocked"
        t = torch.randn(32, 32)
        with patch(
            "torch.Tensor.device",
            new_callable=PropertyMock,
            return_value=torch.device("npu:0"),
        ):
            result = npu_format_cast(t, input_dtype=torch.float8_e4m3fn)
        mock_npu_format_cast.assert_called_once()
        self.assertEqual(result, "mocked")

    @patch("sglang.srt.hardware_backend.npu.utils._is_npu", True)
    @patch("sglang.srt.hardware_backend.npu.utils.torch.ops.npu.npu_format_cast")
    def test_aligned_npu_tensor_calls_format_cast(self, mock_npu_format_cast):
        mock_npu_format_cast.return_value = "mocked"
        t = torch.randn(32, 32, dtype=torch.bfloat16)
        with patch(
            "torch.Tensor.device",
            new_callable=PropertyMock,
            return_value=torch.device("npu:0"),
        ):
            result = npu_format_cast(t, acl_format=NPUACLFormat.ACL_FORMAT_ND)
        mock_npu_format_cast.assert_called_once_with(t, 2)
        self.assertEqual(result, "mocked")

    @patch("sglang.srt.hardware_backend.npu.utils._is_npu", True)
    @patch("sglang.srt.hardware_backend.npu.utils.torch.ops.npu.npu_format_cast")
    def test_default_fractal_nz_on_npu(self, mock_npu_format_cast):
        mock_npu_format_cast.return_value = "mocked"
        t = torch.randn(32, 32, dtype=torch.bfloat16)
        with patch(
            "torch.Tensor.device",
            new_callable=PropertyMock,
            return_value=torch.device("npu:0"),
        ):
            result = npu_format_cast(t)
        mock_npu_format_cast.assert_called_once_with(t, 29)
        self.assertEqual(result, "mocked")


class TestSetDefaultServerArgs(unittest.TestCase):
    def _make_mock_args(self, **kwargs):
        defaults = {
            "attention_backend": None,
            "prefill_attention_backend": None,
            "decode_attention_backend": None,
            "page_size": None,
            "chunked_prefill_size": None,
            "tp_size": 1,
            "disable_custom_all_reduce": False,
            "enable_hierarchical_cache": False,
            "hicache_io_backend": None,
            "hicache_mem_layout": None,
        }
        defaults.update(kwargs)

        decode = MagicMock()
        decode.max_bs = None

        cuda_graph_config = MagicMock()
        cuda_graph_config.decode = decode

        args = MagicMock()
        for k, v in defaults.items():
            setattr(args, k, v)
        args.cuda_graph_config = cuda_graph_config
        args.use_mla_backend = MagicMock(return_value=False)
        return args

    # -- basic attention / page size --

    def test_sets_attention_backend_to_ascend(self):
        args = self._make_mock_args()
        set_default_server_args(args)
        self.assertEqual(args.attention_backend, "ascend")
        self.assertEqual(args.prefill_attention_backend, "ascend")
        self.assertEqual(args.decode_attention_backend, "ascend")

    def test_page_size_default_when_none(self):
        args = self._make_mock_args(page_size=None)
        set_default_server_args(args)
        self.assertEqual(args.page_size, 128)

    def test_page_size_keeps_explicit(self):
        args = self._make_mock_args(page_size=256)
        set_default_server_args(args)
        self.assertEqual(args.page_size, 256)

    def test_page_size_not_set_when_not_none(self):
        args = self._make_mock_args(page_size=64)
        set_default_server_args(args)
        self.assertEqual(args.page_size, 64)

    # -- small memory (<= 32 GB) --

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_small_mem_low_tp(self, mock_get_mem):
        mock_get_mem.return_value = 32 * 1024
        args = self._make_mock_args(tp_size=1)
        set_default_server_args(args)
        self.assertEqual(args.chunked_prefill_size, 4 * 1024)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 16)

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_small_mem_high_tp(self, mock_get_mem):
        mock_get_mem.return_value = 32 * 1024
        args = self._make_mock_args(tp_size=4)
        set_default_server_args(args)
        self.assertEqual(args.chunked_prefill_size, 4 * 1024)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 64)

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_small_mem_exact_boundary(self, mock_get_mem):
        mock_get_mem.return_value = 32 * 1024
        args = self._make_mock_args(tp_size=1)
        set_default_server_args(args)
        self.assertEqual(args.chunked_prefill_size, 4 * 1024)

    # -- medium memory (32 GB < mem <= 64 GB) --

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_large_mem_low_tp(self, mock_get_mem):
        mock_get_mem.return_value = 48 * 1024
        args = self._make_mock_args(tp_size=1)
        set_default_server_args(args)
        self.assertEqual(args.chunked_prefill_size, 8 * 1024)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 64)

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_large_mem_high_tp(self, mock_get_mem):
        mock_get_mem.return_value = 60 * 1024
        args = self._make_mock_args(tp_size=8)
        set_default_server_args(args)
        self.assertEqual(args.chunked_prefill_size, 8 * 1024)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 256)

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_large_mem_exact_boundary(self, mock_get_mem):
        mock_get_mem.return_value = 64 * 1024
        args = self._make_mock_args(tp_size=1)
        set_default_server_args(args)
        self.assertEqual(args.chunked_prefill_size, 8 * 1024)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 64)

    # -- very large memory (> 64 GB) --

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_very_large_mem_no_defaults(self, mock_get_mem):
        mock_get_mem.return_value = 128 * 1024
        args = self._make_mock_args(chunked_prefill_size=None)
        args.cuda_graph_config.decode.max_bs = None
        set_default_server_args(args)
        self.assertIsNone(args.chunked_prefill_size)
        self.assertIsNone(args.cuda_graph_config.decode.max_bs)

    # -- explicit values are preserved --

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_skip_chunked_prefill_when_explicit(self, mock_get_mem):
        mock_get_mem.return_value = 32 * 1024
        args = self._make_mock_args(chunked_prefill_size=16 * 1024)
        set_default_server_args(args)
        self.assertEqual(args.chunked_prefill_size, 16 * 1024)

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_skip_max_bs_when_explicit(self, mock_get_mem):
        mock_get_mem.return_value = 32 * 1024
        args = self._make_mock_args()
        args.cuda_graph_config.decode.max_bs = 128
        set_default_server_args(args)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 128)

    # -- always-set fields --

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_disable_custom_all_reduce(self, mock_get_mem):
        mock_get_mem.return_value = 32 * 1024
        args = self._make_mock_args()
        set_default_server_args(args)
        self.assertTrue(args.disable_custom_all_reduce)

    # -- hierarchical cache --

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_hierarchical_cache_mla(self, mock_get_mem):
        mock_get_mem.return_value = 32 * 1024
        args = self._make_mock_args(enable_hierarchical_cache=True)
        args.use_mla_backend = MagicMock(return_value=True)
        set_default_server_args(args)
        self.assertEqual(args.hicache_io_backend, "kernel_ascend")
        self.assertEqual(args.hicache_mem_layout, "page_first_kv_split")

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_hierarchical_cache_non_mla(self, mock_get_mem):
        mock_get_mem.return_value = 32 * 1024
        args = self._make_mock_args(enable_hierarchical_cache=True)
        args.use_mla_backend = MagicMock(return_value=False)
        set_default_server_args(args)
        self.assertEqual(args.hicache_io_backend, "kernel_ascend")
        self.assertEqual(args.hicache_mem_layout, "page_first_direct")

    def test_no_hierarchical_cache_skipped(self):
        args = self._make_mock_args(enable_hierarchical_cache=False)
        set_default_server_args(args)
        self.assertIsNone(args.hicache_io_backend)
        self.assertIsNone(args.hicache_mem_layout)
        # The disable_custom_all_reduce is still set unconditionally
        self.assertTrue(args.disable_custom_all_reduce)

    # -- edge cases --

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_exact_32gb_tp_edge(self, mock_get_mem):
        mock_get_mem.return_value = 32 * 1024
        args = self._make_mock_args(tp_size=4)
        set_default_server_args(args)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 64)

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_tp_less_than_4(self, mock_get_mem):
        mock_get_mem.return_value = 32 * 1024
        args = self._make_mock_args(tp_size=3)
        set_default_server_args(args)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 16)

    @patch("sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity")
    def test_tp_greater_than_4(self, mock_get_mem):
        mock_get_mem.return_value = 64 * 1024
        args = self._make_mock_args(tp_size=16)
        set_default_server_args(args)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 256)


class TestGetIndexerWeightStream(unittest.TestCase):
    def setUp(self):
        import sglang.srt.hardware_backend.npu.utils as npu_utils

        npu_utils.indexer_weight_stream = None

    def test_creates_and_reuses_stream(self):
        stream1 = get_indexer_weight_stream()
        stream2 = get_indexer_weight_stream()
        self.assertIs(stream1, stream2)
        self.assertIsNotNone(stream1)

    def test_returns_same_on_multiple_calls(self):
        streams = [get_indexer_weight_stream() for _ in range(5)]
        self.assertTrue(all(s is streams[0] for s in streams))


class TestShareRoutedStream(unittest.TestCase):
    def setUp(self):
        import sglang.srt.hardware_backend.npu.utils as npu_utils

        npu_utils.share_stream = None
        npu_utils.routed_stream = None

    def test_share_stream_default_none(self):
        self.assertIsNone(get_share_stream())

    def test_share_stream_get_set(self):
        stream = object()
        set_share_stream(stream)
        self.assertIs(get_share_stream(), stream)

    def test_share_stream_replaces(self):
        stream1 = object()
        stream2 = object()
        set_share_stream(stream1)
        set_share_stream(stream2)
        self.assertIs(get_share_stream(), stream2)

    def test_routed_stream_default_none(self):
        self.assertIsNone(get_routed_stream())

    def test_routed_stream_get_set(self):
        stream = object()
        set_routed_stream(stream)
        self.assertIs(get_routed_stream(), stream)

    def test_routed_stream_replaces(self):
        stream1 = object()
        stream2 = object()
        set_routed_stream(stream1)
        set_routed_stream(stream2)
        self.assertIs(get_routed_stream(), stream2)

    def test_share_and_routed_independent(self):
        share = object()
        routed = object()
        set_share_stream(share)
        set_routed_stream(routed)
        self.assertIs(get_share_stream(), share)
        self.assertIs(get_routed_stream(), routed)


class TestWaitShareRoutedStream(unittest.TestCase):
    def test_wait_share_stream_noop_when_none(self):
        # Ensure stream is None
        import sglang.srt.hardware_backend.npu.utils as npu_utils

        npu_utils.share_stream = None
        # Should not raise
        from sglang.srt.hardware_backend.npu.utils import wait_share_stream

        wait_share_stream()

    def test_wait_routed_stream_noop_when_none(self):
        import sglang.srt.hardware_backend.npu.utils as npu_utils

        npu_utils.routed_stream = None
        from sglang.srt.hardware_backend.npu.utils import wait_routed_stream

        wait_routed_stream()

    @patch("sglang.srt.hardware_backend.npu.utils.torch.get_device_module")
    def test_wait_share_stream_calls_wait_stream(self, mock_get_dev):
        cur_stream = MagicMock()
        mock_dev = MagicMock()
        mock_dev.current_stream.return_value = cur_stream
        mock_get_dev.return_value = mock_dev

        share = MagicMock()
        set_share_stream(share)

        from sglang.srt.hardware_backend.npu.utils import wait_share_stream

        wait_share_stream()
        cur_stream.wait_stream.assert_called_once_with(share)

    @patch("sglang.srt.hardware_backend.npu.utils.torch.get_device_module")
    def test_wait_routed_stream_calls_wait_stream(self, mock_get_dev):
        cur_stream = MagicMock()
        mock_dev = MagicMock()
        mock_dev.current_stream.return_value = cur_stream
        mock_get_dev.return_value = mock_dev

        routed = MagicMock()
        set_routed_stream(routed)

        from sglang.srt.hardware_backend.npu.utils import wait_routed_stream

        wait_routed_stream()
        cur_stream.wait_stream.assert_called_once_with(routed)


class TestProcessExpert(unittest.TestCase):
    def setUp(self):
        import sglang.srt.hardware_backend.npu.utils as npu_utils

        npu_utils.share_stream = None
        npu_utils.routed_stream = None

    @patch("sglang.srt.hardware_backend.npu.utils.torch.get_device_module")
    def test_process_shared_expert_creates_stream(self, mock_get_dev):
        mock_dev = MagicMock()
        new_stream = MagicMock()
        mock_dev.Stream.return_value = new_stream
        current_stream = MagicMock()
        mock_dev.current_stream.return_value = current_stream
        mock_get_dev.return_value = mock_dev

        from sglang.srt.hardware_backend.npu.utils import process_shared_expert

        hidden = MagicMock()
        forward_func = MagicMock(return_value="output")
        result = process_shared_expert(hidden, forward_func)

        # A new stream should have been created
        mock_dev.Stream.assert_called_once()
        # wait_stream is called on the new stream with the current stream as arg
        new_stream.wait_stream.assert_called_once_with(current_stream)
        # forward_func should have been called
        forward_func.assert_called_once_with(hidden)
        self.assertEqual(result, "output")

    @patch("sglang.srt.hardware_backend.npu.utils.torch.get_device_module")
    def test_process_shared_expert_reuses_stream(self, mock_get_dev):
        mock_dev = MagicMock()
        existing_stream = MagicMock()
        current_stream = MagicMock()
        mock_dev.current_stream.return_value = current_stream
        mock_get_dev.return_value = mock_dev

        set_share_stream(existing_stream)

        from sglang.srt.hardware_backend.npu.utils import process_shared_expert

        hidden = MagicMock()
        forward_func = MagicMock(return_value="output")
        result = process_shared_expert(hidden, forward_func)

        # No new stream should be created
        mock_dev.Stream.assert_not_called()
        # wait_stream is called on the existing stream with the current stream as arg
        existing_stream.wait_stream.assert_called_once_with(current_stream)
        forward_func.assert_called_once_with(hidden)
        self.assertEqual(result, "output")

    @patch("sglang.srt.hardware_backend.npu.utils.torch.get_device_module")
    def test_process_routed_expert_creates_stream(self, mock_get_dev):
        mock_dev = MagicMock()
        new_stream = MagicMock()
        mock_dev.Stream.return_value = new_stream
        current_stream = MagicMock()
        mock_dev.current_stream.return_value = current_stream
        mock_get_dev.return_value = mock_dev

        from sglang.srt.hardware_backend.npu.utils import process_routed_expert

        hidden = MagicMock()
        topk_output = MagicMock()
        forward_func = MagicMock(return_value="output")
        result = process_routed_expert(hidden, topk_output, forward_func)

        mock_dev.Stream.assert_called_once()
        new_stream.wait_stream.assert_called_once_with(current_stream)
        forward_func.assert_called_once_with(hidden, topk_output)
        self.assertEqual(result, "output")

    @patch("sglang.srt.hardware_backend.npu.utils.torch.get_device_module")
    def test_process_routed_expert_reuses_stream(self, mock_get_dev):
        mock_dev = MagicMock()
        existing_stream = MagicMock()
        current_stream = MagicMock()
        mock_dev.current_stream.return_value = current_stream
        mock_get_dev.return_value = mock_dev

        set_routed_stream(existing_stream)

        from sglang.srt.hardware_backend.npu.utils import process_routed_expert

        hidden = MagicMock()
        topk_output = MagicMock()
        forward_func = MagicMock(return_value="output")
        result = process_routed_expert(hidden, topk_output, forward_func)

        mock_dev.Stream.assert_not_called()
        existing_stream.wait_stream.assert_called_once_with(current_stream)
        forward_func.assert_called_once_with(hidden, topk_output)
        self.assertEqual(result, "output")

    @patch("sglang.srt.hardware_backend.npu.utils.torch.get_device_module")
    def test_shared_and_routed_streams_independent(self, mock_get_dev):
        mock_dev = MagicMock()
        share_stream_mock = MagicMock()
        routed_stream_mock = MagicMock()
        current_stream = MagicMock()
        mock_dev.current_stream.return_value = current_stream
        mock_get_dev.return_value = mock_dev

        def stream_side_effect():
            if not hasattr(stream_side_effect, "call_count"):
                stream_side_effect.call_count = 0
            stream_side_effect.call_count += 1
            if stream_side_effect.call_count == 1:
                return share_stream_mock
            return routed_stream_mock

        mock_dev.Stream.side_effect = stream_side_effect

        from sglang.srt.hardware_backend.npu.utils import (
            process_routed_expert,
            process_shared_expert,
        )

        process_shared_expert(MagicMock(), MagicMock(return_value="s"))
        process_routed_expert(MagicMock(), MagicMock(), MagicMock(return_value="r"))

        # Two different streams were created
        self.assertEqual(mock_dev.Stream.call_count, 2)


if __name__ == "__main__":
    unittest.main()
