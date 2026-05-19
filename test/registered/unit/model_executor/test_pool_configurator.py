"""Unit tests for pool_configurator.py -- CPU only, no GPU required.

Tests the end-to-end computation: available_bytes -> MemoryPoolConfig,
verifying tokens are correct, constraints are respected, and memory
invariants hold (tokens * per_token_cost <= available_bytes).
"""

import contextlib
import importlib.abc
import importlib.machinery
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci


def maybe_stub_sgl_kernel():
    try:
        import sgl_kernel  # noqa: F401

        return
    except (ImportError, OSError):
        pass

    class _SglKernelLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__getattr__ = lambda name: MagicMock()

    class _SglKernelFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "sgl_kernel" or fullname.startswith("sgl_kernel."):
                return importlib.machinery.ModuleSpec(
                    fullname,
                    _SglKernelLoader(),
                    is_package=True,
                )
            return None

    sys.meta_path.insert(0, _SglKernelFinder())


maybe_stub_sgl_kernel()

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@contextlib.contextmanager
def mock_cpu_env(kv_size=2, tp_size=1):
    """Mock GPU-dependent functions for CPU-only testing."""
    import sglang.srt.model_executor.pool_configurator  # noqa: F401

    def element_size(dtype):
        if dtype is torch.uint8:
            return 1
        return kv_size

    with (
        patch("torch._utils._element_size", side_effect=element_size),
        patch(
            "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
            return_value=tp_size,
        ),
    ):
        yield


def _make_model_runner(
    *,
    num_kv_heads=4,
    head_dim=64,
    v_head_dim=64,
    num_layers=32,
    use_mla_backend=False,
    is_hybrid_swa=False,
    full_attention_layer_ids=None,
    swa_attention_layer_ids=None,
    swa_num_kv_heads=None,
    swa_head_dim=None,
    swa_v_head_dim=None,
    swa_full_tokens_ratio=0.5,
    page_size=1,
    mambaish_config=None,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    index_head_dim=128,
    hf_config=None,
    enable_hisparse=False,
    max_running_requests=None,
    hisparse_config=None,
    kv_cache_dtype="fake_bf16",
):
    """Create a mock ModelRunner with the fields configurators need."""
    mr = MagicMock()

    mr.use_mla_backend = use_mla_backend
    mr.is_draft_worker = False
    mr.num_effective_layers = num_layers
    mr.start_layer = 0
    mr.end_layer = num_layers
    mr.mambaish_config = mambaish_config
    mr.is_hybrid_swa = is_hybrid_swa
    mr.enable_hisparse = enable_hisparse

    mc = SimpleNamespace()
    mc.head_dim = head_dim
    mc.v_head_dim = v_head_dim
    mc.kv_lora_rank = kv_lora_rank
    mc.qk_rope_head_dim = qk_rope_head_dim
    mc.index_head_dim = index_head_dim
    mc.is_hybrid_swa = is_hybrid_swa
    mc.full_attention_layer_ids = (
        full_attention_layer_ids
        if full_attention_layer_ids is not None
        else list(range(num_layers))
    )
    mc.swa_attention_layer_ids = (
        swa_attention_layer_ids if swa_attention_layer_ids is not None else []
    )
    mc.swa_head_dim = swa_head_dim or head_dim
    mc.swa_v_head_dim = swa_v_head_dim or v_head_dim
    mc.get_num_kv_heads = lambda tp_size: num_kv_heads
    mc.get_swa_num_kv_heads = lambda tp_size: swa_num_kv_heads or num_kv_heads
    mc.hf_config = hf_config or SimpleNamespace(architectures=["LlamaForCausalLM"])
    mr.model_config = mc

    mr.kv_cache_dtype = kv_cache_dtype

    sa = SimpleNamespace()
    sa.swa_full_tokens_ratio = swa_full_tokens_ratio
    sa.page_size = page_size
    sa.max_running_requests = max_running_requests
    sa.hisparse_config = hisparse_config
    mr.server_args = sa

    spec = MagicMock()
    spec.is_dflash.return_value = False
    spec.is_none.return_value = True
    mr.spec_algorithm = spec

    return mr


KV_SIZE = 2  # bf16


def _full_per_token(mr):
    mc = mr.model_config
    return mc.get_num_kv_heads(1) * (mc.head_dim + mc.v_head_dim) * KV_SIZE


def _swa_per_token(mr):
    mc = mr.model_config
    return mc.get_swa_num_kv_heads(1) * (mc.swa_head_dim + mc.swa_v_head_dim) * KV_SIZE


def _actual_memory_used(mr, config):
    """Compute actual memory consumed by the pool sizes in config."""
    mc = mr.model_config
    full_pt = _full_per_token(mr)
    swa_pt = _swa_per_token(mr)
    nf = len(mc.full_attention_layer_ids)
    ns = len(mc.swa_attention_layer_ids)

    if mr.is_hybrid_swa:
        full = config.full_max_total_num_tokens or 0
        swa = config.swa_max_total_num_tokens or 0
        return full * full_pt * nf + swa * swa_pt * ns
    else:
        return config.max_total_num_tokens * full_pt * (nf + ns)


class TestDefaultConfigurator(unittest.TestCase):
    """Default (MHA): available_bytes -> tokens, memory invariant holds."""

    def _run(self, available_bytes, page_size=1, **kwargs):
        mr = _make_model_runner(page_size=page_size, **kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, cfg, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_page_alignment(self):
        available = 10_000_000
        _, _, config = self._run(available, page_size=128)
        self.assertEqual(config.max_total_num_tokens % 128, 0)

    def test_constraint_respected(self):
        """calculate_pool_sizes_from_max_tokens respects the limit."""
        mr, cfg, config = self._run(10_000_000)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(100, page_size=1)
        self.assertEqual(constrained.max_total_num_tokens, 100)

    def test_constraint_page_aligned(self):
        mr, cfg, _ = self._run(10_000_000, page_size=128)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(1000, page_size=128)
        self.assertEqual(constrained.max_total_num_tokens, 896)  # 1000 // 128 * 128

    def test_no_swa_fields(self):
        _, _, config = self._run(10_000_000)
        self.assertIsNone(config.full_max_total_num_tokens)
        self.assertIsNone(config.swa_max_total_num_tokens)


class TestDSAModelConfigurator(unittest.TestCase):
    """GLM-5 DSA/NSA MLA memory sizing against pool allocation shapes."""

    PAGE_SIZE = 64
    NUM_LAYERS = 78
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    INDEX_HEAD_DIM = 128
    AVAILABLE_GPU_GB = (32, 64, 96)
    DEVICE_BUFFER_SIZES = (4096, 6144, 8192)
    BATCH_SIZES = (32, 64, 96)
    HOST_TO_DEVICE_RATIOS = (2, 5, 8)

    class _FakeTensor:
        _next_ptr = 1

        def __init__(self, shape, dtype=torch.float32, device="cpu"):
            if isinstance(shape, torch.Size):
                shape = tuple(shape)
            elif isinstance(shape, int):
                shape = (shape,)
            elif not isinstance(shape, tuple):
                shape = tuple(shape)

            self.shape = tuple(int(dim) for dim in shape)
            self.dtype = dtype
            self.device = device
            self._data_ptr = TestDSAModelConfigurator._FakeTensor._next_ptr
            TestDSAModelConfigurator._FakeTensor._next_ptr += max(1, self.nbytes)

        @property
        def nbytes(self):
            numel = 1
            for dim in self.shape:
                numel *= dim
            return numel * self.dtype.itemsize

        def numel(self):
            numel = 1
            for dim in self.shape:
                numel *= dim
            return numel

        def element_size(self):
            return self.dtype.itemsize

        def data_ptr(self):
            return self._data_ptr

        def __len__(self):
            return self.shape[0]

        def __getitem__(self, key):
            if not isinstance(key, tuple):
                key = (key,)

            next_dim = 0
            new_shape = []
            for item in key:
                if item is Ellipsis:
                    remaining = len(self.shape) - (len(key) - 1)
                    new_shape.extend(self.shape[next_dim : next_dim + remaining])
                    next_dim += remaining
                elif isinstance(item, int):
                    next_dim += 1
                elif isinstance(item, slice):
                    start, stop, step = item.indices(self.shape[next_dim])
                    new_shape.append(len(range(start, stop, step)))
                    next_dim += 1
                else:
                    new_shape.append(self.shape[next_dim])
                    next_dim += 1
            new_shape.extend(self.shape[next_dim:])
            return TestDSAModelConfigurator._FakeTensor(
                tuple(new_shape), self.dtype, self.device
            )

        def transpose(self, dim0, dim1):
            shape = list(self.shape)
            shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
            return TestDSAModelConfigurator._FakeTensor(
                tuple(shape), self.dtype, self.device
            )

    @staticmethod
    def _make_dsa_hf_config(index_head_dim=INDEX_HEAD_DIM):
        return SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=index_head_dim,
        )

    def _make_dsa_runner(
        self,
        *,
        enable_hisparse=False,
        max_running_requests=None,
        device_buffer_size=None,
        host_to_device_ratio=None,
    ):
        hisparse_config = None
        if enable_hisparse:
            hisparse_config = json.dumps(
                {
                    "top_k": 2048,
                    "device_buffer_size": device_buffer_size,
                    "host_to_device_ratio": host_to_device_ratio,
                }
            )

        return _make_model_runner(
            num_layers=self.NUM_LAYERS,
            use_mla_backend=True,
            page_size=self.PAGE_SIZE,
            kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM,
            index_head_dim=self.INDEX_HEAD_DIM,
            hf_config=self._make_dsa_hf_config(),
            enable_hisparse=enable_hisparse,
            max_running_requests=max_running_requests,
            hisparse_config=hisparse_config,
            kv_cache_dtype=torch.bfloat16,
        )

    @contextlib.contextmanager
    def _pool_shape_only_allocation(self):
        def fake_tensor_from_args(args, kwargs):
            if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
                shape = tuple(args[0])
            else:
                shape = tuple(args)
            return self._FakeTensor(
                shape,
                kwargs.get("dtype", torch.float32),
                kwargs.get("device", "cpu"),
            )

        def fake_arange(*args, **kwargs):
            if len(args) == 1:
                start, end, step = 0, args[0], 1
            else:
                start = args[0]
                end = args[1]
                step = args[2] if len(args) > 2 else 1
            length = max(0, (end - start + step - 1) // step)
            return self._FakeTensor(
                (length,),
                kwargs.get("dtype", torch.int64),
                kwargs.get("device", "cpu"),
            )

        with (
            patch("torch.empty", side_effect=fake_tensor_from_args),
            patch("torch.zeros", side_effect=fake_tensor_from_args),
            patch("torch.arange", side_effect=fake_arange),
        ):
            yield

    def _calculate_config(self, mr, available_bytes):
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, mr.server_args.page_size)
        return cfg, config

    def _make_nsa_pool(self, max_total_num_tokens):
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

        return NSATokenToKVPool(
            size=max_total_num_tokens,
            page_size=self.PAGE_SIZE,
            kv_lora_rank=self.KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM,
            layer_num=self.NUM_LAYERS,
            device="cpu",
            index_head_dim=self.INDEX_HEAD_DIM,
            enable_memory_saver=False,
            kv_cache_dim=self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
        )

    def _make_hisparse_nsa_pool(self, max_total_num_tokens, host_to_device_ratio):
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseNSATokenToKVPool

        return HiSparseNSATokenToKVPool(
            size=max_total_num_tokens,
            page_size=self.PAGE_SIZE,
            kv_lora_rank=self.KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM,
            layer_num=self.NUM_LAYERS,
            device="cpu",
            index_head_dim=self.INDEX_HEAD_DIM,
            enable_memory_saver=False,
            kv_cache_dim=self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            host_to_device_ratio=host_to_device_ratio,
        )

    def _make_hisparse_host_pool(
        self, device_pool, host_to_device_ratio, available_cpu_bytes
    ):
        from sglang.srt.mem_cache.memory_pool_host import (
            HICACHE_HOST_MEMORY_RESERVE_BYTES,
            MLATokenToKVPoolHost,
        )

        with patch(
            "sglang.srt.mem_cache.memory_pool_host.psutil.virtual_memory",
            return_value=SimpleNamespace(
                available=available_cpu_bytes + HICACHE_HOST_MEMORY_RESERVE_BYTES
            ),
        ):
            return MLATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=1,
                layout="layer_first",
                pin_memory=False,
                override_kv_cache_dim=device_pool.kv_cache_dim,
            )

    @staticmethod
    def _device_pool_bytes(pool):
        return sum(buf.nbytes for buf in pool.kv_buffer) + sum(
            buf.nbytes for buf in pool.index_k_with_scale_buffer
        )

    @staticmethod
    def _host_pool_bytes(pool):
        return pool.kv_buffer.nbytes

    def _main_kv_cell_size(self):
        return (
            (self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM)
            * self.NUM_LAYERS
            * torch._utils._element_size(torch.bfloat16)
        )

    @staticmethod
    def _align_up(value, page_size):
        return ((value + page_size - 1) // page_size) * page_size

    def _expected_hisparse_pool_size(self, buffer_size, batch_size):
        return self._align_up(buffer_size * batch_size, self.PAGE_SIZE)

    def test_non_hisparse_full_gpu_pool_fits_available_memory(self):
        mr = self._make_dsa_runner()
        for available_gpu_gb in self.AVAILABLE_GPU_GB:
            available_gpu = available_gpu_gb * 1024**3
            with self.subTest(available_gpu_gb=available_gpu_gb):
                _, config = self._calculate_config(mr, available_gpu)
                with self._pool_shape_only_allocation():
                    pool = self._make_nsa_pool(config.max_total_num_tokens)

                actual_device_bytes = self._device_pool_bytes(pool)

                self.assertLessEqual(actual_device_bytes, available_gpu)
                self.assertEqual(config.max_total_num_tokens % self.PAGE_SIZE, 0)

    def test_hisparse_gpu_and_cpu_pool_fit_memory_budget(self):
        for available_gpu_gb in self.AVAILABLE_GPU_GB:
            available_gpu = available_gpu_gb * 1024**3
            for buffer_size in self.DEVICE_BUFFER_SIZES:
                for batch_size in self.BATCH_SIZES:
                    hot_tokens = self._expected_hisparse_pool_size(
                        buffer_size, batch_size
                    )
                    hot_main_bytes = hot_tokens * self._main_kv_cell_size()
                    if hot_main_bytes > available_gpu:
                        continue
                    for host_to_device_ratio in self.HOST_TO_DEVICE_RATIOS:
                        available_cpu = available_gpu * host_to_device_ratio
                        with self._pool_shape_only_allocation():
                            expected_device_pool = self._make_hisparse_nsa_pool(
                                hot_tokens, host_to_device_ratio
                            )
                            expected_host_pool = self._make_hisparse_host_pool(
                                expected_device_pool,
                                host_to_device_ratio,
                                available_cpu,
                            )
                        if (
                            self._device_pool_bytes(expected_device_pool)
                            > available_gpu
                        ):
                            continue
                        if self._host_pool_bytes(expected_host_pool) > available_cpu:
                            continue

                        with self.subTest(
                            available_gpu_gb=available_gpu_gb,
                            buffer_size=buffer_size,
                            batch_size=batch_size,
                            host_to_device_ratio=host_to_device_ratio,
                        ):
                            mr = self._make_dsa_runner(
                                enable_hisparse=True,
                                max_running_requests=batch_size,
                                device_buffer_size=buffer_size,
                                host_to_device_ratio=host_to_device_ratio,
                            )
                            _, config = self._calculate_config(mr, available_gpu)
                            with self._pool_shape_only_allocation():
                                device_pool = self._make_hisparse_nsa_pool(
                                    config.max_total_num_tokens,
                                    host_to_device_ratio,
                                )
                                host_pool = self._make_hisparse_host_pool(
                                    device_pool,
                                    host_to_device_ratio,
                                    available_cpu,
                                )

                            actual_gpu_bytes = self._device_pool_bytes(device_pool)
                            actual_cpu_bytes = self._host_pool_bytes(host_pool)

                            self.assertGreaterEqual(
                                config.max_total_num_tokens, hot_tokens
                            )
                            self.assertLessEqual(actual_gpu_bytes, available_gpu)
                            self.assertLessEqual(actual_cpu_bytes, available_cpu)
                            self.assertEqual(
                                config.max_total_num_tokens % self.PAGE_SIZE, 0
                            )


class TestHybridSWAConfigurator(unittest.TestCase):
    """Hybrid SWA: full/swa split, ratio, memory invariant."""

    def _make_swa_runner(self, full_layers=16, swa_layers=16, ratio=0.5, page_size=1):
        return _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(full_layers)),
            swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
            swa_num_kv_heads=4,
            page_size=page_size,
            swa_full_tokens_ratio=ratio,
        )

    def _run(self, available_bytes, **kwargs):
        mr = self._make_swa_runner(**kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, mr.server_args.page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_ratio_respected(self):
        """swa_tokens ~= full_tokens * ratio (within page alignment)"""
        available = 10_000_000
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            mr, _, config = self._run(available, ratio=ratio, page_size=1)
            full = config.full_max_total_num_tokens
            swa = config.swa_max_total_num_tokens
            self.assertEqual(swa, int(full * ratio), f"ratio={ratio}")

    def test_ratio_with_page_alignment(self):
        """With page alignment, swa_tokens = align(full_tokens * ratio)"""
        available = 10_000_000
        mr, _, config = self._run(available, ratio=0.5, page_size=128)
        full = config.full_max_total_num_tokens
        swa = config.swa_max_total_num_tokens
        self.assertEqual(full % 128, 0)
        self.assertEqual(swa % 128, 0)
        self.assertEqual(swa, (int(full * 0.5) // 128) * 128)

    def test_max_total_equals_full(self):
        """For hybrid, max_total_num_tokens = full_max_total_num_tokens"""
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.full_max_total_num_tokens)

    def test_constraint_respected(self):
        """full_tokens = constrained value after re-run"""
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(200, page_size=1)
        self.assertEqual(config.full_max_total_num_tokens, 200)
        self.assertEqual(config.swa_max_total_num_tokens, 100)

    def test_constraint_memory_within_budget(self):
        """After constraint, memory <= original budget (but less than profiled due to constraint)."""
        available = 10_000_000
        mr, cfg, original = self._run(available, page_size=1)
        user_limit = original.full_max_total_num_tokens // 2
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(
                user_limit, mr.server_args.page_size
            )
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        # constrained should use roughly half the memory
        original_used = _actual_memory_used(mr, original)
        self.assertAlmostEqual(used / original_used, 0.5, delta=0.01)

    def test_different_layer_counts(self):
        """Asymmetric full/swa layer counts"""
        available = 10_000_000
        mr, _, config = self._run(available, full_layers=24, swa_layers=8, ratio=0.5)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertEqual(
            config.swa_max_total_num_tokens,
            int(config.full_max_total_num_tokens * 0.5),
        )


class TestAllSWAConfigurator(unittest.TestCase):
    """All-SWA (full_layers=0): special case."""

    def _run(self, available_bytes, ratio=0.5, page_size=1):
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=[],
            swa_attention_layer_ids=list(range(32)),
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=ratio,
            page_size=page_size,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_full_max_is_zero(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.full_max_total_num_tokens, 0)

    def test_max_total_equals_swa(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.swa_max_total_num_tokens)

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        swa_pt = _swa_per_token(mr)
        ns = len(mr.model_config.swa_attention_layer_ids)
        used = config.swa_max_total_num_tokens * swa_pt * ns
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_constraint_respected(self):
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(500, page_size=1)
        self.assertEqual(config.max_total_num_tokens, 500)
        self.assertEqual(config.swa_max_total_num_tokens, 500)


class TestFactory(unittest.TestCase):
    def test_default_for_non_swa(self):
        mr = _make_model_runner(is_hybrid_swa=False)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                DefaultPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, DefaultPoolConfigurator)

    def test_swa_for_hybrid(self):
        mr = _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(16)),
            swa_attention_layer_ids=list(range(16, 32)),
            swa_num_kv_heads=4,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                HybridSWAPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, HybridSWAPoolConfigurator)


if __name__ == "__main__":
    unittest.main()
