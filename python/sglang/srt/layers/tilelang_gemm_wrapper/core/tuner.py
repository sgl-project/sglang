"""TileLang GEMM Tuner with Ray multi-GPU support."""
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import triton

logger = logging.getLogger(__name__)

ray = None


def _ensure_ray():
    global ray
    if ray is None:
        import ray as _ray
        ray = _ray


def _tflops(M: int, N: int, K: int, latency_ms: float) -> float:
    return 2.0 * M * N * K / latency_ms / 1e9


def _matmul_configs(M: int, N: int, K: int) -> List[dict]:
    """Generate config search space for standard GEMM."""
    tiles_M = [64] if M < 128 else [64, 128]
    tiles_N = [16, 32, 64, 128]
    tiles_K = [128]
    stages = [1, 2, 3, 4]
    threads = [128, 256]
    return [
        dict(block_M=BM, block_N=BN, block_K=BK, num_stages=S, threads=TH)
        for BM in tiles_M
        for BN in tiles_N
        for BK in tiles_K
        for S in stages
        for TH in threads 
        if (BM * BK + BN * BK) * S < 256 * 1024   # check for shared memory
    ]


def _matmul_splitk_configs(M: int, N: int, K: int) -> List[dict]:
    """Generate config search space for Split-K GEMM."""
    tiles_M = [64] if M < 128 else [64, 128]
    tiles_N = [16, 32, 64, 128]
    tiles_K = [128]
    stages = [0, 1, 2]
    threads = [128, 256]
    split_ks = [2, 4, 8]
    
    configs = []
    for BM in tiles_M:
        for BN in tiles_N:
            for BK in tiles_K:
                for S in stages:
                    for TH in threads:
                        for SK in split_ks:
                            if K % SK != 0:
                                continue
                            K_per_split = K // SK
                            if K_per_split % BK != 0:
                                continue
                            if (BM * BK + BN * BK) * max(S, 1) < 256 * 1024:    # check for shared memory
                                configs.append(dict(
                                    block_M=BM, block_N=BN, block_K=BK,
                                    num_stages=S, threads=TH, split_k=SK
                                ))
    return configs


def _create_benchmark_worker_class():
    """Create Ray Worker class dynamically."""
    _ensure_ray()

    @ray.remote(num_gpus=1)
    class BenchmarkWorker:
        def __init__(self, bench_rep: int = 20):
            self.bench_rep = bench_rep
            self.gpu_id = ray.get_gpu_ids()[0] if ray.get_gpu_ids() else 0

            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(self.gpu_id))
            torch.cuda.set_device(0)

            self._data_cache: Dict[Tuple[int, int, int], tuple] = {}
            print(f"[Worker] Initialized on GPU {self.gpu_id}")

        def _get_data(self, M: int, N: int, K: int):
            from sglang.srt.layers.tilelang_gemm_wrapper.core.quant_utils import (
                prepare_gemm_inputs,
            )

            key = (M, N, K)
            if key not in self._data_cache:
                A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
                B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

                A_fp8, B_fp8, A_scale, B_scale = prepare_gemm_inputs(A, B)
                C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

                self._data_cache[key] = (A_fp8, B_fp8, A_scale, B_scale, C)

            return self._data_cache[key]

        def benchmark_kernels(
            self,
            kernel_type: str,
            cfgs: List[dict],
            compile_cfgs: List[dict],
            indices: List[int],
            M: int,
            N: int,
            K: int,
        ) -> List[Optional[dict]]:
            from sglang.srt.layers.tilelang_gemm_wrapper.core.kernel_registry import (
                get_kernel_factory,
            )

            info = get_kernel_factory(kernel_type)
            factory = info["factory"]
            has_split_k = info["has_split_k"]
            is_swap_ab = info["is_swap_ab"]

            try:
                kernels = factory.par_compile(compile_cfgs, num_workers=16)
            except Exception as e:
                print(f"[Worker GPU {self.gpu_id}] Compilation failed: {e}")
                return [None] * len(cfgs)

            A_fp8, B_fp8, A_scale, B_scale, C = self._get_data(M, N, K)
            C_tl = torch.zeros_like(C)

            results = []
            for kernel, cfg, config_idx in zip(kernels, cfgs, indices):
                try:
                    latency_ms = self._benchmark_kernel(
                        kernel, cfg, has_split_k, is_swap_ab,
                        M, N, K, A_fp8, B_fp8, A_scale, B_scale, C_tl
                    )

                    results.append({
                        "config_idx": config_idx,
                        "kernel_type": kernel_type,
                        "cfg": cfg,
                        "latency_ms": latency_ms,
                        "tflops": _tflops(M, N, K, latency_ms),
                    })
                except Exception:
                    results.append(None)

            del kernels
            torch.cuda.empty_cache()

            return results

        def _benchmark_kernel(
            self,
            kernel,
            cfg: dict,
            has_split_k: bool,
            is_swap_ab: bool,
            M: int,
            N: int,
            K: int,
            A_fp8: torch.Tensor,
            B_fp8: torch.Tensor,
            A_scale: torch.Tensor,
            B_scale: torch.Tensor,
            C: torch.Tensor,
        ) -> float:
            if has_split_k:
                split_k = cfg["split_k"]
                if is_swap_ab:
                    C_partial = torch.zeros(
                        split_k, N, M, device="cuda", dtype=torch.float32
                    )

                    def bench_fn():
                        kernel(B_fp8, A_fp8, C_partial, C, B_scale, A_scale)
                else:
                    C_partial = torch.zeros(
                        split_k, M, N, device="cuda", dtype=torch.float32
                    )

                    def bench_fn():
                        kernel(A_fp8, B_fp8, C_partial, C, A_scale, B_scale)
            else:
                if is_swap_ab:
                    def bench_fn():
                        kernel(B_fp8, A_fp8, C, B_scale, A_scale)
                else:
                    def bench_fn():
                        kernel(A_fp8, B_fp8, C, A_scale, B_scale)

            ms, _, _ = triton.testing.do_bench_cudagraph(bench_fn, rep=self.bench_rep, quantiles=[0.5, 0.2, 0.8])
            return ms

        def clear_cache(self):
            self._data_cache.clear()
            torch.cuda.empty_cache()

    return BenchmarkWorker


class GEMMTuner:
    """GEMM Tuner with Ray multi-GPU support."""

    def __init__(
        self,
        config_dir: str = None,
        m_values: List[int] = None,
        num_gpus: int = None,
        bench_rep: int = 20,
    ):
        from sglang.srt.layers.tilelang_gemm_wrapper.core.config_loader import (
            ConfigLoader,
            get_default_m_values,
        )

        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), "config")

        self.config_dir = config_dir
        self.m_values = m_values or get_default_m_values()
        self.bench_rep = bench_rep
        self.config_loader = ConfigLoader(config_dir)

        os.makedirs(config_dir, exist_ok=True)

        _ensure_ray()
        if not ray.is_initialized():
            ray.init()

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        self.num_gpus = min(num_gpus, torch.cuda.device_count())

        logger.info(f"[GEMMTuner] Using {self.num_gpus} GPUs")

        BenchmarkWorker = _create_benchmark_worker_class()
        self.workers = [
            BenchmarkWorker.remote(bench_rep=bench_rep)
            for _ in range(self.num_gpus)
        ]

    def tune_single(
        self,
        M: int,
        N: int,
        K: int,
        kernel_types: List[str] = None,
        verbose: bool = True,
    ) -> Optional[dict]:
        """Tune a single (M, N, K) using multi-GPU parallelism."""
        from sglang.srt.layers.tilelang_gemm_wrapper.core.kernel_registry import (
            KERNEL_REGISTRY,
            _initialize_registry,
        )

        _initialize_registry()

        if kernel_types is None:
            kernel_types = list(KERNEL_REGISTRY.keys())

        if not kernel_types:
            if verbose:
                print("  [error] No kernel types available. Is tilelang installed?")
            return None

        pruned = []
        if M > 32:
            kernel_types = [kt for kt in kernel_types if "swapAB" not in kt]
            pruned.append("swapAB (M > 32)")
        if M > 128:
            kernel_types = [kt for kt in kernel_types if "splitK" not in kt]
            pruned.append("splitK (M > 128)")

        if verbose and pruned:
            print(f"  [prune] Skipping: {', '.join(pruned)}")

        all_results = []

        for kernel_type in kernel_types:
            if verbose:
                print(f"  [{kernel_type}] Tuning with {self.num_gpus} GPUs...")

            results = self._tune_kernel_type_parallel(
                kernel_type, M, N, K, verbose=verbose
            )
            all_results.extend(results)

        if not all_results:
            return None

        best = min(all_results, key=lambda x: x["latency_ms"])

        return {
            "kernel_type": best["kernel_type"],
            **best["cfg"],
            "latency_ms": best["latency_ms"],
            "tflops": best["tflops"],
        }

    def _tune_kernel_type_parallel(
        self,
        kernel_type: str,
        M: int,
        N: int,
        K: int,
        verbose: bool = True,
    ) -> List[dict]:
        """Tune all configs for a kernel type in parallel."""
        from sglang.srt.layers.tilelang_gemm_wrapper.core.kernel_registry import (
            get_kernel_factory,
        )

        info = get_kernel_factory(kernel_type)
        has_split_k = info["has_split_k"]
        is_swap_ab = info["is_swap_ab"]
        scale_shm_key = info["scale_shm_key"]

        if has_split_k:
            base_cfgs = _matmul_splitk_configs(M, N, K)
        else:
            base_cfgs = _matmul_configs(M, N, K)

        if not base_cfgs:
            if verbose:
                print(f"  [{kernel_type}] No valid base configs")
            return []

        cfgs = [
            {**cfg, "c_scale_local": csl, scale_shm_key: ssh}
            for cfg in base_cfgs
            for csl in [False, True]
            for ssh in [False, True]
        ]

        compile_M = N if is_swap_ab else M
        compile_N = M if is_swap_ab else N

        compile_cfgs = [
            {**cfg, "M": compile_M, "N": compile_N, "K": K,
             "out_dtype": "bfloat16", "accum_dtype": "float32"}
            for cfg in cfgs
        ]

        if verbose:
            print(f"  [{kernel_type}] Distributing {len(cfgs)} configs "
                  f"to {self.num_gpus} GPUs...")

        chunks: List[List[dict]] = [[] for _ in range(self.num_gpus)]
        chunk_compile_cfgs: List[List[dict]] = [[] for _ in range(self.num_gpus)]
        chunk_indices: List[List[int]] = [[] for _ in range(self.num_gpus)]

        for idx, (cfg, compile_cfg) in enumerate(zip(cfgs, compile_cfgs)):
            worker_idx = idx % self.num_gpus
            chunks[worker_idx].append(cfg)
            chunk_compile_cfgs[worker_idx].append(compile_cfg)
            chunk_indices[worker_idx].append(idx)

        futures = []
        for worker_idx, worker in enumerate(self.workers):
            if chunks[worker_idx]:
                future = worker.benchmark_kernels.remote(
                    kernel_type,
                    chunks[worker_idx],
                    chunk_compile_cfgs[worker_idx],
                    chunk_indices[worker_idx],
                    M, N, K,
                )
                futures.append(future)

        all_results = []
        for results in ray.get(futures):
            all_results.extend([r for r in results if r is not None])

        if verbose:
            success_rate = len(all_results) / len(cfgs) * 100 if cfgs else 0
            print(f"  [{kernel_type}] {len(all_results)}/{len(cfgs)} "
                  f"configs succeeded ({success_rate:.1f}%)")

            if all_results:
                best = min(all_results, key=lambda x: x["latency_ms"])
                print(f"  [{kernel_type}] Best: latency={best['latency_ms']:.4f}ms, "
                      f"tflops={best['tflops']:.2f}")

        return all_results

    def tune_for_nk(
        self,
        N: int,
        K: int,
        kernel_types: List[str] = None,
        verbose: bool = True,
    ) -> Dict[int, dict]:
        """Tune all M values for given (N, K)."""
        configs: Dict[int, dict] = {}

        for M in self.m_values:
            if verbose:
                print(f"\n{'='*60}")
                print(f"[tune] Tuning M={M}, N={N}, K={K}")
                print(f"{'='*60}")

            result = self.tune_single(M, N, K, kernel_types, verbose=verbose)

            if result:
                configs[M] = result

                if verbose:
                    print(f"\n[tune] Best for M={M}: "
                          f"{result['kernel_type']}, "
                          f"latency={result['latency_ms']:.4f}ms, "
                          f"tflops={result['tflops']:.2f}")
            else:
                if verbose:
                    print(f"\n[tune] No valid config found for M={M}")

            ray.get([w.clear_cache.remote() for w in self.workers])

        config_path = self.config_loader.save_config(N, K, configs)

        if verbose:
            print(f"\n{'='*60}")
            print(f"[tune] Config saved to {config_path}")
            print(f"[tune] Total M values tuned: {len(configs)}/{len(self.m_values)}")
            print(f"{'='*60}")

        return configs

    def shutdown(self):
        """Shutdown Ray."""
        _ensure_ray()
        ray.shutdown()
