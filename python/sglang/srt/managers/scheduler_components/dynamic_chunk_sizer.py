from __future__ import annotations

import logging
import math
import time
from array import array
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
    is_dp_attention_enabled,
    set_is_extend_in_batch,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import broadcast_pyobj
from sglang.srt.utils.common import get_device_module

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class DynamicChunkSizer:
    """Sizes PP prefill chunks from a profiled quadratic latency model."""

    def __init__(
        self,
        *,
        model_runner: ModelRunner,
        model_config: ModelConfig,
        tree_cache: BasePrefixCache,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        spec_algorithm: SpeculativeAlgorithm,
        chunked_prefill_size: int,
        max_prefill_tokens: int,
        page_size: int,
        device: str,
        pp_group: GroupCoordinator,
        world_group: GroupCoordinator,
        pp_rank: int,
    ):
        self.model_runner = model_runner
        self.model_config = model_config
        self.tree_cache = tree_cache
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.spec_algorithm = spec_algorithm
        self.chunked_prefill_size = chunked_prefill_size
        self.max_prefill_tokens = max_prefill_tokens
        self.page_size = page_size
        self.device = device
        self.pp_group = pp_group
        self.world_group = world_group
        self.pp_rank = pp_rank
        self.predictor = ChunkSizePredictor()

    def profile_and_fit(self) -> bool:
        """PP0 profiles synthetic prefills and every rank fits the same samples;
        returns whether the predictor is ready."""
        samples: Optional[Tuple[List[int], List[float]]] = None

        if self.pp_group.is_first_rank:
            try:
                samples = self._profile_prefill_latency()
            except Exception as e:
                logger.warning(
                    f"[PP Dynamic Chunk] Failed to profile prefill latency: {e!r}. "
                    "Dynamic chunking will be disabled."
                )

        # The samples are global, so one broadcast from global rank 0 (a PP0 rank)
        # reaches every stage and attention rank; a failure travels as None.
        samples = broadcast_pyobj(
            [samples], self.world_group.rank, self.world_group.cpu_group, src=0
        )[0]

        if samples is None:
            return False

        seq_lens, latencies = samples
        # Quadratic model: f(l) = al^2 + bl + c
        try:
            self.predictor.fit(seq_lens, latencies)
        except Exception as e:
            # Every rank fits the same samples, so this fails on all of them alike.
            logger.warning(
                f"[PP Dynamic Chunk] Failed to fit the chunk-size predictor: {e!r}. "
                "Dynamic chunking will be disabled."
            )
            return False
        self.predictor.set_target_latency(self.chunked_prefill_size)
        self.predictor.is_ready = True
        logger.info(
            f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predictor ready (quadratic). "
            f"Target latency: {self.predictor.target_latency:.2f}ms"
        )
        return True

    def predict(self, history_len: int) -> Optional[int]:
        """Chunk size for the next prefill step, or None to keep the static size."""
        if not self.predictor.is_ready:
            return None

        max_chunk_size = self.max_prefill_tokens
        predicted_size = self.predictor.predict_next_chunk_size(
            history_len=history_len,
            base_chunk_size=self.chunked_prefill_size,
            page_size=self.page_size,
            context_len=self.model_config.context_len,
            max_chunk_size=max_chunk_size,
        )

        if predicted_size is not None:
            logger.debug(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predicted chunk size: "
                f"{predicted_size} (history_len={history_len})"
            )

        return predicted_size

    def _profile_prefill_latency(self) -> Tuple[List[int], List[float]]:
        seq_lens: List[int] = []
        latencies: List[float] = []
        model_runner = self.model_runner
        model_config = model_runner.model_config
        input_ids_list: List[array[int]] = []
        for i in range(128):
            chunk_size = int(
                self.chunked_prefill_size * 1.25
                - i * (self.chunked_prefill_size * 1.25 // 128)
            )
            if chunk_size <= 0:
                break
            input_ids = array(
                "q",
                np.random.randint(0, 10000, size=chunk_size, dtype=np.int64).tobytes(),
            )
            input_ids_list.append(input_ids)

        sampling_params = SamplingParams(
            temperature=0,
            max_new_tokens=1,
        )
        # Create and profile requests
        for i, input_ids in enumerate(
            tqdm(
                input_ids_list,
                desc="Profiling prefill latency for dynamic chunking",
            )
        ):
            req = Req(
                rid=str(i),
                origin_input_text="",
                origin_input_ids=input_ids,
                sampling_params=sampling_params,
            )
            # Walk the same match -> lock -> alloc lifecycle as a scheduled
            # request so release_kv_cache can release it symmetrically.
            req.init_next_round_input(self.tree_cache)
            lock = self.tree_cache.inc_lock_ref(req.last_node)
            req.swa_uuid_for_lock = lock.swa_uuid_for_lock
            req.set_extend_range(
                len(req.prefix_indices), len(req.full_untruncated_fill_ids)
            )

            # Prepare batch
            batch = ScheduleBatch.init_new(
                [req],
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                self.tree_cache,
                self.model_config,
                False,
                self.spec_algorithm,
            )

            current_seq_len = req.extend_range.end

            if is_dp_attention_enabled():
                # Profiling runs one request on this rank; other DP ranks report 0.
                dp_size = get_attention_dp_size()
                global_num_tokens = [0] * dp_size
                dp_rank = get_attention_dp_rank()
                global_num_tokens[dp_rank] = current_seq_len
                batch.global_num_tokens = global_num_tokens
                batch.global_num_tokens_for_logprob = global_num_tokens

            hs = (
                getattr(model_config, "hc_hidden_size", None)
                or model_config.hidden_size
            )
            proxy_tensors = {
                "hidden_states": torch.zeros(
                    (current_seq_len, hs),
                    dtype=model_config.dtype,
                    device=self.device,
                ),
                "residual": torch.zeros(
                    (current_seq_len, model_config.hidden_size),
                    dtype=model_config.dtype,
                    device=self.device,
                ),
            }
            pp_proxy_topk_size = model_runner.get_pp_proxy_topk_size()
            if pp_proxy_topk_size is not None:
                proxy_tensors["topk_indices"] = torch.zeros(
                    (current_seq_len, pp_proxy_topk_size),
                    dtype=torch.int32,
                    device=self.device,
                )

            pp_proxy = PPProxyTensors(proxy_tensors)

            # Measure latency with device synchronization for accurate timing
            device_module = get_device_module()
            # Synchronize before starting timing to ensure clean measurement
            device_module.synchronize()

            start = time.perf_counter()
            batch.prepare_for_extend()

            # Resolve deferred H2D: prepare_for_extend now leaves input_ids=None
            if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
                batch.input_ids = batch.prefill_input_ids_cpu.to(
                    self.device, non_blocking=True
                )
                batch.prefill_input_ids_cpu = None

            forward_batch = ForwardBatch.init_new(
                batch,
                model_runner,
                return_hidden_states_before_norm=False,
            )
            set_is_extend_in_batch(batch.forward_mode.is_extend())

            _ = model_runner.forward(
                forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
            )

            # Synchronize after forward to ensure GPU operations complete
            device_module.synchronize()

            latency_seconds = time.perf_counter() - start
            latency_ms = latency_seconds * 1e3  # Convert to milliseconds
            seq_lens.append(len(input_ids))
            latencies.append(latency_ms)

            # Release KV and Mamba cache
            if req.kv.holds_kv:
                release_kv_cache(req, self.tree_cache, is_insert=False)

        logger.info(
            f"[PP Dynamic Chunk] [PP0] Profiled {len(seq_lens)} samples: "
            f"seq_lens={seq_lens}, latencies_ms={latencies}"
        )
        return seq_lens, latencies


class ChunkSizePredictor:
    """Quadratic latency model f(l) = a*l^2 + b*l + c; predicts the chunk x with
    f(L + x) - f(L) = target_latency."""

    def __init__(self):
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = 0.0
        self.constant_coeff_c = 0.0
        self.target_latency: Optional[float] = None
        self.is_ready = False

    def fit(self, seq_lens: List[int], latencies: List[float]):
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points."""
        # Skip the first data point to reduce fitting bias, as the first run is slower without warmup
        L = np.array(seq_lens[1:], dtype=np.float64)
        T = np.array(latencies[1:], dtype=np.float64)

        if len(L) < 8:
            raise ValueError(
                f"Not enough data points for quadratic fitting ({len(L)} < 8). "
                "Need at least 8 samples with different sequence lengths."
            )

        # Build design matrix for f(l) = al^2 + bl + c
        X = np.column_stack([L * L, L, np.ones_like(L)])  # [l^2, l, 1]

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
            if len(coeffs) >= 3:
                fitted_a = float(coeffs[0])  # quadratic coefficient
                fitted_b = float(coeffs[1])  # linear coefficient
                fitted_c = float(coeffs[2])  # constant coefficient
            else:
                raise ValueError("Failed to fit coefficients: insufficient rank")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to fit f(l) = al^2 + bl + c: {e}")

        # Validate coefficients
        if fitted_a <= 0:
            raise ValueError(
                f"Fitted quadratic coefficient a={fitted_a:.2e} is not positive. "
                "Attention has O(n^2) complexity, so a must be positive. "
                "Check warmup data quality."
            )

        if fitted_b < 0:
            logger.warning(
                f"Fitted linear coefficient b={fitted_b:.2e} is negative. Setting b=0."
            )
            fitted_b = 0.0

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            f"[ChunkSizePredictor] Fitted coefficients: a={fitted_a:.2e}, "
            f"b={fitted_b:.2e}, c={fitted_c:.2e}"
        )

    def set_target_latency(self, base_chunk_size: int):
        """Set target latency based on base chunk size: target = f(base_chunk_size) - f(0)."""

        def f(length: float) -> float:
            """Total latency function: f(length) = a*length^2 + b*length + c."""
            return (
                self.quadratic_coeff_a * length * length
                + self.linear_coeff_b * length
                + self.constant_coeff_c
            )

        self.target_latency = f(float(base_chunk_size)) - f(0.0)

        if self.target_latency <= 0:
            raise ValueError(
                f"Calculated target_latency={self.target_latency:.2f}ms is not positive. "
                "Check warmup data quality."
            )

        logger.info(
            f"[ChunkSizePredictor] Target latency: {self.target_latency:.2f}ms "
            f"(base_chunk_size={base_chunk_size})"
        )

    def predict_next_chunk_size(
        self,
        history_len: int,
        base_chunk_size: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
    ) -> Optional[int]:
        """Chunk size x with f(L + x) - f(L) = target_latency for L = history_len,
        or None when the model cannot say."""
        if not self.is_ready or self.target_latency is None:
            return None

        # Handle quadratic model: f(l) = al^2 + bl + c
        if self.quadratic_coeff_a <= 0:
            return None

        # f(L+x) - f(L) = T expands to a*x^2 + (2aL + b)*x - T = 0.
        A = self.quadratic_coeff_a
        B = 2 * self.quadratic_coeff_a * history_len + self.linear_coeff_b
        C = -self.target_latency

        discriminant = B * B - 4 * A * C

        if discriminant < 0:
            logger.warning(
                f"Discriminant is negative ({discriminant:.2e}). "
                f"No real solution for chunk size. L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        sqrt_discriminant = math.sqrt(discriminant)
        calculated_chunk_size_float = (-B + sqrt_discriminant) / (2 * A)

        if calculated_chunk_size_float <= 0:
            logger.warning(
                f"Calculated chunk size is non-positive ({calculated_chunk_size_float:.2f}). "
                f"L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        # Use a smooth coefficient to reduce the abrupt decrease in chunk size
        smooth_coeff = envs.SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR.get()
        smoothed_chunk_size = base_chunk_size + smooth_coeff * (
            calculated_chunk_size_float - base_chunk_size
        )
        # Make sure the dynamic chunk size is at least 1/4 of the base chunk size
        calculated_chunk_size = max(int(smoothed_chunk_size), base_chunk_size // 4)

        # Align to page_size (minimum alignment size is 64)
        alignment_size = max(page_size, 64)
        dynamic_chunk_size = (calculated_chunk_size // alignment_size) * alignment_size

        # Ensure aligned size is at least alignment_size
        if dynamic_chunk_size < alignment_size:
            dynamic_chunk_size = alignment_size

        # Apply constraints
        max_allowed = context_len - history_len - 100  # Leave 100 tokens margin
        if max_chunk_size is not None:
            max_allowed = min(max_allowed, max_chunk_size)
        dynamic_chunk_size = min(dynamic_chunk_size, max_allowed)

        # Align again after min operation
        dynamic_chunk_size = (dynamic_chunk_size // alignment_size) * alignment_size

        if dynamic_chunk_size < alignment_size:
            return None

        return dynamic_chunk_size
