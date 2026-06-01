from __future__ import annotations

from typing import ClassVar

from sglang.test.kv_canary.consts import SWA_POOL_SERVER_ARGS
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

# Pipeline parallelism across 2 GPUs (tp=1, pp=2). A PP rank keeps pp_loop_size
# micro-batches as host-side scheduling slots, but runs all their forwards
# serially on its single forward_stream, so canary's single-slot phase checker
# stays valid under PP exactly as it does under the default overlap scheduler.
# These fixtures launch one SWA server with pp_size=2 and kv-canary enabled to
# prove a clean PP run is violation-free and an injected perturbation still
# surfaces.
PP_SIZE: int = 2


class CanaryPPFixture(CanaryE2EBase):
    """Single SWA server launched with pp_size=2 and kv-canary enabled."""

    model_mode: ClassVar[str] = "swa"

    @classmethod
    def setUpClass(cls) -> None:
        cls.extra_server_args = (
            "--pp-size",
            str(PP_SIZE),
            # The SWA fixture model (gemma per-layer-input embeddings) cannot run
            # PP under CUDA graph: the runner's PP proxy schema only carries
            # {hidden_states, residual} and drops per_layer_inputs. This is an
            # sglang-core limitation unrelated to kv-canary, so fall back to eager
            # replay. Canary still brackets every forward in eager mode, and PP
            # serializes all micro-batch forwards on one forward_stream either way.
            "--disable-cuda-graph",
            *SWA_POOL_SERVER_ARGS,
            *cls.extra_server_args,
        )
        super().setUpClass()

    def maybe_assert_swa_divergence_observed(self) -> None:
        # swa_out_of_window_tokens is sampled from whichever forward batch is live
        # at the divergence interval. Under PP's micro-batch scheduling that sampled
        # forward may not contain a request whose in-window prefix has been evicted
        # to SWA slot 0, so the count is a sampling-instant artifact here. The SWA
        # path is still proven exercised by verify_swa << verify_full and
        # swa_full_idx_divergence > 0, so only the out-of-window floor is relaxed.
        self.assert_swa_divergence_observed(min_swa_out_of_window_tokens=0)
