# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in gfx1250 uTLX attention using AITER's cache and metadata lifecycle."""

import inspect
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

logger = logging.getLogger(__name__)

TLX_PAGE_SIZES = (16, 64, 128)
TLX_PHASES = ("decode", "prefill_window", "prefill_short", "prefill_long")
# AITER's full-attention prefill is faster on gfx1250 from about 1024 query tokens.
LONG_PREFILL_MIN_Q = 1024


class TlxAttnBackend(AiterAttnBackend):
    def __init__(self, model_runner, **kwargs):
        if torch.version.hip is None or not torch.cuda.get_device_properties(
            model_runner.device
        ).gcnArchName.startswith("gfx1250"):
            raise ValueError("The tlx attention backend requires AMD gfx1250")
        if model_runner.use_mla_backend:
            raise ValueError("The tlx attention backend supports MHA/GQA, not MLA")
        # The optional plugin is loaded only when this backend is requested.
        try:
            from sglang.kernels.ops.attention.tlx.paged import (
                unified_attention,
                unsupported_reason,
            )
        except ImportError as error:
            raise ImportError(
                "The tlx attention backend needs the uTLX Triton plugin "
                "(utlx_plugin from triton-ext extensions/utlx, with tdm_async_load "
                "and workgroup_barrier)"
            ) from error

        self._tlx_attention = unified_attention
        self._tlx_parameters = frozenset(
            inspect.signature(unified_attention).parameters
        )
        self._tlx_unsupported_reason = unsupported_reason
        self._tlx_phases = frozenset(
            phase.strip()
            for phase in envs.SGLANG_TLX_PHASES.get().split(",")
            if phase.strip()
        )
        unknown = sorted(self._tlx_phases - set(TLX_PHASES))
        if unknown:
            raise ValueError(
                f"SGLANG_TLX_PHASES has unknown phases {unknown}; choose from {TLX_PHASES}"
            )
        self._tlx_fallback_reasons = set()
        self._tlx_routed_phases = set()
        self._tlx_dispatched = set()
        super().__init__(model_runner, force_unified_attention=True, **kwargs)
        # Both would send every call to AITER; fail instead of running it silently.
        if self.page_size not in TLX_PAGE_SIZES:
            raise ValueError(
                f"The tlx attention backend needs --page-size in {TLX_PAGE_SIZES}, "
                f"got {self.page_size}"
            )
        if self.kv_cache_is_vectorized_5d:
            raise ValueError(
                "The tlx attention backend does not support the vectorized 5D "
                "AITER KV cache layout"
            )

    def _run_unified_attention(self, **kwargs):
        phase = self._phase(kwargs)
        if phase not in self._tlx_phases:
            if phase not in self._tlx_routed_phases:
                logger.info("uTLX attention routes %s to AITER", phase)
                self._tlx_routed_phases.add(phase)
            return super()._run_unified_attention(**kwargs)
        # AITER-only tuning arguments have no uTLX equivalent; keep AITER for them.
        extra = sorted(kwargs.keys() - self._tlx_parameters)
        reason = (
            f"AITER-only arguments {extra}"
            if extra
            else self._tlx_unsupported_reason(**kwargs)
        )
        if reason is None:
            self._log_dispatch(kwargs)
            return self._tlx_attention(**kwargs)
        if reason not in self._tlx_fallback_reasons:
            logger.warning("uTLX attention falls back to AITER: %s", reason)
            self._tlx_fallback_reasons.add(reason)
        return super()._run_unified_attention(**kwargs)

    @staticmethod
    def _phase(kwargs):
        max_seqlen_q = kwargs["max_seqlen_q"]
        if max_seqlen_q <= 16:
            return "decode"
        if kwargs["window_size"][0] >= 0:
            return "prefill_window"
        return "prefill_long" if max_seqlen_q >= LONG_PREFILL_MIN_Q else "prefill_short"

    def _log_dispatch(self, kwargs):
        # Host-side metadata only; under graph capture this runs once per captured shape.
        max_seqlen_q = kwargs["max_seqlen_q"]
        mode = (
            "decode"
            if max_seqlen_q == 1
            else "verify"
            if max_seqlen_q <= 16
            else "prefill"
        )
        key = (
            mode,
            tuple(kwargs["window_size"]),
            kwargs["q"].dtype,
            kwargs["k"].dtype,
            kwargs["q"].shape[1:],
            kwargs["k"].shape[1:],
        )
        if key not in self._tlx_dispatched:
            self._tlx_dispatched.add(key)
            logger.info(
                "uTLX attention: %s window=%s q=%s%s kv=%s%s",
                key[0],
                key[1],
                key[2],
                tuple(key[4]),
                key[3],
                tuple(key[5]),
            )
