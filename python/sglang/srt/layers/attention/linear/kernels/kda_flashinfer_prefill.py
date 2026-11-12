# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""FlashInfer 0.7 CuTe DSL KDA prefill with radix-cache checkpoints."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import sys
import tempfile
from itertools import accumulate
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.kda_input_pack import contiguous_kda_inputs
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def build_flashinfer_kda_checkpoint_plan(
    forward_batch: ForwardBatch,
    metadata: ForwardMetadata,
    device: torch.device,
    chunk_size: int,
) -> None:
    """Plan scheduler-generated tracked boundaries, asserting their validity.

    Host track metadata follows build_prefill_track_plan's unaligned-row order.
    """
    extend_lens = forward_batch.extend_seq_lens_cpu
    track_lens = forward_batch.mamba_track_seqlens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    track_mask = forward_batch.mamba_prefill_track_mask_cpu

    checkpoint_counts = [length // chunk_size for length in extend_lens]
    checkpoint_starts = list(accumulate(checkpoint_counts, initial=0))
    # NOTE(kpham-sgl): FlashInfer GDN writes all periodic checkpoints for SGLang
    # to select from; FlashInfer KDA writes only mapped destinations (-1 skips).
    checkpoint_destinations = [-1] * checkpoint_starts[-1]
    num_tracked_checkpoints = 0
    for row, tracked in enumerate(track_mask):
        if not tracked:
            continue
        relative_track_len = track_lens[row] - prefix_lens[row]
        if relative_track_len % chunk_size == 0:
            continue
        completed_chunks = relative_track_len // chunk_size
        assert 0 < completed_chunks <= checkpoint_counts[row], (
            f"Invalid KDA checkpoint metadata: row={row}, "
            f"relative_track_len={relative_track_len}, extend_len={extend_lens[row]}, "
            f"chunk_size={chunk_size}"
        )
        checkpoint_destinations[checkpoint_starts[row] + completed_chunks - 1] = (
            num_tracked_checkpoints
        )
        num_tracked_checkpoints += 1

    assert num_tracked_checkpoints == metadata.track_ssm_h_batch_src.numel(), (
        "KDA checkpoint rows disagree with the prefill tracking metadata"
    )
    metadata.state_checkpoint_cu_starts = torch.tensor(
        checkpoint_starts, dtype=torch.int64, pin_memory=True
    ).to(device, non_blocking=True)
    metadata.num_state_checkpoints = num_tracked_checkpoints
    metadata.state_checkpoint_every_n_tokens = chunk_size
    metadata.state_checkpoint_indices = torch.tensor(
        checkpoint_destinations, dtype=torch.int32, pin_memory=True
    ).to(device, non_blocking=True)


class FlashInferKDAPrefillKernel(LinearAttnKernelBase):
    """Execute KDA prefill; FlashInfer validates the tensor contract."""

    uses_state_checkpoints = True
    # FlashInfer memoizes cu_seqlens host contents on tensor identity, but casts an
    # int32 one to a new int64 tensor per call; one int64 tensor per forward hits it.
    wants_int64_query_start_loc = True
    supports_track_state_snapshot = True
    supports_safe_gate = True

    def __init__(self):
        capability = torch.cuda.get_device_capability()
        if capability not in prefill_capabilities():
            raise ValueError(
                "FlashInfer KDA prefill requires SM100 or SM103 "
                f"(SM107 with {ENV_FLAG}=1)"
            )
        prepare_flashinfer_prefill(capability)
        from flashinfer import kda

        self._recurrent_kda = kda.recurrent_kda
        # Older FlashInfer installations retain the eager BCG break.
        self.prefill_graph_wrapper_cls = getattr(
            kda, "RecurrentKDAPrefillWrapper", None
        )

    def decode(self, *args, **kwargs):
        raise NotImplementedError("FlashInferKDAPrefillKernel is prefill-only")

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        query_start_loc_i64: Optional[torch.Tensor] = None,
        return_intermediate_states: bool = False,
        state_checkpoint_cu_starts: Optional[torch.Tensor] = None,
        num_state_checkpoints: int = 0,
        state_checkpoint_every_n_tokens: int = 0,
        state_checkpoint_indices: Optional[torch.Tensor] = None,
        track_ssm_h_batch_src: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        needs_checkpoint = (
            return_intermediate_states and kwargs.get("track_state") is not None
        )
        checkpoints = (
            ssm_states.new_empty((num_state_checkpoints, *ssm_states.shape[1:]))
            if needs_checkpoint
            else None
        )
        q, k, v, g, beta = contiguous_kda_inputs(
            q, k, v, g[:, : q.shape[1]], beta[:, : q.shape[1]]
        )
        result = self._recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log.reshape(-1).float().contiguous(),
            # FlashInfer's grouped-CTA recurrent kernel (extends with a cached prefix)
            # indexes dt_bias as a flat [H*K] tensor and asserts a unit leading stride.
            dt_bias=dt_bias.reshape(-1).float().contiguous(),
            initial_state=ssm_states,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=lower_bound,
            cu_seqlens=(
                query_start_loc if query_start_loc_i64 is None else query_start_loc_i64
            ),
            ssm_state_indices=cache_indices.to(torch.int32),
            beta_is_logit=True,
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=(
                state_checkpoint_cu_starts if needs_checkpoint else None
            ),
            checkpoint_state_indices=(
                state_checkpoint_indices if needs_checkpoint else None
            ),
            checkpoint_every_n_tokens=(
                state_checkpoint_every_n_tokens if needs_checkpoint else 0
            ),
            backend="cute-dsl",
        )
        if needs_checkpoint:
            kwargs["track_state"][track_ssm_h_batch_src] = checkpoints.float()
        output = result[0]
        return (output, None) if return_intermediate_states else output


# ===== Opt-in FlashInfer CuTe DSL KDA prefill on SM107 (VR200, CC 10.7) =====
#
# Opt-in FlashInfer CuTe DSL KDA prefill on SM107 (VR200, CC 10.7).
#
# Off unless ``SGLANG_K31_KDA_PREFILL_FI_SM107=1``. With it on, and only on a CC 10.7
# device, the FlashInfer KDA prefill (``recurrent_kda(backend="cute-dsl")``, the same
# safe-gate KDA math SGLang already runs on SM100/SM103) is admitted where SGLang and
# FlashInfer 0.7.2 accept only CC 10.0/10.3. Every other device, and every process
# without the variable, behaves exactly as before.
#
# Two FlashInfer fixes are backported in memory (no FlashInfer file changes on disk):
#
# - flashinfer-ai/flashinfer#6201 (059c01ac, first released in v0.7.3): add (10, 7) to
#   ``flashinfer.kda_prefill_cute._SUPPORTED_COMPUTE_CAPABILITIES``.
# - flashinfer-ai/flashinfer#8684 (0102aa9f, main 2026-11-05, in no release yet): double-buffer
#   ``update_ready`` in ``kernel_chain_dv2``. Without it the decomp route (taken when
#   ``2 * num_seqs * heads <= num_SMs``) can hang. The
#   fixed module source is written once to a cache file and registered as
#   ``flashinfer.kda_kernels.kda_chunked_bt16`` before FlashInfer first imports it.
#   If the hunks do not apply and the module is not already fixed, this raises rather
#   than run the unfixed kernel.
#
# Third-party notices.
# - The #8684 hunks below (``_PR8684_HUNKS``) are copied verbatim from FlashInfer
#   (https://github.com/flashinfer-ai/flashinfer) flashinfer/kda_kernels/kda_chunked_bt16.py,
#   which carries this notice:
#     SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#     SPDX-License-Identifier: BSD-3-Clause
#   Redistribution and use in source and binary forms, with or without modification, are
#   permitted provided that the following conditions are met:
#   1. Redistributions of source code must retain the above copyright notice, this list of
#      conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice, this list of
#      conditions and the following disclaimer in the documentation and/or other materials
#      provided with the distribution.
#   3. Neither the name of the copyright holder nor the names of its contributors may be used
#      to endorse or promote products derived from this software without specific prior
#      written permission.
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
#   OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
#   TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# - The #6201 change (CC 10.7 in the supported set) is reproduced in
#   ``prepare_flashinfer_prefill``, from flashinfer/kda_prefill_cute.py: Copyright (c) 2026 by
#   FlashInfer team, licensed under the Apache License, Version 2.0
#   (http://www.apache.org/licenses/LICENSE-2.0).

logger = logging.getLogger(__name__)

ENV_FLAG = "SGLANG_K31_KDA_PREFILL_FI_SM107"
SM107 = (10, 7)
_BASE_CAPABILITIES = ((10, 0), (10, 3))
_BT16_MODULE = "flashinfer.kda_kernels.kda_chunked_bt16"
_FIXED_MARKER = "update_ready_mbar.subview(chunk % 2)"

# flashinfer-ai/flashinfer#8684, flashinfer/kda_kernels/kda_chunked_bt16.py, verbatim
# hunks (trailing whitespace stripped; an empty line is an empty context line).
_PR8684_HUNKS = r"""
@@ -8712,8 +8712,10 @@ def kernel_chain_dv2(
     u_input_ready_mbar = cutlass.Array(
         cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
     )
+    # Two slots: warp 12 commits stateq_done before its wait, so one slot can
+    # complete twice before warp 12 polls it.
     update_ready_mbar = cutlass.Array(
-        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
+        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
     )
     shared_acc_ready_mbar = cutlass.Array(
         cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
@@ -8822,7 +8824,8 @@ def kernel_chain_dv2(
                 prims.mbarrier_init(stateq_done_mbar.subview(stage), 1)
             # CG1 fuses the right-half pack/scale while idle CG0 warps 4-7
             # fuse the left half. Both groups must finish before delta.
-            prims.mbarrier_init(update_ready_mbar, 8)
+            for stage in cutlass.range_constexpr(2):
+                prims.mbarrier_init(update_ready_mbar.subview(stage), 8)
             for stage in cutlass.range_constexpr(K2_QSTATE_STAGE_COUNT):
                 prims.mbarrier_init(qstate_acc_ready_mbar.subview(stage), 1)
             prims.mbarrier_init(final_state_stored_mbar, 8)
@@ -8920,7 +8923,6 @@ def kernel_chain_dv2(
     elif warp_idx == ROLES.super_mma:
         tmem_raw_addr = tmem_ptr_i32.load()
         si_phase12 = cutlass.Int32(0)
-        upd_phase12 = cutlass.Int32(0)
         for chunk in cutlass.range(num_chunks, unroll=1):
             raw_stage = chunk % K2_RAW_STAGE_COUNT
             qstate_stage = chunk % K2_QSTATE_STAGE_COUNT
@@ -8957,7 +8959,7 @@ def kernel_chain_dv2(
                 True,
                 DV_HALF,
             )
-            upd_phase12 = update_ready_wait(update_ready_mbar, upd_phase12)
+            update_ready_wait(update_ready_mbar.subview(chunk % 2), (chunk // 2) % 2)
             prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
             xpack_col12 = tcgen05_shared_input_tmem_col_offset(acc_stage)
             tcgen05_chain_issue_qkv_mma(
@@ -8975,7 +8977,6 @@ def kernel_chain_dv2(
         tmem_raw_addr = tmem_ptr_i32.load()
         si_l_phase = cutlass.Int32(0)
         si_phase = cutlass.Int32(0)
-        upd_phase = cutlass.Int32(0)
         for chunk in cutlass.range(num_chunks, unroll=1):
             raw_stage = chunk % K2_RAW_STAGE_COUNT
             raw_phase = (chunk // K2_RAW_STAGE_COUNT) % 2
@@ -9017,7 +9018,7 @@ def kernel_chain_dv2(
             )

             xpack_col = tcgen05_shared_input_tmem_col_offset(acc_stage)
-            upd_phase = update_ready_wait(update_ready_mbar, upd_phase)
+            update_ready_wait(update_ready_mbar.subview(chunk % 2), (chunk // 2) % 2)
             prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
             tcgen05_chain_issue_delta_half_mma(
                 w_stage_smem, tmem_raw_addr, xpack_col, input_dtype, 0
@@ -9152,7 +9153,7 @@ def kernel_chain_dv2(
                 input_dtype,
             )
             post_scale_lane = cute.arch.lane_idx()
-            update_ready_arrive(update_ready_mbar)
+            update_ready_arrive(update_ready_mbar.subview(0))
             if cutlass.const_expr(state_ckpt is not None):
                 # Peeled chunk 0's checkpoint (stride 1 only).
                 checkpoint_due = ckpt_next == cutlass.Int32(1)
@@ -9232,7 +9233,7 @@ def kernel_chain_dv2(
                 input_dtype,
             )
             post_scale_lane = cute.arch.lane_idx()
-            update_ready_arrive(update_ready_mbar)
+            update_ready_arrive(update_ready_mbar.subview(chunk % 2))
             if cutlass.const_expr(state_ckpt is not None):
                 # Same contract as the engine-side checkpoint; both M=64 TMEM
                 # halves wait their own k_restore parity slot (the same wait
@@ -9339,7 +9340,7 @@ def kernel_chain_dv2(
                     cutlass.Int32(0),
                 )
                 prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
-                update_ready_arrive(update_ready_mbar)
+                update_ready_arrive(update_ready_mbar.subview(0))
             for chunk in cutlass.range(1, num_chunks, 1, unroll=1):
                 prev = chunk - cutlass.Int32(1)
                 raw_stage = chunk % K2_RAW_STAGE_COUNT
@@ -9378,7 +9379,7 @@ def kernel_chain_dv2(
                     state_input_ready_l_mbar,
                     0,
                 )
-                update_ready_arrive(update_ready_mbar)
+                update_ready_arrive(update_ready_mbar.subview(chunk % 2))


 @cute.jit
"""

_prepared = False


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").strip().lower() in ("1", "true", "yes", "on")


def prefill_capabilities() -> tuple:
    """Device capabilities that may run FlashInfer KDA prefill in this process."""
    return _BASE_CAPABILITIES + ((SM107,) if enabled() else ())


def _hunks(text: str) -> list[tuple[str, str]]:
    """(before, after) blocks of a unified diff's hunks, context lines included."""
    blocks: list[tuple[list[str], list[str]]] = []
    for line in text.strip("\n").split("\n"):
        if line.startswith("@@"):
            blocks.append(([], []))
            continue
        before, after = blocks[-1]
        tag, body = (line[:1], line[1:]) if line else (" ", "")
        if tag in (" ", "-"):
            before.append(body)
        if tag in (" ", "+"):
            after.append(body)
    return [("\n".join(b) + "\n", "\n".join(a) + "\n") for b, a in blocks]


def apply_pr8684(source: str) -> Optional[str]:
    """The source with #8684 applied, ``source`` itself if already fixed, else None.

    Each hunk's before-block (its context and removed lines) must occur exactly once.
    """
    if _FIXED_MARKER in source:
        return source
    for before, after in _hunks(_PR8684_HUNKS):
        if source.count(before) != 1:
            return None
        source = source.replace(before, after, 1)
    return source if _FIXED_MARKER in source else None


def _install_fixed_bt16() -> str:
    """Register the #8684-fixed kda_chunked_bt16 under FlashInfer's module name."""
    module = sys.modules.get(_BT16_MODULE)
    if module is not None:
        if getattr(module, "_sglang_pr8684", False):
            return "already installed"
        path = getattr(module, "__file__", None)
        if path and _FIXED_MARKER in open(path, encoding="utf-8").read():
            return "already fixed upstream"
        raise RuntimeError(
            f"{ENV_FLAG}=1: {_BT16_MODULE} was imported before the #8684 backport "
            "could be installed; refusing to run the unfixed decomp kernel"
        )
    spec = importlib.util.find_spec(_BT16_MODULE)
    if spec is None or not spec.origin:
        raise RuntimeError(f"{ENV_FLAG}=1: cannot locate {_BT16_MODULE}")
    with open(spec.origin, encoding="utf-8") as f:
        source = f.read()
    fixed = apply_pr8684(source)
    if fixed is None:
        raise RuntimeError(
            f"{ENV_FLAG}=1: flashinfer#8684 does not apply to {spec.origin}; "
            "refusing to run the unfixed decomp kernel"
        )
    if fixed is source:
        return "already fixed upstream"
    digest = hashlib.sha256(fixed.encode()).hexdigest()[:16]
    cache_dir = os.environ.get("SGLANG_CACHE_DIR") or tempfile.gettempdir()
    target_dir = os.path.join(cache_dir, "k31_fi_kda_bt16")
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"kda_chunked_bt16_pr8684_{digest}.py")
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(fixed)
        os.replace(tmp, path)  # atomic: concurrent ranks write identical bytes
    fixed_spec = importlib.util.spec_from_file_location(_BT16_MODULE, path)
    module = importlib.util.module_from_spec(fixed_spec)
    module._sglang_pr8684 = True
    sys.modules[_BT16_MODULE] = module
    try:
        fixed_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_BT16_MODULE, None)
        raise
    import flashinfer.kda_kernels as package

    setattr(package, "kda_chunked_bt16", module)
    return f"installed {path}"


def prepare_flashinfer_prefill(capability: tuple) -> None:
    """Once per process, on CC 10.7 with the flag on: backport #6201 and #8684."""
    global _prepared
    if _prepared or not enabled() or tuple(capability) != SM107:
        return
    from flashinfer import kda_prefill_cute

    supported = kda_prefill_cute._SUPPORTED_COMPUTE_CAPABILITIES
    pr6201 = "already present" if SM107 in supported else "added"
    supported.add(SM107)
    pr8684 = _install_fixed_bt16()
    _prepared = True
    logger.info(
        "K31 FlashInfer KDA prefill on SM107 engaged (%s=1): flashinfer#6201 %s, "
        "flashinfer#8684 %s",
        ENV_FLAG,
        pr6201,
        pr8684,
    )
