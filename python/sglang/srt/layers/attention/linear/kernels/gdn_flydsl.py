"""AITER FlyDSL prefill kernels for GDN linear attention."""

from collections.abc import Sequence

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.utils.common import is_gfx95_supported


def _load_aiter_flydsl_gdn():
    try:
        from aiter.ops.prefill_batch_metadata import (
            build_gated_delta_rule_prefill_metadata,
        )
        from aiter.ops.triton.gated_delta_net import (
            chunk_gated_delta_rule_opt_vk,
        )
    except ImportError as exc:
        raise ImportError(
            "The FlyDSL GDN prefill backend requires the AITER version shipped "
            "with the SGLang ROCm image."
        ) from exc

    return chunk_gated_delta_rule_opt_vk, build_gated_delta_rule_prefill_metadata


class FlyDSLGDNKernel(TritonGDNKernel):
    """GDN prefill using AITER's fused FlyDSL prepare and state-scan kernels.

    The AITER path consumes SGLang's native ``[N, H, V, K]`` state pool and
    updates indexed slots in place. Batches that require per-chunk states, or
    inputs outside the validated bf16 K=V=128 domain, retain the Triton path.
    Decode and target verification are inherited from Triton.
    """

    def __init__(self):
        if not is_gfx95_supported():
            raise RuntimeError(
                "The FlyDSL GDN prefill backend requires an AMD gfx95 GPU."
            )
        self._prefill_fn, self._metadata_builder = _load_aiter_flydsl_gdn()

    def build_prefill_metadata(
        self,
        seq_lens_cpu: Sequence[int],
        *,
        cu_seqlens: torch.Tensor,
    ):
        return self._metadata_builder(
            seq_lens_cpu,
            cu_seqlens=cu_seqlens,
            chunk_size=64,
        )

    @staticmethod
    def _supports_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ssm_states: torch.Tensor,
    ) -> bool:
        return (
            q.dtype == k.dtype == v.dtype == torch.bfloat16
            and k.shape[-1] == 128
            and v.shape[-1] == 128
            and ssm_states.dtype in (torch.bfloat16, torch.float32)
        )

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
        inplace_update: bool = True,
        **kwargs,
    ) -> tuple:
        prefill_metadata = kwargs.get("prefill_metadata")
        if (
            prefill_metadata is None
            or not inplace_update
            or not self._supports_inputs(q, k, v, ssm_states)
        ):
            return super().extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                inplace_update=inplace_update,
                **kwargs,
            )

        # DP-attention MLP-sync rows use -1 as a padding marker. AITER's
        # indexed K5 path expects an in-bounds slot, so send those ignored rows
        # to the pool's reserved sentinel slot.
        ssm_cache_indices = torch.where(
            cache_indices >= 0,
            cache_indices,
            ssm_states.shape[0] - 1,
        ).to(torch.int32)
        output, _ = self._prefill_fn(
            q=q,
            k=k,
            v=v,
            o=kwargs.get("output"),
            g=g,
            beta=beta,
            initial_state=ssm_states,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
            use_prepare_flydsl=True,
            use_chunk_flydsl=True,
            state_dtype=ssm_states.dtype,
            snapshot_dtype=torch.bfloat16,
            use_exp2=True,
            prefill_metadata=prefill_metadata,
            initial_state_indices=ssm_cache_indices,
            inplace_final_state=True,
        )
        return output, None, None
