"""Residue-aware NVFP4 dense linear method (mext_r1, extended_k, standard).

Subclasses ModelOptFp4LinearMethod. A layer whose prefix resolves to no
residue spec follows the parent implementation exactly (same create_weights,
same process_weights_after_loading, same apply) -- the stock path stays
bit-for-bit and layout-for-layout unchanged. Only layers the metadata names
enter residue code.

mext_r1 layers keep the parent weight contract (weight stays at the original
K; the residue lives in the activation's M dimension at runtime), so
create_weights is inherited unchanged. extended_k layers store a K-extended
weight, so create_weights overrides the input sizes and, for row-parallel
layers, installs the two-range TP shard plan. Their PWAL also derives a
base-K scale layout: decode can use MExt fold over a strided weight prefix,
while larger M retains the full K-ext path. Both modes route apply() through
the opaque residue linear op.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.quantization.fp4_utils import (
    Fp4GemmRunnerBackend,
    get_fp4_gemm_runner_backend,
)
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4LinearMethod
from sglang.srt.layers.quantization.residue_nvfp4.metadata import (
    ResidueLayerSpec,
    ResidueMode,
)
from sglang.srt.layers.quantization.residue_nvfp4.tp import (
    SF_VEC_SIZE,
    ResidueShardPlan,
    current_tp_rank,
    interleave_extended_for_tp,
    plan_from_partition,
)

# Backends whose process_weights_after_loading produces the cutlass-dialect
# weight/scale layout the residue kernels consume. This is a LAYOUT
# compatibility set, not a preference: marlin repacks, trtllm shuffles, and
# neither layout can feed the fold or the residue quant chain. AUTO is
# accepted because it only survives to PWAL when the runner config was never
# initialized (tests / offline tools), and the parent PWAL routes AUTO
# through the cutlass-layout branch.
_CUTLASS_LAYOUT_BACKENDS = (
    Fp4GemmRunnerBackend.AUTO,
    Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
    Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
)


def _require_cutlass_layout_backend(layer_name: str) -> None:
    backend = get_fp4_gemm_runner_backend()
    if backend not in _CUTLASS_LAYOUT_BACKENDS:
        raise ValueError(
            f"residue NVFP4 layer {layer_name!r} needs a cutlass-layout FP4 "
            f"GEMM backend; the server selected {backend.value!r}. Pin one "
            "with --fp4-gemm-runner-backend flashinfer_cutlass (or "
            "flashinfer_cutedsl)."
        )


def _wrap_extended_weight_loaders(layer, plan: ResidueShardPlan) -> None:
    """Interleave extended columns before the stock loader slices them.

    `weight` is packed FP4 (two nibbles per byte, so K_ext/2 columns) and
    `weight_scale` holds one scale per 16 channels -- hence the differing
    `scale` factors. A loaded tensor whose last dim is not the extended width
    is left alone: it is not a K-extended tensor.
    """
    targets = (("weight", 2), ("weight_scale", SF_VEC_SIZE))
    for name, scale in targets:
        param = getattr(layer, name, None)
        if param is None:
            continue
        orig = getattr(param, "weight_loader", None)
        if orig is None:
            continue

        def make(orig_loader, col_scale):
            def loader(param_, loaded_weight, *args, **kwargs):
                if loaded_weight.shape[-1] == plan.k_ext // col_scale:
                    loaded_weight = interleave_extended_for_tp(
                        loaded_weight, plan, scale=col_scale
                    )
                return orig_loader(param_, loaded_weight, *args, **kwargs)

            return loader

        # weight_loader is a read-only property backed by _weight_loader on
        # sglang's parameter classes.
        param._weight_loader = make(orig, scale)


class ModelOptFp4ResidueLinearMethod(ModelOptFp4LinearMethod):
    """ModelOptFp4LinearMethod with residue dispatch for metadata-named layers."""

    def __init__(self, quant_config, prefix: str = ""):
        super().__init__(quant_config)
        self.prefix = prefix
        self.layer_spec: Optional[ResidueLayerSpec] = (
            quant_config.residue_spec.spec_for(prefix) if prefix else None
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        spec = self.layer_spec
        shard_plan: Optional[ResidueShardPlan] = None

        if spec is not None and spec.mode is ResidueMode.K_EXT:
            # The stored weight is K-extended. Row-parallel layers arrive with
            # the input already sharded (input_size_per_partition <
            # input_size IS the row-parallel signal), so the extended K has to
            # be sharded too; column-parallel layers get the whole extension.
            shard_plan = plan_from_partition(
                extended_dim=spec.k_ext,
                input_size_per_partition=input_size_per_partition,
                input_size=input_size,
                num_salient=spec.num_salient,
                tp_rank=current_tp_rank(),
            )
            if shard_plan is not None:
                input_size_per_partition = shard_plan.k_ext_shard
                input_size = spec.k_ext
            else:
                if input_size != spec.k_base:
                    raise ValueError(
                        f"layer {self.prefix!r}: metadata K_base={spec.k_base} "
                        f"does not match the layer input size {input_size}"
                    )
                input_size_per_partition = spec.k_ext
                input_size = spec.k_ext

        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        layer._residue_spec = spec

        if shard_plan is not None:
            # Row-parallel: permute the checkpoint's extended columns so the
            # stock loader's contiguous narrow lands on this rank's two
            # ranges. Far less invasive than replacing the loader.
            layer._residue_shard_plan = shard_plan
            _wrap_extended_weight_loaders(layer, shard_plan)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        spec = self.layer_spec
        if spec is None:
            super().process_weights_after_loading(layer)
            return

        _require_cutlass_layout_backend(self.prefix)
        if spec.mode is ResidueMode.MEXT_R1:
            self._pwal_mext_r1(layer, spec)
        else:
            assert spec.mode is ResidueMode.K_EXT
            self._pwal_extended_k(layer, spec)

    def _pwal_mext_r1(self, layer: torch.nn.Module, spec: ResidueLayerSpec) -> None:
        k = layer.input_size_per_partition
        n = layer.output_size_per_partition
        if spec.k_base != k and (spec.k_base % k != 0):
            raise ValueError(
                f"layer {self.prefix!r}: metadata K={spec.k_base} does not "
                f"match or shard onto the loaded weight K={k}"
            )
        # The fold consumes the weight and scale in their checkpoint geometry;
        # requiring these alignments keeps the parent PWAL from padding
        # (padding would desync the swizzled base scale from the raw layout).
        if k % 64 != 0 or n % 32 != 0:
            raise ValueError(
                f"layer {self.prefix!r}: mext_r1 requires K % 64 == 0 and "
                f"N % 32 == 0, got K={k}, N={n}"
            )

        super().process_weights_after_loading(layer)

        assert getattr(layer, "weights_padding_cols", 0) == 0, (
            f"layer {self.prefix!r}: parent PWAL padded the weight "
            "unexpectedly; the fold layout contract is violated"
        )

        # Fold layout: full K is the only K -- the fold reads the entire
        # weight, so the full swizzled scale doubles as its own base scale.
        layer._mext_decode_k_base = int(k)
        layer.weight_scale_base = layer.weight_scale_interleaved
        layer._residue_fold_eligible = True
        layer._residue_num_salient = 0
        layer._residue_channel_mask = None

        # Tell the pre-capture warmup this shape exists. k_ext is the STORED
        # width in FP4 elements (== k_base here; ext-K layers differ).
        from sglang.srt.layers.quantization.residue_nvfp4.warmup import (
            register_fold_shape,
        )

        register_fold_shape(
            int(layer.weight.shape[0]),
            int(k),
            int(layer.weight.shape[1]) * 2,
            is_mext_r1=True,
        )

    def _pwal_extended_k(self, layer: torch.nn.Module, spec: ResidueLayerSpec) -> None:
        from sglang.kernels.ops.quantization.residue_nvfp4_quant import (
            indices_to_channel_masks,
        )
        from sglang.srt.layers.quantization.utils import swizzle_blockscale

        # The weight this rank holds is K_ext (or K_ext/tp) wide, so the
        # salient indices must be this rank's too -- otherwise the channel
        # mask would be built over the wrong K.
        shard_plan: Optional[ResidueShardPlan] = getattr(
            layer, "_residue_shard_plan", None
        )
        indices = torch.tensor(spec.salient_indices, dtype=torch.int64)
        if shard_plan is not None:
            indices = shard_plan.local_salient_indices(indices)
            k_base_local = shard_plan.base_shard
            num_salient_local = shard_plan.salient_shard
        else:
            k_base_local = spec.k_base
            num_salient_local = spec.num_salient

        stored_k = int(layer.weight.shape[1]) * 2  # packed FP4: 2/byte
        if stored_k != k_base_local + num_salient_local:
            raise ValueError(
                f"layer {self.prefix!r}: stored weight K={stored_k} does not "
                f"match K_base+S = {k_base_local}+{num_salient_local}"
            )

        if k_base_local % SF_VEC_SIZE != 0:
            raise ValueError(
                f"layer {self.prefix!r}: base K={k_base_local} is not aligned "
                f"to the NVFP4 scale group size {SF_VEC_SIZE}"
            )

        # The parent PWAL may alias weight_scale_interleaved back onto the
        # original Parameter storage. Preserve the checkpoint-order scale
        # first: decode fold consumes only the base-K prefix, while the
        # large-M K-ext GEMM still consumes the full extended-K scale.
        base_scale_groups = int(k_base_local) // SF_VEC_SIZE
        if layer.weight_scale.shape[-1] < base_scale_groups:
            raise ValueError(
                f"layer {self.prefix!r}: raw weight scale width "
                f"{layer.weight_scale.shape[-1]} cannot cover base K="
                f"{k_base_local}"
            )
        raw_base_weight_scale = (
            layer.weight_scale[..., :base_scale_groups].detach().clone()
        )

        # Parent PWAL pads/swizzles for the extended-K GEMM; the shard-plan
        # validation guarantees K_ext%32==0 so no K padding actually fires,
        # but N padding is fine (the op slices the output back).
        super().process_weights_after_loading(layer)

        # Reproduce the parent's CUTLASS scale layout for the BASE prefix.
        # The packed weight stays in one allocation; nvfp4_linear narrows it
        # to a strided base-K view for fold and retains the full tensor for
        # large-M K-ext.
        layer.weight_scale_base = swizzle_blockscale(raw_base_weight_scale)
        layer._mext_decode_k_base = int(k_base_local)
        layer._residue_fold_eligible = True
        layer._residue_num_salient = int(num_salient_local)
        layer._residue_k_base = int(k_base_local)
        layer._residue_channel_mask = indices_to_channel_masks(
            indices.to(layer.weight.device), k_base_local
        )

        from sglang.srt.layers.quantization.residue_nvfp4.warmup import (
            register_fold_shape,
        )

        register_fold_shape(
            int(layer.weight.shape[0]),
            int(k_base_local),
            int(layer.weight.stride(0)) * 2,
            is_mext_r1=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        spec = self.layer_spec
        if spec is None:
            return super().apply(layer, x, bias)

        from sglang.kernels.ops.gemm.residue_nvfp4_linear import nvfp4_linear

        output_size = layer.output_size_per_partition
        output_shape = [*x.shape[:-1], output_size]
        x_2d = x.reshape(-1, x.shape[-1])

        is_mext_r1 = spec.mode is ResidueMode.MEXT_R1
        channel_mask = getattr(layer, "_residue_channel_mask", None)
        if channel_mask is None:
            # The op signature requires a mask tensor; mext_r1 has no channel
            # selection, so pass a 1-byte placeholder.
            channel_mask = x_2d.new_zeros(1, dtype=torch.uint8)

        output = nvfp4_linear(
            x_2d,
            layer.weight,
            layer.input_scale_inv,
            layer.weight_scale_interleaved,
            layer.weight_scale_base,
            channel_mask,
            layer.alpha,
            int(getattr(layer, "_mext_decode_k_base", 0)),
            int(getattr(layer, "_residue_num_salient", 0)),
            int(getattr(layer, "weights_padding_cols", 0)),
            int(output_size),
            bool(getattr(layer, "_residue_fold_eligible", False)),
            is_mext_r1,
        )

        if bias is not None:
            output = output + bias
        return output.view(*output_shape)
