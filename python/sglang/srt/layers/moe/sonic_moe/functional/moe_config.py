# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import math
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from quack.tile_scheduler import RasterOrderOption

from sglang.srt.layers.moe.sonic_moe.enums import ActivationType, is_glu

from .grouped_gemm import HopperWgmma_MoE_kernel

LIBRARY_NAME = "cutedsl_kernels"


def ceil_div(a: int, b: int):
    return int(math.ceil(a / b))


@dataclass
class HopperGEMMConfig:
    tile_shape_mnk: cutlass.Constexpr[cute.Shape] = (128, 256, 64)
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape] = (2, 1)
    epi_tile_size: cutlass.Constexpr[int] = 32
    ## assume we always use persistent kernel
    # is_persistent: cutlass.Constexpr[bool] = True
    is_pingpong: cutlass.Constexpr[bool] = False
    raster_order: RasterOrderOption = RasterOrderOption.Heuristic
    L2_group_size: int = 8
    initial_d_epi_stage: cutlass.Constexpr[int] = 4


class HopperWgmma_MoE_Up_proj_Fwd:
    def __init__(
        self,
        E: int,
        H: int,
        I: int,
        activation_type: ActivationType,
        inference_mode=False,
    ):
        super().__init__()
        is_glu_activation = is_glu(activation_type)
        if is_glu_activation:
            assert (
                H % 64 == 0 and H >= 512 and I % 64 == 0
            ), f"{LIBRARY_NAME} only supports GLU MoE with H % 64 == 0 (H >= 512) and I % 64 == 0"
        else:
            assert (
                H % 64 == 0 and H >= 512 and I % 128 == 0
            ), f"{LIBRARY_NAME} only supports non-GLU MoE with H % 64 == 0 (H >= 512) and I % 128 == 0"
        # TODO: this assertion does not mean that the MoE impl prohibits such config.
        # Instead, we just do not search for the best configs manually yet for small-shaped MoE
        if (I >= 128 and is_glu_activation) or (I >= 256 and not is_glu_activation):
            up_config = HopperGEMMConfig(
                tile_shape_mnk=(128, 256, 64),
                cluster_shape_mnk=(2, 1),
                epi_tile_size=(32 if not inference_mode else 64),
                is_pingpong=False,
                initial_d_epi_stage=2,
                raster_order=RasterOrderOption.AlongM,
            )
        elif (I == 64 and is_glu_activation) or (I == 128 and not is_glu_activation):
            up_config = HopperGEMMConfig(
                tile_shape_mnk=(192, 128, 64),
                cluster_shape_mnk=(1, 1),
                epi_tile_size=(32 if not inference_mode else 64),
                is_pingpong=True,
                initial_d_epi_stage=8,
                raster_order=RasterOrderOption.AlongM,
            )
        else:
            raise NotImplementedError()

        compute_swiglu = False
        compute_geglu = False
        compute_reglu = False

        compute_relu_sq = False
        compute_silu = False
        compute_relu = False
        compute_gelu = False

        if activation_type == ActivationType.SWIGLU:
            compute_swiglu = True
        elif activation_type == ActivationType.GEGLU:
            compute_geglu = True
        elif activation_type == ActivationType.REGLU:
            compute_reglu = True

        elif activation_type == ActivationType.RELU_SQ:
            compute_relu_sq = True
        elif activation_type == ActivationType.RELU:
            compute_relu = True
        elif activation_type == ActivationType.SILU:
            compute_silu = True
        elif activation_type == ActivationType.GELU:
            compute_gelu = True

        else:
            raise NotImplementedError(
                f"Activation function {activation_type} not supported yet!"
            )

        self.module = HopperWgmma_MoE_kernel(
            E,
            cutlass.Float32,
            up_config.tile_shape_mnk,
            (*up_config.cluster_shape_mnk, 1),
            pingpong=up_config.is_pingpong,
            is_persistent=True,
            compute_swiglu=compute_swiglu,
            compute_reglu=compute_reglu,
            compute_geglu=compute_geglu,
            compute_relu_sq=compute_relu_sq,
            compute_relu=compute_relu,
            compute_silu=compute_silu,
            compute_gelu=compute_gelu,
            is_A_gather=True,
            epi_tile_size=up_config.epi_tile_size,
            initial_d_epi_stage=up_config.initial_d_epi_stage,
            inference_mode=inference_mode,
        )
        self.max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            up_config.cluster_shape_mnk[0] * up_config.cluster_shape_mnk[1]
        )
        self.current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    @cute.jit
    def __call__(
        self,
        mX,
        mW1,
        mZ,
        mY1,
        mB1,
        mE_offset,
        mX_gather,
        mD_tensormap,
        mY1_tensormap,
        mE_permute_order,
        stream,
    ):
        return self.module(
            mX,
            mW1,
            None,
            mB1,
            mZ,
            mY1,
            None,
            None,
            mE_offset,
            mX_gather,
            None,
            None,
            None,
            None,
            None,
            mD_tensormap,
            mY1_tensormap,
            None,
            mE_permute_order,
            const_expr(self.max_active_clusters),
            stream,
        )


class HopperWgmma_MoE_Down_proj_Fwd:
    def __init__(self, E: int, H: int, I: int):
        super().__init__()
        assert (
            H % 64 == 0 and H >= 512 and I % 64 == 0
        ), f"{LIBRARY_NAME} only supports MoE with H % 64 == 0 (H >= 512) and I % 64 == 0"
        if I >= 1024:
            down_config = HopperGEMMConfig(
                tile_shape_mnk=(128, 256, 64),
                cluster_shape_mnk=(2, 1),
                epi_tile_size=32,
                is_pingpong=False,
                initial_d_epi_stage=4,
                raster_order=RasterOrderOption.AlongN,
            )
        elif I >= 256:
            down_config = HopperGEMMConfig(
                tile_shape_mnk=(128, 192, 64),
                cluster_shape_mnk=(2, 1),
                epi_tile_size=(96 if H % 96 == 0 else 64),
                is_pingpong=True,
                initial_d_epi_stage=5,
                raster_order=RasterOrderOption.AlongN,
            )
        elif I >= 64:
            down_config = HopperGEMMConfig(
                tile_shape_mnk=(128, 192, 64),
                cluster_shape_mnk=(1, 2),
                epi_tile_size=64,
                is_pingpong=True,
                initial_d_epi_stage=8,
                raster_order=RasterOrderOption.AlongN,
            )
        else:
            raise NotImplementedError()

        self.module = HopperWgmma_MoE_kernel(
            E,
            cutlass.Float32,
            down_config.tile_shape_mnk,
            (*down_config.cluster_shape_mnk, 1),
            pingpong=down_config.is_pingpong,
            is_persistent=True,
            compute_swiglu=False,
            is_A_gather=False,
            epi_tile_size=down_config.epi_tile_size,
            initial_d_epi_stage=down_config.initial_d_epi_stage,
        )
        self.max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            down_config.cluster_shape_mnk[0] * down_config.cluster_shape_mnk[1]
        )

    @cute.jit
    def __call__(
        self,
        mY1,
        mW2,
        mY2,
        mB2,
        mE_offset,
        mX_gather,
        mD_tensormap,
        mE_permute_order,
        stream,
    ):
        # we are not really using mX_gather in the Grouped GEMM,
        # but CuTe-DSL compiler disallows dynamic flow so we still need to pass this argument
        return self.module(
            mY1,
            mW2,
            None,
            mB2,
            mY2,
            None,
            None,
            None,
            mE_offset,
            mX_gather,
            None,
            None,
            None,
            None,
            None,
            mD_tensormap,
            None,
            None,
            mE_permute_order,
            const_expr(self.max_active_clusters),
            stream,
        )


class HopperWgmma_MoE_Down_proj_ActGrad_Bwd:
    def __init__(self, E: int, H: int, I: int, activation_type: ActivationType):
        super().__init__()
        is_glu_activation = is_glu(activation_type)
        if is_glu_activation:
            assert (
                H % 64 == 0 and H >= 512 and I % 64 == 0
            ), f"{LIBRARY_NAME} only supports GLU MoE with H % 64 == 0 (H >= 512) and I % 64 == 0"
        else:
            assert (
                H % 64 == 0 and H >= 512 and I % 128 == 0
            ), f"{LIBRARY_NAME} only supports non-GLU MoE with H % 64 == 0 (H >= 512) and I % 128 == 0"

        # heavy register pressure due to pingpong + heavy epilogue
        #   effectively no alternatives to this config
        dz_partial_ds_config = HopperGEMMConfig(
            tile_shape_mnk=(128, 128, 64),
            cluster_shape_mnk=(2, 1),
            epi_tile_size=32,
            initial_d_epi_stage=4,
            is_pingpong=True,
            raster_order=RasterOrderOption.Heuristic,
        )

        compute_swiglu = False
        compute_geglu = False
        compute_reglu = False

        compute_relu_sq = False
        compute_silu = False
        compute_relu = False
        compute_gelu = False

        if activation_type == ActivationType.SWIGLU:
            compute_swiglu = True
        elif activation_type == ActivationType.GEGLU:
            compute_geglu = True
        elif activation_type == ActivationType.REGLU:
            compute_reglu = True

        elif activation_type == ActivationType.RELU_SQ:
            compute_relu_sq = True
        elif activation_type == ActivationType.RELU:
            compute_relu = True
        elif activation_type == ActivationType.SILU:
            compute_silu = True
        elif activation_type == ActivationType.GELU:
            compute_gelu = True

        else:
            raise NotImplementedError(
                f"Activation function {activation_type} not supported yet!"
            )

        self.module = HopperWgmma_MoE_kernel(
            E,
            cutlass.Float32,
            dz_partial_ds_config.tile_shape_mnk,
            (*dz_partial_ds_config.cluster_shape_mnk, 1),
            pingpong=dz_partial_ds_config.is_pingpong,
            is_persistent=True,
            compute_swiglu=compute_swiglu,
            compute_reglu=compute_reglu,
            compute_geglu=compute_geglu,
            compute_relu_sq=compute_relu_sq,
            compute_relu=compute_relu,
            compute_silu=compute_silu,
            compute_gelu=compute_gelu,
            compute_dz_and_partial_ds_and_y1s=True,
            is_A_gather=True,
            epi_tile_size=dz_partial_ds_config.epi_tile_size,
            initial_d_epi_stage=dz_partial_ds_config.initial_d_epi_stage,
        )
        self.max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            dz_partial_ds_config.cluster_shape_mnk[0]
            * dz_partial_ds_config.cluster_shape_mnk[1]
        )

    @cute.jit
    def __call__(
        self,
        mDout,
        mW2_trans,
        mZ_FP32_if_GLU_else_BF16,
        mDz_FP32_if_GLU_else_BF16,
        mY1S,
        mS,
        mDS_partial,
        mE_offset,
        mX_gather,
        mS_scatter,
        tensormaps,
        mE_permute_order,
        stream,
    ):
        return self.module(
            mDout,
            mW2_trans,
            mZ_FP32_if_GLU_else_BF16,
            None,
            mDz_FP32_if_GLU_else_BF16,
            mY1S,
            mS,
            mDS_partial,
            mE_offset,
            mX_gather,
            None,
            mS_scatter,
            None,
            None,
            tensormaps[0],
            tensormaps[1],
            tensormaps[2],
            None,
            mE_permute_order,
            const_expr(self.max_active_clusters),
            stream,
        )


class HopperWgmma_MoE_Down_proj_WeightGrad_Bwd:
    def __init__(self, E: int, H: int, I: int):
        super().__init__()
        assert (
            H % 64 == 0 and H >= 512 and I % 64 == 0
        ), f"{LIBRARY_NAME} only supports MoE with H % 64 == 0 (H >= 512) and I % 64 == 0"

        if I >= 128:
            dw2_config = HopperGEMMConfig(
                tile_shape_mnk=(128, 256, 64),
                cluster_shape_mnk=(2, 1),
                epi_tile_size=16,
                is_pingpong=False,
                initial_d_epi_stage=6,
                raster_order=RasterOrderOption.AlongN,
            )
        elif I == 64:
            dw2_config = HopperGEMMConfig(
                tile_shape_mnk=(64, 192, 64),
                cluster_shape_mnk=(2, 1),
                epi_tile_size=32,
                is_pingpong=True,
                initial_d_epi_stage=6,
                raster_order=RasterOrderOption.AlongN,
            )
        else:
            raise NotImplementedError()

        self.module = HopperWgmma_MoE_kernel(
            E,
            cutlass.Float32,
            dw2_config.tile_shape_mnk,
            (*dw2_config.cluster_shape_mnk, 1),
            pingpong=dw2_config.is_pingpong,
            is_persistent=True,
            compute_swiglu=False,
            compute_weight_gradient=True,
            compute_dz_and_partial_ds_and_y1s=False,
            is_A_gather=True,
            epi_tile_size=dw2_config.epi_tile_size,
            initial_d_epi_stage=dw2_config.initial_d_epi_stage,
        )
        self.max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            dw2_config.cluster_shape_mnk[0] * dw2_config.cluster_shape_mnk[1]
        )

    @cute.jit
    def __call__(
        self,
        mDout_trans,
        mY1S_trans,
        mDw2,
        mE_offset,
        mX_gather,
        tensormaps,
        mE_permute_order,
        stream,
    ):
        return self.module(
            mDout_trans,
            mY1S_trans,
            None,
            None,
            mDw2,
            None,
            None,
            None,
            mE_offset,
            mX_gather,
            None,
            None,
            None,
            tensormaps[0],
            None,
            None,
            None,
            None,
            mE_permute_order,
            const_expr(self.max_active_clusters),
            stream,
        )


class HopperWgmma_MoE_Up_proj_ActGrad_Bwd:
    def __init__(self, E: int, H: int, I: int, is_glu_activation: bool):
        super().__init__()
        if is_glu_activation:
            assert (
                H % 64 == 0 and H >= 512 and I % 64 == 0
            ), f"{LIBRARY_NAME} only supports GLU MoE with H % 64 == 0 (H >= 512) and I % 64 == 0"
        else:
            assert (
                H % 64 == 0 and H >= 512 and I % 128 == 0
            ), f"{LIBRARY_NAME} only supports non-GLU MoE with H % 64 == 0 (H >= 512) and I % 128 == 0"

        if (I >= 512 and is_glu_activation) or (I >= 1024 and not is_glu_activation):
            dx_config = HopperGEMMConfig(
                tile_shape_mnk=(128, 256, 64),
                cluster_shape_mnk=(2, 1),
                epi_tile_size=32,
                is_pingpong=False,
                initial_d_epi_stage=4,
                raster_order=RasterOrderOption.AlongN,
            )
        elif (I >= 64 and is_glu_activation) or (I >= 128 and not is_glu_activation):
            dx_config = HopperGEMMConfig(
                tile_shape_mnk=(128, 192, 64),
                cluster_shape_mnk=(2, 1),
                epi_tile_size=64,
                is_pingpong=True,
                initial_d_epi_stage=8,
                raster_order=RasterOrderOption.AlongN,
            )
        else:
            raise NotImplementedError()

        self.module = HopperWgmma_MoE_kernel(
            E,
            cutlass.Float32,
            dx_config.tile_shape_mnk,
            (*dx_config.cluster_shape_mnk, 1),
            pingpong=dx_config.is_pingpong,
            is_persistent=True,
            compute_swiglu=False,
            compute_dz_and_partial_ds_and_y1s=False,
            is_A_gather=False,
            epi_tile_size=dx_config.epi_tile_size,
        )

        self.max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            dx_config.cluster_shape_mnk[0] * dx_config.cluster_shape_mnk[1]
        )
        self.current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    @cute.jit
    def __call__(
        self,
        mDz,
        mW1_trans,
        mDx_expanded,
        mE_offset,
        mX_gather,
        mS_scatter,
        tensormaps,
        mE_permute_order,
        stream,
    ):
        return self.module(
            mDz,
            mW1_trans,
            None,
            None,
            mDx_expanded,
            None,
            None,
            None,
            mE_offset,
            mX_gather,
            None,
            mS_scatter,
            None,
            None,
            None,
            tensormaps[0],
            tensormaps[1],
            None,
            mE_permute_order,
            const_expr(self.max_active_clusters),
            stream,
        )


class HopperWgmma_MoE_Up_proj_WeightGrad_Bwd:
    def __init__(self, E: int, H: int, I: int, is_glu_activation: bool):
        super().__init__()
        if is_glu_activation:
            assert (
                H % 64 == 0 and H >= 512 and I % 64 == 0
            ), f"{LIBRARY_NAME} only supports GLU MoE with H % 64 == 0 (H >= 512) and I % 64 == 0"
        else:
            assert (
                H % 64 == 0 and H >= 512 and I % 128 == 0
            ), f"{LIBRARY_NAME} only supports non-GLU MoE with H % 64 == 0 (H >= 512) and I % 128 == 0"

        if (I >= 128 and is_glu_activation) or (I >= 256 and not is_glu_activation):
            dw1_config = HopperGEMMConfig(
                tile_shape_mnk=(128, 256, 64),
                cluster_shape_mnk=(2, 1),
                epi_tile_size=16,
                is_pingpong=False,
                initial_d_epi_stage=6,
                raster_order=RasterOrderOption.Heuristic,
            )
        elif (I == 64 and is_glu_activation) or (I == 128 and not is_glu_activation):
            dw1_config = HopperGEMMConfig(
                tile_shape_mnk=(256, 128, 64),
                cluster_shape_mnk=(2, 1),
                epi_tile_size=16,
                is_pingpong=False,
                initial_d_epi_stage=6,
                raster_order=RasterOrderOption.AlongN,
            )
        else:
            raise NotImplementedError()

        self.module = HopperWgmma_MoE_kernel(
            E,
            cutlass.Float32,
            dw1_config.tile_shape_mnk,
            (*dw1_config.cluster_shape_mnk, 1),
            pingpong=dw1_config.is_pingpong,
            is_persistent=True,
            compute_swiglu=False,
            compute_weight_gradient=True,
            compute_dz_and_partial_ds_and_y1s=False,
            is_A_gather=True,
            epi_tile_size=dw1_config.epi_tile_size,
        )

        self.max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            dw1_config.cluster_shape_mnk[0] * dw1_config.cluster_shape_mnk[1]
        )

    @cute.jit
    def __call__(
        self,
        mX_trans,
        mDz_trans,
        mDw1_trans,
        mE_offset,
        mX_gather,
        tensormaps,
        mE_permute_order,
        stream,
    ):
        return self.module(
            mX_trans,
            mDz_trans,
            None,
            None,
            mDw1_trans,
            None,
            None,
            None,
            mE_offset,
            mX_gather,
            None,
            None,
            None,
            tensormaps[0],
            None,
            None,
            None,
            None,
            mE_permute_order,
            const_expr(self.max_active_clusters),
            stream,
        )
