"""
Sparse Video Gen 2 (SAP) attention backend.

This is a baseline integration that wires the backend into the
attention framework.

Adapted from https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/models/wan/attention.py
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from svg.kernels.triton.permute import (
        apply_inverse_permutation_triton,
        permute_tensor_by_labels_triton,
    )
    from svg.kmeans_utils import (
        batch_kmeans_Euclid,
        dynamic_block_sparse_fwd_flashinfer,
        identify_dynamic_map,
    )

    svg2_available = True
except ImportError:
    svg2_available = False

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SparseVideoGen2AttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SPARSE_VIDEO_GEN_2_ATTN

    @staticmethod
    def get_impl_cls() -> type["SparseVideoGen2AttentionImpl"]:
        return SparseVideoGen2AttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["SparseVideoGen2AttentionMetadata"]:
        return SparseVideoGen2AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["SparseVideoGen2AttentionMetadataBuilder"]:
        return SparseVideoGen2AttentionMetadataBuilder


@dataclass
class Svg2LayerCache:
    # centroids for kmeans clustering
    q_centroids: torch.Tensor | None = None
    k_centroids: torch.Tensor | None = None
    centroids_initialized: bool = False


@dataclass
class Svg2Cache:
    layers: dict[int, Svg2LayerCache] = field(default_factory=dict)

    def get_layer(self, layer_idx: int) -> Svg2LayerCache:
        layer_cache = self.layers.get(layer_idx)
        if layer_cache is None:
            layer_cache = Svg2LayerCache()
            self.layers[layer_idx] = layer_cache
        return layer_cache


@dataclass
class SparseVideoGen2AttentionMetadata(AttentionMetadata):
    current_timestep: int
    num_q_centroids: int
    num_k_centroids: int
    top_p_kmeans: float
    min_kc_ratio: float
    kmeans_iter_init: int
    kmeans_iter_step: int
    zero_step_kmeans_init: bool
    first_layers_fp: float
    first_times_fp: float
    context_length: int
    num_frame: int
    frame_size: int
    cache: Svg2Cache
    prompt_length: int | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None


def _require_kwarg(kwargs: dict[str, Any], name: str) -> Any:
    if name not in kwargs:
        raise ValueError(
            f"Missing required argument for SparseVideoGen2Attention: {name}"
        )
    return kwargs[name]


class SparseVideoGen2AttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore[override]
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, ...],
        patch_size: tuple[int, int, int],
        cache: Svg2Cache,
        num_q_centroids: int,
        num_k_centroids: int,
        top_p_kmeans: float,
        min_kc_ratio: float,
        kmeans_iter_init: int,
        kmeans_iter_step: int,
        zero_step_kmeans_init: bool,
        first_layers_fp: float,
        first_times_fp: float,
        context_length: int = 0,
        prompt_length: int | None = None,
        **kwargs: dict[str, Any],
    ) -> SparseVideoGen2AttentionMetadata:
        raw_shape = tuple(raw_latent_shape)
        if len(raw_shape) == 5:
            t, h, w = raw_shape[2:5]
        elif len(raw_shape) == 3:
            t, h, w = raw_shape
        else:
            raise ValueError(
                "raw_latent_shape must be (T, H, W) or (B, C, T, H, W) for SAP attention"
            )
        pt, ph, pw = patch_size
        if t % pt != 0 or h % ph != 0 or w % pw != 0:
            raise ValueError(
                "raw_latent_shape must be divisible by patch_size for SAP attention"
            )

        num_frame = t // pt
        frame_size = (h // ph) * (w // pw)

        return SparseVideoGen2AttentionMetadata(
            current_timestep=current_timestep,
            num_q_centroids=num_q_centroids,
            num_k_centroids=num_k_centroids,
            top_p_kmeans=top_p_kmeans,
            min_kc_ratio=min_kc_ratio,
            kmeans_iter_init=kmeans_iter_init,
            kmeans_iter_step=kmeans_iter_step,
            zero_step_kmeans_init=zero_step_kmeans_init,
            first_layers_fp=first_layers_fp,
            first_times_fp=first_times_fp,
            context_length=context_length,
            prompt_length=prompt_length,
            num_frame=num_frame,
            frame_size=frame_size,
            cache=cache,
        )


class SparseVideoGen2AttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if causal:
            raise ValueError(
                "Sparse Video Gen 2 attention does not support causal attention"
            )
        if not svg2_available:
            raise ImportError(
                "Sparse Video Gen 2 attention backend requires svg package to be installed"
                "Please install it by following the instructions at "
                "https://github.com/svg-project/Sparse-VideoGen"
            )
        self.prefix = prefix
        self.layer_idx = self._get_layer_idx(prefix)

    def _get_layer_idx(self, prefix: str) -> int:
        parts = prefix.split(".")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid prefix for SparseVideoGen2AttentionImpl: {prefix}"
            )
        return int(parts[-3])

    def kmeans_init(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.reshape(cfg * num_heads, seq_len, dim),
            n_clusters=attn_metadata.num_q_centroids,
            max_iters=attn_metadata.kmeans_iter_init,
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.reshape(cfg * num_heads, seq_len, dim),
            n_clusters=attn_metadata.num_k_centroids,
            max_iters=attn_metadata.kmeans_iter_init,
        )

        layer_cache = attn_metadata.cache.get_layer(self.layer_idx)
        layer_cache.q_centroids = qcentroids
        layer_cache.k_centroids = kcentroids

        return (
            qlabels,
            qcentroids,
            qcluster_sizes,
            qiter,
            klabels,
            kcentroids,
            kcluster_sizes,
            kiter,
        )

    def kmeans_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ):
        cfg, num_heads, seq_len, dim = query.size()
        layer_cache = attn_metadata.cache.get_layer(self.layer_idx)
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.reshape(cfg * num_heads, seq_len, dim),
            n_clusters=attn_metadata.num_q_centroids,
            max_iters=attn_metadata.kmeans_iter_step,
            init_centroids=layer_cache.q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.reshape(cfg * num_heads, seq_len, dim),
            n_clusters=attn_metadata.num_k_centroids,
            max_iters=attn_metadata.kmeans_iter_step,
            init_centroids=layer_cache.k_centroids,
        )

        layer_cache.q_centroids = qcentroids
        layer_cache.k_centroids = kcentroids

        return (
            qlabels,
            qcentroids,
            qcluster_sizes,
            qiter,
            klabels,
            kcentroids,
            kcluster_sizes,
            kiter,
        )

    def kmeans_clustering(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ):
        layer_cache = attn_metadata.cache.get_layer(self.layer_idx)
        if not layer_cache.centroids_initialized:
            (
                qlabels,
                qcentroids,
                qcluster_sizes,
                qiter,
                klabels,
                kcentroids,
                kcluster_sizes,
                kiter,
            ) = self.kmeans_init(query, key, attn_metadata)
            layer_cache.centroids_initialized = True
            logger.debug(
                "Centroids initialized at layer %s (init iters: %s).",
                self.layer_idx,
                attn_metadata.kmeans_iter_init,
            )
        else:
            (
                qlabels,
                qcentroids,
                qcluster_sizes,
                qiter,
                klabels,
                kcentroids,
                kcluster_sizes,
                kiter,
            ) = self.kmeans_step(query, key, attn_metadata)

        return (
            qlabels,
            qcentroids,
            qcluster_sizes,
            qiter,
            klabels,
            kcentroids,
            kcluster_sizes,
            kiter,
        )

    def semantic_aware_permutation(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ):
        cfg, num_heads, seq_len, dim = query.size()

        # 1. Kmeans clustering
        (
            qlabels,
            qcentroids,
            qcluster_sizes,
            qiter,
            klabels,
            kcentroids,
            kcluster_sizes,
            kiter,
        ) = self.kmeans_clustering(query, key, attn_metadata)

        # 2. Identify dynamic map
        q_cluster_sizes = qcluster_sizes.view(
            cfg, num_heads, attn_metadata.num_q_centroids
        )
        k_cluster_sizes = kcluster_sizes.view(
            cfg, num_heads, attn_metadata.num_k_centroids
        )

        dynamic_map = identify_dynamic_map(
            qcentroids.view(cfg, num_heads, attn_metadata.num_q_centroids, dim),
            kcentroids.view(cfg, num_heads, attn_metadata.num_k_centroids, dim),
            q_cluster_sizes,
            k_cluster_sizes,
            attn_metadata.top_p_kmeans,
            attn_metadata.min_kc_ratio,
        )

        # 3. Permute the query, key, value
        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(
            query, qlabels, dim=2
        )
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(
            key, klabels, dim=2
        )
        v_permuted, v_sorted_indices = permute_tensor_by_labels_triton(
            value, klabels, dim=2, sorted_indices=k_sorted_indices
        )

        return (
            q_permuted,
            k_permuted,
            v_permuted,
            dynamic_map,
            q_cluster_sizes,
            k_cluster_sizes,
            q_sorted_indices,
        )

    def _hunyuan_dynamic_map_post_processing(
        self,
        q_perm: torch.Tensor,
        k_perm: torch.Tensor,
        v_perm: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dyn_map: torch.Tensor,
        qc_sz_s: torch.Tensor,
        kc_sz_s: torch.Tensor,
        q_sorted_indices: torch.Tensor,
        video_length: int,
        context_length: int,
        prompt_length: int,
        unprompt_length: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Place the permuted video tokens back and keep text tokens at the tail.
        query[:, :, :-context_length, :] = q_perm
        key[:, :, :-context_length, :] = k_perm
        value[:, :, :-context_length, :] = v_perm

        # Add prompt/unprompt clusters to the dynamic map.
        dyn_map = F.pad(dyn_map, (0, 2, 0, 2), value=0)
        dyn_map[:, :, -2, :-1] = True
        dyn_map[:, :, :-1, -2] = True
        dyn_map[:, :, -1, -1] = True

        qc_sz_s = F.pad(qc_sz_s, (0, 2), value=0)
        qc_sz_s[:, :, -2] = prompt_length
        qc_sz_s[:, :, -1] = unprompt_length
        kc_sz_s = F.pad(kc_sz_s, (0, 2), value=0)
        kc_sz_s[:, :, -2] = prompt_length
        kc_sz_s[:, :, -1] = unprompt_length

        q_sorted_indices = F.pad(q_sorted_indices, (0, context_length), value=0)
        q_sorted_indices[:, video_length:] = torch.arange(
            video_length,
            video_length + context_length,
            device=q_sorted_indices.device,
        )
        return query, key, value, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ) -> torch.Tensor:
        torch.backends.cuda.preferred_linalg_library(backend="magma")
        res = None
        # bshd -> bhsd
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        batch_size, num_heads, seq_len, dim = query.size()

        context_length, num_frame, frame_size = (
            attn_metadata.context_length,
            attn_metadata.num_frame,
            attn_metadata.frame_size,
        )
        prompt_length = attn_metadata.prompt_length
        if prompt_length is None:
            prompt_length = context_length

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        # Determine if we use Full Attention to calculate
        full_attention_flag = False

        if self.layer_idx < attn_metadata.first_layers_fp:
            full_attention_flag = True
        if attn_metadata.current_timestep > attn_metadata.first_times_fp:
            full_attention_flag = True

        if full_attention_flag:
            if attn_metadata.zero_step_kmeans_init:
                video_length = attn_metadata.num_frame * attn_metadata.frame_size
                query_video = query[:, :, :video_length, :].contiguous()
                key_video = key[:, :, :video_length, :].contiguous()
                self.kmeans_clustering(query_video, key_video, attn_metadata)

            with sdpa_kernel(
                SDPBackend.CUDNN_ATTENTION
            ):  # not sure why we need to force cudnn here, but it's faster than flash attention
                output_hidden_states = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )

            res = output_hidden_states.reshape(
                batch_size, num_heads, seq_len, dim
            ).transpose(1, 2)
        else:
            if context_length > 0:
                video_length = num_frame * frame_size
                unprompt_length = max(context_length - prompt_length, 0)
                query_video = query[:, :, :video_length, :].contiguous()
                key_video = key[:, :, :video_length, :].contiguous()
                value_video = value[:, :, :video_length, :].contiguous()

                (
                    q_perm,
                    k_perm,
                    v_perm,
                    dyn_map,
                    qc_sz_s,
                    kc_sz_s,
                    q_sorted_indices,
                ) = self.semantic_aware_permutation(
                    query_video, key_video, value_video, attn_metadata
                )
                (
                    q_perm,
                    k_perm,
                    v_perm,
                    dyn_map,
                    qc_sz_s,
                    kc_sz_s,
                    q_sorted_indices,
                ) = self._hunyuan_dynamic_map_post_processing(
                    q_perm,
                    k_perm,
                    v_perm,
                    query,
                    key,
                    value,
                    dyn_map,
                    qc_sz_s,
                    kc_sz_s,
                    q_sorted_indices,
                    video_length,
                    context_length,
                    prompt_length,
                    unprompt_length,
                )
            else:
                (
                    q_perm,
                    k_perm,
                    v_perm,
                    dyn_map,
                    qc_sz_s,
                    kc_sz_s,
                    q_sorted_indices,
                ) = self.semantic_aware_permutation(query, key, value, attn_metadata)

            output_permuted = dynamic_block_sparse_fwd_flashinfer(
                q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, is_cpu=False
            )

            attn_output = apply_inverse_permutation_triton(
                output_permuted, q_sorted_indices, dim=2
            )

            res = attn_output.reshape(batch_size, num_heads, seq_len, dim).transpose(
                1, 2
            )

        torch.backends.cuda.preferred_linalg_library(
            backend="default"
        )  # reset to default
        return res.contiguous()
