# SPDX-License-Identifier: Apache-2.0
"""
3D mesh utilities for Hunyuan3D texture generation.

This module provides:
- Mesh I/O and conversion utilities
- Camera projection utilities
- Mesh rendering with CUDA rasterization
- Texture baking and inpainting

Adapted from Hunyuan3D-2: https://github.com/Tencent/Hunyuan3D-2
"""

from __future__ import annotations

import math
import os
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange, repeat
from PIL import Image

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# =============================================================================
# Camera Utilities
# =============================================================================


def transform_pos(
    mtx: Union[np.ndarray, torch.Tensor],
    pos: torch.Tensor,
    keepdim: bool = False,
) -> torch.Tensor:
    """Transform positions by a matrix."""
    t_mtx = torch.from_numpy(mtx).to(pos.device) if isinstance(mtx, np.ndarray) else mtx

    if pos.shape[-1] == 3:
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
    else:
        posw = pos

    if keepdim:
        return torch.matmul(posw, t_mtx.t())[...]
    else:
        return torch.matmul(posw, t_mtx.t())[None, ...]


def get_mv_matrix(
    elev: float,
    azim: float,
    camera_distance: float,
    center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute model-view matrix from camera parameters."""
    elev = -elev
    azim += 90

    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim)

    camera_position = np.array(
        [
            camera_distance * math.cos(elev_rad) * math.cos(azim_rad),
            camera_distance * math.cos(elev_rad) * math.sin(azim_rad),
            camera_distance * math.sin(elev_rad),
        ]
    )

    if center is None:
        center = np.array([0, 0, 0])
    else:
        center = np.array(center)

    lookat = center - camera_position
    lookat = lookat / np.linalg.norm(lookat)

    up = np.array([0, 0, 1.0])
    right = np.cross(lookat, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, lookat)
    up = up / np.linalg.norm(up)

    c2w = np.concatenate(
        [np.stack([right, up, -lookat], axis=-1), camera_position[:, None]], axis=-1
    )

    w2c = np.zeros((4, 4))
    w2c[:3, :3] = np.transpose(c2w[:3, :3], (1, 0))
    w2c[:3, 3:] = -np.matmul(np.transpose(c2w[:3, :3], (1, 0)), c2w[:3, 3:])
    w2c[3, 3] = 1.0

    return w2c.astype(np.float32)


def get_orthographic_projection_matrix(
    left: float = -1,
    right: float = 1,
    bottom: float = -1,
    top: float = 1,
    near: float = 0,
    far: float = 2,
) -> np.ndarray:
    """Compute orthographic projection matrix."""
    ortho_matrix = np.eye(4, dtype=np.float32)
    ortho_matrix[0, 0] = 2 / (right - left)
    ortho_matrix[1, 1] = 2 / (top - bottom)
    ortho_matrix[2, 2] = -2 / (far - near)
    ortho_matrix[0, 3] = -(right + left) / (right - left)
    ortho_matrix[1, 3] = -(top + bottom) / (top - bottom)
    ortho_matrix[2, 3] = -(far + near) / (far - near)
    return ortho_matrix


def get_perspective_projection_matrix(
    fovy: float,
    aspect_wh: float,
    near: float,
    far: float,
) -> np.ndarray:
    """Compute perspective projection matrix."""
    fovy_rad = math.radians(fovy)
    return np.array(
        [
            [1.0 / (math.tan(fovy_rad / 2.0) * aspect_wh), 0, 0, 0],
            [0, 1.0 / math.tan(fovy_rad / 2.0), 0, 0],
            [0, 0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
            [0, 0, -1, 0],
        ]
    ).astype(np.float32)


# =============================================================================
# Mesh I/O Utilities
# =============================================================================


def export_to_trimesh(mesh_output: Any) -> Any:
    """Convert mesh output to trimesh format.

    Args:
        mesh_output: Mesh output from VAE decoder, can be single mesh or list.

    Returns:
        trimesh.Trimesh object(s).
    """
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                # Reverse face winding
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_obj = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_obj)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        return trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)


def load_mesh(path: str) -> Any:
    """Load a mesh from file.

    Args:
        path: Path to mesh file (OBJ, GLB, etc.).

    Returns:
        Loaded trimesh object.
    """
    return trimesh.load(path)


def save_mesh(
    mesh: Any,
    path: str,
    file_type: Optional[str] = None,
) -> str:
    """Save a mesh to file.

    Args:
        mesh: trimesh.Trimesh object.
        path: Output path.
        file_type: Optional file type override.

    Returns:
        Path to saved file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mesh.export(path, file_type=file_type)
    return path


def convert_obj_to_glb(obj_path: str, glb_path: Optional[str] = None) -> str:
    """Convert OBJ file to GLB format.

    Args:
        obj_path: Path to input OBJ file.
        glb_path: Optional output GLB path. Defaults to same name with .glb extension.

    Returns:
        Path to output GLB file.
    """
    if glb_path is None:
        glb_path = obj_path.rsplit(".", 1)[0] + ".glb"

    mesh = load_mesh(obj_path)
    return save_mesh(mesh, glb_path)


def mesh_uv_wrap(mesh: Any) -> Any:
    """Apply UV unwrapping to mesh.

    Args:
        mesh: Input trimesh object.

    Returns:
        Mesh with UV coordinates.
    """
    try:
        import xatlas
    except ImportError:
        logger.warning("xatlas not available, skipping UV unwrap")
        return mesh

    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

    # Remap vertices according to xatlas output
    # vmapping contains indices into the original vertex array
    new_vertices = mesh.vertices[vmapping]
    new_faces = indices

    # Create new mesh with remapped vertices and faces
    try:
        # Try to get existing material
        if hasattr(mesh.visual, "material") and mesh.visual.material is not None:
            material = mesh.visual.material
        else:
            # Create a default material
            material = trimesh.visual.material.SimpleMaterial()

        # Create new mesh with UV coordinates
        new_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False,  # Don't merge vertices
        )
        new_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)

        return new_mesh

    except Exception as e:
        logger.warning(f"Failed to create UV-mapped mesh: {e}")
        # Fallback: try to just add UVs to original mesh
        try:
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs[: len(mesh.vertices)])
            return mesh
        except Exception as e2:
            logger.warning(f"Failed to apply UV mapping: {e2}")
            return mesh


def remesh_mesh(
    input_path: str,
    output_path: str,
    target_faces: int = 50000,
) -> str:
    """Remesh a mesh to target face count.

    Args:
        input_path: Path to input mesh.
        output_path: Path to output mesh.
        target_faces: Target number of faces.

    Returns:
        Path to remeshed mesh.
    """
    try:
        import pymeshlab
    except ImportError:
        logger.warning("pymeshlab not available, copying mesh without remeshing")
        import shutil

        shutil.copy(input_path, output_path)
        return output_path

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)

    # Simplify mesh
    current_faces = ms.current_mesh().face_number()
    if current_faces > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)

    ms.save_current_mesh(output_path)
    return output_path


# =============================================================================
# Mesh Texture Inpainting
# =============================================================================


def meshVerticeInpaint_smooth(
    texture: np.ndarray,
    mask: np.ndarray,
    vtx_pos: np.ndarray,
    vtx_uv: np.ndarray,
    pos_idx: np.ndarray,
    uv_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inpaint texture using mesh vertex connectivity.

    This function propagates colors from painted vertices to unpainted vertices
    using the mesh connectivity graph, weighted by inverse distance.

    Args:
        texture: Input texture [H, W, C], float32 in [0, 1]
        mask: Valid region mask [H, W], uint8 (255 = valid)
        vtx_pos: Vertex positions [V, 3]
        vtx_uv: UV coordinates [V, 2]
        pos_idx: Face vertex indices [F, 3]
        uv_idx: Face UV indices [F, 3]

    Returns:
        new_texture: Inpainted texture [H, W, C]
        new_mask: Updated mask [H, W]
    """
    texture_height, texture_width, texture_channel = texture.shape
    vtx_num = vtx_pos.shape[0]

    # Initialize vertex data
    vtx_mask = np.zeros(vtx_num, dtype=np.float32)
    vtx_color = [np.zeros(texture_channel, dtype=np.float32) for _ in range(vtx_num)]
    uncolored_vtxs = []

    # Build adjacency graph
    G = [[] for _ in range(vtx_num)]

    # Sample colors from texture for each vertex
    for i in range(uv_idx.shape[0]):
        for k in range(3):
            vtx_uv_idx = uv_idx[i, k]
            vtx_idx = pos_idx[i, k]
            uv_v = int(round(vtx_uv[vtx_uv_idx, 0] * (texture_width - 1)))
            uv_u = int(round((1.0 - vtx_uv[vtx_uv_idx, 1]) * (texture_height - 1)))

            # Clamp UV coordinates
            uv_u = max(0, min(uv_u, texture_height - 1))
            uv_v = max(0, min(uv_v, texture_width - 1))

            if mask[uv_u, uv_v] > 0:
                vtx_mask[vtx_idx] = 1.0
                vtx_color[vtx_idx] = texture[uv_u, uv_v].copy()
            else:
                uncolored_vtxs.append(vtx_idx)

            # Add edge to adjacency graph
            G[pos_idx[i, k]].append(pos_idx[i, (k + 1) % 3])

    # Iteratively propagate colors
    smooth_count = 2
    last_uncolored_vtx_count = 0

    while smooth_count > 0:
        uncolored_vtx_count = 0

        for vtx_idx in uncolored_vtxs:
            sum_color = np.zeros(texture_channel, dtype=np.float32)
            total_weight = 0.0
            vtx_0 = vtx_pos[vtx_idx]

            for connected_idx in G[vtx_idx]:
                if vtx_mask[connected_idx] > 0:
                    vtx1 = vtx_pos[connected_idx]
                    dist = np.sqrt(np.sum((vtx_0 - vtx1) ** 2))
                    dist_weight = 1.0 / max(dist, 1e-4)
                    dist_weight *= dist_weight
                    sum_color += vtx_color[connected_idx] * dist_weight
                    total_weight += dist_weight

            if total_weight > 0:
                vtx_color[vtx_idx] = sum_color / total_weight
                vtx_mask[vtx_idx] = 1.0
            else:
                uncolored_vtx_count += 1

        if last_uncolored_vtx_count == uncolored_vtx_count:
            smooth_count -= 1
        else:
            smooth_count += 1
        last_uncolored_vtx_count = uncolored_vtx_count

    # Write back colors to texture
    new_texture = texture.copy()
    new_mask = mask.copy()

    for face_idx in range(uv_idx.shape[0]):
        for k in range(3):
            vtx_uv_idx = uv_idx[face_idx, k]
            vtx_idx = pos_idx[face_idx, k]

            if vtx_mask[vtx_idx] == 1.0:
                uv_v = int(round(vtx_uv[vtx_uv_idx, 0] * (texture_width - 1)))
                uv_u = int(round((1.0 - vtx_uv[vtx_uv_idx, 1]) * (texture_height - 1)))

                # Clamp UV coordinates
                uv_u = max(0, min(uv_u, texture_height - 1))
                uv_v = max(0, min(uv_v, texture_width - 1))

                new_texture[uv_u, uv_v] = vtx_color[vtx_idx]
                new_mask[uv_u, uv_v] = 255

    return new_texture, new_mask


def meshVerticeInpaint(
    texture: np.ndarray,
    mask: np.ndarray,
    vtx_pos: np.ndarray,
    vtx_uv: np.ndarray,
    pos_idx: np.ndarray,
    uv_idx: np.ndarray,
    method: str = "smooth",
) -> Tuple[np.ndarray, np.ndarray]:
    """Inpaint texture using mesh connectivity.

    Args:
        texture: Input texture [H, W, C]
        mask: Valid region mask [H, W]
        vtx_pos: Vertex positions [V, 3]
        vtx_uv: UV coordinates [V, 2]
        pos_idx: Face vertex indices [F, 3]
        uv_idx: Face UV indices [F, 3]
        method: Inpainting method ("smooth")

    Returns:
        new_texture: Inpainted texture
        new_mask: Updated mask
    """
    if method == "smooth":
        return meshVerticeInpaint_smooth(
            texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx
        )
    else:
        raise ValueError(f"Invalid method '{method}'. Use 'smooth'.")


# =============================================================================
# Mesh Rendering Utilities
# =============================================================================


def stride_from_shape(shape: Tuple[int, ...]) -> List[int]:
    """Compute stride from shape for scatter operations."""
    stride = [1]
    for x in reversed(shape[1:]):
        stride.append(stride[-1] * x)
    return list(reversed(stride))


def scatter_add_nd_with_count(
    input: torch.Tensor,
    count: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter add with counting for texture baking."""
    D = indices.shape[-1]
    C = input.shape[-1]
    size = input.shape[:-1]
    stride = stride_from_shape(size)

    assert len(size) == D

    input = input.view(-1, C)
    count = count.view(-1, 1)

    flatten_indices = (
        indices * torch.tensor(stride, dtype=torch.long, device=indices.device)
    ).sum(-1)

    if weights is None:
        weights = torch.ones_like(values[..., :1])

    input.scatter_add_(0, flatten_indices.unsqueeze(1).repeat(1, C), values)
    count.scatter_add_(0, flatten_indices.unsqueeze(1), weights)

    return input.view(*size, C), count.view(*size, 1)


def linear_grid_put_2d(
    H: int,
    W: int,
    coords: torch.Tensor,
    values: torch.Tensor,
    return_count: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Put values into a 2D grid using linear interpolation."""
    C = values.shape[-1]

    indices = coords * torch.tensor(
        [H - 1, W - 1], dtype=torch.float32, device=coords.device
    )
    indices_00 = indices.floor().long()
    indices_00[:, 0].clamp_(0, H - 2)
    indices_00[:, 1].clamp_(0, W - 2)
    indices_01 = indices_00 + torch.tensor(
        [0, 1], dtype=torch.long, device=indices.device
    )
    indices_10 = indices_00 + torch.tensor(
        [1, 0], dtype=torch.long, device=indices.device
    )
    indices_11 = indices_00 + torch.tensor(
        [1, 1], dtype=torch.long, device=indices.device
    )

    h = indices[..., 0] - indices_00[..., 0].float()
    w = indices[..., 1] - indices_00[..., 1].float()
    w_00 = (1 - h) * (1 - w)
    w_01 = (1 - h) * w
    w_10 = h * (1 - w)
    w_11 = h * w

    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)
    weights = torch.ones_like(values[..., :1])

    result, count = scatter_add_nd_with_count(
        result,
        count,
        indices_00,
        values * w_00.unsqueeze(1),
        weights * w_00.unsqueeze(1),
    )
    result, count = scatter_add_nd_with_count(
        result,
        count,
        indices_01,
        values * w_01.unsqueeze(1),
        weights * w_01.unsqueeze(1),
    )
    result, count = scatter_add_nd_with_count(
        result,
        count,
        indices_10,
        values * w_10.unsqueeze(1),
        weights * w_10.unsqueeze(1),
    )
    result, count = scatter_add_nd_with_count(
        result,
        count,
        indices_11,
        values * w_11.unsqueeze(1),
        weights * w_11.unsqueeze(1),
    )

    if return_count:
        return result, count

    mask = count.squeeze(-1) > 0
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


class MeshRender:
    """Mesh renderer using CUDA rasterization for texture generation.

    This class provides rendering utilities for multi-view texture generation:
    - Normal map rendering
    - Position map rendering
    - Texture back-projection
    - Texture baking from multiple views

    Uses custom CUDA rasterizer for fast GPU-based rendering.
    """

    def __init__(
        self,
        camera_distance: float = 1.45,
        camera_type: str = "orth",
        default_resolution: int = 1024,
        texture_size: int = 1024,
        bake_mode: str = "linear",
        device: str = "cuda",
    ):
        """Initialize the mesh renderer.

        Args:
            camera_distance: Distance from camera to object center
            camera_type: "orth" for orthographic, "perspective" for perspective
            default_resolution: Default rendering resolution
            texture_size: Texture map resolution
            bake_mode: Texture baking mode ("linear")
            device: Device to use for tensor operations
        """
        self.device = device

        self.set_default_render_resolution(default_resolution)
        self.set_default_texture_resolution(texture_size)

        self.camera_distance = camera_distance
        self.camera_type = camera_type
        self.bake_angle_thres = 75
        self.bake_unreliable_kernel_size = int(
            (2 / 512) * max(self.default_resolution[0], self.default_resolution[1])
        )
        self.bake_mode = bake_mode

        # Set up camera projection matrix
        if camera_type == "orth":
            self.ortho_scale = 1.2
            self.camera_proj_mat = get_orthographic_projection_matrix(
                left=-self.ortho_scale * 0.5,
                right=self.ortho_scale * 0.5,
                bottom=-self.ortho_scale * 0.5,
                top=self.ortho_scale * 0.5,
                near=0.1,
                far=100,
            )
        elif camera_type == "perspective":
            self.camera_proj_mat = get_perspective_projection_matrix(
                49.13,
                self.default_resolution[1] / self.default_resolution[0],
                0.01,
                100.0,
            )
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")

        # Mesh data
        self.vtx_pos = None
        self.pos_idx = None
        self.vtx_uv = None
        self.uv_idx = None
        self.tex = None
        self.mesh_copy = None
        self.scale_factor = 1.0

    def set_default_render_resolution(
        self, default_resolution: Union[int, Tuple[int, int]]
    ):
        """Set default rendering resolution."""
        if isinstance(default_resolution, int):
            default_resolution = (default_resolution, default_resolution)
        self.default_resolution = default_resolution

    def set_default_texture_resolution(self, texture_size: Union[int, Tuple[int, int]]):
        """Set default texture resolution."""
        if isinstance(texture_size, int):
            texture_size = (texture_size, texture_size)
        self.texture_size = texture_size

    def _rasterize(
        self,
        pos_clip: torch.Tensor,
        tri: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        """Rasterize using CUDA rasterizer.

        Args:
            pos_clip: Clip-space positions [N, 4]
            tri: Triangle indices [F, 3]
            resolution: (height, width)

        Returns:
            rast_out: Rasterization output [1, H, W, 4] (barycentric + face_idx)
        """
        from sglang.multimodal_gen.csrc.render.hunyuan3d_rasterizer import rasterize

        if pos_clip.dim() == 2:
            pos_clip = pos_clip.unsqueeze(0)

        findices, barycentric = rasterize(pos_clip, tri, resolution)
        rast_out = torch.cat((barycentric, findices.unsqueeze(-1).float()), dim=-1)
        rast_out = rast_out.unsqueeze(0)
        return rast_out

    def _interpolate(
        self,
        attr: torch.Tensor,
        rast_out: torch.Tensor,
        tri: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate vertex attributes.

        Args:
            attr: Vertex attributes [1, N, C]
            rast_out: Rasterization output [1, H, W, 4]
            tri: Triangle indices [F, 3]

        Returns:
            Interpolated attributes [1, H, W, C]
        """
        from sglang.multimodal_gen.csrc.render.hunyuan3d_rasterizer import interpolate

        barycentric = rast_out[0, ..., :-1]
        findices = rast_out[0, ..., -1].int()

        if attr.dim() == 2:
            attr = attr.unsqueeze(0)

        result = interpolate(attr, findices, barycentric, tri)
        return result

    def load_mesh(
        self,
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        scale_factor: float = 1.15,
        auto_center: bool = True,
    ):
        """Load a mesh for rendering.

        Args:
            mesh: Input trimesh object
            scale_factor: Scale factor for normalization
            auto_center: Whether to auto-center the mesh
        """
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        self.mesh_copy = mesh.copy()

        vtx_pos = mesh.vertices.astype(np.float32)
        pos_idx = mesh.faces.astype(np.int32)

        # Get UV coordinates if available
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            vtx_uv = mesh.visual.uv.astype(np.float32)
            uv_idx = pos_idx.copy()
        else:
            vtx_uv = None
            uv_idx = None

        self.vtx_pos = torch.from_numpy(vtx_pos).to(self.device).float()
        self.pos_idx = torch.from_numpy(pos_idx).to(self.device).to(torch.int32)

        if vtx_uv is not None and uv_idx is not None:
            self.vtx_uv = torch.from_numpy(vtx_uv).to(self.device).float()
            self.uv_idx = torch.from_numpy(uv_idx).to(self.device).to(torch.int32)
        else:
            self.vtx_uv = None
            self.uv_idx = None

        # Coordinate transformation (Y-up to Z-up)
        self.vtx_pos[:, [0, 1]] = -self.vtx_pos[:, [0, 1]]
        self.vtx_pos[:, [1, 2]] = self.vtx_pos[:, [2, 1]]
        if self.vtx_uv is not None:
            self.vtx_uv[:, 1] = 1.0 - self.vtx_uv[:, 1]

        if auto_center:
            max_bb = (self.vtx_pos - 0).max(0)[0]
            min_bb = (self.vtx_pos - 0).min(0)[0]
            center = (max_bb + min_bb) / 2
            scale = torch.norm(self.vtx_pos - center, dim=1).max() * 2.0
            self.vtx_pos = (self.vtx_pos - center) * (scale_factor / float(scale))
            self.scale_factor = scale_factor

    def save_mesh(self) -> trimesh.Trimesh:
        """Save mesh with current texture."""
        texture_data = self.get_texture()
        texture_img = Image.fromarray((texture_data * 255).astype(np.uint8))

        # Get mesh data with inverse transformation
        vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()

        mesh = trimesh.Trimesh(
            vertices=vtx_pos,
            faces=pos_idx,
            process=False,
        )

        if vtx_uv is not None:
            material = trimesh.visual.material.SimpleMaterial(image=texture_img)
            mesh.visual = trimesh.visual.TextureVisuals(uv=vtx_uv, material=material)

        return mesh

    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get mesh data with inverse coordinate transformation."""
        vtx_pos = self.vtx_pos.cpu().numpy().copy()
        pos_idx = self.pos_idx.cpu().numpy()

        # Inverse coordinate transformation
        vtx_pos[:, [1, 2]] = vtx_pos[:, [2, 1]]
        vtx_pos[:, [0, 1]] = -vtx_pos[:, [0, 1]]

        if self.vtx_uv is not None:
            vtx_uv = self.vtx_uv.cpu().numpy().copy()
            vtx_uv[:, 1] = 1.0 - vtx_uv[:, 1]
            uv_idx = self.uv_idx.cpu().numpy()
        else:
            vtx_uv = None
            uv_idx = None

        return vtx_pos, pos_idx, vtx_uv, uv_idx

    def set_texture(self, tex: Union[np.ndarray, torch.Tensor, Image.Image]):
        """Set texture for the mesh."""
        if isinstance(tex, np.ndarray):
            if tex.max() <= 1.0:
                tex = (tex * 255).astype(np.uint8)
            tex = Image.fromarray(tex.astype(np.uint8))
        elif isinstance(tex, torch.Tensor):
            tex_np = tex.cpu().numpy()
            if tex_np.max() <= 1.0:
                tex_np = (tex_np * 255).astype(np.uint8)
            tex = Image.fromarray(tex_np.astype(np.uint8))

        tex = tex.resize(self.texture_size).convert("RGB")
        tex = np.array(tex) / 255.0
        self.tex = torch.from_numpy(tex).to(self.device).float()

    def get_texture(self) -> np.ndarray:
        """Get current texture as numpy array."""
        if self.tex is None:
            return np.ones((*self.texture_size, 3), dtype=np.float32)
        return self.tex.cpu().numpy()

    def _get_pos_from_mvp(
        self,
        elev: float,
        azim: float,
        camera_distance: Optional[float] = None,
        center: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get camera-space and clip-space positions."""
        proj = self.camera_proj_mat
        r_mv = get_mv_matrix(
            elev=elev,
            azim=azim,
            camera_distance=(
                self.camera_distance if camera_distance is None else camera_distance
            ),
            center=center,
        )

        pos_camera = transform_pos(r_mv, self.vtx_pos, keepdim=True)
        pos_clip = transform_pos(proj, pos_camera)

        return pos_camera, pos_clip

    def render_normal(
        self,
        elev: float,
        azim: float,
        camera_distance: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        resolution: Optional[Tuple[int, int]] = None,
        bg_color: List[float] = [1, 1, 1],
        use_abs_coor: bool = False,
        normalize_rgb: bool = True,
        return_type: str = "th",
    ) -> Union[torch.Tensor, np.ndarray, Image.Image]:
        """Render normal map from a viewpoint.

        Args:
            elev: Elevation angle in degrees
            azim: Azimuth angle in degrees
            camera_distance: Camera distance (optional)
            center: Look-at center (optional)
            resolution: Render resolution (optional)
            bg_color: Background color
            use_abs_coor: Use absolute coordinates for normals
            normalize_rgb: Normalize normals to RGB range [0, 1]
            return_type: "th" for tensor, "np" for numpy, "pl" for PIL

        Returns:
            Normal map in specified format
        """
        pos_camera, pos_clip = self._get_pos_from_mvp(
            elev, azim, camera_distance, center
        )

        if resolution is None:
            resolution = self.default_resolution
        if isinstance(resolution, (int, float)):
            resolution = (int(resolution), int(resolution))

        rast_out = self._rasterize(pos_clip, self.pos_idx, resolution)

        # Compute face normals
        if use_abs_coor:
            mesh_triangles = self.vtx_pos[self.pos_idx[:, :3].long(), :]
        else:
            pos_camera_3d = pos_camera[:, :3] / pos_camera[:, 3:4]
            mesh_triangles = pos_camera_3d[self.pos_idx[:, :3].long(), :]

        face_normals = F.normalize(
            torch.cross(
                mesh_triangles[:, 1, :] - mesh_triangles[:, 0, :],
                mesh_triangles[:, 2, :] - mesh_triangles[:, 0, :],
                dim=-1,
            ),
            dim=-1,
        )

        # Compute vertex normals
        vertex_normals = trimesh.geometry.mean_vertex_normals(
            vertex_count=self.vtx_pos.shape[0],
            faces=self.pos_idx.cpu().numpy(),
            face_normals=face_normals.cpu().numpy(),
        )
        vertex_normals = (
            torch.from_numpy(vertex_normals).float().to(self.device).contiguous()
        )

        # Interpolate normals
        normal = self._interpolate(vertex_normals[None, ...], rast_out, self.pos_idx)

        # Apply visibility mask
        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
        bg_tensor = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        normal = normal * visible_mask + bg_tensor * (1 - visible_mask)

        if normalize_rgb:
            normal = (normal + 1) * 0.5

        image = normal[0, ...]

        if return_type == "np":
            image = image.cpu().numpy()
        elif return_type == "pl":
            image = image.cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))

        return image

    def render_position(
        self,
        elev: float,
        azim: float,
        camera_distance: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        resolution: Optional[Tuple[int, int]] = None,
        bg_color: List[float] = [1, 1, 1],
        return_type: str = "th",
    ) -> Union[torch.Tensor, np.ndarray, Image.Image]:
        """Render position map from a viewpoint.

        Args:
            elev: Elevation angle in degrees
            azim: Azimuth angle in degrees
            camera_distance: Camera distance (optional)
            center: Look-at center (optional)
            resolution: Render resolution (optional)
            bg_color: Background color
            return_type: "th" for tensor, "np" for numpy, "pl" for PIL

        Returns:
            Position map in specified format
        """
        pos_camera, pos_clip = self._get_pos_from_mvp(
            elev, azim, camera_distance, center
        )

        if resolution is None:
            resolution = self.default_resolution
        if isinstance(resolution, (int, float)):
            resolution = (int(resolution), int(resolution))

        rast_out = self._rasterize(pos_clip, self.pos_idx, resolution)

        # Position colors (normalized vertex positions)
        tex_position = 0.5 - self.vtx_pos[:, :3] / self.scale_factor
        tex_position = tex_position.contiguous()

        # Interpolate positions
        position = self._interpolate(tex_position[None, ...], rast_out, self.pos_idx)

        # Apply visibility mask
        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
        bg_tensor = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        position = position * visible_mask + bg_tensor * (1 - visible_mask)

        image = position[0, ...]

        if return_type == "np":
            image = image.cpu().numpy()
        elif return_type == "pl":
            image = image.cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))

        return image

    def render_depth(
        self,
        elev: float,
        azim: float,
        camera_distance: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        resolution: Optional[Tuple[int, int]] = None,
        return_type: str = "th",
    ) -> Union[torch.Tensor, np.ndarray, Image.Image]:
        """Render depth map from a viewpoint."""
        pos_camera, pos_clip = self._get_pos_from_mvp(
            elev, azim, camera_distance, center
        )

        if resolution is None:
            resolution = self.default_resolution
        if isinstance(resolution, (int, float)):
            resolution = (int(resolution), int(resolution))

        rast_out = self._rasterize(pos_clip, self.pos_idx, resolution)

        pos_camera_3d = pos_camera[:, :3] / pos_camera[:, 3:4]
        tex_depth = pos_camera_3d[:, 2].reshape(1, -1, 1).contiguous()

        depth = self._interpolate(tex_depth, rast_out, self.pos_idx)

        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
        depth_masked = depth[visible_mask > 0]
        if depth_masked.numel() > 0:
            depth_max, depth_min = depth_masked.max(), depth_masked.min()
            depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        depth = depth * visible_mask

        image = depth[0, ...]

        if return_type == "np":
            image = image.cpu().numpy()
        elif return_type == "pl":
            image = image.squeeze(-1).cpu().numpy() * 255
            image = Image.fromarray(image.astype(np.uint8))

        return image

    def render_normal_multiview(
        self,
        camera_elevs: List[float],
        camera_azims: List[float],
        use_abs_coor: bool = True,
    ) -> List[Image.Image]:
        """Render normal maps from multiple viewpoints."""
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type="pl"
            )
            normal_maps.append(normal_map)
        return normal_maps

    def render_position_multiview(
        self,
        camera_elevs: List[float],
        camera_azims: List[float],
    ) -> List[Image.Image]:
        """Render position maps from multiple viewpoints."""
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render_position(elev, azim, return_type="pl")
            position_maps.append(position_map)
        return position_maps

    def _render_sketch_from_depth(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Render sketch from depth using edge detection."""
        depth_image_np = depth_image.cpu().numpy()
        depth_image_np = (depth_image_np * 255).astype(np.uint8)
        depth_edges = cv2.Canny(depth_image_np, 30, 80)
        sketch_image = (
            torch.from_numpy(depth_edges).to(depth_image.device).float() / 255.0
        )
        sketch_image = sketch_image.unsqueeze(-1)
        return sketch_image

    def back_project(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        elev: float,
        azim: float,
        camera_distance: Optional[float] = None,
        center: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Back-project an image onto mesh UV space."""
        if isinstance(image, Image.Image):
            image = torch.tensor(np.array(image) / 255.0)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image)
        if image.dim() == 2:
            image = image.unsqueeze(-1)
        image = image.float().to(self.device)
        resolution = image.shape[:2]
        channel = image.shape[-1]

        pos_camera, pos_clip = self._get_pos_from_mvp(
            elev, azim, camera_distance, center
        )

        rast_out = self._rasterize(pos_clip, self.pos_idx, resolution)
        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ...]

        # Compute vertex normals for angle-based weighting
        pos_camera_3d = pos_camera[:, :3] / pos_camera[:, 3:4]
        v0 = pos_camera_3d[self.pos_idx[:, 0].long(), :]
        v1 = pos_camera_3d[self.pos_idx[:, 1].long(), :]
        v2 = pos_camera_3d[self.pos_idx[:, 2].long(), :]
        face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)

        vertex_normals = trimesh.geometry.mean_vertex_normals(
            vertex_count=self.vtx_pos.shape[0],
            faces=self.pos_idx.cpu().numpy(),
            face_normals=face_normals.cpu().numpy(),
        )
        vertex_normals = (
            torch.from_numpy(vertex_normals).float().to(self.device).contiguous()
        )

        # Interpolate normals and UVs
        normal = self._interpolate(vertex_normals[None, ...], rast_out, self.pos_idx)
        normal = normal[0, ...]

        if self.vtx_uv is not None:
            uv = self._interpolate(self.vtx_uv[None, ...], rast_out, self.uv_idx)
        else:
            # No UV coordinates
            texture = torch.zeros(
                self.texture_size[1], self.texture_size[0], channel, device=self.device
            )
            cos_map = torch.zeros(
                self.texture_size[1], self.texture_size[0], 1, device=self.device
            )
            boundary_map = torch.zeros_like(cos_map)
            return texture, cos_map, boundary_map

        # Compute depth for sketch
        tex_depth = pos_camera_3d[:, 2].reshape(1, -1, 1).contiguous()
        depth = self._interpolate(tex_depth, rast_out, self.pos_idx)[0, ...]
        depth_masked = depth[visible_mask > 0]
        if depth_masked.numel() > 0:
            depth_max, depth_min = depth_masked.max(), depth_masked.min()
            depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        else:
            depth_normalized = depth
        depth_image = depth_normalized * visible_mask

        sketch_image = self._render_sketch_from_depth(depth_image)

        # Cosine weighting
        lookat = torch.tensor([[0, 0, -1]], device=self.device)
        cos_image = torch.nn.functional.cosine_similarity(lookat, normal.view(-1, 3))
        cos_image = cos_image.view(normal.shape[0], normal.shape[1], 1)

        cos_thres = np.cos(self.bake_angle_thres / 180 * np.pi)
        cos_image[cos_image < cos_thres] = 0

        # Shrink visible mask
        kernel_size = self.bake_unreliable_kernel_size * 2 + 1
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32).to(
            sketch_image.device
        )

        visible_mask_proc = visible_mask.permute(2, 0, 1).unsqueeze(0).float()
        visible_mask_proc = F.conv2d(
            1.0 - visible_mask_proc, kernel, padding=kernel_size // 2
        )
        visible_mask_proc = 1.0 - (visible_mask_proc > 0).float()
        visible_mask_proc = visible_mask_proc.squeeze(0).permute(1, 2, 0)

        sketch_proc = sketch_image.permute(2, 0, 1).unsqueeze(0)
        sketch_proc = F.conv2d(sketch_proc, kernel, padding=kernel_size // 2)
        sketch_proc = (sketch_proc > 0).float()
        sketch_proc = sketch_proc.squeeze(0).permute(1, 2, 0)
        visible_mask_proc = visible_mask_proc * (sketch_proc < 0.5)

        cos_image[visible_mask_proc == 0] = 0

        # Linear baking
        proj_mask = (visible_mask_proc != 0).view(-1)
        uv_flat = uv.squeeze(0).contiguous().view(-1, 2)[proj_mask]
        image_flat = image.squeeze(0).contiguous().view(-1, channel)[proj_mask]
        cos_flat = cos_image.contiguous().view(-1, 1)[proj_mask]
        sketch_flat = sketch_image.contiguous().view(-1, 1)[proj_mask]

        texture = linear_grid_put_2d(
            self.texture_size[1], self.texture_size[0], uv_flat[..., [1, 0]], image_flat
        )
        cos_map = linear_grid_put_2d(
            self.texture_size[1], self.texture_size[0], uv_flat[..., [1, 0]], cos_flat
        )
        boundary_map = linear_grid_put_2d(
            self.texture_size[1],
            self.texture_size[0],
            uv_flat[..., [1, 0]],
            sketch_flat,
        )

        return texture, cos_map, boundary_map

    def bake_from_multiview(
        self,
        views: List[Image.Image],
        camera_elevs: List[float],
        camera_azims: List[float],
        view_weights: List[float],
        method: str = "fast",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bake texture from multiple views."""
        project_textures, project_weighted_cos_maps = [], []
        bake_exp = 4

        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights
        ):
            project_texture, project_cos_map, _ = self.back_project(
                view, camera_elev, camera_azim
            )
            project_cos_map = weight * (project_cos_map**bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)

        if method == "fast":
            texture, ori_trust_map = self.fast_bake_texture(
                project_textures, project_weighted_cos_maps
            )
        else:
            raise ValueError(f"Unknown bake method: {method}")

        return texture, ori_trust_map > 1e-8

    @torch.no_grad()
    def fast_bake_texture(
        self,
        textures: List[torch.Tensor],
        cos_maps: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast texture baking by weighted averaging."""
        channel = textures[0].shape[-1]
        texture_merge = torch.zeros(self.texture_size + (channel,)).to(self.device)
        trust_map_merge = torch.zeros(self.texture_size + (1,)).to(self.device)

        for texture, cos_map in zip(textures, cos_maps):
            view_sum = (cos_map > 0).sum()
            painted_sum = ((cos_map > 0) * (trust_map_merge > 0)).sum()
            if view_sum > 0 and painted_sum / view_sum > 0.99:
                continue
            texture_merge += texture * cos_map
            trust_map_merge += cos_map

        texture_merge = texture_merge / torch.clamp(trust_map_merge, min=1e-8)

        return texture_merge, trust_map_merge > 1e-8

    def texture_inpaint(
        self,
        texture: torch.Tensor,
        mask: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """Inpaint missing regions in UV texture using mesh-aware method.

        Args:
            texture: Texture tensor [H, W, C]
            mask: Mask of valid regions

        Returns:
            Inpainted texture
        """
        if isinstance(texture, torch.Tensor):
            texture_np = texture.cpu().numpy()
        else:
            texture_np = texture

        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

        # Ensure proper format
        if texture_np.max() <= 1.0:
            texture_np = texture_np.astype(np.float32)
        else:
            texture_np = (texture_np / 255.0).astype(np.float32)

        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(-1)
        mask_uint8 = (mask_np.astype(np.float32) * 255).astype(np.uint8)

        # Get mesh data for mesh-aware inpainting
        vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()

        if vtx_uv is not None and uv_idx is not None:
            # Use mesh-aware inpainting
            texture_np, mask_uint8 = meshVerticeInpaint(
                texture_np, mask_uint8, vtx_pos, vtx_uv, pos_idx, uv_idx
            )

        # Final OpenCV inpainting for remaining holes
        texture_uint8 = (texture_np * 255).astype(np.uint8)
        inpaint_mask = 255 - mask_uint8
        texture_inpainted = cv2.inpaint(texture_uint8, inpaint_mask, 3, cv2.INPAINT_NS)

        return torch.from_numpy(texture_inpainted / 255.0).float().to(self.device)

    # Alias for compatibility
    uv_inpaint = texture_inpaint


# =============================================================================
# Hunyuan3D Image Processors
# =============================================================================
# Adapted from https://github.com/Tencent-Hunyuan/Hunyuan3D-2
# Licensed under TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT


def array_to_tensor(np_array):
    """Convert numpy array to normalized tensor."""
    image_pt = torch.tensor(np_array).float()
    image_pt = image_pt / 255 * 2 - 1
    image_pt = rearrange(image_pt, "h w c -> c h w")
    image_pts = repeat(image_pt, "c h w -> b c h w", b=1)
    return image_pts


class ImageProcessorV2:
    """Image processor for Hunyuan3D single-view input."""

    # External module path aliases for compatibility with Hunyuan3D configs
    _aliases = [
        "hy3dshape.preprocessors.ImageProcessorV2",
        "hy3dgen.shapegen.preprocessors.ImageProcessorV2",
    ]

    def __init__(self, size=512, border_ratio=None):
        self.size = size
        self.border_ratio = border_ratio

    @staticmethod
    def recenter(image, border_ratio: float = 0.2):
        """recenter an image to leave some empty space at the image border.

        Args:
            image (ndarray): input image, float/uint8 [H, W, 3/4]
            mask (ndarray): alpha mask, bool [H, W]
            border_ratio (float, optional): border ratio, image will be resized to (1 - border_ratio). Defaults to 0.2.

        Returns:
            ndarray: output image, float/uint8 [H, W, 3/4]
        """

        if image.shape[-1] == 4:
            mask = image[..., 3]
        else:
            mask = np.ones_like(image[..., 0:1]) * 255
            image = np.concatenate([image, mask], axis=-1)
            mask = mask[..., 0]

        height, width, channels = image.shape

        size = max(height, width)
        result = np.zeros((size, size, channels), dtype=np.uint8)

        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        crop_h = x_max - x_min
        crop_w = y_max - y_min
        if crop_h == 0 or crop_w == 0:
            raise ValueError("input image is empty")
        desired_size = int(size * (1 - border_ratio))
        scale = desired_size / max(crop_h, crop_w)
        scaled_h = int(crop_h * scale)
        scaled_w = int(crop_w * scale)
        x2_min = (size - scaled_h) // 2
        x2_max = x2_min + scaled_h

        y2_min = (size - scaled_w) // 2
        y2_max = y2_min + scaled_w

        result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
            image[x_min:x_max, y_min:y_max],
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_AREA,
        )

        bg = np.ones((result.shape[0], result.shape[1], 3), dtype=np.uint8) * 255

        mask = result[..., 3:].astype(np.float32) / 255
        result = result[..., :3] * mask + bg * (1 - mask)

        mask = mask * 255
        result = result.clip(0, 255).astype(np.uint8)
        mask = mask.clip(0, 255).astype(np.uint8)
        return result, mask

    def load_image(self, image, border_ratio=0.15, to_tensor=True):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            image, mask = self.recenter(image, border_ratio=border_ratio)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = image.convert("RGBA")
            image = np.asarray(image)
            image, mask = self.recenter(image, border_ratio=border_ratio)

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = mask[..., np.newaxis]

        if to_tensor:
            image = array_to_tensor(image)
            mask = array_to_tensor(mask)
        return image, mask

    def __call__(self, image, border_ratio=0.15, to_tensor=True, **kwargs):
        if self.border_ratio is not None:
            border_ratio = self.border_ratio
        image, mask = self.load_image(
            image, border_ratio=border_ratio, to_tensor=to_tensor
        )
        outputs = {"image": image, "mask": mask}
        return outputs


class MVImageProcessorV2(ImageProcessorV2):
    """Multi-view image processor for Hunyuan3D.

    View order: front, front clockwise 90, back, front clockwise 270.
    """

    # External module path aliases for compatibility with Hunyuan3D configs
    _aliases = [
        "hy3dshape.preprocessors.MVImageProcessorV2",
    ]

    return_view_idx = True

    def __init__(self, size=512, border_ratio=None):
        super().__init__(size, border_ratio)
        self.view2idx = {"front": 0, "left": 1, "back": 2, "right": 3}

    def __call__(self, image_dict, border_ratio=0.15, to_tensor=True, **kwargs):
        if self.border_ratio is not None:
            border_ratio = self.border_ratio

        images = []
        masks = []
        view_idxs = []
        for view_tag, image in image_dict.items():
            view_idxs.append(self.view2idx[view_tag])
            image, mask = self.load_image(
                image, border_ratio=border_ratio, to_tensor=to_tensor
            )
            images.append(image)
            masks.append(mask)

        zipped_lists = zip(view_idxs, images, masks)
        sorted_zipped_lists = sorted(zipped_lists)
        view_idxs, images, masks = zip(*sorted_zipped_lists)

        image = torch.cat(images, 0).unsqueeze(0)
        mask = torch.cat(masks, 0).unsqueeze(0)
        outputs = {"image": image, "mask": mask, "view_idxs": view_idxs}
        return outputs


IMAGE_PROCESSORS = {
    "v2": ImageProcessorV2,
    "mv_v2": MVImageProcessorV2,
}

DEFAULT_IMAGEPROCESSOR = "v2"


# All tool classes available in this module for resolution
TOOL_CLASSES = (
    ImageProcessorV2,
    MVImageProcessorV2,
)


def resolve_hunyuan3d_tool(target: str):
    """Resolve a Hunyuan3D tool class by target string."""
    # First, try to match against _aliases
    for cls in TOOL_CLASSES:
        aliases = getattr(cls, "_aliases", [])
        if target in aliases:
            return cls

    # Then, try to match against class names
    for cls in TOOL_CLASSES:
        if cls.__name__ == target:
            return cls

    return None


__all__ = [
    # Camera utilities
    "transform_pos",
    "get_mv_matrix",
    "get_orthographic_projection_matrix",
    "get_perspective_projection_matrix",
    # Mesh I/O utilities
    "export_to_trimesh",
    "load_mesh",
    "save_mesh",
    "convert_obj_to_glb",
    "mesh_uv_wrap",
    "remesh_mesh",
    # Mesh inpainting
    "meshVerticeInpaint",
    "meshVerticeInpaint_smooth",
    # Rendering utilities
    "stride_from_shape",
    "scatter_add_nd_with_count",
    "linear_grid_put_2d",
    "MeshRender",
    # Hunyuan3D image processors
    "array_to_tensor",
    "ImageProcessorV2",
    "MVImageProcessorV2",
    "IMAGE_PROCESSORS",
    "DEFAULT_IMAGEPROCESSOR",
    "TOOL_CLASSES",
    "resolve_hunyuan3d_tool",
]
