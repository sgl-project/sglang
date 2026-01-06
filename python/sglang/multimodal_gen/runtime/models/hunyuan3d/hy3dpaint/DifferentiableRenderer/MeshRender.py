# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import cv2
import torch
import trimesh
import numpy as np
from PIL import Image
import torch.nn.functional as F
from typing import Union, Optional, Tuple, List, Any, Callable
from dataclasses import dataclass
from enum import Enum
from .camera_utils import (
    transform_pos,
    get_mv_matrix,
    get_orthographic_projection_matrix,
    get_perspective_projection_matrix,
)

try:
    from .mesh_utils import load_mesh, save_mesh
except:
    print("Bpy IO CAN NOT BE Imported!!!")

try:
    from .mesh_inpaint_processor import meshVerticeInpaint  # , meshVerticeColor
except:
    print("InPaint Function CAN NOT BE Imported!!!")


class RenderMode(Enum):
    """Rendering mode enumeration."""
    NORMAL = "normal"
    POSITION = "position"
    ALPHA = "alpha"
    UV_POS = "uvpos"


class ReturnType(Enum):
    """Return type enumeration."""
    TENSOR = "th"
    NUMPY = "np"
    PIL = "pl"


class TextureType(Enum):
    """Texture type enumeration."""
    DIFFUSE = "diffuse"
    METALLIC_ROUGHNESS = "mr"
    NORMAL = "normal"


@dataclass
class RenderConfig:
    """Unified rendering configuration."""
    elev: float = 0
    azim: float = 0
    camera_distance: Optional[float] = None
    center: Optional[List[float]] = None
    resolution: Optional[Union[int, Tuple[int, int]]] = None
    bg_color: List[float] = None
    return_type: str = "th"
    
    def __post_init__(self):
        if self.bg_color is None:
            self.bg_color = [1, 1, 1]


@dataclass
class ViewState:
    """Camera view state for rendering pipeline."""
    proj_mat: torch.Tensor
    mv_mat: torch.Tensor
    pos_camera: torch.Tensor
    pos_clip: torch.Tensor
    resolution: Tuple[int, int]


def stride_from_shape(shape):
    """
    Calculate stride values from a given shape for multi-dimensional indexing.
    
    Args:
        shape: Tuple or list representing tensor dimensions
        
    Returns:
        List of stride values for each dimension
    """
    stride = [1]
    for x in reversed(shape[1:]):
        stride.append(stride[-1] * x)
    return list(reversed(stride))


def scatter_add_nd_with_count(input, count, indices, values, weights=None):
    """
    Perform scatter-add operation on N-dimensional tensors with counting.
    
    Args:
        input: Input tensor [..., C] with D dimensions + C channels
        count: Count tensor [..., 1] with D dimensions  
        indices: Index tensor [N, D] of type long
        values: Value tensor [N, C] to scatter
        weights: Optional weight tensor [N, C], defaults to ones if None
        
    Returns:
        Tuple of (updated_input, updated_count) tensors
    """
    # input: [..., C], D dimension + C channel
    # count: [..., 1], D dimension
    # indices: [N, D], long
    # values: [N, C]

    D = indices.shape[-1]
    C = input.shape[-1]
    size = input.shape[:-1]
    stride = stride_from_shape(size)

    assert len(size) == D

    input = input.view(-1, C)  # [HW, C]
    count = count.view(-1, 1)

    flatten_indices = (indices * torch.tensor(stride, dtype=torch.long, device=indices.device)).sum(-1)  # [N]

    if weights is None:
        weights = torch.ones_like(values[..., :1])

    input.scatter_add_(0, flatten_indices.unsqueeze(1).repeat(1, C), values)
    count.scatter_add_(0, flatten_indices.unsqueeze(1), weights)

    return input.view(*size, C), count.view(*size, 1)


def linear_grid_put_2d(H, W, coords, values, return_count=False):
    """
    Place values on a 2D grid using bilinear interpolation.
    
    Args:
        H: Grid height
        W: Grid width  
        coords: Coordinate tensor [N, 2] with values in range [0, 1]
        values: Value tensor [N, C] to place on grid
        return_count: Whether to return count information
        
    Returns:
        2D grid tensor [H, W, C] with interpolated values, optionally with count tensor
    """
    # coords: [N, 2], float in [0, 1]
    # values: [N, C]

    C = values.shape[-1]

    indices = coords * torch.tensor([H - 1, W - 1], dtype=torch.float32, device=coords.device)
    indices_00 = indices.floor().long()  # [N, 2]
    indices_00[:, 0].clamp_(0, H - 2)
    indices_00[:, 1].clamp_(0, W - 2)
    indices_01 = indices_00 + torch.tensor([0, 1], dtype=torch.long, device=indices.device)
    indices_10 = indices_00 + torch.tensor([1, 0], dtype=torch.long, device=indices.device)
    indices_11 = indices_00 + torch.tensor([1, 1], dtype=torch.long, device=indices.device)

    h = indices[..., 0] - indices_00[..., 0].float()
    w = indices[..., 1] - indices_00[..., 1].float()
    w_00 = (1 - h) * (1 - w)
    w_01 = (1 - h) * w
    w_10 = h * (1 - w)
    w_11 = h * w

    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)  # [H, W, 1]
    weights = torch.ones_like(values[..., :1])  # [N, 1]

    result, count = scatter_add_nd_with_count(
        result, count, indices_00, values * w_00.unsqueeze(1), weights * w_00.unsqueeze(1)
    )
    result, count = scatter_add_nd_with_count(
        result, count, indices_01, values * w_01.unsqueeze(1), weights * w_01.unsqueeze(1)
    )
    result, count = scatter_add_nd_with_count(
        result, count, indices_10, values * w_10.unsqueeze(1), weights * w_10.unsqueeze(1)
    )
    result, count = scatter_add_nd_with_count(
        result, count, indices_11, values * w_11.unsqueeze(1), weights * w_11.unsqueeze(1)
    )

    if return_count:
        return result, count

    mask = count.squeeze(-1) > 0
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


def mipmap_linear_grid_put_2d(H, W, coords, values, min_resolution=128, return_count=False):
    """
    Place values on 2D grid using mipmap-based multiresolution interpolation to fill holes.
    
    Args:
        H: Grid height
        W: Grid width
        coords: Coordinate tensor [N, 2] with values in range [0, 1] 
        values: Value tensor [N, C] to place on grid
        min_resolution: Minimum resolution for mipmap levels
        return_count: Whether to return count information
        
    Returns:
        2D grid tensor [H, W, C] with filled values, optionally with count tensor
    """
    # coords: [N, 2], float in [0, 1]
    # values: [N, C]

    C = values.shape[-1]

    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)  # [H, W, 1]

    cur_H, cur_W = H, W

    while min(cur_H, cur_W) > min_resolution:

        # try to fill the holes
        mask = count.squeeze(-1) == 0
        if not mask.any():
            break

        cur_result, cur_count = linear_grid_put_2d(cur_H, cur_W, coords, values, return_count=True)
        result[mask] = (
            result[mask]
            + F.interpolate(
                cur_result.permute(2, 0, 1).unsqueeze(0).contiguous(), (H, W), mode="bilinear", align_corners=False
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .contiguous()[mask]
        )
        count[mask] = (
            count[mask]
            + F.interpolate(cur_count.view(1, 1, cur_H, cur_W), (H, W), mode="bilinear", align_corners=False).view(
                H, W, 1
            )[mask]
        )
        cur_H //= 2
        cur_W //= 2

    if return_count:
        return result, count

    mask = count.squeeze(-1) > 0
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


# ============ Core utility functions for reducing duplication ============

def _normalize_image_input(image: Union[np.ndarray, torch.Tensor, Image.Image]) -> Union[np.ndarray, torch.Tensor]:
    """Normalize image input to consistent format."""
    if isinstance(image, Image.Image):
        return np.array(image) / 255.0
    elif isinstance(image, torch.Tensor):
        return image.cpu().numpy() if image.is_cuda else image
    return image


def _convert_texture_format(tex: Union[np.ndarray, torch.Tensor, Image.Image], 
                          texture_size: Tuple[int, int], device: str, force_set: bool = False) -> torch.Tensor:
    """Unified texture format conversion logic."""
    if not force_set:
        if isinstance(tex, np.ndarray):
            tex = Image.fromarray((tex * 255).astype(np.uint8))
        elif isinstance(tex, torch.Tensor):
            tex_np = tex.cpu().numpy()
            tex = Image.fromarray((tex_np * 255).astype(np.uint8))
        
        tex = tex.resize(texture_size).convert("RGB")
        tex = np.array(tex) / 255.0
        return torch.from_numpy(tex).to(device).float()
    else:
        if isinstance(tex, np.ndarray):
            tex = torch.from_numpy(tex)
        return tex.to(device).float()


def _format_output(image: torch.Tensor, return_type: str) -> Union[torch.Tensor, np.ndarray, Image.Image]:
    """Convert output to requested format."""
    if return_type == ReturnType.NUMPY.value:
        return image.cpu().numpy()
    elif return_type == ReturnType.PIL.value:
        img_np = image.cpu().numpy() * 255
        return Image.fromarray(img_np.astype(np.uint8))
    return image


def _ensure_resolution_format(resolution: Optional[Union[int, Tuple[int, int]]], 
                             default: Tuple[int, int]) -> Tuple[int, int]:
    """Ensure resolution is in (height, width) format."""
    if resolution is None:
        return default
    if isinstance(resolution, (int, float)):
        return (int(resolution), int(resolution))
    return tuple(resolution)


def _apply_background_mask(content: torch.Tensor, visible_mask: torch.Tensor, 
                          bg_color: List[float], device: str) -> torch.Tensor:
    """Apply background color to masked regions."""
    bg_tensor = torch.tensor(bg_color, dtype=torch.float32, device=device)
    return content * visible_mask + bg_tensor * (1 - visible_mask)


class MeshRender:
    def __init__(
        self,
        camera_distance=1.45,
        camera_type="orth",
        default_resolution=1024,
        texture_size=1024,
        use_antialias=True,
        max_mip_level=None,
        filter_mode="linear-mipmap-linear",
        bake_mode="back_sample",
        raster_mode="cr",
        shader_type="face",
        use_opengl=False,
        device="cuda",
    ):
        """
        Initialize mesh renderer with configurable parameters.
        
        Args:
            camera_distance: Distance from camera to object center
            camera_type: Type of camera projection ("orth" or "perspective") 
            default_resolution: Default rendering resolution
            texture_size: Size of texture maps
            use_antialias: Whether to use antialiasing
            max_mip_level: Maximum mipmap level for texture filtering
            filter_mode: Texture filtering mode
            bake_mode: Texture baking method ("back_sample", "linear", "mip-map")
            raster_mode: Rasterization backend ("cr" for custom rasterizer)
            shader_type: Shading type ("face" or "vertex")
            use_opengl: Whether to use OpenGL backend (deprecated)
            device: Computing device ("cuda" or "cpu")
        """

        self.device = device

        self.set_default_render_resolution(default_resolution)
        self.set_default_texture_resolution(texture_size)

        self.camera_distance = camera_distance
        self.use_antialias = use_antialias
        self.max_mip_level = max_mip_level
        self.filter_mode = filter_mode
        self.bake_angle_thres = 75
        self.set_boundary_unreliable_scale(2)
        self.bake_mode = bake_mode
        self.shader_type = shader_type

        self.raster_mode = raster_mode
        if self.raster_mode == "cr":
            import custom_rasterizer as cr

            self.raster = cr
        else:
            raise f"No raster named {self.raster_mode}"

        if camera_type == "orth":
            self.set_orth_scale(1.2)
        elif camera_type == "perspective":
            self.camera_proj_mat = get_perspective_projection_matrix(
                49.13, self.default_resolution[1] / self.default_resolution[0], 0.01, 100.0
            )
        else:
            raise f"No camera type {camera_type}"

        # Removed multiprocessing components for single-threaded version

    def _create_view_state(self, config: RenderConfig) -> ViewState:
        """Create unified view state for rendering pipeline."""
        proj = self.camera_proj_mat
        r_mv = get_mv_matrix(
            elev=config.elev,
            azim=config.azim,
            camera_distance=self.camera_distance if config.camera_distance is None else config.camera_distance,
            center=config.center,
        )
        
        pos_camera = transform_pos(r_mv, self.vtx_pos, keepdim=True)
        pos_clip = transform_pos(proj, pos_camera)
        resolution = _ensure_resolution_format(config.resolution, self.default_resolution)
        
        return ViewState(proj, r_mv, pos_camera, pos_clip, resolution)

    def _compute_face_normals(self, triangles: torch.Tensor) -> torch.Tensor:
        """Compute face normals from triangle vertices."""
        return F.normalize(
            torch.cross(
                triangles[:, 1, :] - triangles[:, 0, :],
                triangles[:, 2, :] - triangles[:, 0, :],
                dim=-1,
            ),
            dim=-1,
        )

    def _get_normals_for_shading(self, view_state: ViewState, use_abs_coor: bool = False) -> torch.Tensor:
        """Get normals based on shader type and coordinate system."""
        if use_abs_coor:
            mesh_triangles = self.vtx_pos[self.pos_idx[:, :3], :]
        else:
            pos_camera = view_state.pos_camera[:, :3] / view_state.pos_camera[:, 3:4]
            mesh_triangles = pos_camera[self.pos_idx[:, :3], :]
        
        face_normals = self._compute_face_normals(mesh_triangles)
        
        # Common rasterization
        rast_out, _ = self.raster_rasterize(view_state.pos_clip, self.pos_idx, resolution=view_state.resolution)
        
        if self.shader_type == "vertex":
            vertex_normals = trimesh.geometry.mean_vertex_normals(
                vertex_count=self.vtx_pos.shape[0],
                faces=self.pos_idx.cpu(),
                face_normals=face_normals.cpu(),
            )
            vertex_normals = torch.from_numpy(vertex_normals).float().to(self.device).contiguous()
            normal, _ = self.raster_interpolate(vertex_normals[None, ...], rast_out, self.pos_idx)
        
        elif self.shader_type == "face":
            tri_ids = rast_out[..., 3]
            tri_ids_mask = tri_ids > 0
            tri_ids = ((tri_ids - 1) * tri_ids_mask).long()
            normal = torch.zeros(rast_out.shape[0], rast_out.shape[1], rast_out.shape[2], 3).to(rast_out)
            normal.reshape(-1, 3)[tri_ids_mask.view(-1)] = face_normals.reshape(-1, 3)[tri_ids[tri_ids_mask].view(-1)]
        
        return normal, rast_out

    def _unified_render_pipeline(self, config: RenderConfig, mode: RenderMode, **kwargs) -> torch.Tensor:
        """Unified rendering pipeline for all render modes."""
        view_state = self._create_view_state(config)
        
        if mode == RenderMode.ALPHA:
            rast_out, _ = self.raster_rasterize(view_state.pos_clip, self.pos_idx, resolution=view_state.resolution)
            return rast_out[..., -1:].long()
        
        elif mode == RenderMode.UV_POS:
            return self.uv_feature_map(self.vtx_pos * 0.5 + 0.5)
        
        elif mode == RenderMode.NORMAL:
            use_abs_coor = kwargs.get('use_abs_coor', False)
            normalize_rgb = kwargs.get('normalize_rgb', True)
            
            normal, rast_out = self._get_normals_for_shading(view_state, use_abs_coor)
            visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
            
            result = _apply_background_mask(normal, visible_mask, config.bg_color, self.device)
            
            if normalize_rgb:
                result = (result + 1) * 0.5
            
            if self.use_antialias:
                result = self.raster_antialias(result, rast_out, view_state.pos_clip, self.pos_idx)
            
            return result[0, ...]
        
        elif mode == RenderMode.POSITION:
            rast_out, _ = self.raster_rasterize(view_state.pos_clip, self.pos_idx, resolution=view_state.resolution)
            
            tex_position = 0.5 - self.vtx_pos[:, :3] / self.scale_factor
            tex_position = tex_position.contiguous()
            
            position, _ = self.raster_interpolate(tex_position[None, ...], rast_out, self.pos_idx)
            visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
            
            result = _apply_background_mask(position, visible_mask, config.bg_color, self.device)
            
            if self.use_antialias:
                result = self.raster_antialias(result, rast_out, view_state.pos_clip, self.pos_idx)
            
            return result[0, ...]

    def set_orth_scale(self, ortho_scale):
        """
        Set the orthographic projection scale and update camera projection matrix.
        
        Args:
            ortho_scale: Scale factor for orthographic projection
        """
        self.ortho_scale = ortho_scale
        self.camera_proj_mat = get_orthographic_projection_matrix(
            left=-self.ortho_scale * 0.5,
            right=self.ortho_scale * 0.5,
            bottom=-self.ortho_scale * 0.5,
            top=self.ortho_scale * 0.5,
            near=0.1,
            far=100,
        )

    def raster_rasterize(self, pos, tri, resolution, ranges=None, grad_db=True):
        """
        Rasterize triangular mesh using the configured rasterization backend.
        
        Args:
            pos: Vertex positions in clip space
            tri: Triangle indices
            resolution: Rendering resolution [height, width]
            ranges: Optional rendering ranges (unused in current implementation)
            grad_db: Whether to compute gradients (unused in current implementation)
            
        Returns:
            Tuple of (rasterization_output, gradient_info)
        """

        if self.raster_mode == "cr":
            rast_out_db = None
            if pos.dim() == 2:
                pos = pos.unsqueeze(0)

            # 确保pos是float32类型
            if pos.dtype == torch.float64:
                pos = pos.to(torch.float32)

            # 确保tri是int32类型
            if tri.dtype == torch.int64:
                tri = tri.to(torch.int32)

            findices, barycentric = self.raster.rasterize(pos, tri, resolution)
            rast_out = torch.cat((barycentric, findices.unsqueeze(-1)), dim=-1)
            rast_out = rast_out.unsqueeze(0)
        else:
            raise f"No raster named {self.raster_mode}"

        return rast_out, rast_out_db

    def raster_interpolate(self, uv, rast_out, uv_idx):
        """
        Interpolate texture coordinates or vertex attributes across rasterized triangles.
        
        Args:
            uv: UV coordinates or vertex attributes to interpolate
            rast_out: Rasterization output containing barycentric coordinates
            uv_idx: UV or vertex indices for triangles
            
        Returns:
            Tuple of (interpolated_values, gradient_info)
        """

        if self.raster_mode == "cr":
            textd = None
            barycentric = rast_out[0, ..., :-1]
            findices = rast_out[0, ..., -1]
            if uv.dim() == 2:
                uv = uv.unsqueeze(0)
            textc = self.raster.interpolate(uv, findices, barycentric, uv_idx)
        else:
            raise f"No raster named {self.raster_mode}"

        return textc, textd

    def raster_antialias(self, color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
        """
        Apply antialiasing to rendered colors (currently returns input unchanged).
        
        Args:
            color: Input color values
            rast: Rasterization output
            pos: Vertex positions
            tri: Triangle indices  
            topology_hash: Optional topology hash for optimization
            pos_gradient_boost: Gradient boosting factor
            
        Returns:
            Antialiased color values
        """

        if self.raster_mode == "cr":
            color = color
        else:
            raise f"No raster named {self.raster_mode}"

        return color

    def set_boundary_unreliable_scale(self, scale):
        """
        Set the kernel size for boundary unreliable region detection during texture baking.
        
        Args:
            scale: Scale factor relative to 512 resolution baseline
        """
        self.bake_unreliable_kernel_size = int(
            (scale / 512) * max(self.default_resolution[0], self.default_resolution[1])
        )

    def load_mesh(
        self,
        mesh,
        scale_factor=1.15,
        auto_center=True,
    ):
        """
        Load mesh from file and set up rendering data structures.
        
        Args:
            mesh: Path to mesh file or mesh object
            scale_factor: Scaling factor for mesh normalization
            auto_center: Whether to automatically center the mesh
        """
        vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data = load_mesh(mesh)
        self.set_mesh(
            vtx_pos, pos_idx, vtx_uv=vtx_uv, uv_idx=uv_idx, scale_factor=scale_factor, auto_center=auto_center
        )
        if texture_data is not None:
            self.set_texture(texture_data)

    def save_mesh(self, mesh_path, downsample=False):
        """
        Save current mesh with textures to file.
        
        Args:
            mesh_path: Output file path
            downsample: Whether to downsample textures by half
        """

        vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh(normalize=False)
        texture_data = self.get_texture()
        texture_metallic, texture_roughness = self.get_texture_mr()
        texture_normal = self.get_texture_normal()
        if downsample:
            texture_data = cv2.resize(texture_data, (texture_data.shape[1] // 2, texture_data.shape[0] // 2))
            if texture_metallic is not None:
                texture_metallic = cv2.resize(
                    texture_metallic, (texture_metallic.shape[1] // 2, texture_metallic.shape[0] // 2)
                )
            if texture_roughness is not None:
                texture_roughness = cv2.resize(
                    texture_roughness, (texture_roughness.shape[1] // 2, texture_roughness.shape[0] // 2)
                )
            if texture_normal is not None:
                texture_normal = cv2.resize(
                    texture_normal, (texture_normal.shape[1] // 2, texture_normal.shape[0] // 2)
                )

        save_mesh(
            mesh_path,
            vtx_pos,
            pos_idx,
            vtx_uv,
            uv_idx,
            texture_data,
            metallic=texture_metallic,
            roughness=texture_roughness,
            normal=texture_normal,
        )

    def set_mesh(self, vtx_pos, pos_idx, vtx_uv=None, uv_idx=None, scale_factor=1.15, auto_center=True):
        """
        Set mesh geometry data and perform coordinate transformations.
        
        Args:
            vtx_pos: Vertex positions [N, 3]
            pos_idx: Triangle vertex indices [F, 3]
            vtx_uv: UV coordinates [N, 2], optional
            uv_idx: Triangle UV indices [F, 3], optional  
            scale_factor: Scaling factor for mesh normalization
            auto_center: Whether to automatically center and scale the mesh
        """
        self.vtx_pos = torch.from_numpy(vtx_pos).to(self.device)
        self.pos_idx = torch.from_numpy(pos_idx).to(self.device)

        # 确保顶点位置是float32类型
        if self.vtx_pos.dtype == torch.float64:
            self.vtx_pos = self.vtx_pos.to(torch.float32)

        # 确保索引类型为int32
        if self.pos_idx.dtype == torch.int64:
            self.pos_idx = self.pos_idx.to(torch.int32)

        if (vtx_uv is not None) and (uv_idx is not None):
            self.vtx_uv = torch.from_numpy(vtx_uv).to(self.device)
            self.uv_idx = torch.from_numpy(uv_idx).to(self.device)

            # 确保UV坐标是float32类型
            if self.vtx_uv.dtype == torch.float64:
                self.vtx_uv = self.vtx_uv.to(torch.float32)

            # 确保UV索引类型为int32
            if self.uv_idx.dtype == torch.int64:
                self.uv_idx = self.uv_idx.to(torch.int32)
        else:
            self.vtx_uv = None
            self.uv_idx = None

        self.vtx_pos[:, [0, 1]] = -self.vtx_pos[:, [0, 1]]
        self.vtx_pos[:, [1, 2]] = self.vtx_pos[:, [2, 1]]
        if (vtx_uv is not None) and (uv_idx is not None):
            self.vtx_uv[:, 1] = 1.0 - self.vtx_uv[:, 1]
            pass

        if auto_center:
            max_bb = (self.vtx_pos - 0).max(0)[0]
            min_bb = (self.vtx_pos - 0).min(0)[0]
            center = (max_bb + min_bb) / 2
            scale = torch.norm(self.vtx_pos - center, dim=1).max() * 2.0
            self.vtx_pos = (self.vtx_pos - center) * (scale_factor / float(scale))
            self.scale_factor = scale_factor
            self.mesh_normalize_scale_factor = scale_factor / float(scale)
            self.mesh_normalize_scale_center = center.unsqueeze(0).cpu().numpy()
        else:
            self.scale_factor = 1.0
            self.mesh_normalize_scale_factor = 1.0
            self.mesh_normalize_scale_center = np.array([[0, 0, 0]])

        if uv_idx is not None:
            self.extract_textiles()

    def _set_texture_unified(self, tex: Union[np.ndarray, torch.Tensor, Image.Image], 
                           texture_type: TextureType, force_set: bool = False):
        """Unified texture setting method."""
        converted_tex = _convert_texture_format(tex, self.texture_size, self.device, force_set)
        
        if texture_type == TextureType.DIFFUSE:
            self.tex = converted_tex
        elif texture_type == TextureType.METALLIC_ROUGHNESS:
            self.tex_mr = converted_tex
        elif texture_type == TextureType.NORMAL:
            self.tex_normalMap = converted_tex

    def set_texture(self, tex, force_set=False):
        """Set the main diffuse texture for the mesh."""
        self._set_texture_unified(tex, TextureType.DIFFUSE, force_set)

    def set_texture_mr(self, mr, force_set=False):
        """Set metallic-roughness texture for PBR rendering."""
        self._set_texture_unified(mr, TextureType.METALLIC_ROUGHNESS, force_set)

    def set_texture_normal(self, normal, force_set=False):
        """Set normal map texture for surface detail."""
        self._set_texture_unified(normal, TextureType.NORMAL, force_set)

    def set_default_render_resolution(self, default_resolution):
        """
        Set the default resolution for rendering operations.
        
        Args:
            default_resolution: Resolution as int (square) or tuple (height, width)
        """
        if isinstance(default_resolution, int):
            default_resolution = (default_resolution, default_resolution)
        self.default_resolution = default_resolution

    def set_default_texture_resolution(self, texture_size):
        """
        Set the default texture resolution for UV mapping operations.
        
        Args:
            texture_size: Texture size as int (square) or tuple (height, width)
        """
        if isinstance(texture_size, int):
            texture_size = (texture_size, texture_size)
        self.texture_size = texture_size

    def get_face_num(self):
        """
        Get the number of triangular faces in the mesh.
        
        Returns:
            Number of faces as integer
        """
        return self.pos_idx.shape[0]

    def get_vertex_num(self):
        """
        Get the number of vertices in the mesh.
        
        Returns:
            Number of vertices as integer
        """
        return self.vtx_pos.shape[0]

    def get_face_areas(self, from_one_index=False):
        """
        Calculate the area of each triangular face in the mesh.
        
        Args:
            from_one_index: If True, insert zero at beginning for 1-indexed face IDs
            
        Returns:
            Numpy array of face areas
        """
        v0 = self.vtx_pos[self.pos_idx[:, 0], :]
        v1 = self.vtx_pos[self.pos_idx[:, 1], :]
        v2 = self.vtx_pos[self.pos_idx[:, 2], :]

        # 计算两个边向量
        edge1 = v1 - v0
        edge2 = v2 - v0

        # 计算叉积的模长的一半即为面积
        areas = torch.norm(torch.cross(edge1, edge2, dim=-1), dim=-1) * 0.5

        areas = areas.cpu().numpy()

        if from_one_index:
            # 在数组前面插入一个0,因为三角片索引是从1开始的
            areas = np.insert(areas, 0, 0)

        return areas

    def get_mesh(self, normalize=True):
        """
        Get mesh geometry with optional coordinate denormalization.
        
        Args:
            normalize: Whether to keep normalized coordinates (True) or restore original scale (False)
            
        Returns:
            Tuple of (vertex_positions, face_indices, uv_coordinates, uv_indices)
        """
        vtx_pos = self.vtx_pos.cpu().numpy()
        pos_idx = self.pos_idx.cpu().numpy()
        vtx_uv = self.vtx_uv.cpu().numpy()
        uv_idx = self.uv_idx.cpu().numpy()

        # 坐标变换的逆变换
        if not normalize:
            vtx_pos = vtx_pos / self.mesh_normalize_scale_factor
            vtx_pos = vtx_pos + self.mesh_normalize_scale_center
        vtx_pos[:, [1, 2]] = vtx_pos[:, [2, 1]]
        vtx_pos[:, [0, 1]] = -vtx_pos[:, [0, 1]]

        vtx_uv[:, 1] = 1.0 - vtx_uv[:, 1]
        return vtx_pos, pos_idx, vtx_uv, uv_idx

    def get_texture(self):
        """
        Get the current diffuse texture as numpy array.
        
        Returns:
            Texture as numpy array in range [0, 1]
        """
        return self.tex.cpu().numpy()

    def get_texture_mr(self):
        """
        Get metallic and roughness textures as separate channels.
        
        Returns:
            Tuple of (metallic_texture, roughness_texture) as numpy arrays, or (None, None) if not set
        """
        metallic, roughness = None, None
        if hasattr(self, "tex_mr"):
            mr = self.tex_mr.cpu().numpy()
            metallic = np.repeat(mr[:, :, 0:1], repeats=3, axis=2)
            roughness = np.repeat(mr[:, :, 1:2], repeats=3, axis=2)
        return metallic, roughness

    def get_texture_normal(self):
        """
        Get the normal map texture as numpy array.
        
        Returns:
            Normal map as numpy array, or None if not set
        """
        normal = None
        if hasattr(self, "tex_normalMap"):
            normal = self.tex_normalMap.cpu().numpy()
        return normal

    def to(self, device):
        """
        Move all tensor attributes to the specified device.
        
        Args:
            device: Target device ("cuda", "cpu", etc.)
        """
        self.device = device

        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(self.device))

    def color_rgb_to_srgb(self, image):
        """
        Convert RGB color values to sRGB color space using gamma correction.
        
        Args:
            image: Input image as PIL Image, numpy array, or torch tensor
            
        Returns:
            sRGB corrected image in same format as input
        """
        if isinstance(image, Image.Image):
            image_rgb = torch.tesnor(np.array(image) / 255.0).float().to(self.device)
        elif isinstance(image, np.ndarray):
            image_rgb = torch.tensor(image).float()
        else:
            image_rgb = image.to(self.device)

        image_srgb = torch.where(
            image_rgb <= 0.0031308, 12.92 * image_rgb, 1.055 * torch.pow(image_rgb, 1 / 2.4) - 0.055
        )

        if isinstance(image, Image.Image):
            image_srgb = Image.fromarray((image_srgb.cpu().numpy() * 255).astype(np.uint8))
        elif isinstance(image, np.ndarray):
            image_srgb = image_srgb.cpu().numpy()
        else:
            image_srgb = image_srgb.to(image.device)

        return image_srgb

    def extract_textiles(self):
        """
        Extract texture-space position and normal information by rasterizing 
        the mesh in UV coordinate space. Creates texture-space geometry mappings.
        """

        vnum = self.vtx_uv.shape[0]
        vtx_uv = torch.cat(
            (self.vtx_uv, torch.zeros_like(self.vtx_uv[:, 0:1]), torch.ones_like(self.vtx_uv[:, 0:1])), axis=1
        )
        vtx_uv = vtx_uv.view(1, vnum, 4) * 2 - 1

        rast_out, rast_out_db = self.raster_rasterize(vtx_uv, self.uv_idx, resolution=self.texture_size)
        position, _ = self.raster_interpolate(self.vtx_pos, rast_out, self.pos_idx)

        v0 = self.vtx_pos[self.pos_idx[:, 0], :]
        v1 = self.vtx_pos[self.pos_idx[:, 1], :]
        v2 = self.vtx_pos[self.pos_idx[:, 2], :]
        face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
        vertex_normals = trimesh.geometry.mean_vertex_normals(
            vertex_count=self.vtx_pos.shape[0],
            faces=self.pos_idx.cpu(),
            face_normals=face_normals.cpu(),
        )
        vertex_normals = torch.from_numpy(vertex_normals).to(self.vtx_pos).contiguous()
        position_normal, _ = self.raster_interpolate(vertex_normals[None, ...], rast_out, self.pos_idx)
        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ..., 0]
        position = position[0]
        position_normal = position_normal[0]
        tri_ids = rast_out[0, ..., 3]
        tri_ids_mask = tri_ids > 0
        tri_ids = ((tri_ids - 1) * tri_ids_mask).long()
        position_normal.reshape(-1, 3)[tri_ids_mask.view(-1)] = face_normals.reshape(-1, 3)[
            tri_ids[tri_ids_mask].view(-1)
        ]

        row = torch.arange(position.shape[0]).to(visible_mask.device)
        col = torch.arange(position.shape[1]).to(visible_mask.device)
        grid_i, grid_j = torch.meshgrid(row, col, indexing="ij")

        mask = visible_mask.reshape(-1) > 0
        position = position.reshape(-1, 3)[mask]
        position_normal = position_normal.reshape(-1, 3)[mask]
        position = torch.cat((position, torch.ones_like(position[:, :1])), axis=-1)
        grid = torch.stack((grid_i, grid_j), -1).reshape(-1, 2)[mask]

        texture_indices = (
            torch.ones(self.texture_size[0], self.texture_size[1], device=self.device, dtype=torch.long) * -1
        )
        texture_indices.view(-1)[grid[:, 0] * self.texture_size[1] + grid[:, 1]] = torch.arange(grid.shape[0]).to(
            device=self.device, dtype=torch.long
        )

        self.tex_position = position
        self.tex_normal = position_normal
        self.tex_grid = grid
        self.texture_indices = texture_indices

    def render_normal(self, elev, azim, camera_distance=None, center=None, resolution=None,
                     bg_color=[1, 1, 1], use_abs_coor=False, normalize_rgb=True, return_type="th"):
        """Render surface normals of the mesh from specified viewpoint."""
        config = RenderConfig(elev, azim, camera_distance, center, resolution, bg_color, return_type)
        image = self._unified_render_pipeline(config, RenderMode.NORMAL, 
                                            use_abs_coor=use_abs_coor, normalize_rgb=normalize_rgb)
        return _format_output(image, return_type)

    def convert_normal_map(self, image):
        """
        Convert normal map from standard format to renderer's coordinate system.
        Applies coordinate transformations for proper normal interpretation.
        
        Args:
            image: Input normal map as PIL Image or numpy array
            
        Returns:
            Converted normal map as PIL Image
        """
        # blue is front, red is left, green is top
        if isinstance(image, Image.Image):
            image = np.array(image)
        mask = (image == [255, 255, 255]).all(axis=-1)

        image = (image / 255.0) * 2.0 - 1.0

        image[..., [1]] = -image[..., [1]]
        image[..., [1, 2]] = image[..., [2, 1]]
        image[..., [0]] = -image[..., [0]]

        image = (image + 1.0) * 0.5

        image = (image * 255).astype(np.uint8)
        image[mask] = [127, 127, 255]

        return Image.fromarray(image)

    def render_position(self, elev, azim, camera_distance=None, center=None, resolution=None, 
                       bg_color=[1, 1, 1], return_type="th"):
        """Render world-space positions of visible mesh surface points."""
        config = RenderConfig(elev, azim, camera_distance, center, resolution, bg_color, return_type)
        image = self._unified_render_pipeline(config, RenderMode.POSITION)
        
        if return_type == ReturnType.PIL.value:
            image = image.squeeze(-1).cpu().numpy() * 255
            return Image.fromarray(image.astype(np.uint8))
        return _format_output(image, return_type)

    def render_uvpos(self, return_type="th"):
        """Render vertex positions mapped to UV texture space."""
        config = RenderConfig(return_type=return_type)
        image = self._unified_render_pipeline(config, RenderMode.UV_POS)
        return _format_output(image, return_type)

    def render_alpha(self, elev, azim, camera_distance=None, center=None, resolution=None, return_type="th"):
        """Render binary alpha mask indicating visible mesh regions."""
        config = RenderConfig(elev, azim, camera_distance, center, resolution, return_type=return_type)
        image = self._unified_render_pipeline(config, RenderMode.ALPHA)
        
        if return_type == ReturnType.PIL.value:
            raise Exception("PIL format not supported for alpha rendering")
        return _format_output(image, return_type)

    def uv_feature_map(self, vert_feat, bg=None):
        """
        Map per-vertex features to UV texture space using mesh topology.
        
        Args:
            vert_feat: Per-vertex feature tensor [N, C]
            bg: Background value for unmapped regions (optional)
            
        Returns:
            Feature map in UV texture space [H, W, C]
        """
        vtx_uv = self.vtx_uv * 2 - 1.0
        vtx_uv = torch.cat([vtx_uv, torch.zeros_like(self.vtx_uv)], dim=1).unsqueeze(0)
        vtx_uv[..., -1] = 1
        uv_idx = self.uv_idx
        rast_out, rast_out_db = self.raster_rasterize(vtx_uv, uv_idx, resolution=self.texture_size)
        feat_map, _ = self.raster_interpolate(vert_feat[None, ...], rast_out, uv_idx)
        feat_map = feat_map[0, ...]
        if bg is not None:
            visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ...]
            feat_map[visible_mask == 0] = bg
        return feat_map

    def render_sketch_from_geometry(self, normal_image, depth_image):
        """
        Generate sketch-style edge image from rendered normal and depth maps.
        
        Args:
            normal_image: Rendered normal map tensor
            depth_image: Rendered depth map tensor
            
        Returns:
            Binary edge sketch image as tensor
        """
        normal_image_np = normal_image.cpu().numpy()
        depth_image_np = depth_image.cpu().numpy()

        normal_image_np = (normal_image_np * 255).astype(np.uint8)
        depth_image_np = (depth_image_np * 255).astype(np.uint8)
        normal_image_np = cv2.cvtColor(normal_image_np, cv2.COLOR_RGB2GRAY)

        normal_edges = cv2.Canny(normal_image_np, 80, 150)
        depth_edges = cv2.Canny(depth_image_np, 30, 80)

        combined_edges = np.maximum(normal_edges, depth_edges)

        sketch_image = torch.from_numpy(combined_edges).to(normal_image.device).float() / 255.0
        sketch_image = sketch_image.unsqueeze(-1)

        return sketch_image

    def render_sketch_from_depth(self, depth_image):
        """
        Generate sketch-style edge image from depth map using edge detection.
        
        Args:
            depth_image: Input depth map tensor
            
        Returns:
            Binary edge sketch image as tensor
        """
        depth_image_np = depth_image.cpu().numpy()
        depth_image_np = (depth_image_np * 255).astype(np.uint8)
        depth_edges = cv2.Canny(depth_image_np, 30, 80)
        combined_edges = depth_edges
        sketch_image = torch.from_numpy(combined_edges).to(depth_image.device).float() / 255.0
        sketch_image = sketch_image.unsqueeze(-1)
        return sketch_image

    def back_project(self, image, elev, azim, camera_distance=None, center=None, method=None):
        """
        Back-project a rendered image onto the mesh's UV texture space.
        Handles visibility, viewing angle, and boundary detection for texture baking.
        
        Args:
            image: Input image to back-project (PIL Image, numpy array, or tensor)
            elev: Camera elevation angle in degrees used for rendering
            azim: Camera azimuth angle in degrees used for rendering
            camera_distance: Camera distance (uses default if None)
            center: Camera focus center (uses origin if None) 
            method: Back-projection method ("linear", "mip-map", "back_sample", uses default if None)
            
        Returns:
            Tuple of (texture, cosine_map, boundary_map) tensors in UV space
        """

        if isinstance(image, Image.Image):
            image = torch.tensor(np.array(image) / 255.0)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image)
        if image.dim() == 2:
            image = image.unsqueeze(-1)
        image = image.float().to(self.device)
        resolution = image.shape[:2]
        channel = image.shape[-1]
        texture = torch.zeros(self.texture_size + (channel,)).to(self.device)
        cos_map = torch.zeros(self.texture_size + (1,)).to(self.device)

        proj = self.camera_proj_mat
        r_mv = get_mv_matrix(
            elev=elev,
            azim=azim,
            camera_distance=self.camera_distance if camera_distance is None else camera_distance,
            center=center,
        )
        pos_camera = transform_pos(r_mv, self.vtx_pos, keepdim=True)
        pos_clip = transform_pos(proj, pos_camera)
        pos_camera = pos_camera[:, :3] / pos_camera[:, 3:4]

        v0 = pos_camera[self.pos_idx[:, 0], :]
        v1 = pos_camera[self.pos_idx[:, 1], :]
        v2 = pos_camera[self.pos_idx[:, 2], :]
        face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)

        tex_depth = pos_camera[:, 2].reshape(1, -1, 1).contiguous()
        rast_out, rast_out_db = self.raster_rasterize(pos_clip, self.pos_idx, resolution=resolution)
        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ...]

        if self.shader_type == "vertex":
            vertex_normals = trimesh.geometry.mean_vertex_normals(
                vertex_count=self.vtx_pos.shape[0],
                faces=self.pos_idx.cpu(),
                face_normals=face_normals.cpu(),
            )
            vertex_normals = torch.from_numpy(vertex_normals).float().to(self.device).contiguous()
            normal, _ = self.raster_interpolate(vertex_normals[None, ...], rast_out, self.pos_idx)
        elif self.shader_type == "face":
            tri_ids = rast_out[..., 3]
            tri_ids_mask = tri_ids > 0
            tri_ids = ((tri_ids - 1) * tri_ids_mask).long()
            normal = torch.zeros(rast_out.shape[0], rast_out.shape[1], rast_out.shape[2], 3).to(rast_out)
            normal.reshape(-1, 3)[tri_ids_mask.view(-1)] = face_normals.reshape(-1, 3)[tri_ids[tri_ids_mask].view(-1)]

        normal = normal[0, ...]
        uv, _ = self.raster_interpolate(self.vtx_uv[None, ...], rast_out, self.uv_idx)
        depth, _ = self.raster_interpolate(tex_depth, rast_out, self.pos_idx)
        depth = depth[0, ...]

        depth_max, depth_min = depth[visible_mask > 0].max(), depth[visible_mask > 0].min()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        depth_image = depth_normalized * visible_mask  # Mask out background.

        sketch_image = self.render_sketch_from_depth(depth_image)

        lookat = torch.tensor([[0, 0, -1]], device=self.device)
        cos_image = torch.nn.functional.cosine_similarity(lookat, normal.view(-1, 3))
        cos_image = cos_image.view(normal.shape[0], normal.shape[1], 1)

        cos_thres = np.cos(self.bake_angle_thres / 180 * np.pi)
        cos_image[cos_image < cos_thres] = 0

        # shrink
        if self.bake_unreliable_kernel_size > 0:
            kernel_size = self.bake_unreliable_kernel_size * 2 + 1
            kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32).to(sketch_image.device)

            visible_mask = visible_mask.permute(2, 0, 1).unsqueeze(0).float()
            visible_mask = F.conv2d(1.0 - visible_mask, kernel, padding=kernel_size // 2)
            visible_mask = 1.0 - (visible_mask > 0).float()  # 二值化
            visible_mask = visible_mask.squeeze(0).permute(1, 2, 0)

            sketch_image = sketch_image.permute(2, 0, 1).unsqueeze(0)
            sketch_image = F.conv2d(sketch_image, kernel, padding=kernel_size // 2)
            sketch_image = (sketch_image > 0).float()  # 二值化
            sketch_image = sketch_image.squeeze(0).permute(1, 2, 0)
            visible_mask = visible_mask * (sketch_image < 0.5)

        cos_image[visible_mask == 0] = 0

        method = self.bake_mode if method is None else method

        if method == "linear":
            proj_mask = (visible_mask != 0).view(-1)
            uv = uv.squeeze(0).contiguous().view(-1, 2)[proj_mask]
            image = image.squeeze(0).contiguous().view(-1, channel)[proj_mask]
            cos_image = cos_image.contiguous().view(-1, 1)[proj_mask]
            sketch_image = sketch_image.contiguous().view(-1, 1)[proj_mask]

            texture = linear_grid_put_2d(self.texture_size[1], self.texture_size[0], uv[..., [1, 0]], image)
            cos_map = linear_grid_put_2d(self.texture_size[1], self.texture_size[0], uv[..., [1, 0]], cos_image)
            boundary_map = linear_grid_put_2d(self.texture_size[1], self.texture_size[0], uv[..., [1, 0]], sketch_image)
        elif method == "mip-map":
            proj_mask = (visible_mask != 0).view(-1)
            uv = uv.squeeze(0).contiguous().view(-1, 2)[proj_mask]
            image = image.squeeze(0).contiguous().view(-1, channel)[proj_mask]
            cos_image = cos_image.contiguous().view(-1, 1)[proj_mask]

            texture = mipmap_linear_grid_put_2d(
                self.texture_size[1], self.texture_size[0], uv[..., [1, 0]], image, min_resolution=128
            )
            cos_map = mipmap_linear_grid_put_2d(
                self.texture_size[1], self.texture_size[0], uv[..., [1, 0]], cos_image, min_resolution=256
            )

            if self.vtx_map is not None:
                vertex_normals = vertex_normals[self.vtx_map, :]
            normal_map = self.uv_feature_map(vertex_normals)
            cos_map_uv = torch.nn.functional.cosine_similarity(lookat, normal_map.view(-1, 3))  # .abs()
            cos_map_uv = cos_map_uv.view(1, 1, normal_map.shape[0], normal_map.shape[1])
            cos_map_uv = torch.nn.functional.max_pool2d(cos_map_uv, kernel_size=3, stride=1, padding=1)
            cos_map_uv = cos_map_uv.reshape(self.texture_size[0], self.texture_size[1], 1)
            cos_map_uv[cos_map_uv < cos_thres] = 0
            # cos_map = torch.min(cos_map, cos_map_uv)
            cos_map[cos_map_uv < cos_thres] = 0
        elif method == "back_sample":

            img_proj = torch.from_numpy(
                np.array(((proj[0, 0], 0, 0, 0), (0, proj[1, 1], 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
            ).to(self.tex_position)
            w2c = torch.from_numpy(r_mv).to(self.tex_position)
            v_proj = self.tex_position @ w2c.T @ img_proj
            inner_mask = (v_proj[:, 0] <= 1.0) & (v_proj[:, 0] >= -1.0) & (v_proj[:, 1] <= 1.0) & (v_proj[:, 1] >= -1.0)
            inner_valid_idx = torch.where(inner_mask)[0].long()
            img_x = torch.clamp(
                ((v_proj[:, 0].clamp(-1, 1) * 0.5 + 0.5) * (resolution[0])).long(), 0, resolution[0] - 1
            )
            img_y = torch.clamp(
                ((v_proj[:, 1].clamp(-1, 1) * 0.5 + 0.5) * (resolution[1])).long(), 0, resolution[1] - 1
            )

            indices = img_y * resolution[0] + img_x
            sampled_z = depth.reshape(-1)[indices]
            sampled_m = visible_mask.reshape(-1)[indices]
            v_z = v_proj[:, 2]

            sampled_w = cos_image.reshape(-1)[indices]
            depth_thres = 3e-3

            # valid_idx = torch.where((torch.abs(v_z - sampled_z) < depth_thres) * (sampled_m*sampled_w>0))[0]
            valid_idx = torch.where((torch.abs(v_z - sampled_z) < depth_thres) & (sampled_m * sampled_w > 0))[0]

            intersection_mask = torch.isin(valid_idx, inner_valid_idx)
            valid_idx = valid_idx[intersection_mask].to(inner_valid_idx)

            indices = indices[valid_idx]
            sampled_b = sketch_image.reshape(-1)[indices]
            sampled_w = sampled_w[valid_idx]

            # bilinear sampling rgb
            wx = ((v_proj[:, 0] * 0.5 + 0.5) * resolution[0] - img_x)[valid_idx].reshape(-1, 1)
            wy = ((v_proj[:, 1] * 0.5 + 0.5) * resolution[1] - img_y)[valid_idx].reshape(-1, 1)
            img_x = img_x[valid_idx]
            img_y = img_y[valid_idx]
            img_x_r = torch.clamp(img_x + 1, 0, resolution[0] - 1)
            img_y_r = torch.clamp(img_y + 1, 0, resolution[1] - 1)
            indices_lr = img_y * resolution[0] + img_x_r
            indices_rl = img_y_r * resolution[0] + img_x
            indices_rr = img_y_r * resolution[0] + img_x_r
            rgb = image.reshape(-1, channel)
            sampled_rgb = (rgb[indices] * (1 - wx) + rgb[indices_lr] * wx) * (1 - wy) + (
                rgb[indices_rl] * (1 - wx) + rgb[indices_rr] * wx
            ) * wy

            # return sampled_rgb, sampled_w, sampled_b, valid_idx
            texture = torch.zeros(self.texture_size[0], self.texture_size[1], channel, device=self.device).reshape(
                -1, channel
            )
            cos_map = torch.zeros(self.texture_size[0], self.texture_size[1], 1, device=self.device).reshape(-1)
            boundary_map = torch.zeros(self.texture_size[0], self.texture_size[1], 1, device=self.device).reshape(-1)

            valid_tex_indices = self.tex_grid[valid_idx, 0] * self.texture_size[1] + self.tex_grid[valid_idx, 1]
            texture[valid_tex_indices, :] = sampled_rgb
            cos_map[valid_tex_indices] = sampled_w
            boundary_map[valid_tex_indices] = sampled_b

            texture = texture.view(self.texture_size[0], self.texture_size[1], channel)
            cos_map = cos_map.view(self.texture_size[0], self.texture_size[1], 1)
            # texture = torch.clamp(texture,0,1)

        else:
            raise f"No bake mode {method}"
        return texture, cos_map, boundary_map

    def bake_texture(self, colors, elevs, azims, camera_distance=None, center=None, exp=6, weights=None):
        """
        Bake multiple view images into a single UV texture using weighted blending.
        
        Args:
            colors: List of input images (tensors, numpy arrays, or PIL Images)
            elevs: List of elevation angles for each view
            azims: List of azimuth angles for each view
            camera_distance: Camera distance (uses default if None)
            center: Camera focus center (uses origin if None)
            exp: Exponent for cosine weighting (higher values favor front-facing views)
            weights: Optional per-view weights (defaults to 1.0 for all views)
            
        Returns:
            Tuple of (merged_texture, trust_map) tensors in UV space
        """
        if isinstance(colors, torch.Tensor):
            colors = [colors[i, ...].float().permute(1, 2, 0) for i in range(colors.shape[0])]
        else:
            for i in range(len(colors)):
                if isinstance(colors[i], Image.Image):
                    colors[i] = torch.tensor(np.array(colors[i]) / 255.0, device=self.device).float()
        if weights is None:
            weights = [1.0 for _ in range(len(colors))]
        textures = []
        cos_maps = []
        for color, elev, azim, weight in zip(colors, elevs, azims, weights):
            texture, cos_map, _ = self.back_project(color, elev, azim, camera_distance, center)
            cos_map = weight * (cos_map**exp)
            textures.append(texture)
            cos_maps.append(cos_map)

        texture_merge, trust_map_merge = self.fast_bake_texture(textures, cos_maps)
        return texture_merge, trust_map_merge

    @torch.no_grad()
    def fast_bake_texture(self, textures, cos_maps):
        """
        Efficiently merge multiple textures using cosine-weighted blending.
        Optimizes by skipping views that don't contribute new information.
        
        Args:
            textures: List of texture tensors to merge
            cos_maps: List of corresponding cosine weight maps
            
        Returns:
            Tuple of (merged_texture, valid_mask) tensors
        """

        channel = textures[0].shape[-1]
        texture_merge = torch.zeros(self.texture_size + (channel,)).to(self.device)
        trust_map_merge = torch.zeros(self.texture_size + (1,)).to(self.device)
        for texture, cos_map in zip(textures, cos_maps):
            view_sum = (cos_map > 0).sum()
            painted_sum = ((cos_map > 0) * (trust_map_merge > 0)).sum()
            if painted_sum / view_sum > 0.99:
                continue
            texture_merge += texture * cos_map
            trust_map_merge += cos_map
        texture_merge = texture_merge / torch.clamp(trust_map_merge, min=1e-8)

        return texture_merge, trust_map_merge > 1e-8

    @torch.no_grad()
    def uv_inpaint(self, texture, mask, vertex_inpaint=True, method="NS", return_float=False):
        """
        Inpaint missing regions in UV texture using mesh-aware and traditional methods.
        
        Args:
            texture: Input texture as tensor, numpy array, or PIL Image
            mask: Binary mask indicating regions to inpaint (1 = keep, 0 = inpaint)
            vertex_inpaint: Whether to use mesh vertex connectivity for inpainting
            method: Inpainting method ("NS" for Navier-Stokes)
            return_float: Whether to return float values (False returns uint8)
            
        Returns:
            Inpainted texture as numpy array
        """

        if isinstance(texture, torch.Tensor):
            texture_np = texture.cpu().numpy()
        elif isinstance(texture, np.ndarray):
            texture_np = texture
        elif isinstance(texture, Image.Image):
            texture_np = np.array(texture) / 255.0

        if isinstance(mask, torch.Tensor):
            mask = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        if vertex_inpaint:
            vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()
            texture_np, mask = meshVerticeInpaint(texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)

        if method == "NS":
            texture_np = cv2.inpaint((texture_np * 255).astype(np.uint8), 255 - mask, 3, cv2.INPAINT_NS)
            assert return_float == False

        return texture_np
