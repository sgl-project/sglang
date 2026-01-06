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

import os
import cv2
import bpy
import math
import numpy as np
from io import StringIO
from typing import Optional, Tuple, Dict, Any


def _safe_extract_attribute(obj: Any, attr_path: str, default: Any = None) -> Any:
    """Extract nested attribute safely from object."""
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return default


def _convert_to_numpy(data: Any, dtype: np.dtype) -> Optional[np.ndarray]:
    """Convert data to numpy array with specified dtype, handling None values."""
    if data is None:
        return None
    return np.asarray(data, dtype=dtype)


def load_mesh(mesh):
    """Load mesh data including vertices, faces, UV coordinates and texture."""
    # Extract vertex positions and face indices
    vtx_pos = _safe_extract_attribute(mesh, "vertices")
    pos_idx = _safe_extract_attribute(mesh, "faces")

    # Extract UV coordinates (reusing face indices for UV indices)
    vtx_uv = _safe_extract_attribute(mesh, "visual.uv")
    uv_idx = pos_idx  # Reuse face indices for UV mapping

    # Convert to numpy arrays with appropriate dtypes
    vtx_pos = _convert_to_numpy(vtx_pos, np.float32)
    pos_idx = _convert_to_numpy(pos_idx, np.int32)
    vtx_uv = _convert_to_numpy(vtx_uv, np.float32)
    uv_idx = _convert_to_numpy(uv_idx, np.int32)

    texture_data = None
    return vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data


def _get_base_path_and_name(mesh_path: str) -> Tuple[str, str]:
    """Get base path without extension and mesh name."""
    base_path = os.path.splitext(mesh_path)[0]
    name = os.path.basename(base_path)
    return base_path, name


def _save_texture_map(
    texture: np.ndarray,
    base_path: str,
    suffix: str = "",
    image_format: str = ".jpg",
    color_convert: Optional[int] = None,
) -> str:
    """Save texture map with optional color conversion."""
    path = f"{base_path}{suffix}{image_format}"
    processed_texture = (texture * 255).astype(np.uint8)

    if color_convert is not None:
        processed_texture = cv2.cvtColor(processed_texture, color_convert)
        cv2.imwrite(path, processed_texture)
    else:
        cv2.imwrite(path, processed_texture[..., ::-1])  # RGB to BGR

    return os.path.basename(path)


def _write_mtl_properties(f, properties: Dict[str, Any]):
    """Write material properties to MTL file."""
    for key, value in properties.items():
        if isinstance(value, (list, tuple)):
            f.write(f"{key} {' '.join(map(str, value))}\n")
        else:
            f.write(f"{key} {value}\n")


def _create_obj_content(
    vtx_pos: np.ndarray, vtx_uv: np.ndarray, pos_idx: np.ndarray, uv_idx: np.ndarray, name: str
) -> str:
    """Create OBJ file content."""
    buffer = StringIO()

    # Write header and vertices
    buffer.write(f"mtllib {name}.mtl\no {name}\n")
    np.savetxt(buffer, vtx_pos, fmt="v %.6f %.6f %.6f")
    np.savetxt(buffer, vtx_uv, fmt="vt %.6f %.6f")
    buffer.write("s 0\nusemtl Material\n")

    # Write faces
    pos_idx_plus1 = pos_idx + 1
    uv_idx_plus1 = uv_idx + 1
    face_format = np.frompyfunc(lambda *x: f"{int(x[0])}/{int(x[1])}", 2, 1)
    faces = face_format(pos_idx_plus1, uv_idx_plus1)
    face_strings = [f"f {' '.join(face)}" for face in faces]
    buffer.write("\n".join(face_strings) + "\n")

    return buffer.getvalue()


def save_obj_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture, metallic=None, roughness=None, normal=None):
    """Save mesh as OBJ file with textures and material."""
    # Convert inputs to numpy arrays
    vtx_pos = _convert_to_numpy(vtx_pos, np.float32)
    vtx_uv = _convert_to_numpy(vtx_uv, np.float32)
    pos_idx = _convert_to_numpy(pos_idx, np.int32)
    uv_idx = _convert_to_numpy(uv_idx, np.int32)

    base_path, name = _get_base_path_and_name(mesh_path)

    # Create and save OBJ content
    obj_content = _create_obj_content(vtx_pos, vtx_uv, pos_idx, uv_idx, name)
    with open(mesh_path, "w") as obj_file:
        obj_file.write(obj_content)

    # Save texture maps
    texture_maps = {}
    texture_maps["diffuse"] = _save_texture_map(texture, base_path)

    if metallic is not None:
        texture_maps["metallic"] = _save_texture_map(metallic, base_path, "_metallic", color_convert=cv2.COLOR_RGB2GRAY)
    if roughness is not None:
        texture_maps["roughness"] = _save_texture_map(
            roughness, base_path, "_roughness", color_convert=cv2.COLOR_RGB2GRAY
        )
    if normal is not None:
        texture_maps["normal"] = _save_texture_map(normal, base_path, "_normal")

    # Create MTL file
    _create_mtl_file(base_path, texture_maps, metallic is not None)


def _create_mtl_file(base_path: str, texture_maps: Dict[str, str], is_pbr: bool):
    """Create MTL material file."""
    mtl_path = f"{base_path}.mtl"

    with open(mtl_path, "w") as f:
        f.write("newmtl Material\n")

        if is_pbr:
            # PBR material properties
            properties = {
                "Kd": [0.800, 0.800, 0.800],
                "Ke": [0.000, 0.000, 0.000],  # 鐜鍏夐伄钄�
                "Ni": 1.500,  # 鎶樺皠绯绘暟
                "d": 1.0,  # 閫忔槑搴�
                "illum": 2,  # 鍏夌収妯″瀷
                "map_Kd": texture_maps["diffuse"],
            }
            _write_mtl_properties(f, properties)

            # Additional PBR maps
            map_configs = [("metallic", "map_Pm"), ("roughness", "map_Pr"), ("normal", "map_Bump -bm 1.0")]

            for texture_key, mtl_key in map_configs:
                if texture_key in texture_maps:
                    f.write(f"{mtl_key} {texture_maps[texture_key]}\n")
        else:
            # Standard material properties
            properties = {
                "Ns": 250.000000,
                "Ka": [0.200, 0.200, 0.200],
                "Kd": [0.800, 0.800, 0.800],
                "Ks": [0.500, 0.500, 0.500],
                "Ke": [0.000, 0.000, 0.000],
                "Ni": 1.500,
                "d": 1.0,
                "illum": 3,
                "map_Kd": texture_maps["diffuse"],
            }
            _write_mtl_properties(f, properties)


def save_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture, metallic=None, roughness=None, normal=None):
    """Save mesh using OBJ format."""
    save_obj_mesh(
        mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture, metallic=metallic, roughness=roughness, normal=normal
    )


def _setup_blender_scene():
    """Setup Blender scene for conversion."""
    if "convert" not in bpy.data.scenes:
        bpy.data.scenes.new("convert")
    bpy.context.window.scene = bpy.data.scenes["convert"]


def _clear_scene_objects():
    """Clear all objects from current Blender scene."""
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
        bpy.data.objects.remove(obj, do_unlink=True)


def _select_mesh_objects():
    """Select all mesh objects in scene."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)


def _merge_vertices_if_needed(merge_vertices: bool):
    """Merge duplicate vertices if requested."""
    if not merge_vertices:
        return

    for obj in bpy.context.selected_objects:
        if obj.type == "MESH":
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.remove_doubles()
            bpy.ops.object.mode_set(mode="OBJECT")


def _apply_shading(shade_type: str, auto_smooth_angle: float):
    """Apply shading to selected objects."""
    shading_ops = {
        "SMOOTH": lambda: bpy.ops.object.shade_smooth(),
        "FLAT": lambda: bpy.ops.object.shade_flat(),
        "AUTO_SMOOTH": lambda: _apply_auto_smooth(auto_smooth_angle),
    }

    if shade_type in shading_ops:
        shading_ops[shade_type]()


def _apply_auto_smooth(auto_smooth_angle: float):
    """Apply auto smooth based on Blender version."""
    angle_rad = math.radians(auto_smooth_angle)

    if bpy.app.version < (4, 1, 0):
        bpy.ops.object.shade_smooth(use_auto_smooth=True, auto_smooth_angle=angle_rad)
    elif bpy.app.version < (4, 2, 0):
        bpy.ops.object.shade_smooth_by_angle(angle=angle_rad)
    else:
        bpy.ops.object.shade_auto_smooth(angle=angle_rad)


def convert_obj_to_glb(
    obj_path: str,
    glb_path: str,
    shade_type: str = "SMOOTH",
    auto_smooth_angle: float = 60,
    merge_vertices: bool = False,
) -> bool:
    """Convert OBJ file to GLB format using Blender."""
    try:
        _setup_blender_scene()
        _clear_scene_objects()

        # Import OBJ file
        bpy.ops.wm.obj_import(filepath=obj_path)
        _select_mesh_objects()

        # Process meshes
        _merge_vertices_if_needed(merge_vertices)
        _apply_shading(shade_type, auto_smooth_angle)

        # Export to GLB
        bpy.ops.export_scene.gltf(filepath=glb_path, use_active_scene=True)
        return True
    except Exception:
        return False
