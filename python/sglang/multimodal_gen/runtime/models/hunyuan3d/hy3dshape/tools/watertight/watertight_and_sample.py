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

import argparse
import igl
import numpy as np
import os
from scipy.stats import truncnorm
import trimesh

def random_sample_pointcloud(mesh, num = 30000):
    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]
    rng = np.random.default_rng()
    index = rng.choice(num, num, replace=False)
    return points[index], normals[index]

def sharp_sample_pointcloud(mesh, num=16384):
    V = mesh.vertices
    N = mesh.face_normals
    VN = mesh.vertex_normals
    F = mesh.faces
    VN2 = np.ones(V.shape[0])
    for i in range(3):
        dot = np.stack((VN2[F[:,i]], np.sum(VN[F[:,i]] * N, axis=-1)), axis=-1)
        VN2[F[:,i]] = np.min(dot, axis=-1)

    sharp_mask = VN2<0.985
    # collect edge
    edge_a = np.concatenate((F[:,0],F[:,1],F[:,2]))
    edge_b = np.concatenate((F[:,1],F[:,2],F[:,0]))
    sharp_edge = ((sharp_mask[edge_a] * sharp_mask[edge_b]))
    edge_a = edge_a[sharp_edge>0]
    edge_b = edge_b[sharp_edge>0]

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = np.random.rand(num)
    w = np.random.rand(num,1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]
    return samples, normals

def sample_sdf(mesh, random_surface, sharp_surface):
    n_volume_points = sharp_surface.shape[0] * 2
    vol_points = (np.random.rand(n_volume_points, 3) - 0.5) * 2 * 1.05

    a, b = -0.25, 0.25
    mu = 0

    # get near points (add offset on surface points)
    offset1 = truncnorm.rvs((a - mu) / 0.005, (b - mu) / 0.005, loc=mu, scale=0.005, size=(len(random_surface), 3))
    offset2 = truncnorm.rvs((a - mu) / 0.05, (b - mu) / 0.05, loc=mu, scale=0.05,  size=(len(random_surface), 3))
    random_near_points = np.concatenate([
        random_surface + offset1,
        random_surface + offset2
    ], axis=0)

    unit_num = len(sharp_surface) // 6
    sharp_near_points = np.concatenate([
        sharp_surface[:unit_num] + np.random.normal(scale=0.001, size=(unit_num, 3)),
        sharp_surface[unit_num:unit_num*2] + np.random.normal(scale=0.003, size=(unit_num,3)),
        sharp_surface[unit_num*2:unit_num*3] + np.random.normal(scale=0.06, size=(unit_num,3)),
        sharp_surface[unit_num*3:unit_num*4] + np.random.normal(scale=0.01, size=(unit_num,3)),
        sharp_surface[unit_num*4:unit_num*5] + np.random.normal(scale=0.02, size=(unit_num,3)),
        sharp_surface[unit_num*5:] + np.random.normal(scale=0.04, size=(len(sharp_surface)-5*unit_num,3))
    ], axis=0)

    np.random.shuffle(random_near_points)
    np.random.shuffle(sharp_near_points)

    sign_type = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    try:
        vol_sdf, I, C = igl.signed_distance(
            vol_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            return_normals=False,
            sign_type=sign_type)
    except:
        vol_sdf, I, C = igl.signed_distance(
            vol_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            return_normals=False)
    try:
        random_near_sdf, I, C = igl.signed_distance(
            random_near_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            return_normals=False,
            sign_type=sign_type)
    except:
        random_near_sdf, I, C = igl.signed_distance(
            random_near_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            return_normals=False)
    try:
        sharp_near_sdf, I, C = igl.signed_distance(
            sharp_near_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            return_normals=False,
            sign_type=sign_type)
    except:
        sharp_near_sdf, I, C = igl.signed_distance(
            sharp_near_points.astype(np.float32), 
            mesh.vertices, mesh.faces, 
            return_normals=False)
        
    vol_label = -vol_sdf
    random_near_label = -random_near_sdf
    sharp_near_label = -sharp_near_sdf

    data = {
        "vol_points": vol_points.astype(np.float16),
        "vol_label": vol_label.astype(np.float16),
        "random_near_points": random_near_points.astype(np.float16),
        "random_near_label": random_near_label.astype(np.float16),
        "sharp_near_points": sharp_near_points.astype(np.float16),
        "sharp_near_label": sharp_near_label.astype(np.float16)
    }
    return data

def SampleMesh(V, F):
    mesh = trimesh.Trimesh(vertices=V, faces=F)

    area = mesh.area
    sample_num = 499712//4

    random_surface, random_normal = random_sample_pointcloud(mesh, num=sample_num)
    random_sharp_surface, sharp_normal = sharp_sample_pointcloud(mesh, num=sample_num)

    #save_surface
    surface = np.concatenate((random_surface, random_normal), axis = 1).astype(np.float16)
    sharp_surface = np.concatenate((random_sharp_surface, sharp_normal), axis=1).astype(np.float16)

    surface_data = {
        "random_surface": surface,
        "sharp_surface": sharp_surface,
    }

    sdf_data = sample_sdf(mesh, random_surface, random_sharp_surface)
    return surface_data, sdf_data

def normalize_to_unit_box(V):
    """
    Normalize the vertices V to fit inside a unit bounding box [0,1]^3.
    V: (n,3) numpy array of vertex positions.
    Returns: normalized V
    """
    V_min = V.min(axis=0)
    V_max = V.max(axis=0)
    scale = (V_max - V_min).max() * 1.01
    V_normalized = (V - V_min) / scale
    return V_normalized

# Given: V (n x 3 array of vertices), F (m x 3 array of faces)
# Parameters epsilon/grid_res
def Watertight(V, F, epsilon = 2.0/256, grid_res = 256):
    # Compute bounding box
    min_corner = V.min(axis=0)
    max_corner = V.max(axis=0)
    padding = 0.05 * (max_corner - min_corner)
    min_corner -= padding
    max_corner += padding

    # Create a uniform grid
    x = np.linspace(min_corner[0], max_corner[0], grid_res)
    y = np.linspace(min_corner[1], max_corner[1], grid_res)
    z = np.linspace(min_corner[2], max_corner[2], grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Compute SDF at grid points using igl.signed_distance with pseudo normals
    sdf, _, _ = igl.signed_distance(
        grid_points, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )
 
    # igl.marching_cubes returns (vertices, faces)
    mc_verts, mc_faces = igl.marching_cubes(epsilon - np.abs(sdf), grid_points, grid_res, grid_res, grid_res, 0.0)

    # mc_verts: (k x 3) array of vertices of the epsilon contour
    # mc_faces: (l x 3) array of faces of the epsilon contour
    return mc_verts, mc_faces

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an OBJ file and output surface and SDF data.')
    parser.add_argument('--input_obj', type=str, help='Path to the input OBJ file')
    parser.add_argument('--output_prefix', type=str, default=None, 
        help='Base name for output files (default: input OBJ filename without extension)')
    args = parser.parse_args()

    input_obj = args.input_obj
    name = args.output_prefix

    V, F = igl.read_triangle_mesh(input_obj)
    V = normalize_to_unit_box(V)

    mc_verts, mc_faces = Watertight(V, F)
    surface_data, sdf_data = SampleMesh(mc_verts, mc_faces)

    parent_folder = os.path.dirname(args.output_prefix)
    os.makedirs(parent_folder, exist_ok=True)
    export_surface = f'{name}_surface.npz'
    np.savez(export_surface, **surface_data)
    export_sdf = f'{name}_sdf.npz'
    np.savez(export_sdf, **sdf_data)
    igl.write_obj(f'{name}_watertight.obj', mc_verts, mc_faces)
