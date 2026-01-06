import numpy as np

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    try:
        from sklearn.neighbors import NearestNeighbors as SKNearest
        KDTree = None
    except Exception:
        raise ImportError("Requires scipy.spatial.cKDTree or sklearn.neighbors. Install scipy or scikit-learn.")

def normalize_point_cloud_dimension(points):
    """
    将点云数据按维度独立归一化到[-1, 1]范围。
    
    参数:
        points (np.ndarray): 输入点云，形状为(n, 3)。
        
    返回:
        np.ndarray: 归一化后的点云。
        tuple: 每个维度的最小值，用于反归一化。
        tuple: 每个维度的最大值，用于反归一化。
    """
    # 计算每个维度（列）的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    
    # 计算每个维度的范围，避免除以零
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1e-8  # 如果某个维度值全相同，则范围设为一个小值
    
    normalized_points = (points - min_vals) / ranges  # 先归一化到[0,1]
    normalized_points = normalized_points * 2 - 1    # 再映射到[-1,1]
    
    return normalized_points, min_vals, max_vals

def sample_points_from_mesh(vertices: np.ndarray, faces: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Uniformly sample points on mesh surface.

    vertices: (n,3) array or flattened (n*3,)
    faces: (f,3) array of indices or flattened (f*3,)
    n_samples: number of points to sample

    Returns: (n_samples, 3) sampled points (float32)
    """
    v = np.asarray(vertices).reshape(-1, 3).astype(np.float64)
    v, _, _ = normalize_point_cloud_dimension(v)
    f = np.asarray(faces).reshape(-1, 3).astype(np.int64)

    v0 = v[f[:, 0], :]
    v1 = v[f[:, 1], :]
    v2 = v[f[:, 2], :]

    # triangle areas
    tri_edges = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(tri_edges, axis=1)
    area_sum = tri_areas.sum()
    if area_sum == 0:
        # Degenerate mesh: return repeated vertices
        idx = np.random.randint(0, v.shape[0], size=n_samples)
        return v[idx].astype(np.float32)

    # probabilities
    probs = tri_areas / area_sum

    # sample triangle indices according to area
    tri_indices = np.random.choice(len(f), size=n_samples, p=probs)

    # sample barycentric coordinates
    r1 = np.sqrt(np.random.rand(n_samples))
    r2 = np.random.rand(n_samples)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2

    pts = (a[:, None] * v0[tri_indices] +
           b[:, None] * v1[tri_indices] +
           c[:, None] * v2[tri_indices])

    return pts.astype(np.float32)


def _nn_distances(a_pts: np.ndarray, b_pts: np.ndarray):
    """
    Compute nearest-neighbor Euclidean distances from each point in a_pts to nearest in b_pts.

    Returns distances (not squared).
    """
    if a_pts.shape[0] == 0:
        return np.array([], dtype=np.float32)
    if b_pts.shape[0] == 0:
        # return inf
        return np.full((a_pts.shape[0],), np.inf, dtype=np.float32)

    if KDTree is not None:
        tree = KDTree(b_pts)
        dists, _ = tree.query(a_pts, k=1)
        return dists.astype(np.float32)
    else:
        # fallback to sklearn
        nbrs = SKNearest(n_neighbors=1, algorithm='auto').fit(b_pts)
        dists, _ = nbrs.kneighbors(a_pts)
        return dists[:, 0].astype(np.float32)


def chamfer_distance_from_meshes(pred_vertices: np.ndarray,
                                 pred_faces: np.ndarray,
                                 gt_vertices: np.ndarray,
                                 gt_faces: np.ndarray,
                                 n_samples: int = 100000,
                                 return_raw: bool = False):
    """
    Compute Chamfer distance between predicted mesh and ground-truth mesh.

    pred_vertices/pred_faces: mesh A (prediction)
    gt_vertices/gt_faces: mesh B (ground truth)
    n_samples: number of samples per mesh (default 100k). Lower for speed, e.g. 10k.
    return_raw: if True, also return the sampled point clouds and per-point distances.

    Returns:
      If return_raw is False:
        dict with keys:
          'cd_l2_sq'      : bidirectional mean squared L2 (mean of squared distances)
          'cd_l2'         : bidirectional mean L2 (mean of distances)
          'A_to_B_l2_sq'  : mean squared distances from A->B
          'B_to_A_l2_sq'  : mean squared distances from B->A
          'A_to_B_l2'     : mean distances A->B
          'B_to_A_l2'     : mean distances B->A
      If return_raw is True:
        (metrics_dict, pts_pred, pts_gt, dists_pred_to_gt, dists_gt_to_pred)
    """
    pts_pred = sample_points_from_mesh(pred_vertices, pred_faces, n_samples)
    pts_gt = sample_points_from_mesh(gt_vertices, gt_faces, n_samples)

    d_pred_to_gt = _nn_distances(pts_pred, pts_gt)  # distances from pred samples to nearest gt
    d_gt_to_pred = _nn_distances(pts_gt, pts_pred)  # distances from gt samples to nearest pred

    # L2 (distances) and L2^2 (squared)
    A_to_B_l2 = float(np.mean(d_pred_to_gt))
    B_to_A_l2 = float(np.mean(d_gt_to_pred))
    cd_l2 = 0.5 * (A_to_B_l2 + B_to_A_l2)

    metrics = {
        'cd_l2': cd_l2,
        'A_to_B_l2': A_to_B_l2,
        'B_to_A_l2': B_to_A_l2,
        'n_samples_per_mesh': n_samples,
    }

    if return_raw:
        return metrics, pts_pred, pts_gt, d_pred_to_gt, d_gt_to_pred
    return metrics


if __name__ == "__main__":
    # Quick example using a simple triangle meshes (triangles)
    # Pred: unit right triangle in XY plane
    pred_verts = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [0, 1, 0]], dtype=np.float32)
    pred_faces = np.array([[0, 1, 2]], dtype=np.int32)

    # GT: slightly translated triangle
    gt_verts = pred_verts + np.array([0.00, 0.00, 0.5], dtype=np.float32)
    gt_faces = pred_faces.copy()

    metrics = chamfer_distance_from_meshes(pred_verts, pred_faces, gt_verts, gt_faces,
                                           n_samples=100000)
    print("Chamfer metrics:", metrics)
