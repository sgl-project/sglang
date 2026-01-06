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

import math

import numpy as np
import torch


def transform_pos(mtx, pos, keepdim=False):
    t_mtx = torch.from_numpy(mtx).to(pos.device) if isinstance(mtx, np.ndarray) else mtx
    if pos.shape[-1] == 3:
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
    else:
        posw = pos

    if keepdim:
        return torch.matmul(posw, t_mtx.t())[...]
    else:
        return torch.matmul(posw, t_mtx.t())[None, ...]


def get_mv_matrix(elev, azim, camera_distance, center=None):
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

    c2w = np.concatenate([np.stack([right, up, -lookat], axis=-1), camera_position[:, None]], axis=-1)

    w2c = np.zeros((4, 4))
    w2c[:3, :3] = np.transpose(c2w[:3, :3], (1, 0))
    w2c[:3, 3:] = -np.matmul(np.transpose(c2w[:3, :3], (1, 0)), c2w[:3, 3:])
    w2c[3, 3] = 1.0

    return w2c.astype(np.float32)


def get_orthographic_projection_matrix(left=-1, right=1, bottom=-1, top=1, near=0, far=2):
    """
    计算正交投影矩阵。

    参数:
        left (float): 投影区域左侧边界。
        right (float): 投影区域右侧边界。
        bottom (float): 投影区域底部边界。
        top (float): 投影区域顶部边界。
        near (float): 投影区域近裁剪面距离。
        far (float): 投影区域远裁剪面距离。

    返回:
        numpy.ndarray: 正交投影矩阵。
    """
    ortho_matrix = np.eye(4, dtype=np.float32)
    ortho_matrix[0, 0] = 2 / (right - left)
    ortho_matrix[1, 1] = 2 / (top - bottom)
    ortho_matrix[2, 2] = -2 / (far - near)
    ortho_matrix[0, 3] = -(right + left) / (right - left)
    ortho_matrix[1, 3] = -(top + bottom) / (top - bottom)
    ortho_matrix[2, 3] = -(far + near) / (far - near)
    return ortho_matrix


def get_perspective_projection_matrix(fovy, aspect_wh, near, far):
    fovy_rad = math.radians(fovy)
    return np.array(
        [
            [1.0 / (math.tan(fovy_rad / 2.0) * aspect_wh), 0, 0, 0],
            [0, 1.0 / math.tan(fovy_rad / 2.0), 0, 0],
            [0, 0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
            [0, 0, -1, 0],
        ]
    ).astype(np.float32)
