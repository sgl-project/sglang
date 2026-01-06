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

import torch
import numpy as np


class ViewProcessor:
    def __init__(self, config, render):
        self.config = config
        self.render = render

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(elev, azim, use_abs_coor=use_abs_coor, return_type="pl")
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(elev, azim, return_type="pl")
            position_maps.append(position_map)

        return position_maps

    def bake_view_selection(
        self, candidate_camera_elevs, candidate_camera_azims, candidate_view_weights, max_selected_view_num
    ):

        original_resolution = self.render.default_resolution
        self.render.set_default_render_resolution(1024)

        selected_camera_elevs = []
        selected_camera_azims = []
        selected_view_weights = []
        selected_alpha_maps = []
        viewed_tri_idxs = []
        viewed_masks = []

        # 计算每个三角片的面积
        face_areas = self.render.get_face_areas(from_one_index=True)
        total_area = face_areas.sum()
        face_area_ratios = face_areas / total_area

        candidate_view_num = len(candidate_camera_elevs)
        self.render.set_boundary_unreliable_scale(2)

        for elev, azim in zip(candidate_camera_elevs, candidate_camera_azims):
            viewed_tri_idx = self.render.render_alpha(elev, azim, return_type="np")
            viewed_tri_idxs.append(set(np.unique(viewed_tri_idx.flatten())))
            viewed_masks.append(viewed_tri_idx[0, :, :, 0] > 0)

        is_selected = [False for _ in range(candidate_view_num)]
        total_viewed_tri_idxs = set()
        total_viewed_area = 0.0

        for idx in range(6):
            selected_camera_elevs.append(candidate_camera_elevs[idx])
            selected_camera_azims.append(candidate_camera_azims[idx])
            selected_view_weights.append(candidate_view_weights[idx])
            selected_alpha_maps.append(viewed_masks[idx])
            is_selected[idx] = True
            total_viewed_tri_idxs.update(viewed_tri_idxs[idx])

        total_viewed_area = face_area_ratios[list(total_viewed_tri_idxs)].sum()
        for iter in range(max_selected_view_num - len(selected_view_weights)):
            max_inc = 0
            max_idx = -1

            for idx, (elev, azim, weight) in enumerate(
                zip(candidate_camera_elevs, candidate_camera_azims, candidate_view_weights)
            ):
                if is_selected[idx]:
                    continue
                new_tri_idxs = viewed_tri_idxs[idx] - total_viewed_tri_idxs
                new_inc_area = face_area_ratios[list(new_tri_idxs)].sum()

                if new_inc_area > max_inc:
                    max_inc = new_inc_area
                    max_idx = idx

            if max_inc > 0.01:
                is_selected[max_idx] = True
                selected_camera_elevs.append(candidate_camera_elevs[max_idx])
                selected_camera_azims.append(candidate_camera_azims[max_idx])
                selected_view_weights.append(candidate_view_weights[max_idx])
                selected_alpha_maps.append(viewed_masks[max_idx])
                total_viewed_tri_idxs = total_viewed_tri_idxs.union(viewed_tri_idxs[max_idx])
                total_viewed_area += max_inc
            else:
                break

        self.render.set_default_render_resolution(original_resolution)

        return selected_camera_elevs, selected_camera_azims, selected_view_weights

    def bake_from_multiview(self, views, camera_elevs, camera_azims, view_weights):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []

        for view, camera_elev, camera_azim, weight in zip(views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim
            )
            project_cos_map = weight * (project_cos_map**self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)
            texture, ori_trust_map = self.render.fast_bake_texture(project_textures, project_weighted_cos_maps)
        return texture, ori_trust_map > 1e-8

    def texture_inpaint(self, texture, mask, defualt=None):
        if defualt is not None:
            mask = mask.astype(bool)
            inpaint_value = torch.tensor(defualt, dtype=texture.dtype, device=texture.device)
            texture[~mask] = inpaint_value
        else:
            texture_np = self.render.uv_inpaint(texture, mask)
            texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture
