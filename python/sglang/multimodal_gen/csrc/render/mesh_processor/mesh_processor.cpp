// SPDX-License-Identifier: Apache-2.0
// Adapted from Hunyuan3D-2: https://github.com/Tencent/Hunyuan3D-2
// Original license: TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT

#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

std::pair<py::array_t<float>,
  py::array_t<uint8_t>>  meshVerticeInpaint_smooth(py::array_t<float> texture,
py::array_t<uint8_t> mask,
                 py::array_t<float> vtx_pos, py::array_t<float> vtx_uv, 
                 py::array_t<int> pos_idx, py::array_t<int> uv_idx) {
    auto texture_buf = texture.request();
    auto mask_buf = mask.request();
    auto vtx_pos_buf = vtx_pos.request();
    auto vtx_uv_buf = vtx_uv.request();
    auto pos_idx_buf = pos_idx.request();
    auto uv_idx_buf = uv_idx.request();

    int texture_height = texture_buf.shape[0];
    int texture_width = texture_buf.shape[1];
    int texture_channel = texture_buf.shape[2];
    float* texture_ptr = static_cast<float*>(texture_buf.ptr);
    uint8_t* mask_ptr = static_cast<uint8_t*>(mask_buf.ptr);

    int vtx_num = vtx_pos_buf.shape[0];
    float* vtx_pos_ptr = static_cast<float*>(vtx_pos_buf.ptr);
    float* vtx_uv_ptr = static_cast<float*>(vtx_uv_buf.ptr);
    int* pos_idx_ptr = static_cast<int*>(pos_idx_buf.ptr);
    int* uv_idx_ptr = static_cast<int*>(uv_idx_buf.ptr);

    vector<float> vtx_mask(vtx_num, 0.0f);
    vector<vector<float>> vtx_color(vtx_num, vector<float>(texture_channel, 0.0f));
    vector<int> uncolored_vtxs;

    vector<vector<int>> G(vtx_num);

    for (int i = 0; i < uv_idx_buf.shape[0]; ++i) {
        for (int k = 0; k < 3; ++k) {
            int vtx_uv_idx = uv_idx_ptr[i * 3 + k];
            int vtx_idx = pos_idx_ptr[i * 3 + k];
            int uv_v = round(vtx_uv_ptr[vtx_uv_idx * 2] * (texture_width - 1));
            int uv_u = round((1.0 - vtx_uv_ptr[vtx_uv_idx * 2 + 1]) * (texture_height - 1));

            if (mask_ptr[uv_u * texture_width + uv_v] > 0) {
                vtx_mask[vtx_idx] = 1.0f;
                for (int c = 0; c < texture_channel; ++c) {
                    vtx_color[vtx_idx][c] = texture_ptr[(uv_u * texture_width + uv_v) * texture_channel + c];
                }
            }else{
                uncolored_vtxs.push_back(vtx_idx);
            }

            G[pos_idx_ptr[i * 3 + k]].push_back(pos_idx_ptr[i * 3 + (k + 1) % 3]);
        }
    }

    int smooth_count = 2;
    int last_uncolored_vtx_count = 0;
    while (smooth_count>0) {
        int uncolored_vtx_count = 0;

        for (int vtx_idx : uncolored_vtxs) {

            vector<float> sum_color(texture_channel, 0.0f);
            float total_weight = 0.0f;

            array<float, 3> vtx_0 = {vtx_pos_ptr[vtx_idx * 3],
vtx_pos_ptr[vtx_idx * 3 + 1], vtx_pos_ptr[vtx_idx * 3 + 2]};
            for (int connected_idx : G[vtx_idx]) {
                if (vtx_mask[connected_idx] > 0) {
                    array<float, 3> vtx1 = {vtx_pos_ptr[connected_idx * 3],
                    vtx_pos_ptr[connected_idx * 3 + 1], vtx_pos_ptr[connected_idx * 3 + 2]};
                    float dist_weight = 1.0f / max(sqrt(pow(vtx_0[0] - vtx1[0], 2) + pow(vtx_0[1] - vtx1[1], 2) + \
                     pow(vtx_0[2] - vtx1[2], 2)), 1E-4);
                    dist_weight = dist_weight * dist_weight;
                    for (int c = 0; c < texture_channel; ++c) {
                        sum_color[c] += vtx_color[connected_idx][c] * dist_weight;
                    }
                    total_weight += dist_weight;
                }
            }

            if (total_weight > 0.0f) {
                for (int c = 0; c < texture_channel; ++c) {
                    vtx_color[vtx_idx][c] = sum_color[c] / total_weight;
                }
                vtx_mask[vtx_idx] = 1.0f;
            } else {
                uncolored_vtx_count++;
            }
            
        }

        if(last_uncolored_vtx_count==uncolored_vtx_count){
            smooth_count--;
        }else{
            smooth_count++;
        }
        last_uncolored_vtx_count = uncolored_vtx_count;
    }

    py::array_t<float> new_texture(texture_buf.size);
    py::array_t<uint8_t> new_mask(mask_buf.size);

    auto new_texture_buf = new_texture.request();
    auto new_mask_buf = new_mask.request();

    float* new_texture_ptr = static_cast<float*>(new_texture_buf.ptr);
    uint8_t* new_mask_ptr = static_cast<uint8_t*>(new_mask_buf.ptr);
    std::copy(texture_ptr, texture_ptr + texture_buf.size, new_texture_ptr);
    std::copy(mask_ptr, mask_ptr + mask_buf.size, new_mask_ptr);

    for (int face_idx = 0; face_idx < uv_idx_buf.shape[0]; ++face_idx) {
        for (int k = 0; k < 3; ++k) {
            int vtx_uv_idx = uv_idx_ptr[face_idx * 3 + k];
            int vtx_idx = pos_idx_ptr[face_idx * 3 + k];

            if (vtx_mask[vtx_idx] == 1.0f) {
                int uv_v = round(vtx_uv_ptr[vtx_uv_idx * 2] * (texture_width - 1));
                int uv_u = round((1.0 - vtx_uv_ptr[vtx_uv_idx * 2 + 1]) * (texture_height - 1));

                for (int c = 0; c < texture_channel; ++c) {
                    new_texture_ptr[(uv_u * texture_width + uv_v) * texture_channel + c] = vtx_color[vtx_idx][c];
                }
                new_mask_ptr[uv_u * texture_width + uv_v] = 255;
            }
        }
    }

    new_texture.resize({texture_height, texture_width, 3});
    new_mask.resize({texture_height, texture_width});
  return std::make_pair(new_texture, new_mask);
}


std::pair<py::array_t<float>, py::array_t<uint8_t>> meshVerticeInpaint(py::array_t<float> texture,
          py::array_t<uint8_t> mask,
          py::array_t<float> vtx_pos, py::array_t<float> vtx_uv,
          py::array_t<int> pos_idx, py::array_t<int> uv_idx, const std::string& method = "smooth") {
    if (method == "smooth") {
        return meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx);
    } else {
        throw std::invalid_argument("Invalid method. Use 'smooth'.");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("meshVerticeInpaint", &meshVerticeInpaint, "Mesh-aware texture inpainting",
          py::arg("texture"), py::arg("mask"),
          py::arg("vtx_pos"), py::arg("vtx_uv"),
          py::arg("pos_idx"), py::arg("uv_idx"),
          py::arg("method") = "smooth");
}
