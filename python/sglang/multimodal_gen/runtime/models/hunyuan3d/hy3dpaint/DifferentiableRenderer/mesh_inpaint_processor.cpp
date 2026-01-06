#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>
#include <functional>

namespace py = pybind11;
using namespace std;

namespace {
// 内部数据结构，避免重复的buffer获取和指针设置
struct MeshData {
    int texture_height, texture_width, texture_channel;
    int vtx_num;
    float* texture_ptr;
    uint8_t* mask_ptr;
    float* vtx_pos_ptr;
    float* vtx_uv_ptr;
    int* pos_idx_ptr;
    int* uv_idx_ptr;
    
    // 存储buffer以防止被销毁
    py::buffer_info texture_buf, mask_buf, vtx_pos_buf, vtx_uv_buf, pos_idx_buf, uv_idx_buf;
    
    MeshData(py::array_t<float>& texture, py::array_t<uint8_t>& mask, 
             py::array_t<float>& vtx_pos, py::array_t<float>& vtx_uv,
             py::array_t<int>& pos_idx, py::array_t<int>& uv_idx) {
        
        texture_buf = texture.request();
        mask_buf = mask.request();
        vtx_pos_buf = vtx_pos.request();
        vtx_uv_buf = vtx_uv.request();
        pos_idx_buf = pos_idx.request();
        uv_idx_buf = uv_idx.request();

        texture_height = texture_buf.shape[0];
        texture_width = texture_buf.shape[1];
        texture_channel = texture_buf.shape[2];
        texture_ptr = static_cast<float*>(texture_buf.ptr);
        mask_ptr = static_cast<uint8_t*>(mask_buf.ptr);

        vtx_num = vtx_pos_buf.shape[0];
        vtx_pos_ptr = static_cast<float*>(vtx_pos_buf.ptr);
        vtx_uv_ptr = static_cast<float*>(vtx_uv_buf.ptr);
        pos_idx_ptr = static_cast<int*>(pos_idx_buf.ptr);
        uv_idx_ptr = static_cast<int*>(uv_idx_buf.ptr);
    }
};

// 公共函数：计算UV坐标
pair<int, int> calculateUVCoordinates(int vtx_uv_idx, const MeshData& data) {
    int uv_v = round(data.vtx_uv_ptr[vtx_uv_idx * 2] * (data.texture_width - 1));
    int uv_u = round((1.0 - data.vtx_uv_ptr[vtx_uv_idx * 2 + 1]) * (data.texture_height - 1));
    return make_pair(uv_u, uv_v);
}

// 公共函数：计算距离权重
float calculateDistanceWeight(const array<float, 3>& vtx_0, const array<float, 3>& vtx1) {
    float dist_weight = 1.0f / max(
        sqrt(
            pow(vtx_0[0] - vtx1[0], 2) + 
            pow(vtx_0[1] - vtx1[1], 2) + 
            pow(vtx_0[2] - vtx1[2], 2)
        ), 1E-4);
    return dist_weight * dist_weight;
}

// 公共函数：获取顶点位置
array<float, 3> getVertexPosition(int vtx_idx, const MeshData& data) {
    return {data.vtx_pos_ptr[vtx_idx * 3], 
            data.vtx_pos_ptr[vtx_idx * 3 + 1], 
            data.vtx_pos_ptr[vtx_idx * 3 + 2]};
}

// 公共函数：构建图结构
void buildGraph(vector<vector<int>>& G, const MeshData& data) {
    G.resize(data.vtx_num);
    for(int i = 0; i < data.uv_idx_buf.shape[0]; ++i) {
        for(int k = 0; k < 3; ++k) {
            G[data.pos_idx_ptr[i * 3 + k]].push_back(data.pos_idx_ptr[i * 3 + (k + 1) % 3]);
        }
    }
}

// 通用初始化函数：处理两种掩码类型（float和int）
template<typename MaskType>
void initializeVertexDataGeneric(const MeshData& data, vector<MaskType>& vtx_mask, 
                                vector<vector<float>>& vtx_color, vector<int>* uncolored_vtxs = nullptr,
                                MaskType mask_value = static_cast<MaskType>(1)) {
    vtx_mask.assign(data.vtx_num, static_cast<MaskType>(0));
    vtx_color.assign(data.vtx_num, vector<float>(data.texture_channel, 0.0f));
    
    if(uncolored_vtxs) {
        uncolored_vtxs->clear();
    }

    for(int i = 0; i < data.uv_idx_buf.shape[0]; ++i) {
        for(int k = 0; k < 3; ++k) {
            int vtx_uv_idx = data.uv_idx_ptr[i * 3 + k];
            int vtx_idx = data.pos_idx_ptr[i * 3 + k];
            auto uv_coords = calculateUVCoordinates(vtx_uv_idx, data);

            if(data.mask_ptr[uv_coords.first * data.texture_width + uv_coords.second] > 0) {
                vtx_mask[vtx_idx] = mask_value;
                for(int c = 0; c < data.texture_channel; ++c) {
                    vtx_color[vtx_idx][c] = data.texture_ptr[(uv_coords.first * data.texture_width + 
                                                            uv_coords.second) * data.texture_channel + c];
                }
            } else if(uncolored_vtxs) {
                uncolored_vtxs->push_back(vtx_idx);
            }
        }
    }
}

// 通用平滑算法：支持不同的掩码类型和检查函数
template<typename MaskType>
void performSmoothingAlgorithm(const MeshData& data, const vector<vector<int>>& G,
                              vector<MaskType>& vtx_mask, vector<vector<float>>& vtx_color, 
                              const vector<int>& uncolored_vtxs,
                              function<bool(MaskType)> is_colored_func,
                              function<void(MaskType&)> set_colored_func) {
    int smooth_count = 2;
    int last_uncolored_vtx_count = 0;
    
    while(smooth_count > 0) {
        int uncolored_vtx_count = 0;

        for(int vtx_idx : uncolored_vtxs) {
            vector<float> sum_color(data.texture_channel, 0.0f);
            float total_weight = 0.0f;

            array<float, 3> vtx_0 = getVertexPosition(vtx_idx, data);
            
            for(int connected_idx : G[vtx_idx]) {
                if(is_colored_func(vtx_mask[connected_idx])) {
                    array<float, 3> vtx1 = getVertexPosition(connected_idx, data);
                    float dist_weight = calculateDistanceWeight(vtx_0, vtx1);
                    
                    for(int c = 0; c < data.texture_channel; ++c) {
                        sum_color[c] += vtx_color[connected_idx][c] * dist_weight;
                    }
                    total_weight += dist_weight;
                }
            }

            if(total_weight > 0.0f) {
                for(int c = 0; c < data.texture_channel; ++c) {
                    vtx_color[vtx_idx][c] = sum_color[c] / total_weight;
                }
                set_colored_func(vtx_mask[vtx_idx]);
            } else {
                uncolored_vtx_count++;
            }
        }

        if(last_uncolored_vtx_count == uncolored_vtx_count) {
            smooth_count--;
        } else {
            smooth_count++;
        }
        last_uncolored_vtx_count = uncolored_vtx_count;
    }
}

// 前向传播算法的通用实现
void performForwardPropagation(const MeshData& data, const vector<vector<int>>& G,
                              vector<float>& vtx_mask, vector<vector<float>>& vtx_color,
                              queue<int>& active_vtxs) {
    while(!active_vtxs.empty()) {
        queue<int> pending_active_vtxs;
        
        while(!active_vtxs.empty()) {
            int vtx_idx = active_vtxs.front();
            active_vtxs.pop();
            array<float, 3> vtx_0 = getVertexPosition(vtx_idx, data);
            
            for(int connected_idx : G[vtx_idx]) {
                if(vtx_mask[connected_idx] > 0) continue;
                
                array<float, 3> vtx1 = getVertexPosition(connected_idx, data);
                float dist_weight = calculateDistanceWeight(vtx_0, vtx1);
                
                for(int c = 0; c < data.texture_channel; ++c) {
                    vtx_color[connected_idx][c] += vtx_color[vtx_idx][c] * dist_weight;
                }
                
                if(vtx_mask[connected_idx] == 0) {
                    pending_active_vtxs.push(connected_idx);
                }
                vtx_mask[connected_idx] -= dist_weight;
            }
        }

        while(!pending_active_vtxs.empty()) {
            int vtx_idx = pending_active_vtxs.front();
            pending_active_vtxs.pop();
            
            for(int c = 0; c < data.texture_channel; ++c) {
                vtx_color[vtx_idx][c] /= -vtx_mask[vtx_idx];
            }
            vtx_mask[vtx_idx] = 1.0f;
            active_vtxs.push(vtx_idx);
        }
    }
}

// 公共函数：创建输出数组
pair<py::array_t<float>, py::array_t<uint8_t>> createOutputArrays(
    const MeshData& data, const vector<float>& vtx_mask, 
    const vector<vector<float>>& vtx_color) {
    
    py::array_t<float> new_texture(data.texture_buf.size);
    py::array_t<uint8_t> new_mask(data.mask_buf.size);

    auto new_texture_buf = new_texture.request();
    auto new_mask_buf = new_mask.request();

    float* new_texture_ptr = static_cast<float*>(new_texture_buf.ptr);
    uint8_t* new_mask_ptr = static_cast<uint8_t*>(new_mask_buf.ptr);
    
    // Copy original texture and mask to new arrays
    copy(data.texture_ptr, data.texture_ptr + data.texture_buf.size, new_texture_ptr);
    copy(data.mask_ptr, data.mask_ptr + data.mask_buf.size, new_mask_ptr);

    for(int face_idx = 0; face_idx < data.uv_idx_buf.shape[0]; ++face_idx) {
        for(int k = 0; k < 3; ++k) {
            int vtx_uv_idx = data.uv_idx_ptr[face_idx * 3 + k];
            int vtx_idx = data.pos_idx_ptr[face_idx * 3 + k];

            if(vtx_mask[vtx_idx] == 1.0f) {
                auto uv_coords = calculateUVCoordinates(vtx_uv_idx, data);
                
                for(int c = 0; c < data.texture_channel; ++c) {
                    new_texture_ptr[
                        (uv_coords.first * data.texture_width + uv_coords.second) * 
                        data.texture_channel + c
                    ] = vtx_color[vtx_idx][c];
                }
                new_mask_ptr[uv_coords.first * data.texture_width + uv_coords.second] = 255;
            }
        }
    }

    // Reshape the new arrays to match the original texture and mask shapes
    new_texture.resize({data.texture_height, data.texture_width, 3});
    new_mask.resize({data.texture_height, data.texture_width});

    return make_pair(new_texture, new_mask);
}

// 创建顶点颜色输出数组的专用函数
pair<py::array_t<float>, py::array_t<uint8_t>> createVertexColorOutput(
    const MeshData& data, const vector<int>& vtx_mask, 
    const vector<vector<float>>& vtx_color) {
    
    py::array_t<float> py_vtx_color({data.vtx_num, data.texture_channel});
    py::array_t<uint8_t> py_vtx_mask({data.vtx_num});

    auto py_vtx_color_buf = py_vtx_color.request();
    auto py_vtx_mask_buf = py_vtx_mask.request();

    float* py_vtx_color_ptr = static_cast<float*>(py_vtx_color_buf.ptr);
    uint8_t* py_vtx_mask_ptr = static_cast<uint8_t*>(py_vtx_mask_buf.ptr);

    for(int i = 0; i < data.vtx_num; ++i) {
        py_vtx_mask_ptr[i] = vtx_mask[i];
        for(int c = 0; c < data.texture_channel; ++c) {
            py_vtx_color_ptr[i * data.texture_channel + c] = vtx_color[i][c];
        }
    }

    return make_pair(py_vtx_color, py_vtx_mask);
}

} // anonymous namespace

// 重构后的 meshVerticeInpaint_smooth 函数
pair<py::array_t<float>, py::array_t<uint8_t>> meshVerticeInpaint_smooth(
    py::array_t<float> texture, py::array_t<uint8_t> mask, py::array_t<float> vtx_pos, py::array_t<float> vtx_uv,
    py::array_t<int> pos_idx, py::array_t<int> uv_idx) {
    
    MeshData data(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx);
    
    vector<float> vtx_mask;
    vector<vector<float>> vtx_color;
    vector<int> uncolored_vtxs;
    vector<vector<int>> G;

    initializeVertexDataGeneric(data, vtx_mask, vtx_color, &uncolored_vtxs, 1.0f);
    buildGraph(G, data);

    // 使用通用平滑算法
    performSmoothingAlgorithm<float>(data, G, vtx_mask, vtx_color, uncolored_vtxs,
        [](float mask_val) { return mask_val > 0; },  // 检查是否着色
        [](float& mask_val) { mask_val = 1.0f; }      // 设置为已着色
    );

    return createOutputArrays(data, vtx_mask, vtx_color);
}

// 重构后的 meshVerticeInpaint_forward 函数
pair<py::array_t<float>, py::array_t<uint8_t>> meshVerticeInpaint_forward(
    py::array_t<float> texture, py::array_t<uint8_t> mask, py::array_t<float> vtx_pos, py::array_t<float> vtx_uv,
    py::array_t<int> pos_idx, py::array_t<int> uv_idx) {
    
    MeshData data(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx);
    
    vector<float> vtx_mask;
    vector<vector<float>> vtx_color;
    vector<vector<int>> G;
    queue<int> active_vtxs;

    // 使用通用初始化（不需要 uncolored_vtxs）
    initializeVertexDataGeneric(data, vtx_mask, vtx_color, nullptr, 1.0f);
    buildGraph(G, data);

    // 收集活跃顶点
    for(int i = 0; i < data.vtx_num; ++i) {
        if(vtx_mask[i] == 1.0f) {
            active_vtxs.push(i);
        }
    }

    // 使用通用前向传播算法
    performForwardPropagation(data, G, vtx_mask, vtx_color, active_vtxs);

    return createOutputArrays(data, vtx_mask, vtx_color);
}

// 主接口函数
pair<py::array_t<float>, py::array_t<uint8_t>> meshVerticeInpaint(
    py::array_t<float> texture, py::array_t<uint8_t> mask, py::array_t<float> vtx_pos, py::array_t<float> vtx_uv,
    py::array_t<int> pos_idx, py::array_t<int> uv_idx, const string& method = "smooth") {
    
    if(method == "smooth") {
        return meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx);
    } else if(method == "forward") {
        return meshVerticeInpaint_forward(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx);
    } else {
        throw invalid_argument("Invalid method. Use 'smooth' or 'forward'.");
    }
}

//============================

// 重构后的 meshVerticeColor_smooth 函数
pair<py::array_t<float>, py::array_t<uint8_t>> meshVerticeColor_smooth(
    py::array_t<float> texture, py::array_t<uint8_t> mask, py::array_t<float> vtx_pos, py::array_t<float> vtx_uv,
    py::array_t<int> pos_idx, py::array_t<int> uv_idx) {
    
    MeshData data(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx);
    
    vector<int> vtx_mask;
    vector<vector<float>> vtx_color;
    vector<int> uncolored_vtxs;
    vector<vector<int>> G;

    initializeVertexDataGeneric(data, vtx_mask, vtx_color, &uncolored_vtxs, 1);
    buildGraph(G, data);

    // 使用通用平滑算法
    performSmoothingAlgorithm<int>(data, G, vtx_mask, vtx_color, uncolored_vtxs,
        [](int mask_val) { return mask_val > 0; },    // 检查是否着色
        [](int& mask_val) { mask_val = 2; }           // 设置为已着色（值为2）
    );

    return createVertexColorOutput(data, vtx_mask, vtx_color);
}

// meshVerticeColor 主接口函数
pair<py::array_t<float>, py::array_t<uint8_t>> meshVerticeColor(
    py::array_t<float> texture, py::array_t<uint8_t> mask, py::array_t<float> vtx_pos, py::array_t<float> vtx_uv,
    py::array_t<int> pos_idx, py::array_t<int> uv_idx, const string& method = "smooth") {
    
    if(method == "smooth") {
        return meshVerticeColor_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx);
    } else {
        throw invalid_argument("Invalid method. Use 'smooth' or 'forward'.");
    }
}

// Python绑定
PYBIND11_MODULE(mesh_inpaint_processor, m) {
    m.def("meshVerticeInpaint", &meshVerticeInpaint, "A function to process mesh", 
          py::arg("texture"), py::arg("mask"), py::arg("vtx_pos"), py::arg("vtx_uv"), 
          py::arg("pos_idx"), py::arg("uv_idx"), py::arg("method") = "smooth");
    m.def("meshVerticeColor", &meshVerticeColor, "A function to process mesh", 
          py::arg("texture"), py::arg("mask"), py::arg("vtx_pos"), py::arg("vtx_uv"), 
          py::arg("pos_idx"), py::arg("uv_idx"), py::arg("method") = "smooth");
} 
