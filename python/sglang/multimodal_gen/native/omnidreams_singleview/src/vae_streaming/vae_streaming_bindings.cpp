/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "vae_streaming/vae_streaming_bindings.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "vae_streaming/lightvae_ops.h"

namespace py = pybind11;

namespace omnidreams_singleview {
namespace {

torch::Tensor dict_tensor(const py::dict& dict, const char* key) {
  TORCH_CHECK(dict.contains(key), "prepared FP8 LightVAE state missing key ", key);
  return dict[py::str(key)].cast<torch::Tensor>();
}

py::dict dict_dict(const py::dict& dict, const char* key) {
  TORCH_CHECK(dict.contains(key), "prepared FP8 LightVAE state missing dict ", key);
  return dict[py::str(key)].cast<py::dict>();
}

c10::optional<torch::Tensor> optional_dict_tensor(const py::dict& dict,
                                                  const char* key) {
  if (!dict.contains(key)) {
    return c10::nullopt;
  }
  py::object obj = dict[py::str(key)].cast<py::object>();
  if (obj.is_none()) {
    return c10::nullopt;
  }
  return obj.cast<torch::Tensor>();
}

torch::Tensor scale_prefix(torch::Tensor scale, int64_t real_channels) {
  if (scale.numel() == 1 || scale.numel() == real_channels) {
    return scale.contiguous();
  }
  TORCH_CHECK(scale.numel() >= real_channels,
              "scale has ", scale.numel(), " values, expected at least ",
              real_channels);
  return scale.narrow(0, 0, real_channels).contiguous();
}

class NativeWanVaeEncoderFp8 {
 public:
  explicit NativeWanVaeEncoderFp8(py::dict state) : state_(std::move(state)) {}

  bool is_cuda() const {
    return dict_tensor(state_, "scale_input").is_cuda();
  }

  void reset_cache() {
    fp8_tensor_cache_.clear();
    fp8_temporal_first_cache_.clear();
  }

  void release_resources() {
    reset_cache();
    lightvae_fp8_sdpa_cleanup();
  }

  torch::Tensor encode(torch::Tensor input, bool use_cache) {
    return lightvae_encode_fp8_native(std::move(input), use_cache);
  }

 private:
  c10::optional<torch::Tensor> cached_tensor(const std::string& key) const {
    auto it = fp8_tensor_cache_.find(key);
    if (it == fp8_tensor_cache_.end() || !it->second.defined()) {
      return c10::nullopt;
    }
    return it->second;
  }

  void save_tail_cache_tin16_inplace(const std::string& key,
                                     torch::Tensor input,
                                     int cache_frames) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kByte,
                "LightVAE FP8 cache input must be CUDA uint8");
    TORCH_CHECK(input.dim() == 5 && input.size(4) == 16,
                "LightVAE FP8 cache input must be [T,C/16,H,W,16], got ",
                input.sizes());
    TORCH_CHECK(input.is_contiguous(), "LightVAE FP8 cache input must be contiguous");
    TORCH_CHECK(cache_frames > 0, "cache_frames must be positive");
    c10::cuda::CUDAGuard guard(input.device());

    auto it = fp8_tensor_cache_.find(key);
    torch::Tensor old;
    int old_frames = 0;
    if (it != fp8_tensor_cache_.end() && it->second.defined()) {
      old = it->second;
      TORCH_CHECK(old.is_cuda() && old.scalar_type() == at::kByte,
                  "LightVAE FP8 existing cache must be CUDA uint8");
      TORCH_CHECK(old.dim() == 5 && old.size(4) == 16,
                  "LightVAE FP8 existing cache must be [T,C/16,H,W,16], got ",
                  old.sizes());
      TORCH_CHECK(old.device() == input.device(),
                  "LightVAE FP8 cache device mismatch");
      TORCH_CHECK(old.is_contiguous(),
                  "LightVAE FP8 existing cache must be contiguous");
      TORCH_CHECK(old.size(1) == input.size(1) && old.size(2) == input.size(2) &&
                      old.size(3) == input.size(3),
                  "LightVAE FP8 cache shape mismatch: cache=", old.sizes(),
                  " input=", input.sizes());
      old_frames = static_cast<int>(old.size(0));
    }

    const int frames = static_cast<int>(input.size(0));
    const int new_frames = std::min(cache_frames, old_frames + frames);
    TORCH_CHECK(new_frames > 0, "LightVAE FP8 cannot cache an empty tensor");
    const bool can_reuse =
        old.defined() && old_frames == new_frames && old.size(1) == input.size(1) &&
        old.size(2) == input.size(2) && old.size(3) == input.size(3);
    torch::Tensor dst = can_reuse
        ? old
        : torch::empty(
              {new_frames, input.size(1), input.size(2), input.size(3),
               input.size(4)},
              input.options());

    const size_t frame_bytes = static_cast<size_t>(input.size(1)) *
                               static_cast<size_t>(input.size(2)) *
                               static_cast<size_t>(input.size(3)) *
                               static_cast<size_t>(input.size(4));
    auto stream = at::cuda::getCurrentCUDAStream(input.device().index()).stream();
    const int tail_start = old_frames + frames - new_frames;
    auto* dst_ptr = dst.data_ptr<uint8_t>();
    const auto* input_ptr = input.data_ptr<uint8_t>();
    const auto* old_ptr = old.defined() ? old.data_ptr<uint8_t>() : nullptr;
    for (int out_t = 0; out_t < new_frames; ++out_t) {
      const int src_ext_t = tail_start + out_t;
      const uint8_t* src_ptr = nullptr;
      if (src_ext_t < old_frames) {
        TORCH_CHECK(old_ptr != nullptr, "LightVAE FP8 cache source missing");
        src_ptr = old_ptr + static_cast<size_t>(src_ext_t) * frame_bytes;
      } else {
        const int src_t = src_ext_t - old_frames;
        TORCH_CHECK(src_t >= 0 && src_t < frames,
                    "LightVAE FP8 cache source frame out of range");
        src_ptr = input_ptr + static_cast<size_t>(src_t) * frame_bytes;
      }
      uint8_t* out_ptr = dst_ptr + static_cast<size_t>(out_t) * frame_bytes;
      if (out_ptr != src_ptr) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            out_ptr, src_ptr, frame_bytes, cudaMemcpyDeviceToDevice, stream));
      }
    }
    fp8_tensor_cache_[key] = dst;
  }

  torch::Tensor fp8_requantize_tin16(torch::Tensor input,
                                     torch::Tensor input_scale,
                                     torch::Tensor output_scale,
                                     int real_channels) {
    if (input_scale.data_ptr() == output_scale.data_ptr() &&
        input_scale.numel() == output_scale.numel()) {
      return input;
    }
    return lightvae_fp8_quantize_bcthw_to_tin16(
        lightvae_fp8_dequantize_tin16_to_bcthw(
            input,
            scale_prefix(input_scale, real_channels),
            real_channels,
            static_cast<int>(input.size(2)),
            static_cast<int>(input.size(3))),
        scale_prefix(output_scale, real_channels),
        static_cast<int>(input.size(2)),
        static_cast<int>(input.size(3)),
        static_cast<int>(input.size(1) * input.size(4)));
  }

  torch::Tensor run_causal_fp8(const std::string& key,
                               torch::Tensor input,
                               torch::Tensor weight,
                               torch::Tensor output_scale,
                               c10::optional<torch::Tensor> bias,
                               bool cached,
                               c10::optional<torch::Tensor> epilogue_scale,
                               c10::optional<torch::Tensor> bias_scaled,
                               bool relu) {
    auto old = cached ? cached_tensor(key) : c10::nullopt;
    torch::Tensor output = epilogue_scale.has_value()
        ? lightvae_fp8_causal_conv3_tin16_warp_mma_scaled_prepared(
              input, old, weight, epilogue_scale.value(), bias_scaled, relu)
        : lightvae_fp8_causal_conv3_tin16_prepared(
              input, old, weight, output_scale, bias, relu);
    if (cached) {
      save_tail_cache_tin16_inplace(key, input, 2);
    }
    return output;
  }

  torch::Tensor run_resblock_fp8(const py::dict& block,
                                 torch::Tensor input,
                                 bool cached) {
    torch::Tensor skip;
    torch::Tensor skip_scale;
    py::object shortcut_obj = block[py::str("shortcut")].cast<py::object>();
    if (shortcut_obj.is_none()) {
      skip = input;
      skip_scale = dict_tensor(block, "input_scale");
    } else {
      py::dict shortcut = shortcut_obj.cast<py::dict>();
      skip = lightvae_fp8_conv1_tin16_prepared(
          input,
          dict_tensor(shortcut, "w"),
          dict_tensor(shortcut, "scale"),
          optional_dict_tensor(shortcut, "b"),
          false);
      skip_scale = dict_tensor(shortcut, "scale");
    }

    const int in_real = block[py::str("in_real")].cast<int>();
    const int out_real = block[py::str("out_real")].cast<int>();
    const std::string path = block[py::str("path")].cast<std::string>();
    torch::Tensor output = lightvae_fp8_rmsnorm_tin16(
        input,
        dict_tensor(block, "input_scale_rms"),
        dict_tensor(block, "norm1_gamma"),
        dict_tensor(block, "norm1_scale_rms"),
        in_real,
        true);
    output = run_causal_fp8(
        path + ".residual.2",
        output,
        dict_tensor(block, "conv1_w"),
        dict_tensor(block, "conv1_scale"),
        optional_dict_tensor(block, "conv1_b"),
        cached,
        optional_dict_tensor(block, "conv1_epilogue_scale"),
        optional_dict_tensor(block, "conv1_bias_scaled"),
        false);
    output = lightvae_fp8_rmsnorm_tin16(
        output,
        dict_tensor(block, "conv1_scale_rms"),
        dict_tensor(block, "norm2_gamma"),
        dict_tensor(block, "norm2_scale_rms"),
        out_real,
        true);
    output = run_causal_fp8(
        path + ".residual.6",
        output,
        dict_tensor(block, "conv2_w"),
        dict_tensor(block, "output_scale"),
        optional_dict_tensor(block, "conv2_b"),
        cached,
        optional_dict_tensor(block, "conv2_epilogue_scale"),
        optional_dict_tensor(block, "conv2_bias_scaled"),
        false);
    return lightvae_fp8_add_tin16(
        output,
        skip,
        scale_prefix(dict_tensor(block, "output_scale"), out_real),
        scale_prefix(skip_scale, out_real),
        scale_prefix(dict_tensor(block, "output_scale"), out_real),
        out_real);
  }

  torch::Tensor run_downsample_fp8(const py::dict& stage,
                                   torch::Tensor input,
                                   bool cached,
                                   const std::string& key) {
    const bool has_spatial = stage.contains("spatial_w");
    torch::Tensor spatial_w =
        has_spatial ? dict_tensor(stage, "spatial_w") : dict_tensor(stage, "w");
    torch::Tensor spatial_scale = has_spatial ? dict_tensor(stage, "spatial_scale")
                                              : dict_tensor(stage, "scale");
    c10::optional<torch::Tensor> spatial_b =
        has_spatial ? optional_dict_tensor(stage, "spatial_b")
                    : optional_dict_tensor(stage, "b");
    c10::optional<torch::Tensor> spatial_epilogue =
        optional_dict_tensor(stage, "spatial_epilogue_scale");
    if (!spatial_epilogue.has_value()) {
      spatial_epilogue = optional_dict_tensor(stage, "epilogue_scale");
    }
    c10::optional<torch::Tensor> spatial_bias_scaled =
        optional_dict_tensor(stage, "spatial_bias_scaled");
    if (!spatial_bias_scaled.has_value()) {
      spatial_bias_scaled = optional_dict_tensor(stage, "bias_scaled");
    }

    torch::Tensor output = spatial_epilogue.has_value()
        ? lightvae_fp8_spatial_conv3_tin16_warp_mma_scaled_prepared(
              input,
              spatial_w,
              spatial_epilogue.value(),
              spatial_bias_scaled,
              2,
              0,
              false,
              true)
        : lightvae_fp8_spatial_conv3_tin16_prepared(
              input, spatial_w, spatial_scale, spatial_b, 2, 0, false, true);
    if (!stage.contains("temporal_w")) {
      return output;
    }

    const std::string first_key = key + ".first";
    const std::string data_key = key + ".cache";
    const bool first_call =
        fp8_temporal_first_cache_.find(first_key) ==
            fp8_temporal_first_cache_.end() ||
        fp8_temporal_first_cache_[first_key];
    if (cached && first_call) {
      fp8_tensor_cache_[data_key] =
          output.narrow(0, output.size(0) - 1, 1).contiguous();
      fp8_temporal_first_cache_[first_key] = false;
      return fp8_requantize_tin16(
          output,
          dict_tensor(stage, "spatial_scale"),
          dict_tensor(stage, "scale"),
          stage[py::str("real")].cast<int>());
    }

    auto old = cached ? cached_tensor(data_key) : c10::nullopt;
    torch::Tensor temporal_output = lightvae_fp8_temporal_conv1_tin16_prepared(
        output,
        old,
        dict_tensor(stage, "temporal_w"),
        dict_tensor(stage, "scale"),
        optional_dict_tensor(stage, "temporal_b"),
        false);
    if (cached) {
      fp8_tensor_cache_[data_key] =
          output.narrow(0, output.size(0) - 1, 1).contiguous();
    }
    return temporal_output;
  }

  torch::Tensor run_attention_fp8(torch::Tensor input,
                                  torch::Tensor input_scale,
                                  const py::dict& attention,
                                  torch::Tensor output_scale) {
    torch::Tensor output = lightvae_fp8_rmsnorm_tin16(
        input,
        dict_tensor(attention, "input_scale_rms"),
        dict_tensor(attention, "norm_gamma"),
        dict_tensor(attention, "norm_scale_rms"),
        96,
        false);
    torch::Tensor qkv = lightvae_fp8_conv1_tin16_prepared(
        output,
        dict_tensor(attention, "qkv_w"),
        dict_tensor(attention, "qkv_scale"),
        optional_dict_tensor(attention, "qkv_b"),
        false);
    auto qkv_tuple = lightvae_fp8_qkv_tin16_to_bmhd(qkv);
    const int frames = static_cast<int>(input.size(0));
    const int height = static_cast<int>(input.size(2));
    const int width = static_cast<int>(input.size(3));
    torch::Tensor sdpa = lightvae_fp8_sdpa_bmhd(
        std::get<0>(qkv_tuple),
        std::get<1>(qkv_tuple),
        std::get<2>(qkv_tuple),
        dict_tensor(attention, "qkv_scale_float"),
        dict_tensor(attention, "sdpa_inverse_scale_float"),
        dict_tensor(attention, "unit_float"),
        frames,
        height * width,
        1.0 / std::sqrt(96.0));
    torch::Tensor sdpa_tin16 =
        lightvae_fp8_bmhd_to_tin16(sdpa, frames, height, width);
    torch::Tensor proj = lightvae_fp8_conv1_tin16_prepared(
        sdpa_tin16,
        dict_tensor(attention, "proj_w"),
        output_scale,
        optional_dict_tensor(attention, "proj_b"),
        false);
    return lightvae_fp8_add_tin16(
        proj,
        input,
        scale_prefix(output_scale, 96),
        scale_prefix(input_scale, 96),
        scale_prefix(output_scale, 96),
        96);
  }

  torch::Tensor lightvae_encode_fp8_native(torch::Tensor input, bool cached) {
    TORCH_CHECK(input.dim() == 5 && input.size(0) == 1 && input.size(1) == 3,
                "LightVAE FP8 input must be [1,3,T,H,W], got ", input.sizes());
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf,
                "LightVAE FP8 input boundary must be CUDA fp16");
    const int h = static_cast<int>(input.size(3));
    const int w = static_cast<int>(input.size(4));
    TORCH_CHECK(h % 8 == 0 && w % 8 == 0,
                "LightVAE FP8 input spatial dimensions must be divisible by 8, got ",
                h, "x", w);
    const int hp = ((h + 7) / 8) * 8;
    const int wp = ((w + 7) / 8) * 8;

    torch::Tensor output = lightvae_fp8_quantize_bcthw_to_tin16(
        input.contiguous(),
        scale_prefix(dict_tensor(state_, "scale_input"), 3),
        hp,
        wp,
        32);
    output = run_causal_fp8(
        "encoder.conv1",
        output,
        dict_tensor(state_, "conv1_w"),
        dict_tensor(state_, "scale_conv1"),
        optional_dict_tensor(state_, "conv1_b"),
        cached,
        optional_dict_tensor(state_, "conv1_epilogue_scale"),
        optional_dict_tensor(state_, "conv1_bias_scaled"),
        false);

    py::list blocks = state_[py::str("blocks")].cast<py::list>();
    output = run_resblock_fp8(blocks[0].cast<py::dict>(), output, cached);
    output = run_resblock_fp8(blocks[1].cast<py::dict>(), output, cached);
    output =
        run_downsample_fp8(dict_dict(state_, "ds0"), output, cached,
                           "encoder.downsamples.2");

    output = run_resblock_fp8(blocks[2].cast<py::dict>(), output, cached);
    output = run_resblock_fp8(blocks[3].cast<py::dict>(), output, cached);
    output =
        run_downsample_fp8(dict_dict(state_, "ds1"), output, cached,
                           "encoder.downsamples.5.time_conv");

    output = run_resblock_fp8(blocks[4].cast<py::dict>(), output, cached);
    output = run_resblock_fp8(blocks[5].cast<py::dict>(), output, cached);
    output =
        run_downsample_fp8(dict_dict(state_, "ds2"), output, cached,
                           "encoder.downsamples.8.time_conv");

    output = run_resblock_fp8(blocks[6].cast<py::dict>(), output, cached);
    output = run_resblock_fp8(blocks[7].cast<py::dict>(), output, cached);
    output = run_resblock_fp8(blocks[8].cast<py::dict>(), output, cached);
    output = run_attention_fp8(
        output,
        dict_tensor(state_, "scale_mid0"),
        dict_dict(state_, "mid_attn"),
        dict_tensor(state_, "scale_mid_attn"));
    output = run_resblock_fp8(blocks[9].cast<py::dict>(), output, cached);

    output = lightvae_fp8_rmsnorm_tin16(
        output,
        dict_tensor(state_, "scale_mid1_rms"),
        dict_tensor(state_, "head_norm_gamma"),
        dict_tensor(state_, "scale_head_norm_rms"),
        96,
        true);
    output = run_causal_fp8(
        "encoder.head.2",
        output,
        dict_tensor(state_, "head_w"),
        dict_tensor(state_, "scale_head_conv"),
        optional_dict_tensor(state_, "head_b"),
        cached,
        optional_dict_tensor(state_, "head_epilogue_scale"),
        optional_dict_tensor(state_, "head_bias_scaled"),
        false);
    output = lightvae_fp8_conv1_tin16_prepared(
        output,
        dict_tensor(state_, "post_w"),
        dict_tensor(state_, "scale_post"),
        optional_dict_tensor(state_, "post_b"),
        false);
    return lightvae_fp8_extract_mu_normalize_tin16(
        output,
        scale_prefix(dict_tensor(state_, "scale_post"), 32),
        dict_tensor(state_, "mean"),
        dict_tensor(state_, "inv_std"),
        h / 8,
        w / 8);
  }

  py::dict state_;
  std::unordered_map<std::string, torch::Tensor> fp8_tensor_cache_;
  std::unordered_map<std::string, bool> fp8_temporal_first_cache_;
};

py::dict vae_backend_status(const std::string& component,
                            const std::string& backend) {
  py::dict status;
  status["component"] = component;
  status["backend"] = backend;
  status["available"] = false;
  status["implementation"] = "warp_mma_tin16_fp8";
  if (backend != "fp8") {
    status["reason"] = "unsupported OmniDreams VAE native backend";
    return status;
  }
  if (component == "vae_encoder") {
    status["available"] = true;
    status["implementation"] = "direct_tin16_fp8_lightvae_encoder";
    status["reason"] = "LightVAE fp8 native encoder is available";
    return status;
  }
  status["reason"] = "unknown OmniDreams VAE native component";
  return status;
}

std::shared_ptr<NativeWanVaeEncoderFp8> create_wan_encoder_fp8(py::dict state) {
  auto encoder = std::make_shared<NativeWanVaeEncoderFp8>(std::move(state));
  TORCH_CHECK(encoder->is_cuda(), "LightVAE fp8 native encoder state must be CUDA");
  return encoder;
}

torch::Tensor encode_wan_fp8(
    const std::shared_ptr<NativeWanVaeEncoderFp8>& encoder,
    torch::Tensor input,
    bool use_cache) {
  TORCH_CHECK(encoder, "LightVAE fp8 native encoder is null");
  return encoder->encode(std::move(input), use_cache);
}

void reset_wan_encoder_fp8(
    const std::shared_ptr<NativeWanVaeEncoderFp8>& encoder) {
  TORCH_CHECK(encoder, "LightVAE fp8 native encoder is null");
  encoder->reset_cache();
}

}  // namespace

void bind_vae_streaming(py::module_& module) {
  bind_lightvae_ops(module);

  py::class_<NativeWanVaeEncoderFp8, std::shared_ptr<NativeWanVaeEncoderFp8>>(
      module, "_OmnidreamsNativeWanVaeEncoderFp8")
      .def("reset_cache", &NativeWanVaeEncoderFp8::reset_cache)
      .def("release_resources", &NativeWanVaeEncoderFp8::release_resources)
      .def("is_cuda", &NativeWanVaeEncoderFp8::is_cuda);

  module.def(
      "omnidreams_vae_backend_status",
      &vae_backend_status,
      py::arg("component"),
      py::arg("backend"),
      "Return availability for an OmniDreams VAE native backend.");
  module.def(
      "omnidreams_vae_create_wan_encoder_fp8",
      &create_wan_encoder_fp8,
      py::arg("state"),
      "Create a cached fp8 native LightVAE encoder state.");
  module.def(
      "omnidreams_vae_reset_wan_encoder_fp8",
      &reset_wan_encoder_fp8,
      py::arg("encoder"),
      "Reset cached fp8 native LightVAE encoder state.");
  module.def(
      "omnidreams_vae_encode_wan_fp8",
      &encode_wan_fp8,
      py::arg("encoder"),
      py::arg("input"),
      py::arg("use_cache") = true,
      "Encode Wan LightVAE input through the fp8 native boundary with fp16 latents.");
}

}  // namespace omnidreams_singleview
