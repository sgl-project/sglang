/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief This extends the contents of cutlass/functional.h with frequently used activation functions.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/constants.h"
#include "cutlass/complex.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/functional.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

// If kIsHeavy is a member, use it.  Otherwise, assume that it's false.
template<class Op, class Enable = void>
struct kIsHeavy_member_or_false {
  static constexpr bool value = false;
};
template<class Op>
struct kIsHeavy_member_or_false<Op, typename cutlass::platform::enable_if<Op::kIsHeavy>::type> {
  static constexpr bool value = Op::kIsHeavy;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Identity operator
template <typename T>
struct Identity {
  static const bool kIsHeavy = false;

  CUTLASS_HOST_DEVICE
  T operator()(T value) const {
    return value;
  }
};

template <typename T, int N>
struct Identity<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> value) const {
    return value;
  }
};

/// Scale operator
template <typename T>
struct Scale {
  struct Arguments {
    using scale_type = T;
    T scale = T(1);
  };

  CUTLASS_HOST_DEVICE
  T operator()(T value, T scale) const {
    multiplies<T> mul;
    return mul(scale, value);
  }

  CUTLASS_HOST_DEVICE
  T operator()(T value, Arguments args = Arguments()) const {
    return this->operator()(value, args.scale);
  }
};

template <typename T, int N>
struct Scale<Array<T, N>> {
  using Arguments = typename Scale<T>::Arguments;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> values, T scale) const {
    multiplies<Array<T, N>> mul;
    return mul(scale, values);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> values, Arguments args = Arguments()) const {
    return this->operator()(values, args.scale);
  }
};

/// Specialization to compose other activations with a defined unary operator
/// e.g. Scale<Identity<T>>
template <template <class> class Activation, typename T>
struct Scale<Activation<T>> {
  using Arguments = typename Scale<T>::Arguments;

  static const bool kIsHeavy = Activation<T>::kIsHeavy;

  CUTLASS_HOST_DEVICE
  T operator()(T value, typename Arguments::scale_type scale) const {
    multiplies<T> mul;
    Activation<T> act;
    return mul(scale, act(value));
  }

  CUTLASS_HOST_DEVICE
  T operator()(T value, Arguments args = Arguments()) const {
    return this->operator()(value, args.scale);
  }
};

/// ReLu operator - propagates NaNs
template <typename T>
struct ReLu {
  static const bool kIsHeavy = false;

  CUTLASS_HOST_DEVICE
  T operator()(T threshold, T value) const {
    constexpr bool PropagateNaN = true;
    maximum<T, PropagateNaN> mx;

    return mx(value, threshold);
  }

  CUTLASS_HOST_DEVICE
  T operator()(T value) const {
    constexpr bool PropagateNaN = true;
    maximum<T, PropagateNaN> mx;

    return mx(value, T(0));
  }
};

template <typename T>
using ReLU = ReLu<T>;

template <typename T, int N>
struct ReLu<Array<T, N>> {
  static const bool kIsHeavy = false;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const & threshold, Array<T, N> const &frag) const {
    constexpr bool PropagateNaN = true;
    maximum<Array<T, N>, PropagateNaN> mx;

    return mx(frag, threshold);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &frag) const {
    constexpr bool PropagateNaN = true;
    maximum<Array<T, N>, PropagateNaN> mx;
    return mx(frag, T(0));
  }
};

// Generic clamp
template <typename T>
struct Clamp {
  struct Arguments {
    T lower_bound = CUTLASS_STL_NAMESPACE::numeric_limits<T>::lowest();
    T upper_bound = CUTLASS_STL_NAMESPACE::numeric_limits<T>::max();
  };

  CUTLASS_HOST_DEVICE
  T operator()(T const& value, T const& lower_bound, T const& upper_bound) const {
    constexpr bool PropagateNaN = true;
    maximum<T, PropagateNaN> mx;
    minimum<T, PropagateNaN> mn;

    return mn(mx(value, lower_bound), upper_bound);
  }

  CUTLASS_HOST_DEVICE
  T operator()(T const& value, Arguments const& args = Arguments()) const {
    return this->operator()(value, args.lower_bound, args.upper_bound);
  }
};

template <typename T, int N>
struct Clamp<Array<T,N>> {
  using Arguments = typename Clamp<T>::Arguments;

  CUTLASS_HOST_DEVICE
  Array<T,N> operator()(Array<T,N> const& values, T const& lower_bound, T const& upper_bound) const {
    constexpr bool PropagateNaN = true;
    maximum<Array<T,N>, PropagateNaN> mx;
    minimum<Array<T,N>, PropagateNaN> mn;

    return mn(mx(values, lower_bound), upper_bound);
  }

  CUTLASS_HOST_DEVICE
  Array<T,N> operator()(Array<T,N> const& values, Arguments const& args = Arguments()) const {
    return this->operator()(values, args.lower_bound, args.upper_bound);
  }
};

// Lower Bound
template <typename T>
struct LowerBound {
  struct Arguments {
    T lower_bound;
  };

  CUTLASS_HOST_DEVICE
  T operator()(T const& value, T const& lower_bound) const {
    constexpr bool PropagateNaN = true;
    maximum<T, PropagateNaN> mx;

    return mx(value, lower_bound);
  }

  CUTLASS_HOST_DEVICE
  T operator()(T const& value, Arguments const& args = Arguments()) const {
    return this->operator()(value, args.lower_bound);
  }
};

template <typename T, int N>
struct LowerBound<Array<T,N>> {
  using Arguments = typename LowerBound<T>::Arguments;

  CUTLASS_HOST_DEVICE
  Array<T,N> operator()(Array<T,N> const& values, T const& lower_bound) const {
    constexpr bool PropagateNaN = true;
    maximum<Array<T,N>, PropagateNaN> mx;

    return mx(values, lower_bound);
  }

  CUTLASS_HOST_DEVICE
  Array<T,N> operator()(Array<T,N> const& values, Arguments const& args = Arguments()) const {
    return this->operator()(values, args.lower_bound);
  }
};

// Leaky Relu operator
template <typename T>
struct LeakyReLU {

  static const bool kIsHeavy = false;

  struct Arguments {
    T leaky_alpha = T(0);
  };

  CUTLASS_HOST_DEVICE
  T operator()(T const& value, T const& leaky_alpha) const {
    T res = value > T(0) ? value : value * leaky_alpha;
    return res;
  }

  CUTLASS_HOST_DEVICE
  T operator()(T const& value, Arguments const& args = Arguments()) const {
    return this->operator()(value, args.leaky_alpha);
  }
};

template <typename T, int N>
struct LeakyReLU<Array<T, N> > {

  static const bool kIsHeavy = false;

  using Arguments = typename LeakyReLU<T>::Arguments;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& values, T const& leaky_alpha) const {
    Array<T, N> y;
    LeakyReLU<T> leaky_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(values.size()); ++i) {
      y[i] = leaky_op(values[i], leaky_alpha);
    }

    return y;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& values, Arguments const& args = Arguments()) const {
    return this->operator()(values, args.leaky_alpha);
  }
};

// Y = min((X <= threshold ? 0 : X), upper_bound)
template <typename T>
struct ThresholdReLU {
  static constexpr bool kIsHeavy = false;

  struct Arguments {
    T threshold = T(0);
    T upper_bound = CUTLASS_STL_NAMESPACE::numeric_limits<T>::max();
  };

  CUTLASS_HOST_DEVICE
  T operator()(T value, T threshold, T upper_bound) const {
    minimum_with_nan_propagation<T> mn;
    
    return mn((value <= threshold ? T(0) : value), upper_bound);
  }

  CUTLASS_HOST_DEVICE
  T operator()(T value, Arguments const& args = Arguments()) const {
    return operator()(value, args.threshold, args.upper_bound);
  }
};

template <typename T, int N>
struct ThresholdReLU<Array<T,N>> {
  static constexpr bool kIsHeavy = false;

  using Arguments = typename ThresholdReLU<T>::Arguments;

  CUTLASS_HOST_DEVICE
  Array<T,N> operator()(Array<T,N> const& values, T threshold, T upper_bound) const {
    ThresholdReLU<T> relu;

    Array<T,N> retvals;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      retvals[i] = relu(values[i], threshold, upper_bound);    
    }

    return retvals;
  }

  CUTLASS_HOST_DEVICE
  Array<T,N> operator()(Array<T,N> const& values, Arguments const& args = Arguments()) const {
    return operator()(values, args.threshold, args.upper_bound);
  }
};

// Tanh operator
template <typename T>
struct Tanh {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  T operator()(T const &value) const {
    return fast_tanh(value);
  }
};

template <typename T, int N>
struct Tanh<Array<T, N> > {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &value) const {
    Array<T, N> y;
    Tanh<T> tanh_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = tanh_op(value[i]);
    }

    return y;
  }
};

template <int N>
struct Tanh<Array<half_t, N>> {
  using T = half_t;
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& z) const {
    fast_tanh_op<Array<T, N>> tanh;
    return tanh(z);
  }
};

// Sigmoid operator
template <typename T>
struct Sigmoid {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  T operator()(T const &value) const {
#if defined(CUTLASS_USE_TANH_FOR_SIGMOID)
    return fast_tanh(value * T(0.5)) * T(0.5) + T(0.5);
#else
    return T(1) / (T(1) + fast_exp(-value));
#endif
  }
};

template <typename T, int N>
struct Sigmoid<Array<T, N>> {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& z) const {
#if defined(CUTLASS_USE_TANH_FOR_SIGMOID)
    multiplies<Array<T, N>> mul;
    multiply_add<Array<T, N>> fma;
    fast_tanh_op<Array<T, N>> tanh;
    return fma(tanh(mul(z, cutlass::constants::half<T>())),
               cutlass::constants::half<T>(),
               cutlass::constants::half<T>());
#else
    plus<Array<T, N>> add;
    divides<Array<T, N>> div;
    negate<Array<T, N>> neg;
    fast_exp_op<Array<T, N>> fast_exp;
    return div(cutlass::constants::one<T>(),
               add(cutlass::constants::one<T>(),
                   fast_exp(neg(z))));
#endif
  }
};

// SiLu (swish) operator introduced by Elfwing et al. in the following paper
// "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning" (2017)
// https://arxiv.org/pdf/1702.03118.pdf
// It is used in EfficientNet and YOLOv5, for example.
// Reference: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
template <typename T>
struct SiLu {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  T operator()(T const &value) const {
    Sigmoid<T> sigmoid;
    return value * sigmoid(value);
  }
};

template <typename T, int N>
struct SiLu<Array<T, N>> {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &value) const {
    Sigmoid<Array<T, N>> sigmoid_op;
    multiplies<Array<T, N>>     mul;
    return mul(value, sigmoid_op(value));
  }
};

template <typename T>
using ScaledSiLu = Scale<SiLu<T>>;

// Hardswish operator introduced by Howard et al. in the following paper
// "Searching for MobileNetV3" (2019)
// https://arxiv.org/pdf/1905.02244.pdf
// It is used in models based on MobilenetNetV3.
// Reference: https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html
template <typename T>
struct HardSwish {
  static const bool kIsHeavy = false;

  CUTLASS_HOST_DEVICE
  T operator()(T const &x) const {
    minimum<T> mn;
    maximum<T> mx;
    T relu6 = mn(mx(x + T(3), T(0)), T(6));
    return x * relu6 / T(6);
  }
};

template <>
struct HardSwish<float> {
  using T = float;
  static const bool kIsHeavy = false;
  static constexpr float kOneSixth = 0.16666667f;

  CUTLASS_HOST_DEVICE
  T operator()(T const &x) const {
    minimum<T> mn;
    maximum<T> mx;
    T relu6 = mn(mx(x + T(3), T(0)), T(6));
    return x * relu6 * kOneSixth;
  }
};

template <>
struct HardSwish<cutlass::half_t> {
  using T = cutlass::half_t;
  static const bool kIsHeavy = false;
  static constexpr float kOneSixth = 0.16666667f;

  CUTLASS_HOST_DEVICE
  T operator()(T const &x) const {
    minimum<T> mn;
    maximum<T> mx;
    T relu6 = mn(mx(x + T(3), T(0)), T(6));
    return x * relu6 * T(kOneSixth);
  }
};

template <typename T, int N>
struct HardSwish<Array<T, N> > {
  static const bool kIsHeavy = false;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &value) const {
    Array<T, N> y;
    HardSwish<T> hardswish_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = hardswish_op(value[i]);
    }

    return y;
  }
};

template <int N>
struct HardSwish<Array<half_t, N> > {
  using T = half_t;
  static const bool kIsHeavy = false;
  static constexpr float kOneSixth = 0.16666667f;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &value) const {
    minimum<Array<T, N> > mn;
    maximum<Array<T, N> > mx;
    multiplies<Array<T, N> > mul;
    plus<Array<T, N> > add;

    return mul(mul(mn(mx(add(value, T(3)), T(0)), T(6)), value), T(kOneSixth));
  }
};

template <typename T>
using ScaledHardSwish = Scale<HardSwish<T>>;

//
// GELU function definitions implemented as described by
//   Hendrycks, D., and Gimpel, K. in
//   "Gaussian Error Linear Units (GELUs)." (2020)
//   https://arxiv.org/pdf/1606.08415.pdf
//
// Floating-point constants are Taylor coefficients described in the paper.
//

// GELU operator
template <typename T>
struct GELU {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  T operator()(T const &value) const {
    return T(cutlass::constants::half<T>() * value *
      (cutlass::constants::one<T>() + (T)erff((float)(value * cutlass::constants::half_root_two<T>()))));
  }
};

template <>
struct GELU<float> {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  float operator()(float const &value) const {
    return cutlass::constants::half<float>() * value *
      (cutlass::constants::one<float>() + erff(value * cutlass::constants::half_root_two<float>() ));
  }
};

template <>
struct GELU<double> {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  double operator()(double const &value) const {
    return cutlass::constants::half<double>() * value *
      (cutlass::constants::one<double>() + erf( value * cutlass::constants::half_root_two<double>() ));
  }
};

template <typename T, int N>
struct GELU<Array<T, N> > {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &value) const {
    Array<T, N> y;
    GELU<T> gelu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = gelu_op(value[i]);
    }

    return y;
  }
};

template <typename T>
using ScaledGELU = Scale<GELU<T>>;

// GELU operator implemented using the Taylor series approximation
template <typename T>
struct GELU_taylor {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  T operator()(T const &z) const {

    T k0 = T(0.7978845608028654);
    T k1 = T(0.044715);

    return T(cutlass::constants::half<T>() * z *
      (cutlass::constants::one<T>() + fast_tanh(k0 * z * (cutlass::constants::one<T>() + k1 * z * z))));
  }
};

template <>
struct GELU_taylor <float>{
  static const bool kIsHeavy = true;
  using T = float;
  CUTLASS_HOST_DEVICE
  T operator()(T const &z) const {
    // 0.5f * (x + x * tanh(x * (0.797885f + 0.0356774f * x * x)));
    T k0 = T(0.7978845608028654);
    T tmp = T(0.044715);
    T k1 = T(k0*tmp);
    multiply_add<T> fma;
    multiplies<T> mul;
    T v0 = mul(k1, z);
    T v1 = fma(v0, z, k0);
    T v2 = mul(z, v1);
    T v3 = fast_tanh(v2);
    T v4 = fma(z, v3, z);
    T v5 = mul(cutlass::constants::half<T>(), v4);
    return v5;
  }
};

template <int N>
struct GELU_taylor<Array<half_t, N> > {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const &z) const {

    using T = half_t;
    Array<half_t, N> y;

    half_t k0 = half_t(0.7978845608028654);
    half_t k1 = half_t(0.044715);

    multiply_add<Array<half_t, N>> fma;
    multiplies<Array<half_t, N>>     mul;
    plus<Array<half_t, N>>         add;

    fast_tanh_op<Array<half_t, N>> tanh;

    Array<half_t, N> u = mul(mul(k0, z), fma(mul(k1, z), z, cutlass::constants::one<T>()));

    y = mul(mul(z, cutlass::constants::half<T>()), add(cutlass::constants::one<T>(), tanh(u)));

    return y;
  }
};

template <int N>
struct GELU_taylor<Array<float, N> > {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  Array<float, N> operator()(Array<float, N> const &value) const {
    multiply_add<Array<float, N>> fma;
    multiplies<Array<float, N>> mul;
    fast_tanh_op<Array<float, N>> tanh;
    // 0.5f * (x + x * tanh(x * (0.797885f + 0.0356774f * x * x)));
    float k0 = float(0.7978845608028654);
    float tmp = float(0.044715);
    float k1 = float(k0*tmp);

    Array<float, N> v0 = mul(k1, value);
    Array<float, N> v1 = fma(v0, value, k0);
    Array<float, N> v2 = mul(value, v1);
    Array<float, N> v3 = tanh(v2);
    Array<float, N> v4 = fma(value, v3, value);
    Array<float, N> v5 = mul(cutlass::constants::half<float>(), v4);
    return v5;
  }
};

template <typename T, int N>
struct GELU_taylor<Array<T, N> > {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &value) const {
    Array<T, N> y;
    GELU_taylor<T> gelu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = gelu_op(value[i]);
    }

    return y;
  }
};

template <typename T>
using ScaledGELU_taylor = Scale<GELU_taylor<T>>;

/// Computes backwards pass for GELU operator assuming d_t is the layer gradient and
/// z is computed from the forward pass.
template <typename T>
struct dGELU {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  T operator()(T const &d_t, T const &z) const {

    T k0 = T(0.7978845608028654);
    T k1 = T(0.044715);
    T k2 = T(0.1070322243);

    T tanh_out = fast_tanh(k0 * z * (1 + k1 * z * z));

    T ff = constants::half<T>() * z * ((1 - tanh_out * tanh_out) * (k0 + k2 * z * z)) +
      constants::half<T>() * (1 + tanh_out);

    return ff * d_t;
  }
};

template <typename T, int N>
struct dGELU<Array<T, N> > {
  static const bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &d_t, Array<T, N> const &z) const {
    Array<T, N> y;
    dGELU<T> gelu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = gelu_op(d_t[i], z[i]);
    }

    return y;
  }
};

template <typename T>
struct dReLU {
  CUTLASS_HOST_DEVICE
  T operator()(T d_t, bool d_relu) const {
    return d_relu ? d_t : T(0);
  }

  template <typename U>
  CUTLASS_HOST_DEVICE
  T operator()(T d_t, U d_relu) const {
    return operator()(d_t, static_cast<bool>(d_relu));
  }
};

template <typename T, int N>
struct dReLU<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& d_t, bool const (&d_relu)[N]) const {
    Array<T, N> y;
    dReLU<T> relu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = relu_op(d_t[i], d_relu[i]);
    }

    return y;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& d_t, Array<uint1b_t, N> const& d_relu) const {
    UnpackPredicates<N> unpack_op;

    bool preds[N];
    unpack_op(preds, d_relu);

    return operator()(d_t, preds);
  }

  template <typename U>
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& d_t, Array<U, N> const& d_relu) const {
    Array<T, N> y;
    dReLU<T> relu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = relu_op(d_t[i], d_relu[i]);
    }

    return y;
  }
};

/// Computes backwards pass for ReLU operator assuming d_t is the layer gradient and
/// z is computed from the forward pass.
template <typename T>
struct dReLU_Z {
  CUTLASS_HOST_DEVICE
  T operator()(T d_t, T z) const {
    return z < 0 ? T(0) : d_t;
  }
};

template <typename T, int N>
struct dReLU_Z<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& d_t, Array<T, N> const& z) const {
    Array<T, N> y;
    dReLU_Z<T> relu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = relu_op(d_t[i], z[i]);
    }

    return y;
  }
};

// ElementwiseFilter operator
// Filters by a specific value and maps it to 0.0
// Used in GEMM + comm
template <typename T>
struct ElementwiseFilter {

  static const bool kIsHeavy = false;

  struct Arguments {
    T value_to_filter = T(-0.0);
    T filtered_value = T(0.0);
  };

  CUTLASS_HOST_DEVICE
  T operator()(T const& value, T const& value_to_filter, T const& filtered_value) const {
    T res = value == value_to_filter ? filtered_value : value;
    return res;
  }

  CUTLASS_HOST_DEVICE
  T operator()(T const& value, Arguments const& args = Arguments()) const {
    return this->operator()(value, args.value_to_filter, args.filtered_value);
  }
};

template <typename T, int N>
struct ElementwiseFilter<Array<T, N> > {

  static const bool kIsHeavy = false;

  using Arguments = typename ElementwiseFilter<T>::Arguments;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& values, T const& value_to_filter, T const& filtered_value) const {
    Array<T, N> y;
    ElementwiseFilter<T> filter_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(values.size()); ++i) {
      y[i] = filter_op(values[i], value_to_filter, filtered_value);
    }

    return y;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& values, Arguments const& args = Arguments()) const {
    return this->operator()(values, args.value_to_filter, args.filtered_value);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
