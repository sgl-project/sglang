/* Copyright @2020-2024 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 *
 * This software ("this software and its documentations" or "the software") is
 * protected by Copyright and the information contained herein is confidential.
 *
 * The software contained herein is PROPRIETARY to Moore Threads and is being
 * provided under the terms and conditions of a form of Moore Threads software
 * license agreement by and between Moore Threads and Licensee ("License
 * Agreement") or electronically accepted by Licensee. Notwithstanding any
 * terms or conditions to the contrary in the License Agreement, copy or
 * disclosure of the software to any third party without the express written
 * consent of Moore Threads is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, MOORE THREADS MAKES NO REPRESENTATION ABOUT ANY WARRANTIES,
 * INCLUDING BUT NOT LIMITED TO THE SUITABILITY OF THE SOFTWARE FOR ANY
 * PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF
 * ANY KIND. MOORE THREADS DISCLAIMS ALL WARRANTIES WITH REGARD TO THE
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL MOORE THREADS BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THE SOFTWARE.
 */

#ifndef INCLUDE_MUDNN_API_INTEGER_SUBBYTE_H_
#define INCLUDE_MUDNN_API_INTEGER_SUBBYTE_H_

#include <limits>
#include <type_traits>

namespace musa::dnn {

// cutlass integer_subbyte class
template <int Bits, bool Signed = true>
struct integer_subbyte {
  using Storage = uint8_t;

  static_assert(Bits <= 8 * sizeof(Storage), "Require a subbyte of bits in integer_subbyte");

  using xint_t = typename std::conditional<Signed, int, unsigned>::type;

  static constexpr Storage bits_mask_ = Storage((1 << Bits) - 1);

  static constexpr Storage sign_mask_ = Storage((Signed ? 1 : 0) << (Bits - 1));

  Storage storage;

  __host__ __device__ constexpr integer_subbyte() {}

  __host__ __device__ constexpr integer_subbyte(int value)
      : storage(reinterpret_cast<Storage const&>(value) & bits_mask_) {}

  __host__ __device__ constexpr integer_subbyte(unsigned value)
      : storage(reinterpret_cast<Storage const&>(value) & bits_mask_) {}
};

}  // namespace musa::dnn

#endif  // INCLUDE_MUDNN_API_INTEGER_SUBBYTE_H_
