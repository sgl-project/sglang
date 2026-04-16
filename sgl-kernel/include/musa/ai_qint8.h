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
#ifndef INCLUDE_MUDNN_API_AI_QINT8_H_
#define INCLUDE_MUDNN_API_AI_QINT8_H_

class qint8_t {
  int8_t _;

 public:
  __device__ int8_t as_int8() const {
    return _;
  }

  __host__ __device__ qint8_t(int8_t val) : _(val) {}
  __host__ __device__ qint8_t() : _(0) {}
  __host__ __device__ qint8_t(const qint8_t& b) {
    _ = b._;
  }
  __host__ __device__ qint8_t(const int& b) {
    _ = b;
  }
  __host__ __device__ qint8_t(const char& b) {
    _ = b;
  }
  __host__ __device__ operator char() const {
    return _;
  }
  __host__ __device__ qint8_t operator+=(const qint8_t& b) {
    return _ += b._;
  }
  __host__ __device__ qint8_t operator-=(const qint8_t& b) {
    return _ -= b._;
  }
  __host__ __device__ qint8_t operator*=(const qint8_t& b) {
    return _ *= b._;
  }

  bool operator<(const qint8_t& b) const {
    return _ < b._;
  }
  bool operator>(const qint8_t& b) const {
    return _ > b._;
  }
  bool operator==(const qint8_t& b) const {
    return _ == b._;
  }
  bool operator!=(const qint8_t& b) const {
    return _ != b._;
  }
};

#endif  // INCLUDE_MUDNN_API_AI_QINT8_H_
