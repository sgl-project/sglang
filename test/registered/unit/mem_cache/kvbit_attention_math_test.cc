#include <cassert>
#include <cmath>
#include <iostream>

#include "packed_layout.h"
#include "sm90/decode/sparse_fp8/attention_math.h"

int main() {
  using namespace kvbit::dsv4;
  static_assert(NOPE_DIM / 2 == CODE_BYTES);
  static_assert(NOPE_DIM / GROUP_SIZE == NUM_GROUPS);
  static_assert(ROPE_OFFSET + ROPE_BYTES == PAYLOAD_BYTES);
  static_assert(COMPACT_ROW_BYTES == 368 && ALIGNED_ROW_BYTES == 384);
  int checked = 0;
  for (float sum : {0.0f, 1.0f, 2.0f, 64.0f}) {
    for (float maximum : {-2000.0f, -1.0f, 0.0f, 1.0f, 2000.0f}) {
      for (float sink : {-INFINITY, -4000.0f, -1.0f, 0.0f, 1.0f, 4000.0f}) {
        const float actual = no_split_lse(sum, maximum, sink);
        if (sum == 0.0f && sink == -INFINITY) {
          assert(actual == INFINITY);
        } else {
          const double ln2 = std::log(2.0);
          const double kv =
              sum == 0.0f ? -INFINITY
                          : maximum * ln2 + std::log(static_cast<double>(sum));
          const double sink_ln = sink * ln2;
          const double high = std::fmax(kv, sink_ln);
          const double expected =
              high + std::log(std::exp(kv - high) + std::exp(sink_ln - high));
          assert(std::isfinite(actual));
          assert(std::abs(actual - expected) < 5e-4);
        }
        ++checked;
      }
    }
  }
  std::cout << checked << " sink/LSE cases passed\n";
}
