#include <cassert>
#include <fstream>
#include <vector>

#include "fuzzer-common.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::ifstream in(argv[i]);
    assert(in);
    in.seekg(0, std::ios_base::end);
    const auto size = in.tellg();
    assert(size >= 0);
    in.seekg(0, std::ios_base::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    in.read(buf.data(), size);
    assert(in.gcount() == size);
    LLVMFuzzerTestOneInput(as_bytes(buf.data()), buf.size());
  }
}
