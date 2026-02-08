#include <fmt/compile.h>

__attribute__((visibility("default"))) std::string foo() {
  return fmt::format(FMT_COMPILE("foo bar {}"), 4242);
}
