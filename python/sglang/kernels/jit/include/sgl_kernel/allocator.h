#pragma once

#include <concepts>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace sglang::host {

/**
 * \brief Allocate only once for a given function.
 * \tparam kThreadSafe Whether to make the allocation thread-safe. Default is true.
 * \tparam kSalt A salt value to avoid cache collision. Default is 0.
 * \param key The key to identify the allocation. It should be unique for each allocation.
 * \param callback The callback function to perform the allocation. It should return the allocated value.
 * \note The `Fn` type must be unique. It's typically a lambda type that's evaluated only once.
 * Otherwise, different call-sites may hit the same cache entry.
 * In case where `Fn` is not unique (e.g. std::function), make `kSalt` unique to avoid cache collision.
 * A typically salt value can be `__COUNTER__` or `__LINE__`.
 */
template <bool kThreadSafe = true, uint32_t kSalt = 0, typename Key, std::invocable Fn>
inline auto allocate_once(Key&& key, Fn&& callback) -> std::decay_t<std::invoke_result_t<Fn>>& {
  using Value = std::decay_t<std::invoke_result_t<Fn>>;
  static std::unordered_map<std::decay_t<Key>, Value> s_map;
  const auto alloc = [&]() -> Value& {
    const auto iter = s_map.find(key);
    if (iter != s_map.end()) return iter->second;
    // Evaluate the callback before inserting, so a throwing callback leaves no empty entry behind.
    auto value = std::forward<Fn>(callback)();
    return s_map.emplace(std::forward<Key>(key), std::move(value)).first->second;
  };
  if constexpr (kThreadSafe) {
    static std::mutex s_mutex;
    const auto lock = std::lock_guard{s_mutex};
    return alloc();
  } else {
    return alloc();
  }
}

}  // namespace sglang::host
