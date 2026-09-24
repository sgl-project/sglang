/// \file once.h
/// \brief Host-side "run it only once" helpers.
///
/// C++ spells this `std::call_once` plus a caller-provided `std::once_flag`. The helpers here
/// hide the flag inside the function, keyed off the uniqueness of the callable's type, and hand
/// back the value the callable produced. They come in two forms: one value per call-site, or
/// one value per (call-site, key).

#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace sglang::host {

namespace details {

/// \brief Fold any number of sub-hashes into one; the mixing step is boost's.
/// Defined for zero parts too, so an empty tuple key hashes without a special case.
inline std::size_t hash_combine(std::same_as<std::size_t> auto... parts) {
  std::size_t seed = 0;
  ((seed ^= parts + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2)), ...);
  return seed;
}

/// \brief Hash for keys `std::hash` does not cover: pairs and tuples, recursively.
struct GenericHash {
  template <typename T>
  std::size_t operator()(const T& value) const {
    return std::hash<T>{}(value);
  }
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& value) const {
    return hash_combine((*this)(value.first), (*this)(value.second));
  }
  template <typename... Args>
  std::size_t operator()(const std::tuple<Args...>& value) const {
    return std::apply([this](const auto&... args) { return hash_combine((*this)(args)...); }, value);
  }
};

struct DummyMutex {
  void lock() {}
  void unlock() {}
};

}  // namespace details

template <typename Key, typename Value, bool kThreadSafe = true>
struct CacheMap {
 public:
  CacheMap() = default;
  CacheMap(const CacheMap&) = delete;
  CacheMap& operator=(const CacheMap&) = delete;

  template <typename Fn, typename KeyArg>
  auto get_cached(KeyArg&& key, Fn&& factory) -> Value& {
    const auto lock = std::lock_guard{m_mutex};
    if (const auto iter = m_map.find(key); iter != m_map.end()) return iter->second;
    // Evaluate the factory before inserting, so a throwing factory leaves no entry behind.
    auto value = std::forward<Fn>(factory)();
    return m_map.try_emplace(std::forward<KeyArg>(key), std::move(value)).first->second;
  }

 private:
  [[no_unique_address]]
  std::conditional_t<kThreadSafe, std::mutex, details::DummyMutex> m_mutex;
  std::unordered_map<Key, Value, details::GenericHash> m_map;
};

template <typename Key, bool kThreadSafe>
struct CacheMap<Key, void, kThreadSafe> {
 public:
  CacheMap() = default;
  CacheMap(const CacheMap&) = delete;
  CacheMap& operator=(const CacheMap&) = delete;

  template <typename Fn, typename KeyArg>
  void get_cached(KeyArg&& key, Fn&& factory) {
    const auto lock = std::lock_guard{m_mutex};
    if (m_set.contains(key)) return;
    // Run the factory before inserting, so a throwing factory is retried on the next call.
    std::forward<Fn>(factory)();
    m_set.emplace(std::forward<KeyArg>(key));
  }

 private:
  [[no_unique_address]]
  std::conditional_t<kThreadSafe, std::mutex, details::DummyMutex> m_mutex;
  std::unordered_set<Key, details::GenericHash> m_set;
};

template <typename Key, bool kThreadSafe = true>
using CachedSet = CacheMap<Key, void, kThreadSafe>;

/**
 * \brief A keyed `std::call_once` that also hands back the value.
 * Runs `factory` at most once per (call-site, key) and returns a stable reference to its result.
 * \tparam Tag A tag type to avoid cache collision.
 * \tparam kThreadSafe Whether to make the initialization thread-safe. Turn it off only where the
 * call-site is provably single-threaded: the lock is taken on every call, cache hits included.
 * \param key The key to identify the value. Each distinct key is initialized independently.
 * \param factory The function to produce the value on first use. It should return the value.
 * \note The `Fn` type must be unique. It's typically a lambda type that's evaluated only once.
 * Otherwise, different call-sites may hit the same cache entry.
 * In case where `Fn` is not unique (e.g. std::function), make `Tag` unique to avoid cache collision.
 * \note `factory` runs with the lock held, so it must not re-enter this same instantiation.
 */
template <typename Tag = void, bool kThreadSafe = true, typename Key, std::invocable Fn>
inline auto init_once(Key&& key, Fn&& factory) -> decltype(auto) {
  static CacheMap<std::decay_t<Key>, std::decay_t<std::invoke_result_t<Fn>>, kThreadSafe> s_cache;
  return s_cache.get_cached(std::forward<Key>(key), std::forward<Fn>(factory));
}

/**
 * \brief `std::call_once` without the caller-provided `once_flag`, also handing back the value.
 * Runs `factory` at most once per call-site and returns a stable reference to its result.
 * \tparam Tag A tag type to avoid cache collision.
 * \param factory The function to produce the value on first use. It should return the value.
 * \note The `Fn` type must be unique, for the same reason as the keyed overload above: the static
 * lives in the instantiation, so a non-unique `Fn` (e.g. std::function) makes two call-sites share
 * one value. Make `Tag` unique in that case.
 * \note Always thread-safe, and deliberately has no `kThreadSafe` knob: C++ already guarantees that
 * concurrent callers block until the first one finishes, and the post-initialization fast path is
 * just a guard check, so there is no lock worth skipping.
 * \note If `factory` throws, the value stays uninitialized and the next call retries, matching the
 * keyed overload. `factory` must not re-enter this same instantiation; that is undefined behavior.
 */
template <typename Tag = void, std::invocable Fn>
inline auto init_once(Fn&& factory) -> decltype(auto) {
  if constexpr (std::is_same_v<std::invoke_result_t<Fn>, void>) {
    static std::once_flag s_once;
    return std::call_once(s_once, std::forward<Fn>(factory));
  } else {
    static auto result = std::forward<Fn>(factory)();
    return (result);  // parenthesized: `decltype(auto)` then deduces a reference, not a copy
  }
}

}  // namespace sglang::host
