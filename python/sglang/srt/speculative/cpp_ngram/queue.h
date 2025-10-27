#pragma once

#include <condition_variable>
#include <queue>

namespace utils {

template <typename T>
class Queue {
 public:
  bool enqueue(T&& rhs) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return false;
      }
      queue_.emplace(std::move(rhs));
    }
    cv_.notify_one();
    return true;
  }

  bool enqueue(const T& rhs) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return false;
      }
      queue_.emplace(rhs);
    }
    cv_.notify_one();
    return true;
  }

  bool dequeue(T& rhs) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return queue_.size() || closed_; });
    if (closed_) {
      return false;
    }
    rhs = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool closed_{false};
};

}  // namespace utils
