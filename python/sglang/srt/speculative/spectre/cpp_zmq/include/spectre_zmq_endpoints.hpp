#pragma once

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <msgpack.hpp>
#include <zmq.hpp>

#include "spectre_zmq_logging.hpp"

template <typename T> class BatchVectorPool {
public:
  using Batch = std::vector<T>;

  struct Reclaimer {
    BatchVectorPool *pool = nullptr;

    void operator()(Batch *batch) const {
      if (batch == nullptr) {
        return;
      }
      if (pool == nullptr) {
        delete batch;
        return;
      }
      pool->recycle(batch);
    }
  };

  using Handle = std::unique_ptr<Batch, Reclaimer>;

  explicit BatchVectorPool(size_t max_cached = 128) : max_cached_(max_cached) {}

  Handle acquire() {
    std::unique_ptr<Batch> batch;
    {
      std::lock_guard<std::mutex> lock(pool_mtx_);
      if (!free_list_.empty()) {
        batch = std::move(free_list_.back());
        free_list_.pop_back();
      }
    }
    if (!batch) {
      batch = std::make_unique<Batch>();
    }
    batch->clear();
    return Handle(batch.release(), Reclaimer{this});
  }

private:
  void recycle(Batch *batch) {
    batch->clear();
    std::unique_ptr<Batch> recycled(batch);
    std::lock_guard<std::mutex> lock(pool_mtx_);
    if (free_list_.size() >= max_cached_) {
      return;
    }
    free_list_.push_back(std::move(recycled));
  }

  size_t max_cached_;
  std::mutex pool_mtx_;
  std::vector<std::unique_ptr<Batch>> free_list_;
};

struct DealerRawBuffer {
  zmq::message_t payload;
};

struct RouterRawBuffer {
  std::string id;
  zmq::message_t payload;
};

inline std::string increment_endpoint_port(const std::string &addr, int delta) {
  if (addr.rfind("ipc://", 0) == 0) {
    if (delta == 1) {
      return addr + ".tx";
    }
    if (delta == 2) {
      return addr + ".ctrl";
    }
    throw std::invalid_argument("invalid ipc zmq addr delta");
  }

  auto pos = addr.rfind(':');
  if (pos == std::string::npos || pos + 1 >= addr.size()) {
    throw std::invalid_argument("invalid zmq addr, expected ...:port");
  }
  int port = std::stoi(addr.substr(pos + 1));
  return addr.substr(0, pos + 1) + std::to_string(port + delta);
}

inline bool is_heartbeat_payload(const zmq::message_t &payload) {
  constexpr size_t heartbeat_size = sizeof("DEALER_HEARTBEAT") - 1;
  return payload.size() == heartbeat_size &&
         std::memcmp(payload.data(), "DEALER_HEARTBEAT", heartbeat_size) == 0;
}

template <typename T> class SafeQueue {
public:
  explicit SafeQueue(size_t capacity = 65536)
      : capacity_(capacity), mask_(capacity - 1), head_(0), tail_(0), size_(0),
        wake_epoch_(0) {
    if ((capacity & (capacity - 1)) != 0) {
      capacity_ = 65536;
      mask_ = capacity_ - 1;
    }
    buffer_.resize(capacity_);
  }

  void push(T &&item) {
    uint64_t wait_epoch = wake_epoch_.load(std::memory_order_acquire);
    std::unique_lock<std::mutex> lock(queue_mtx_);

    not_full_cv_.wait(lock, [this, wait_epoch] {
      return size_ < capacity_ ||
             wake_epoch_.load(std::memory_order_acquire) != wait_epoch;
    });
    if (size_ == capacity_) {
      return;
    }

    buffer_[tail_] = std::move(item);
    tail_ = (tail_ + 1) & mask_;
    ++size_;
    lock.unlock();

    not_empty_cv_.notify_one();
  }

  bool pop(T &item, int timeout_ms = 1) {
    uint64_t wait_epoch = wake_epoch_.load(std::memory_order_acquire);
    std::unique_lock<std::mutex> lock(queue_mtx_);

    if (size_ == 0) {
      if (!not_empty_cv_.wait_for(
              lock, std::chrono::milliseconds(timeout_ms), [this, wait_epoch] {
                return size_ > 0 ||
                       wake_epoch_.load(std::memory_order_acquire) !=
                           wait_epoch;
              })) {
        return false;
      }
      if (size_ == 0) {
        return false;
      }
    }

    item = std::move(buffer_[head_]);
    head_ = (head_ + 1) & mask_;
    --size_;
    lock.unlock();
    not_full_cv_.notify_one();
    return true;
  }

  void notify_all() {
    wake_epoch_.fetch_add(1, std::memory_order_acq_rel);
    std::lock_guard<std::mutex> lock(queue_mtx_);
    not_empty_cv_.notify_all();
    not_full_cv_.notify_all();
  }

  bool peek_and_check_timeout(long long timeout_ms) {
    std::lock_guard<std::mutex> lock(queue_mtx_);
    if (size_ == 0) {
      return false;
    }
    auto now = std::chrono::steady_clock::now();
    auto msg_time = buffer_[head_].timestamp;
    auto diff =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - msg_time)
            .count();
    return diff > timeout_ms;
  }

  std::vector<T> pop_all() {
    std::vector<T> res;
    std::unique_lock<std::mutex> lock(queue_mtx_);
    res.reserve(size_);
    while (size_ > 0) {
      res.push_back(std::move(buffer_[head_]));
      head_ = (head_ + 1) & mask_;
      --size_;
    }
    lock.unlock();
    not_full_cv_.notify_all();
    return res;
  }

  size_t size() {
    std::lock_guard<std::mutex> lock(queue_mtx_);
    return size_;
  }

  void clear() {
    std::unique_lock<std::mutex> lock(queue_mtx_);
    head_ = 0;
    tail_ = 0;
    size_ = 0;
    lock.unlock();
    not_full_cv_.notify_all();
  }

private:
  size_t capacity_;
  size_t mask_;
  std::vector<T> buffer_;
  size_t head_;
  size_t tail_;
  size_t size_;
  std::atomic<uint64_t> wake_epoch_;
  std::mutex queue_mtx_;
  std::condition_variable not_empty_cv_;
  std::condition_variable not_full_cv_;
};

template <typename DataT> struct TimestampedObj {
  DataT data;
  std::chrono::steady_clock::time_point timestamp;
};

enum class SocketChannel {
  Rx,
  Ctrl,
};

template <typename Derived, typename RawRecvT = std::string,
          typename RawSendT = std::string>
class [[gnu::visibility("default")]] AsyncZmqEndpointCRTP {
public:
  static constexpr const char *DEALER_HEARTBEAT = "DEALER_HEARTBEAT";
  static constexpr int ENDPOINT_MONITOR_INTERNAL_MS = 1000;
  static constexpr int DEALER_HEARTBEAT_TIMEOUT_MS = 3000;
  static constexpr int PYTHON_READY_QUEUE_TIMEOUT_MS = 30000;

  AsyncZmqEndpointCRTP(const std::string &log_addr, const std::string &tx_addr,
                       const std::string &rx_addr, const std::string &ctrl_addr,
                       bool bind, zmq::socket_type sock_type)
      : internal_ctx_(1), tx_sock_(internal_ctx_, sock_type),
        rx_sock_(internal_ctx_, sock_type),
        ctrl_sock_(internal_ctx_, sock_type), log_addr_(log_addr),
        tx_addr_(tx_addr), rx_addr_(rx_addr), ctrl_addr_(ctrl_addr),
        bind_(bind), running_(false), sock_type_tag(sock_type) {
    endpoint_type = bind_ ? "Router" : "Dealer";
  }

  virtual ~AsyncZmqEndpointCRTP() { stop(); }

  void start() {
    if (running_) {
      return;
    }
    static_cast<Derived *>(this)->do_setup_tx_socket();
    static_cast<Derived *>(this)->do_setup_rx_socket();
    static_cast<Derived *>(this)->do_setup_ctrl_socket();
    do_bind_or_connect(tx_sock_, tx_addr_);
    do_bind_or_connect(rx_sock_, rx_addr_);
    do_bind_or_connect(ctrl_sock_, ctrl_addr_);
    running_ = true;

    zmq_io_tx_thread_ =
        std::thread(&AsyncZmqEndpointCRTP::zmq_io_tx_loop, this);
    zmq_io_rx_thread_ =
        std::thread(&AsyncZmqEndpointCRTP::zmq_io_rx_loop, this);
    zmq_io_ctrl_thread_ =
        std::thread(&AsyncZmqEndpointCRTP::zmq_io_ctrl_loop, this);
    zmq_data_unpack_thread_ =
        std::thread(&AsyncZmqEndpointCRTP::zmq_data_unpack_loop, this);
    endpoint_monitor_thread_ =
        std::thread(&AsyncZmqEndpointCRTP::endpoint_monitor_loop, this);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    spectre_info_log("[ZMQ ", endpoint_type, " Start!] addr:", log_addr_);
  }

  void stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
      return;
    }

    try {
      recv_raw_queue_.notify_all();
      send_raw_queue_.notify_all();
      internal_ctx_.shutdown();
      if (zmq_io_tx_thread_.joinable()) {
        zmq_io_tx_thread_.join();
      }
      if (zmq_io_rx_thread_.joinable()) {
        zmq_io_rx_thread_.join();
      }
      if (zmq_io_ctrl_thread_.joinable()) {
        zmq_io_ctrl_thread_.join();
      }
      if (zmq_data_unpack_thread_.joinable()) {
        zmq_data_unpack_thread_.join();
      }
      if (endpoint_monitor_thread_.joinable()) {
        endpoint_monitor_thread_.join();
      }
      size_t dropped_send = send_raw_queue_.size();
      if (dropped_send != 0) {
        send_raw_queue_.clear();
        record_shutdown_drop("send_raw_queue", dropped_send);
      }
      size_t dropped_recv = recv_raw_queue_.size();
      if (dropped_recv != 0) {
        recv_raw_queue_.clear();
        record_shutdown_drop("recv_raw_queue", dropped_recv);
      }
      static_cast<Derived *>(this)->discard_pending_on_shutdown();
      tx_sock_.close();
      rx_sock_.close();
      ctrl_sock_.close();
      internal_ctx_.close();
    } catch (...) {
    }

    log_endpoint_counters();
    spectre_info_log("[ZMQ ", endpoint_type, " Stop!] addr:", log_addr_);
  }

  void enqueue_send_raw(RawSendT &&obj) {
    if (!running_.load(std::memory_order_acquire)) {
      record_send_after_stop_drop();
      return;
    }
    send_raw_queue_.push(std::move(obj));
  }

  void do_bind_or_connect(zmq::socket_t &sock, const std::string &addr) {
    if (bind_) {
      sock.bind(addr);
    } else {
      sock.connect(addr);
    }
  }

protected:
  void record_malformed_frames(const char *channel, size_t frame_count) {
    auto total =
        malformed_event_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    spectre_warn_log("[ZMQ Warn] ", endpoint_type, " malformed ", channel,
                     " frames dropped on ", log_addr_,
                     ": frame_count=", frame_count, " total=", total);
  }

  void record_unexpected_ctrl_payload(const std::string &id,
                                      size_t payload_size) {
    auto total =
        malformed_event_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    spectre_warn_log("[ZMQ Warn] ", endpoint_type,
                     " unexpected ctrl payload dropped on ", log_addr_,
                     ": id=", id, " bytes=", payload_size, " total=", total);
  }

  void record_incomplete_payload(const char *source, size_t payload_size) {
    auto total =
        incomplete_payload_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    spectre_warn_log("[ZMQ Warn] ", endpoint_type, " incomplete ", source,
                     " payload dropped on ", log_addr_,
                     ": bytes=", payload_size, " total=", total);
  }

  void record_unpack_failure(const char *source, size_t payload_size,
                             const std::string &detail) {
    auto total =
        unpack_failure_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    spectre_warn_log("[ZMQ Warn] ", endpoint_type, " failed to unpack ", source,
                     " payload on ", log_addr_, ": bytes=", payload_size,
                     " detail=", detail, " total=", total);
  }

  void record_timeout_discard(size_t batch_count, size_t request_count) {
    auto total_batches = timeout_discard_batch_count_.fetch_add(
                             batch_count, std::memory_order_relaxed) +
                         batch_count;
    auto total_requests = timeout_discard_request_count_.fetch_add(
                              request_count, std::memory_order_relaxed) +
                          request_count;
    spectre_warn_log(
        "[ZMQ Warn] ", endpoint_type,
        " dropped timed out python_ready_queue items on ", log_addr_,
        ": batches=", batch_count, " requests=", request_count,
        " total_batches=", total_batches, " total_requests=", total_requests);
  }

  void record_shutdown_drop(const char *queue_name, size_t item_count,
                            size_t request_count = 0) {
    auto total_items = shutdown_drop_item_count_.fetch_add(
                           item_count, std::memory_order_relaxed) +
                       item_count;
    auto total_requests = shutdown_drop_request_count_.fetch_add(
                              request_count, std::memory_order_relaxed) +
                          request_count;
    spectre_warn_log("[ZMQ Warn] ", endpoint_type, " dropped pending ",
                     queue_name, " items during shutdown on ", log_addr_,
                     ": items=", item_count, " requests=", request_count,
                     " total_items=", total_items,
                     " total_requests=", total_requests);
  }

  void record_send_after_stop_drop() {
    auto total =
        send_after_stop_drop_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    spectre_warn_log("[ZMQ Warn] ", endpoint_type,
                     " dropped send after stop on ", log_addr_,
                     ": total=", total);
  }

  void log_endpoint_counters() {
    spectre_info_log(
        "[ZMQ Stats] ", endpoint_type, " addr=", log_addr_,
        " malformed_events=",
        malformed_event_count_.load(std::memory_order_relaxed),
        " incomplete_payloads=",
        incomplete_payload_count_.load(std::memory_order_relaxed),
        " unpack_failures=",
        unpack_failure_count_.load(std::memory_order_relaxed),
        " timeout_discarded_batches=",
        timeout_discard_batch_count_.load(std::memory_order_relaxed),
        " timeout_discarded_requests=",
        timeout_discard_request_count_.load(std::memory_order_relaxed),
        " shutdown_dropped_items=",
        shutdown_drop_item_count_.load(std::memory_order_relaxed),
        " shutdown_dropped_requests=",
        shutdown_drop_request_count_.load(std::memory_order_relaxed),
        " send_after_stop_drops=",
        send_after_stop_drop_count_.load(std::memory_order_relaxed));
  }

  void setup_common_socket_options(zmq::socket_t &sock) {
    sock.set(zmq::sockopt::sndhwm, 10000);
    sock.set(zmq::sockopt::rcvhwm, 10000);
    sock.set(zmq::sockopt::linger, 0);
    sock.set(zmq::sockopt::immediate, 0);
    sock.set(zmq::sockopt::sndbuf, 10 * 1024 * 1024);
    sock.set(zmq::sockopt::rcvbuf, 10 * 1024 * 1024);
  }

  void zmq_io_tx_loop() {
    while (running_) {
      try {
        RawSendT to_send;
        if (!send_raw_queue_.pop(to_send, 100)) {
          continue;
        }
        static_cast<Derived *>(this)->handle_tx_send(to_send);
        while (send_raw_queue_.pop(to_send, 0)) {
          static_cast<Derived *>(this)->handle_tx_send(to_send);
        }
      } catch (const zmq::error_t &e) {
        if (e.num() == ETERM || !running_) {
          break;
        }
        handle_tx_error(e);
      }
    }
  }

  void zmq_io_rx_loop() {
    while (running_) {
      try {
        zmq_pollitem_t items[] = {
            {static_cast<void *>(rx_sock_), 0, ZMQ_POLLIN, 0},
        };
        int rc = zmq_poll(items, 1, 100);
        if (rc > 0 && (items[0].revents & ZMQ_POLLIN)) {
          handle_raw_recv_base(rx_sock_, SocketChannel::Rx);
        }
      } catch (const zmq::error_t &e) {
        if (e.num() == ETERM || !running_) {
          break;
        }
        handle_rx_error(e);
      }
    }
  }

  void zmq_io_ctrl_loop() {
    while (running_) {
      try {
        zmq_pollitem_t items[] = {
            {static_cast<void *>(ctrl_sock_), 0, ZMQ_POLLIN, 0},
        };
        int rc = zmq_poll(items, 1, 100);
        if (rc > 0 && (items[0].revents & ZMQ_POLLIN)) {
          handle_raw_recv_base(ctrl_sock_, SocketChannel::Ctrl);
        }
        static_cast<Derived *>(this)->on_ctrl_loop_tick();
      } catch (const zmq::error_t &e) {
        if (e.num() == ETERM || !running_) {
          break;
        }
        handle_ctrl_error(e);
      }
    }
  }

  void zmq_data_unpack_loop() {
    const size_t batch_size = 32;
    std::vector<RawRecvT> batch_raw;
    batch_raw.reserve(batch_size);
    while (running_) {
      RawRecvT raw_data;
      if (recv_raw_queue_.pop(raw_data, 1)) {
        batch_raw.push_back(std::move(raw_data));
      }

      if (!batch_raw.empty() &&
          (batch_raw.size() >= batch_size || recv_raw_queue_.size() == 0)) {
        for (auto &item : batch_raw) {
          static_cast<Derived *>(this)->process_incoming_data(std::move(item));
        }
        batch_raw.clear();
      }
    }
    if (!batch_raw.empty()) {
      record_shutdown_drop("data_unpack_batch", batch_raw.size());
    }
  }

  void endpoint_monitor_loop() {
    while (running_) {
      static_cast<Derived *>(this)->on_monitor_tick();
      std::this_thread::sleep_for(
          std::chrono::milliseconds(ENDPOINT_MONITOR_INTERNAL_MS));
    }
  }

  void handle_raw_recv_base(zmq::socket_t &sock, SocketChannel channel) {
    int max_batch = 100;
    bool should_stop_recv = false;
    while (max_batch-- > 0 && !should_stop_recv) {
      std::vector<zmq::message_t> frames;
      size_t total_bytes = 0;
      auto recv_start = std::chrono::steady_clock::now();
      while (true) {
        zmq::message_t msg;
        auto res = sock.recv(msg, zmq::recv_flags::dontwait);
        if (!res) {
          should_stop_recv = true;
          break;
        }
        total_bytes += msg.size();
        frames.push_back(std::move(msg));
        if (!sock.get(zmq::sockopt::rcvmore)) {
          break;
        }
      }
      if (!frames.empty()) {
        auto recv_end = std::chrono::steady_clock::now();
        spectre_debug_log(
            "[ZMQ LOG C++][RECV] bytes=", total_bytes,
            " time_us=", spectre_duration_us(recv_start, recv_end));
        if (channel == SocketChannel::Rx) {
          static_cast<Derived *>(this)->dispatch_rx_frames(std::move(frames));
        } else {
          static_cast<Derived *>(this)->dispatch_ctrl_frames(std::move(frames));
        }
      }
    }
  }

  void handle_tx_error(const zmq::error_t &e) {
    if (running_) {
      spectre_warn_log("[ZMQ Error] ", log_addr_, ": ", e.what());
      auto err = e.num();
      if (err == ECONNRESET || err == ECONNREFUSED || err == ENETDOWN ||
          err == ENETUNREACH || err == EHOSTUNREACH || err == ETIMEDOUT) {
        reconnect_tx_socket();
      }
    }
  }

  void handle_rx_error(const zmq::error_t &e) {
    if (running_) {
      spectre_warn_log("[ZMQ Error] ", log_addr_, ": ", e.what());
      auto err = e.num();
      if (err == ECONNRESET || err == ECONNREFUSED || err == ENETDOWN ||
          err == ENETUNREACH || err == EHOSTUNREACH || err == ETIMEDOUT) {
        reconnect_rx_socket();
      }
    }
  }

  void handle_ctrl_error(const zmq::error_t &e) {
    if (running_) {
      spectre_warn_log("[ZMQ Error] ", log_addr_, ": ", e.what());
      auto err = e.num();
      if (err == ECONNRESET || err == ECONNREFUSED || err == ENETDOWN ||
          err == ENETUNREACH || err == EHOSTUNREACH || err == ETIMEDOUT) {
        reconnect_ctrl_socket();
      }
    }
  }

  void reconnect_tx_socket() {
    spectre_warn_log("[ZMQ Reconnecting] ", log_addr_);
    tx_sock_.close();
    tx_sock_ = zmq::socket_t(internal_ctx_, sock_type_tag);
    static_cast<Derived *>(this)->do_setup_tx_socket();
    do_bind_or_connect(tx_sock_, tx_addr_);
    static_cast<Derived *>(this)->on_tx_reconnect();
  }

  void reconnect_rx_socket() {
    spectre_warn_log("[ZMQ Reconnecting] ", log_addr_);
    rx_sock_.close();
    rx_sock_ = zmq::socket_t(internal_ctx_, sock_type_tag);
    static_cast<Derived *>(this)->do_setup_rx_socket();
    do_bind_or_connect(rx_sock_, rx_addr_);
    static_cast<Derived *>(this)->on_rx_reconnect();
  }

  void reconnect_ctrl_socket() {
    spectre_warn_log("[ZMQ Reconnecting] ", log_addr_);
    ctrl_sock_.close();
    ctrl_sock_ = zmq::socket_t(internal_ctx_, sock_type_tag);
    static_cast<Derived *>(this)->do_setup_ctrl_socket();
    do_bind_or_connect(ctrl_sock_, ctrl_addr_);
    static_cast<Derived *>(this)->on_ctrl_reconnect();
  }

  std::string endpoint_type;
  zmq::context_t internal_ctx_;
  zmq::socket_t tx_sock_;
  zmq::socket_t rx_sock_;
  zmq::socket_t ctrl_sock_;
  std::string log_addr_;
  std::string tx_addr_;
  std::string rx_addr_;
  std::string ctrl_addr_;
  bool bind_;
  std::atomic<bool> running_;
  zmq::socket_type sock_type_tag;
  std::thread zmq_io_tx_thread_;
  std::thread zmq_io_rx_thread_;
  std::thread zmq_io_ctrl_thread_;
  std::thread zmq_data_unpack_thread_;
  std::thread endpoint_monitor_thread_;
  SafeQueue<RawRecvT> recv_raw_queue_;
  SafeQueue<RawSendT> send_raw_queue_;
  std::atomic<uint64_t> malformed_event_count_{0};
  std::atomic<uint64_t> timeout_discard_batch_count_{0};
  std::atomic<uint64_t> timeout_discard_request_count_{0};
  std::atomic<uint64_t> unpack_failure_count_{0};
  std::atomic<uint64_t> incomplete_payload_count_{0};
  std::atomic<uint64_t> shutdown_drop_item_count_{0};
  std::atomic<uint64_t> shutdown_drop_request_count_{0};
  std::atomic<uint64_t> send_after_stop_drop_count_{0};
};

class [[gnu::visibility("default")]] DealerEndpoint
    : public AsyncZmqEndpointCRTP<DealerEndpoint, DealerRawBuffer,
                                  std::string> {
public:
  using Base =
      AsyncZmqEndpointCRTP<DealerEndpoint, DealerRawBuffer, std::string>;
  using RequestBatchHandle =
      typename BatchVectorPool<spectre::SpectreRequest>::Handle;

  DealerEndpoint(const std::string &addr, const std::string &identity,
                 bool bind = false);

  void do_setup_tx_socket();
  void do_setup_rx_socket();
  void do_setup_ctrl_socket();
  void on_tx_reconnect();
  void on_rx_reconnect();
  void on_ctrl_reconnect();
  void on_ctrl_loop_tick();
  void send_heartbeat();
  void send_objs(const std::vector<spectre::SpectreRequest> &reqs);
  void send_objs(std::vector<spectre::SpectreRequest> &&reqs);
  std::vector<spectre::SpectreRequest> get_received_objs();
  void on_monitor_tick();
  void dispatch_rx_frames(std::vector<zmq::message_t> &&frames);
  void dispatch_ctrl_frames(std::vector<zmq::message_t> &&frames);
  void handle_tx_send(std::string &data);
  void process_incoming_data(DealerRawBuffer &&raw);
  void discard_pending_on_shutdown();

private:
  std::string identity_;
  std::chrono::steady_clock::time_point last_heartbeat_send_time_ =
      std::chrono::steady_clock::time_point::min();
  msgpack::unpacker batch_unpacker_;
  BatchVectorPool<spectre::SpectreRequest> request_batch_pool_;
  SafeQueue<TimestampedObj<RequestBatchHandle>> python_ready_queue_;
};

class [[gnu::visibility("default")]] RouterEndpoint
    : public AsyncZmqEndpointCRTP<RouterEndpoint, RouterRawBuffer,
                                  std::pair<std::string, std::string>> {
public:
  using Base = AsyncZmqEndpointCRTP<RouterEndpoint, RouterRawBuffer,
                                    std::pair<std::string, std::string>>;
  using RequestBatchHandle =
      typename BatchVectorPool<spectre::SpectreRequest>::Handle;

  explicit RouterEndpoint(const std::string &addr, bool bind = true);

  void do_setup_tx_socket();
  void do_setup_rx_socket();
  void do_setup_ctrl_socket();
  void on_tx_reconnect();
  void on_rx_reconnect();
  void on_ctrl_reconnect();
  void on_ctrl_loop_tick();
  void send_objs(const std::string &id,
                 const std::vector<spectre::SpectreRequest> &reqs);
  void send_objs(const std::string &id,
                 std::vector<spectre::SpectreRequest> &&reqs);
  std::vector<std::pair<std::string, spectre::SpectreRequest>>
  get_received_objs();
  void on_monitor_tick();
  void dispatch_rx_frames(std::vector<zmq::message_t> &&frames);
  void dispatch_ctrl_frames(std::vector<zmq::message_t> &&frames);
  void handle_tx_send(std::pair<std::string, std::string> &data);
  void process_incoming_data(RouterRawBuffer &&raw);
  std::vector<std::string> get_all_dealers();
  void discard_pending_on_shutdown();

private:
  std::mutex reg_mtx_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      registered_dealers_;
  msgpack::unpacker batch_unpacker_;
  BatchVectorPool<spectre::SpectreRequest> request_batch_pool_;
  SafeQueue<TimestampedObj<std::pair<std::string, RequestBatchHandle>>>
      python_ready_queue_;
};
