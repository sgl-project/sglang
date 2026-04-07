#include "spectre_zmq_endpoints.hpp"

#include <utility>

#include "spectre_zmq_serialization.hpp"

namespace {

bool has_empty_delimiter(const std::vector<zmq::message_t> &frames,
                         size_t delimiter_index) {
  return frames.size() > delimiter_index && frames[delimiter_index].size() == 0;
}

} // namespace

DealerEndpoint::DealerEndpoint(const std::string &addr,
                               const std::string &identity, bool bind)
    : Base(addr, increment_endpoint_port(addr, 1), addr,
           increment_endpoint_port(addr, 2), bind, zmq::socket_type::dealer),
      identity_(identity) {}

void DealerEndpoint::do_setup_tx_socket() {
  this->tx_sock_.set(zmq::sockopt::routing_id, identity_);
  this->setup_common_socket_options(this->tx_sock_);
}

void DealerEndpoint::do_setup_rx_socket() {
  this->rx_sock_.set(zmq::sockopt::routing_id, identity_);
  this->setup_common_socket_options(this->rx_sock_);
}

void DealerEndpoint::do_setup_ctrl_socket() {
  this->ctrl_sock_.set(zmq::sockopt::routing_id, identity_);
  this->setup_common_socket_options(this->ctrl_sock_);
}

void DealerEndpoint::on_tx_reconnect() {}

void DealerEndpoint::on_rx_reconnect() {}

void DealerEndpoint::on_ctrl_reconnect() {
  last_heartbeat_send_time_ = std::chrono::steady_clock::time_point::min();
  send_heartbeat();
}

void DealerEndpoint::on_ctrl_loop_tick() {
  auto now = std::chrono::steady_clock::now();
  if (last_heartbeat_send_time_ ==
          std::chrono::steady_clock::time_point::min() ||
      std::chrono::duration_cast<std::chrono::milliseconds>(
          now - last_heartbeat_send_time_)
              .count() >= Base::ENDPOINT_MONITOR_INTERNAL_MS) {
    send_heartbeat();
  }
}

void DealerEndpoint::send_heartbeat() {
  this->ctrl_sock_.send(
      zmq::buffer(Base::DEALER_HEARTBEAT, std::strlen(Base::DEALER_HEARTBEAT)),
      zmq::send_flags::none);
  last_heartbeat_send_time_ = std::chrono::steady_clock::now();
  if (spectre_debug_enabled()) {
    spectre_debug_log("[ZMQ LOG C++] ", identity_, " sending heartbeat to ",
                      this->log_addr_);
  }
}

void DealerEndpoint::send_objs(
    const std::vector<spectre::SpectreRequest> &reqs) {
  auto copied_reqs = reqs;
  send_objs(std::move(copied_reqs));
}

void DealerEndpoint::send_objs(std::vector<spectre::SpectreRequest> &&reqs) {
  stamp_request_batch(reqs, &spectre::SpectreRequest::draft_send_time);
  auto pack_start = std::chrono::steady_clock::now();
  auto packed = pack_spectre_batch_payload(reqs);
  auto pack_end = std::chrono::steady_clock::now();
  spectre_debug_log("[ZMQ LOG C++][PACK] nums=", reqs.size(),
                    " bytes=", packed.size(),
                    " time_us=", spectre_duration_us(pack_start, pack_end));
  this->enqueue_send_raw(std::move(packed));
}

std::vector<spectre::SpectreRequest> DealerEndpoint::get_received_objs() {
  auto wrapped_list = python_ready_queue_.pop_all();
  size_t total_size = 0;
  for (auto &wrapped : wrapped_list) {
    total_size += wrapped.data->size();
  }

  std::vector<spectre::SpectreRequest> out;
  out.reserve(total_size);
  for (auto &wrapped : wrapped_list) {
    for (auto &obj : *wrapped.data) {
      out.push_back(std::move(obj));
    }
  }
  return out;
}

void DealerEndpoint::on_monitor_tick() {
  size_t discarded_batches = 0;
  size_t discarded_requests = 0;
  if (python_ready_queue_.peek_and_check_timeout(
          Base::PYTHON_READY_QUEUE_TIMEOUT_MS)) {
    while (python_ready_queue_.peek_and_check_timeout(
        Base::PYTHON_READY_QUEUE_TIMEOUT_MS)) {
      TimestampedObj<RequestBatchHandle> discarded;
      if (python_ready_queue_.pop(discarded, 0)) {
        ++discarded_batches;
        discarded_requests += discarded.data->size();
      }
    }
  }
  if (discarded_batches != 0) {
    this->record_timeout_discard(discarded_batches, discarded_requests);
  }
}

void DealerEndpoint::dispatch_rx_frames(std::vector<zmq::message_t> &&frames) {
  if (frames.size() == 2 && has_empty_delimiter(frames, 0)) {
    frames.erase(frames.begin());
  }

  if (frames.size() != 1) {
    this->record_malformed_frames("dealer-rx", frames.size());
    return;
  }

  DealerRawBuffer raw;
  raw.payload = std::move(frames.back());
  this->recv_raw_queue_.push(std::move(raw));
}

void DealerEndpoint::dispatch_ctrl_frames(
    std::vector<zmq::message_t> &&frames) {
  if (!frames.empty()) {
    this->record_malformed_frames("dealer-ctrl", frames.size());
  }
}

void DealerEndpoint::handle_tx_send(std::string &data) {
  auto start = std::chrono::steady_clock::now();
  this->tx_sock_.send(zmq::buffer(data), zmq::send_flags::none);
  auto end = std::chrono::steady_clock::now();
  spectre_debug_log("[ZMQ LOG C++][SEND] bytes=", data.size(),
                    " time_us=", spectre_duration_us(start, end));
}

void DealerEndpoint::process_incoming_data(DealerRawBuffer &&raw) {
  try {
    auto reqs = request_batch_pool_.acquire();
    auto unpack_start = std::chrono::steady_clock::now();
    bool ok = unpack_spectre_batch_payload(batch_unpacker_, raw.payload.data(),
                                           raw.payload.size(), *reqs);
    auto unpack_end = std::chrono::steady_clock::now();
    if (!ok) {
      this->record_incomplete_payload("dealer", raw.payload.size());
      return;
    }

    spectre_debug_log("[ZMQ LOG C++][UNPACK] nums=", reqs->size(),
                      " bytes=", raw.payload.size(), " time_us=",
                      spectre_duration_us(unpack_start, unpack_end));
    stamp_request_batch(*reqs, &spectre::SpectreRequest::draft_recv_time);
    python_ready_queue_.push(
        {std::move(reqs), std::chrono::steady_clock::now()});
  } catch (const std::exception &e) {
    this->record_unpack_failure("dealer", raw.payload.size(), e.what());
  }
}

void DealerEndpoint::discard_pending_on_shutdown() {
  auto wrapped_list = python_ready_queue_.pop_all();
  if (wrapped_list.empty()) {
    return;
  }

  size_t dropped_requests = 0;
  for (auto &wrapped : wrapped_list) {
    dropped_requests += wrapped.data->size();
  }
  this->record_shutdown_drop("dealer_python_ready_queue", wrapped_list.size(),
                             dropped_requests);
}

RouterEndpoint::RouterEndpoint(const std::string &addr, bool bind)
    : Base(addr, addr, increment_endpoint_port(addr, 1),
           increment_endpoint_port(addr, 2), bind, zmq::socket_type::router) {}

void RouterEndpoint::do_setup_tx_socket() {
  this->setup_common_socket_options(this->tx_sock_);
}

void RouterEndpoint::do_setup_rx_socket() {
  this->setup_common_socket_options(this->rx_sock_);
}

void RouterEndpoint::do_setup_ctrl_socket() {
  this->setup_common_socket_options(this->ctrl_sock_);
}

void RouterEndpoint::on_tx_reconnect() {}

void RouterEndpoint::on_rx_reconnect() {}

void RouterEndpoint::on_ctrl_reconnect() {}

void RouterEndpoint::on_ctrl_loop_tick() {}

void RouterEndpoint::send_objs(
    const std::string &id, const std::vector<spectre::SpectreRequest> &reqs) {
  auto copied_reqs = reqs;
  send_objs(id, std::move(copied_reqs));
}

void RouterEndpoint::send_objs(const std::string &id,
                               std::vector<spectre::SpectreRequest> &&reqs) {
  stamp_request_batch(reqs, &spectre::SpectreRequest::target_send_time);
  auto pack_start = std::chrono::steady_clock::now();
  auto packed = pack_spectre_batch_payload(reqs);
  auto pack_end = std::chrono::steady_clock::now();
  spectre_debug_log("[ZMQ LOG C++][PACK] nums=", reqs.size(),
                    " bytes=", packed.size(),
                    " time_us=", spectre_duration_us(pack_start, pack_end));
  this->enqueue_send_raw({id, std::move(packed)});
}

std::vector<std::pair<std::string, spectre::SpectreRequest>>
RouterEndpoint::get_received_objs() {
  auto wrapped_list = python_ready_queue_.pop_all();
  size_t total_size = 0;
  for (auto &wrapped : wrapped_list) {
    total_size += wrapped.data.second->size();
  }

  std::vector<std::pair<std::string, spectre::SpectreRequest>> out;
  out.reserve(total_size);
  for (auto &wrapped : wrapped_list) {
    const std::string &id = wrapped.data.first;
    for (auto &obj : *wrapped.data.second) {
      out.push_back({id, std::move(obj)});
    }
  }
  return out;
}

void RouterEndpoint::on_monitor_tick() {
  size_t discarded_batches = 0;
  size_t discarded_requests = 0;
  if (python_ready_queue_.peek_and_check_timeout(
          Base::PYTHON_READY_QUEUE_TIMEOUT_MS)) {
    while (python_ready_queue_.peek_and_check_timeout(
        Base::PYTHON_READY_QUEUE_TIMEOUT_MS)) {
      TimestampedObj<std::pair<std::string, RequestBatchHandle>> discarded;
      if (python_ready_queue_.pop(discarded, 0)) {
        ++discarded_batches;
        discarded_requests += discarded.data.second->size();
      }
    }
  }
  if (discarded_batches != 0) {
    this->record_timeout_discard(discarded_batches, discarded_requests);
  }

  auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(reg_mtx_);
  auto it = registered_dealers_.begin();
  while (it != registered_dealers_.end()) {
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second)
            .count() > Base::DEALER_HEARTBEAT_TIMEOUT_MS) {
      spectre_warn_log("[ZMQ Warn] ", it->first,
                       " heartbeat timed out, removing!");
      it = registered_dealers_.erase(it);
    } else {
      ++it;
    }
  }
}

void RouterEndpoint::dispatch_rx_frames(std::vector<zmq::message_t> &&frames) {
  if (frames.size() == 3 && has_empty_delimiter(frames, 1)) {
    frames.erase(frames.begin() + 1);
  }

  if (frames.size() != 2) {
    this->record_malformed_frames("router-rx", frames.size());
    return;
  }

  std::string id(static_cast<char *>(frames[0].data()), frames[0].size());
  auto &payload = frames.back();

  RouterRawBuffer raw;
  raw.id = std::move(id);
  raw.payload = std::move(payload);
  this->recv_raw_queue_.push(std::move(raw));
}

void RouterEndpoint::dispatch_ctrl_frames(
    std::vector<zmq::message_t> &&frames) {
  if (frames.size() == 3 && has_empty_delimiter(frames, 1)) {
    frames.erase(frames.begin() + 1);
  }

  if (frames.size() != 2) {
    this->record_malformed_frames("router-ctrl", frames.size());
    return;
  }

  std::string id(static_cast<char *>(frames[0].data()), frames[0].size());
  auto &payload = frames.back();

  if (is_heartbeat_payload(payload)) {
    std::lock_guard<std::mutex> lk(reg_mtx_);
    registered_dealers_[id] = std::chrono::steady_clock::now();
    if (spectre_debug_enabled()) {
      spectre_debug_log("[ZMQ LOG C++] ", this->log_addr_,
                        " received heartbeat from ", id);
    }
    return;
  }

  this->record_unexpected_ctrl_payload(id, payload.size());
}

void RouterEndpoint::handle_tx_send(std::pair<std::string, std::string> &data) {
  auto start = std::chrono::steady_clock::now();
  this->tx_sock_.send(zmq::buffer(data.first), zmq::send_flags::sndmore);
  this->tx_sock_.send(zmq::buffer(data.second), zmq::send_flags::none);
  auto end = std::chrono::steady_clock::now();
  spectre_debug_log("[ZMQ LOG C++][SEND] bytes=", data.second.size(),
                    " time_us=", spectre_duration_us(start, end));
}

void RouterEndpoint::process_incoming_data(RouterRawBuffer &&raw) {
  try {
    auto reqs = request_batch_pool_.acquire();
    auto unpack_start = std::chrono::steady_clock::now();
    bool ok = unpack_spectre_batch_payload(batch_unpacker_, raw.payload.data(),
                                           raw.payload.size(), *reqs);
    auto unpack_end = std::chrono::steady_clock::now();
    if (!ok) {
      this->record_incomplete_payload("router", raw.payload.size());
      return;
    }

    spectre_debug_log("[ZMQ LOG C++][UNPACK] nums=", reqs->size(),
                      " bytes=", raw.payload.size(), " time_us=",
                      spectre_duration_us(unpack_start, unpack_end));
    stamp_request_batch(*reqs, &spectre::SpectreRequest::target_recv_time);
    python_ready_queue_.push({{std::move(raw.id), std::move(reqs)},
                              std::chrono::steady_clock::now()});
  } catch (const std::exception &e) {
    this->record_unpack_failure("router", raw.payload.size(), e.what());
  }
}

std::vector<std::string> RouterEndpoint::get_all_dealers() {
  std::lock_guard<std::mutex> lock(reg_mtx_);
  std::vector<std::string> res;
  for (auto &p : registered_dealers_) {
    res.push_back(p.first);
  }
  return res;
}

void RouterEndpoint::discard_pending_on_shutdown() {
  auto wrapped_list = python_ready_queue_.pop_all();
  if (!wrapped_list.empty()) {
    size_t dropped_requests = 0;
    for (auto &wrapped : wrapped_list) {
      dropped_requests += wrapped.data.second->size();
    }
    this->record_shutdown_drop("router_python_ready_queue", wrapped_list.size(),
                               dropped_requests);
  }

  std::lock_guard<std::mutex> lock(reg_mtx_);
  registered_dealers_.clear();
}
