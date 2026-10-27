/*
 * sglang-disagg NIXL shim — implementation. See nixl_shim.h for the wrapped
 * NIXL C++ surface and the upstream sglang file references. Object-lifetime
 * patterns mirror the official Rust binding's C wrapper (ai-dynamo/nixl
 * src/bindings/rust/wrapper.cpp): new/delete per opaque handle,
 * malloc+memcpy for outgoing blobs, try/catch -> status.
 */
#include "nixl_shim.h"

#include "nixl.h"
#include "nixl_params.h"
#include "nixl_types.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

nixl_mem_t mem_of(int kind) {
  return kind == SGD_NIXL_MEM_VRAM ? VRAM_SEG : DRAM_SEG;
}

nixl_thread_sync_t sync_of(int mode) {
  switch (mode) {
  case SGD_NIXL_SYNC_STRICT:
    return nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
  case SGD_NIXL_SYNC_RW:
    return nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
  default:
    return nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE;
  }
}

nixl_reg_dlist_t reg_dlist(int mem_kind, const uint64_t *descs3, size_t n) {
  nixl_reg_dlist_t dlist(mem_of(mem_kind));
  for (size_t i = 0; i < n; i++) {
    dlist.addDesc(nixlBlobDesc(static_cast<uintptr_t>(descs3[3 * i]),
                               static_cast<size_t>(descs3[3 * i + 1]),
                               descs3[3 * i + 2], nixl_blob_t()));
  }
  return dlist;
}

nixl_xfer_dlist_t xfer_dlist(int mem_kind, const uint64_t *descs3, size_t n) {
  nixl_xfer_dlist_t dlist(mem_of(mem_kind));
  for (size_t i = 0; i < n; i++) {
    dlist.addDesc(nixlBasicDesc(static_cast<uintptr_t>(descs3[3 * i]),
                                static_cast<size_t>(descs3[3 * i + 1]),
                                descs3[3 * i + 2]));
  }
  return dlist;
}

nixl_opt_args_t notif_args(const uint8_t *notif, size_t notif_len) {
  nixl_opt_args_t args;
  if (notif != nullptr && notif_len > 0) {
    args.notifMsg.assign(reinterpret_cast<const char *>(notif), notif_len);
    args.notif = args.notifMsg;
    args.hasNotif = true;
  }
  return args;
}

/* sgd_nixl_post_write / sgd_nixl_make_xfer shared tail: post and, on a post
 * error, release before reporting (mirrors _post_xfer raising on a failed
 * post). */
int post_or_release(nixlAgent *agent, nixlXferReqH *req,
                    nixl_opt_args_t *args) {
  nixl_status_t st = agent->postXferReq(req, args);
  if (st != NIXL_SUCCESS && st != NIXL_IN_PROG) {
    agent->releaseXferReq(req);
    return static_cast<int>(st);
  }
  return static_cast<int>(st);
}

} // namespace

struct sgd_nixl_agent_s {
  nixlAgent *agent = nullptr;
  std::string name; /* nixlAgent exposes no name getter; kept for
                     * sgd_nixl_agent_name (nixl/conn.py:3472). */
  std::vector<nixlBackendH *> backends; /* owned by agent; never deleted */
  std::mutex mu;
  std::unordered_map<uint64_t, nixlXferReqH *> xfers;
  std::unordered_map<uint64_t, nixlDlistH *> dlists;
  uint64_t next_handle = 1;

  uint64_t track_xfer(nixlXferReqH *req) {
    std::lock_guard<std::mutex> lock(mu);
    uint64_t id = next_handle++;
    xfers.emplace(id, req);
    return id;
  }

  nixlXferReqH *take_xfer(uint64_t id, bool erase) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = xfers.find(id);
    if (it == xfers.end()) {
      return nullptr;
    }
    nixlXferReqH *req = it->second;
    if (erase) {
      xfers.erase(it);
    }
    return req;
  }

  nixlDlistH *lookup_dlist(uint64_t id, bool erase) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = dlists.find(id);
    if (it == dlists.end()) {
      return nullptr;
    }
    nixlDlistH *h = it->second;
    if (erase) {
      dlists.erase(it);
    }
    return h;
  }
};

extern "C" {

int sgd_nixl_agent_create(const char *name, uint64_t pthr_delay_us,
                          int sync_mode, sgd_nixl_agent_t *out) {
  if (name == nullptr || out == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  auto *handle = new (std::nothrow) sgd_nixl_agent_s;
  if (handle == nullptr) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
  try {
    /* nixl/conn.py:597-601: progress thread on, sync_mode STRICT (the
     * transfer worker and bootstrap thread share the agent; sync NONE
     * is thread-unsafe). */
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    cfg.syncMode = sync_of(sync_mode);
    if (pthr_delay_us > 0) {
      cfg.pthrDelay = pthr_delay_us;
    }
    handle->agent = new nixlAgent(std::string(name), cfg);
    handle->name = name;
    *out = handle;
    return NIXL_SUCCESS;
  } catch (...) {
    sgd_nixl_agent_destroy(handle);
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

void sgd_nixl_agent_destroy(sgd_nixl_agent_t agent) {
  if (agent == nullptr) {
    return;
  }
  delete agent->agent;
  delete agent;
}

int sgd_nixl_agent_name(sgd_nixl_agent_t agent, char **name_out) {
  if (agent == nullptr || name_out == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  char *copy = strdup(agent->name.c_str());
  if (copy == nullptr) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
  *name_out = copy;
  return NIXL_SUCCESS;
}

int sgd_nixl_create_backend(sgd_nixl_agent_t agent, const char *backend,
                            const char *const *keys, const char *const *vals,
                            size_t n) {
  if (agent == nullptr || backend == nullptr || (keys == nullptr && n > 0) ||
      (vals == nullptr && n > 0)) {
    return SGD_NIXL_ERR_INVALID;
  }
  try {
    nixl_b_params_t params;
    for (size_t i = 0; i < n; i++) {
      if (keys[i] == nullptr || vals[i] == nullptr) {
        return SGD_NIXL_ERR_INVALID;
      }
      params[keys[i]] = vals[i];
    }
    nixlBackendH *bknd = nullptr;
    nixl_status_t st =
        agent->agent->createBackend(std::string(backend), params, bknd);
    if (st != NIXL_SUCCESS) {
      return static_cast<int>(st);
    }
    agent->backends.push_back(bknd);
    return NIXL_SUCCESS;
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_get_avail_plugins(sgd_nixl_agent_t agent, char **csv_out) {
  if (agent == nullptr || csv_out == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  try {
    std::vector<nixl_backend_t> plugins;
    nixl_status_t st = agent->agent->getAvailPlugins(plugins);
    if (st != NIXL_SUCCESS) {
      return static_cast<int>(st);
    }
    std::string csv;
    for (size_t i = 0; i < plugins.size(); i++) {
      if (i > 0) {
        csv += ',';
      }
      csv += plugins[i];
    }
    char *copy = strdup(csv.c_str());
    if (copy == nullptr) {
      return SGD_NIXL_ERR_EXCEPTION;
    }
    *csv_out = copy;
    return NIXL_SUCCESS;
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_register_mem(sgd_nixl_agent_t agent, int mem_kind,
                          const uint64_t *descs3, size_t n) {
  if (agent == nullptr || (descs3 == nullptr && n > 0)) {
    return SGD_NIXL_ERR_INVALID;
  }
  try {
    nixl_reg_dlist_t dlist = reg_dlist(mem_kind, descs3, n);
    return static_cast<int>(agent->agent->registerMem(dlist, nullptr));
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_deregister_mem(sgd_nixl_agent_t agent, int mem_kind,
                            const uint64_t *descs3, size_t n) {
  if (agent == nullptr || (descs3 == nullptr && n > 0)) {
    return SGD_NIXL_ERR_INVALID;
  }
  try {
    nixl_reg_dlist_t dlist = reg_dlist(mem_kind, descs3, n);
    return static_cast<int>(agent->agent->deregisterMem(dlist, nullptr));
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_get_local_md(sgd_nixl_agent_t agent, uint8_t **data, size_t *len) {
  if (agent == nullptr || data == nullptr || len == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  try {
    nixl_blob_t blob;
    nixl_status_t st = agent->agent->getLocalMD(blob);
    if (st != NIXL_SUCCESS) {
      return static_cast<int>(st);
    }
    void *copy = malloc(blob.size() == 0 ? 1 : blob.size());
    if (copy == nullptr) {
      return SGD_NIXL_ERR_EXCEPTION;
    }
    memcpy(copy, blob.data(), blob.size());
    *data = static_cast<uint8_t *>(copy);
    *len = blob.size();
    return NIXL_SUCCESS;
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

void sgd_nixl_free(void *p) { free(p); }

int sgd_nixl_load_remote_md(sgd_nixl_agent_t agent, const uint8_t *data,
                            size_t len, char **name_out) {
  if (agent == nullptr || (data == nullptr && len > 0) || name_out == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  try {
    nixl_blob_t blob;
    blob.assign(reinterpret_cast<const char *>(data), len);
    std::string name;
    nixl_status_t st = agent->agent->loadRemoteMD(blob, name);
    if (st != NIXL_SUCCESS) {
      return static_cast<int>(st);
    }
    char *copy = strdup(name.c_str());
    if (copy == nullptr) {
      return SGD_NIXL_ERR_EXCEPTION;
    }
    *name_out = copy;
    return NIXL_SUCCESS;
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_remove_remote(sgd_nixl_agent_t agent, const char *name) {
  if (agent == nullptr || name == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  try {
    return static_cast<int>(
        agent->agent->invalidateRemoteMD(std::string(name)));
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_check_remote_md(sgd_nixl_agent_t agent, const char *name) {
  if (agent == nullptr || name == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  try {
    /* Empty dlist: whole-metadata validity, == Python
     * check_remote_metadata(name) with descs=None. */
    nixl_xfer_dlist_t empty(DRAM_SEG);
    return static_cast<int>(
        agent->agent->checkRemoteMD(std::string(name), empty));
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_make_connection(sgd_nixl_agent_t agent, const char *name) {
  if (agent == nullptr || name == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  try {
    return static_cast<int>(
        agent->agent->makeConnection(std::string(name), nullptr));
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_post_write(sgd_nixl_agent_t agent, int mem_kind,
                        const uint64_t *src3, const uint64_t *dst3, size_t n,
                        const char *peer, const uint8_t *notif,
                        size_t notif_len, uint64_t *handle_out) {
  if (agent == nullptr || (src3 == nullptr && n > 0) ||
      (dst3 == nullptr && n > 0) || peer == nullptr || handle_out == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  try {
    nixl_xfer_dlist_t local = xfer_dlist(mem_kind, src3, n);
    nixl_xfer_dlist_t remote = xfer_dlist(mem_kind, dst3, n);
    nixl_opt_args_t args = notif_args(notif, notif_len);
    nixlXferReqH *req = nullptr;
    nixl_status_t st = agent->agent->createXferReq(
        NIXL_WRITE, local, remote, std::string(peer), req, &args);
    if (st != NIXL_SUCCESS) {
      return static_cast<int>(st);
    }
    int rc = post_or_release(agent->agent, req, &args);
    if (rc < 0) {
      return rc;
    }
    *handle_out = agent->track_xfer(req);
    return rc;
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_xfer_status(sgd_nixl_agent_t agent, uint64_t handle) {
  if (agent == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  nixlXferReqH *req = agent->take_xfer(handle, false);
  if (req == nullptr) {
    return SGD_NIXL_ERR_LOOKUP;
  }
  try {
    return static_cast<int>(agent->agent->getXferStatus(req));
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_release_xfer(sgd_nixl_agent_t agent, uint64_t handle) {
  if (agent == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  nixlXferReqH *req = agent->take_xfer(handle, true);
  if (req == nullptr) {
    return SGD_NIXL_ERR_LOOKUP;
  }
  try {
    return static_cast<int>(agent->agent->releaseXferReq(req));
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_get_notifs(sgd_nixl_agent_t agent, sgd_nixl_blob_t **out,
                        size_t *count) {
  if (agent == nullptr || out == nullptr || count == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  *out = nullptr;
  *count = 0;
  try {
    nixl_notifs_t notifs;
    nixl_status_t st = agent->agent->getNotifs(notifs, nullptr);
    if (st != NIXL_SUCCESS) {
      return static_cast<int>(st);
    }
    /* Flatten sorted by agent name: unordered_map iteration order is
     * not deterministic and this drain feeds the decode-side completion
     * path. */
    std::vector<std::string> agents;
    agents.reserve(notifs.size());
    size_t total = 0;
    for (const auto &kv : notifs) {
      agents.push_back(kv.first);
      total += kv.second.size();
    }
    std::sort(agents.begin(), agents.end());
    if (total == 0) {
      return NIXL_SUCCESS;
    }
    auto *blobs =
        static_cast<sgd_nixl_blob_t *>(calloc(total, sizeof(sgd_nixl_blob_t)));
    if (blobs == nullptr) {
      return SGD_NIXL_ERR_EXCEPTION;
    }
    size_t at = 0;
    for (const auto &name : agents) {
      for (const auto &msg : notifs[name]) {
        void *copy = malloc(msg.size() == 0 ? 1 : msg.size());
        if (copy == nullptr) {
          sgd_nixl_free_notifs(blobs, at);
          return SGD_NIXL_ERR_EXCEPTION;
        }
        memcpy(copy, msg.data(), msg.size());
        blobs[at].data = static_cast<uint8_t *>(copy);
        blobs[at].len = msg.size();
        at++;
      }
    }
    *out = blobs;
    *count = total;
    return NIXL_SUCCESS;
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

void sgd_nixl_free_notifs(sgd_nixl_blob_t *blobs, size_t count) {
  if (blobs == nullptr) {
    return;
  }
  for (size_t i = 0; i < count; i++) {
    free(blobs[i].data);
  }
  free(blobs);
}

int sgd_nixl_prep_dlist(sgd_nixl_agent_t agent, const char *agent_name,
                        int mem_kind, const uint64_t *descs3, size_t n,
                        uint64_t *handle_out) {
  if (agent == nullptr || agent_name == nullptr ||
      (descs3 == nullptr && n > 0) || handle_out == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  try {
    nixl_xfer_dlist_t descs = xfer_dlist(mem_kind, descs3, n);
    nixlDlistH *dlist = nullptr;
    /* "" == NIXL_INIT_AGENT: the local side. */
    nixl_status_t st = agent->agent->prepXferDlist(std::string(agent_name),
                                                   descs, dlist, nullptr);
    if (st != NIXL_SUCCESS) {
      return static_cast<int>(st);
    }
    std::lock_guard<std::mutex> lock(agent->mu);
    uint64_t id = agent->next_handle++;
    agent->dlists.emplace(id, dlist);
    *handle_out = id;
    return NIXL_SUCCESS;
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_release_dlist(sgd_nixl_agent_t agent, uint64_t handle) {
  if (agent == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  nixlDlistH *dlist = agent->lookup_dlist(handle, true);
  if (dlist == nullptr) {
    return SGD_NIXL_ERR_LOOKUP;
  }
  try {
    return static_cast<int>(agent->agent->releasedDlistH(dlist));
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

int sgd_nixl_make_xfer(sgd_nixl_agent_t agent, uint64_t local_dlist,
                       const int32_t *local_idx, size_t n_local,
                       uint64_t remote_dlist, const int32_t *remote_idx,
                       size_t n_remote, uint64_t *handle_out) {
  if (agent == nullptr || (local_idx == nullptr && n_local > 0) ||
      (remote_idx == nullptr && n_remote > 0) || handle_out == nullptr) {
    return SGD_NIXL_ERR_INVALID;
  }
  nixlDlistH *local = agent->lookup_dlist(local_dlist, false);
  nixlDlistH *remote = agent->lookup_dlist(remote_dlist, false);
  if (local == nullptr || remote == nullptr) {
    return SGD_NIXL_ERR_LOOKUP;
  }
  try {
    std::vector<int> lidx(local_idx, local_idx + n_local);
    std::vector<int> ridx(remote_idx, remote_idx + n_remote);
    nixlXferReqH *req = nullptr;
    nixl_status_t st = agent->agent->makeXferReq(NIXL_WRITE, *local, lidx,
                                                 *remote, ridx, req, nullptr);
    if (st != NIXL_SUCCESS) {
      return static_cast<int>(st);
    }
    int rc = post_or_release(agent->agent, req, nullptr);
    if (rc < 0) {
      return rc;
    }
    *handle_out = agent->track_xfer(req);
    return rc;
  } catch (...) {
    return SGD_NIXL_ERR_EXCEPTION;
  }
}

} // extern "C"
