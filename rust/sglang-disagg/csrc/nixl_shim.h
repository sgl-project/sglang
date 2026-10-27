/*
 * sglang-disagg NIXL shim — extern "C" opaque-pointer API over the NIXL C++
 * classes.
 *
 * NIXL (github.com/ai-dynamo/nixl) ships no C API; its public surface is the
 * C++ nixlAgent class. This shim wraps exactly the surface SGLang's NIXL
 * KVManager drives (python/sglang/srt/disaggregation/nixl/conn.py), no more:
 *
 *   nixl/conn.py:597-620   agent config (sync_mode=STRICT; the transfer
 *                          worker and bootstrap thread run concurrently so
 *                          sync NONE is unsafe) + explicit create_backend
 *                          with backend_params (num_threads lands here for
 *                          UCX/OBJ/GDS_MT/UCCL, :603-611)
 *   nixl/conn.py:623       get_plugin_list() availability check
 *                          (-> getAvailPlugins)
 *   nixl/conn.py:2030+     _add_remote_peer: add_remote_agent
 *                          (= loadRemoteMD); upstream then connects LAZILY
 *                          at the first transfer — the 60 s LIBFABRIC
 *                          handshake that stalls the Python GIL. The Rust
 *                          registration path calls make_connection EAGERLY
 *                          right after loadRemoteMD, off all hot paths.
 *   nixl/conn.py:2049+     _reload_invalidated_peers: check_remote_metadata
 *                          (-> checkRemoteMD with an empty dlist, 0 means
 *                          valid), remove_remote_agent (= invalidateRemoteMD)
 *   nixl/conn.py:3472,3519 self.agent.name in the registration messages
 *                          (shim stores the ctor name; nixlAgent exposes
 *                          no name getter)
 *   register_buffer_to_engine / _prepare_payload_xfer /
 *   _prepare_state_xfer  registerMem, prepXferDlist ("" ==
 *                          NIXL_INIT_AGENT for the local side), makeXferReq
 *   _send_kvcache_generic  get_xfer_descs + initialize_xfer + transfer
 *                          (= createXferReq + postXferReq; the shim takes
 *                          flat (addr,len,dev) x n arrays instead of
 *                          per-descriptor calls)
 *   _await_handles         check_xfer_state (= getXferStatus) + release
 *
 * Upstream C++ references (ai-dynamo/nixl src/api/cpp/nixl.h, verified
 * against main): createBackend :106, registerMem :118, deregisterMem :131,
 * makeConnection :158, prepXferDlist :184, makeXferReq :257, createXferReq
 * :306, postXferReq :345, getXferStatus :355, releaseXferReq :388,
 * releasedDlistH :445, getNotifs :461, getLocalMD :489, loadRemoteMD :520,
 * invalidateRemoteMD :532, getAvailPlugins :67, checkRemoteMD :616.
 *
 * Status convention: functions return the raw nixl_status_t integer
 * (0 = NIXL_SUCCESS, 1 = NIXL_IN_PROG, negative = NIXL_ERR_* — see
 * nixl_types.h) so the Rust side can map them onto the Python exception
 * classes (nixlBackendError / nixlCancelledError /
 * nixlRemoteDisconnectError; nixl/conn.py imports). Shim-local failures use
 * values below -100 and never collide with NIXL's own codes.
 */
#ifndef SGD_NIXL_SHIM_H
#define SGD_NIXL_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque agent handle: owns the nixlAgent, its backend handles, the agent's
 * own name, and the handle tables for live transfer requests and prepped
 * dlists. */
typedef struct sgd_nixl_agent_s *sgd_nixl_agent_t;

/* Shim-local status codes (NIXL's own nixl_status_t values pass through
 * unchanged; these never overlap). */
#define SGD_NIXL_ERR_EXCEPTION (-100) /* C++ exception escaped a NIXL call */
#define SGD_NIXL_ERR_INVALID (-101)   /* bad shim argument (null pointer) */
#define SGD_NIXL_ERR_LOOKUP (-102)    /* unknown transfer/dlist handle id */

/* Thread-sync modes, mirroring nixl_thread_sync_t. Upstream passes STRICT
 * (nixl/conn.py:600). */
#define SGD_NIXL_SYNC_NONE 0
#define SGD_NIXL_SYNC_STRICT 1
#define SGD_NIXL_SYNC_RW 2

/* Memory kinds, mirroring nixl_mem_t. */
#define SGD_NIXL_MEM_DRAM 0
#define SGD_NIXL_MEM_VRAM 1

/* One outgoing notification message (get_notifs output element). */
typedef struct {
  uint8_t *data; /* malloc'd; released by sgd_nixl_free_notifs */
  size_t len;
} sgd_nixl_blob_t;

/* Create an agent: nixlAgentConfig with useProgThread=true and the given
 * syncMode / pthrDelay (microseconds; 0 = NIXL default — upstream does not
 * set it). No backends are created here: upstream configures backends=[]
 * and creates the one backend explicitly (nixl/conn.py:598,615) so
 * backend_params can be passed. */
int sgd_nixl_agent_create(const char *name, uint64_t pthr_delay_us,
                          int sync_mode, sgd_nixl_agent_t *out);

/* ~nixlAgent (destroys backend handles with it) and the shim struct. */
void sgd_nixl_agent_destroy(sgd_nixl_agent_t agent);

/* The agent's own name (the registration-message agent_name;
 * nixl/conn.py:3472). strdup'd; free with sgd_nixl_free. */
int sgd_nixl_agent_name(sgd_nixl_agent_t agent, char **name_out);

/* createBackend with explicit params (nixl/conn.py:613-620): keys/vals are
 * `n` parallel NUL-terminated strings (e.g. "num_threads" -> "8" for UCX).
 * May block on device setup — upstream wraps this in run_with_deadline. */
int sgd_nixl_create_backend(sgd_nixl_agent_t agent, const char *backend,
                            const char *const *keys, const char *const *vals,
                            size_t n);

/* getAvailPlugins -> comma-separated names, malloc'd (the get_plugin_list
 * availability check, nixl/conn.py:623). Free with sgd_nixl_free. */
int sgd_nixl_get_avail_plugins(sgd_nixl_agent_t agent, char **csv_out);

/* registerMem / deregisterMem over a flat (addr, len, dev) x n u64 array
 * (nixl_reg_dlist_t of nixlBlobDesc with empty metaInfo, mem_kind VRAM|DRAM;
 * register_buffer_to_engine, nixl/conn.py:1972). */
int sgd_nixl_register_mem(sgd_nixl_agent_t agent, int mem_kind,
                          const uint64_t *descs3, size_t n);
int sgd_nixl_deregister_mem(sgd_nixl_agent_t agent, int mem_kind,
                            const uint64_t *descs3, size_t n);

/* getLocalMD -> malloc'd copy (the agent_metadata field of the registration
 * message). Free *data with sgd_nixl_free. */
int sgd_nixl_get_local_md(sgd_nixl_agent_t agent, uint8_t **data, size_t *len);
void sgd_nixl_free(void *p);

/* loadRemoteMD == Python add_remote_agent -> the remote agent's name
 * (strdup'd; free with sgd_nixl_free). invalidateRemoteMD == Python
 * remove_remote_agent. checkRemoteMD with an empty dlist == Python
 * check_remote_metadata: 0 (NIXL_SUCCESS) means the metadata is valid, any
 * other value means invalid-or-error — the Rust layer maps exactly like
 * Python's `== NIXL_SUCCESS`. */
int sgd_nixl_load_remote_md(sgd_nixl_agent_t agent, const uint8_t *data,
                            size_t len, char **name_out);
int sgd_nixl_remove_remote(sgd_nixl_agent_t agent, const char *name);
int sgd_nixl_check_remote_md(sgd_nixl_agent_t agent, const char *name);

/* makeConnection: the blocking lazy-connect handshake (up to ~60 s on
 * LIBFABRIC). Upstream never calls it — connections form at first transfer
 * while the Python GIL is held. The Rust registration path calls it eagerly
 * right after loadRemoteMD, off every hot path. */
int sgd_nixl_make_connection(sgd_nixl_agent_t agent, const char *name);

/* Combined initialize_xfer + transfer (nixl/conn.py _post_xfer): build
 * src/dst nixl_xfer_dlist_t (mem_kind, flat (addr, len, dev) x n),
 * createXferReq (NIXL_WRITE), attach the notif payload via
 * nixl_opt_args_t.notifMsg, then postXferReq. SUCCESS and IN_PROG both mean
 * posted (Python transfer() returns "DONE"/"PROC"); *handle_out identifies
 * the request for status/release. On a post error the request is released
 * and the error status returned. */
int sgd_nixl_post_write(sgd_nixl_agent_t agent, int mem_kind,
                        const uint64_t *src3, const uint64_t *dst3, size_t n,
                        const char *peer, const uint8_t *notif,
                        size_t notif_len, uint64_t *handle_out);

/* getXferStatus == Python check_xfer_state: NIXL_SUCCESS (done),
 * NIXL_IN_PROG, or a negative error code, verbatim. */
int sgd_nixl_xfer_status(sgd_nixl_agent_t agent, uint64_t handle);

/* releaseXferReq and drop the handle from the table. */
int sgd_nixl_release_xfer(sgd_nixl_agent_t agent, uint64_t handle);

/* getNotifs drained and flattened, sorted by agent name (deterministic
 * drain order). *out is a malloc'd array of `*count` blobs; release with
 * sgd_nixl_free_notifs. Empty result: *out = NULL, *count = 0. */
int sgd_nixl_get_notifs(sgd_nixl_agent_t agent, sgd_nixl_blob_t **out,
                        size_t *count);
void sgd_nixl_free_notifs(sgd_nixl_blob_t *blobs, size_t count);

/* Prepped-dlist family (_prepare_payload_xfer / _prepare_state_xfer).
 * prepXferDlist: agent_name "" == NIXL_INIT_AGENT (local side); otherwise
 * the remote agent's name. releasedDlistH frees the handle. */
int sgd_nixl_prep_dlist(sgd_nixl_agent_t agent, const char *agent_name,
                        int mem_kind, const uint64_t *descs3, size_t n,
                        uint64_t *handle_out);
int sgd_nixl_release_dlist(sgd_nixl_agent_t agent, uint64_t handle);

/* makeXferReq(NIXL_WRITE, local, local_idx, remote, remote_idx) +
 * postXferReq — Python make_prepped_xfer + transfer. Same posting semantics
 * as sgd_nixl_post_write. */
int sgd_nixl_make_xfer(sgd_nixl_agent_t agent, uint64_t local_dlist,
                       const int32_t *local_idx, size_t n_local,
                       uint64_t remote_dlist, const int32_t *remote_idx,
                       size_t n_remote, uint64_t *handle_out);

#ifdef __cplusplus
}
#endif

#endif /* SGD_NIXL_SHIM_H */
