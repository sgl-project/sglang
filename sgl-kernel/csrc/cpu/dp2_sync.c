#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <signal.h>
#include <stdalign.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define SGLANG_DP2_SYNC_ABI_VERSION 2U
#define SGLANG_DP2_SYNC_LAYOUT_VERSION 1U
#define SGLANG_DP2_SYNC_MAGIC UINT64_C(0x53484450324d4c50)
#define SGLANG_DP2_SYNC_WORLD_SIZE 2U
#define SGLANG_DP2_SYNC_WIDTH 7U
#define SGLANG_DP2_SYNC_SLOTS 2U
#define SGLANG_DP2_SYNC_ERROR_BUFFER_MIN 128U
#define SGLANG_DP2_SYNC_WAIT_SLICE_NS UINT64_C(100000000)

_Static_assert(ATOMIC_INT_LOCK_FREE == 2, "32-bit atomics must be lock-free");
_Static_assert(
    ATOMIC_LLONG_LOCK_FREE == 2, "64-bit atomics must be lock-free"
);

struct sglang_dp2_sync_stats {
    uint64_t sequence;
    uint64_t total_ns;
    uint64_t peer_wait_ns;
    uint64_t arrival_skew_ns;
    uint64_t post_latest_arrival_ns;
};

struct sglang_dp2_rank_state {
    alignas(64) _Atomic uint64_t published_sequence;
    _Atomic uint32_t futex_epoch;
    _Atomic int32_t pid;
    uint32_t reserved;
    uint64_t arrival_ns[SGLANG_DP2_SYNC_SLOTS];
    int64_t
        payload[SGLANG_DP2_SYNC_SLOTS][SGLANG_DP2_SYNC_WIDTH];
};

struct sglang_dp2_shared_state {
    alignas(64) uint64_t magic;
    uint32_t layout_version;
    uint32_t world_size;
    uint32_t width;
    uint32_t state_size;
    _Atomic uint32_t error_code;
    _Atomic uint32_t unlinked;
    uint8_t header_padding[32];
    struct sglang_dp2_rank_state rank[SGLANG_DP2_SYNC_WORLD_SIZE];
};

struct sglang_dp2_sync_handle {
    struct sglang_dp2_shared_state *shared;
    size_t mapping_size;
    uint64_t timeout_ns;
    uint64_t local_sequence;
    int rank;
    int fd;
    pid_t pid;
    char shm_name[64];
};

_Static_assert(
    offsetof(struct sglang_dp2_shared_state, rank) % 64 == 0,
    "rank state must start on a cache-line boundary"
);
_Static_assert(
    sizeof(struct sglang_dp2_rank_state) % 64 == 0,
    "rank states must not share cache lines"
);

static void set_error(
    char *error_buffer,
    size_t error_buffer_size,
    const char *format,
    ...
) {
    if (error_buffer == NULL || error_buffer_size == 0) {
        return;
    }
    va_list args;
    va_start(args, format);
    (void)vsnprintf(error_buffer, error_buffer_size, format, args);
    va_end(args);
    error_buffer[error_buffer_size - 1] = '\0';
}

static uint64_t monotonic_ns(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) {
        return 0;
    }
    return (uint64_t)timestamp.tv_sec * UINT64_C(1000000000) +
        (uint64_t)timestamp.tv_nsec;
}

static uint64_t fnv1a_64(const char *value) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const unsigned char *cursor = (const unsigned char *)value;
    while (*cursor != '\0') {
        hash ^= (uint64_t)*cursor++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int futex_wait_shared(
    _Atomic uint32_t *address,
    uint32_t expected,
    uint64_t timeout_ns
) {
    struct timespec timeout = {
        .tv_sec = (time_t)(timeout_ns / UINT64_C(1000000000)),
        .tv_nsec = (long)(timeout_ns % UINT64_C(1000000000)),
    };
    return (int)syscall(
        SYS_futex,
        (uint32_t *)address,
        FUTEX_WAIT,
        expected,
        &timeout,
        NULL,
        0
    );
}

static void futex_wake_all(_Atomic uint32_t *address) {
    (void)syscall(
        SYS_futex,
        (uint32_t *)address,
        FUTEX_WAKE,
        INT32_MAX,
        NULL,
        NULL,
        0
    );
}

static void wake_rank(struct sglang_dp2_rank_state *rank) {
    (void)atomic_fetch_add_explicit(
        &rank->futex_epoch, 1U, memory_order_release
    );
    futex_wake_all(&rank->futex_epoch);
}

static void publish_shared_error(
    struct sglang_dp2_sync_handle *handle,
    uint32_t error_code
) {
    uint32_t expected = 0;
    (void)atomic_compare_exchange_strong_explicit(
        &handle->shared->error_code,
        &expected,
        error_code,
        memory_order_acq_rel,
        memory_order_acquire
    );
    for (size_t rank = 0; rank < SGLANG_DP2_SYNC_WORLD_SIZE; ++rank) {
        wake_rank(&handle->shared->rank[rank]);
    }
}

static int validate_shared_state(
    const struct sglang_dp2_shared_state *shared,
    char *error_buffer,
    size_t error_buffer_size
) {
    if (shared->magic != SGLANG_DP2_SYNC_MAGIC) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync has invalid magic 0x%016llx",
            (unsigned long long)shared->magic
        );
        return -1;
    }
    if (shared->layout_version != SGLANG_DP2_SYNC_LAYOUT_VERSION) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync layout mismatch: expected %u, got %u",
            SGLANG_DP2_SYNC_LAYOUT_VERSION,
            shared->layout_version
        );
        return -1;
    }
    if (
        shared->world_size != SGLANG_DP2_SYNC_WORLD_SIZE ||
        shared->width != SGLANG_DP2_SYNC_WIDTH ||
        shared->state_size != sizeof(*shared)
    ) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync geometry mismatch: world=%u width=%u "
            "state_size=%u expected=%zu",
            shared->world_size,
            shared->width,
            shared->state_size,
            sizeof(*shared)
        );
        return -1;
    }
    return 0;
}

uint32_t sglang_dp2_sync_abi_version(void) {
    return SGLANG_DP2_SYNC_ABI_VERSION;
}

int sglang_dp2_sync_open(
    const char *session_id,
    int rank,
    uint64_t timeout_ns,
    void **output_handle,
    char *error_buffer,
    size_t error_buffer_size
) {
    if (error_buffer != NULL && error_buffer_size > 0) {
        error_buffer[0] = '\0';
    }
    if (
        session_id == NULL || session_id[0] == '\0' ||
        output_handle == NULL ||
        error_buffer == NULL ||
        error_buffer_size < SGLANG_DP2_SYNC_ERROR_BUFFER_MIN
    ) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync received invalid open arguments"
        );
        return -1;
    }
    if (rank < 0 || rank >= (int)SGLANG_DP2_SYNC_WORLD_SIZE) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync rank must be 0 or 1, got %d",
            rank
        );
        return -1;
    }
    if (timeout_ns == 0) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync timeout must be positive"
        );
        return -1;
    }

    struct sglang_dp2_sync_handle *handle =
        calloc(1, sizeof(struct sglang_dp2_sync_handle));
    if (handle == NULL) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync handle allocation failed"
        );
        return -1;
    }
    handle->fd = -1;
    handle->mapping_size = sizeof(struct sglang_dp2_shared_state);
    handle->timeout_ns = timeout_ns;
    handle->rank = rank;
    handle->pid = getpid();
    (void)snprintf(
        handle->shm_name,
        sizeof(handle->shm_name),
        "/sglang_dp2_%016llx",
        (unsigned long long)fnv1a_64(session_id)
    );

    handle->fd = shm_open(
        handle->shm_name,
        O_RDWR | O_CREAT | O_CLOEXEC,
        S_IRUSR | S_IWUSR
    );
    if (handle->fd < 0) {
        set_error(
            error_buffer,
            error_buffer_size,
            "shm_open(%s) failed: %s",
            handle->shm_name,
            strerror(errno)
        );
        free(handle);
        return -1;
    }
    if (flock(handle->fd, LOCK_EX) != 0) {
        set_error(
            error_buffer,
            error_buffer_size,
            "flock(%s) failed: %s",
            handle->shm_name,
            strerror(errno)
        );
        close(handle->fd);
        free(handle);
        return -1;
    }

    struct stat file_stat;
    if (fstat(handle->fd, &file_stat) != 0) {
        set_error(
            error_buffer,
            error_buffer_size,
            "fstat(%s) failed: %s",
            handle->shm_name,
            strerror(errno)
        );
        (void)flock(handle->fd, LOCK_UN);
        close(handle->fd);
        free(handle);
        return -1;
    }

    bool initialize = file_stat.st_size == 0;
    if (
        !initialize &&
        (uint64_t)file_stat.st_size != (uint64_t)handle->mapping_size
    ) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync segment has size %lld, expected %zu",
            (long long)file_stat.st_size,
            handle->mapping_size
        );
        (void)flock(handle->fd, LOCK_UN);
        close(handle->fd);
        free(handle);
        return -1;
    }
    if (initialize && ftruncate(handle->fd, (off_t)handle->mapping_size) != 0) {
        set_error(
            error_buffer,
            error_buffer_size,
            "ftruncate(%s) failed: %s",
            handle->shm_name,
            strerror(errno)
        );
        (void)flock(handle->fd, LOCK_UN);
        close(handle->fd);
        (void)shm_unlink(handle->shm_name);
        free(handle);
        return -1;
    }

    void *mapping = mmap(
        NULL,
        handle->mapping_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        handle->fd,
        0
    );
    if (mapping == MAP_FAILED) {
        set_error(
            error_buffer,
            error_buffer_size,
            "mmap(%s) failed: %s",
            handle->shm_name,
            strerror(errno)
        );
        (void)flock(handle->fd, LOCK_UN);
        close(handle->fd);
        if (initialize) {
            (void)shm_unlink(handle->shm_name);
        }
        free(handle);
        return -1;
    }
    handle->shared = mapping;

    if (initialize) {
        memset(handle->shared, 0, handle->mapping_size);
        handle->shared->layout_version = SGLANG_DP2_SYNC_LAYOUT_VERSION;
        handle->shared->world_size = SGLANG_DP2_SYNC_WORLD_SIZE;
        handle->shared->width = SGLANG_DP2_SYNC_WIDTH;
        handle->shared->state_size = (uint32_t)sizeof(*handle->shared);
        atomic_thread_fence(memory_order_release);
        handle->shared->magic = SGLANG_DP2_SYNC_MAGIC;
    } else if (
        validate_shared_state(
            handle->shared, error_buffer, error_buffer_size
        ) != 0
    ) {
        (void)munmap(handle->shared, handle->mapping_size);
        (void)flock(handle->fd, LOCK_UN);
        close(handle->fd);
        free(handle);
        return -1;
    }

    if (mlock(handle->shared, handle->mapping_size) != 0) {
        set_error(
            error_buffer,
            error_buffer_size,
            "mlock(%s) failed: %s",
            handle->shm_name,
            strerror(errno)
        );
        (void)munmap(handle->shared, handle->mapping_size);
        (void)flock(handle->fd, LOCK_UN);
        close(handle->fd);
        if (initialize) {
            (void)shm_unlink(handle->shm_name);
        }
        free(handle);
        return -1;
    }

    _Atomic int32_t *pid_slot = &handle->shared->rank[rank].pid;
    int32_t expected_pid = 0;
    if (!atomic_compare_exchange_strong_explicit(
            pid_slot,
            &expected_pid,
            (int32_t)handle->pid,
            memory_order_acq_rel,
            memory_order_acquire
        )) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync rank %d is already owned by pid %d",
            rank,
            expected_pid
        );
        (void)munlock(handle->shared, handle->mapping_size);
        (void)munmap(handle->shared, handle->mapping_size);
        (void)flock(handle->fd, LOCK_UN);
        close(handle->fd);
        free(handle);
        return -1;
    }

    if (flock(handle->fd, LOCK_UN) != 0) {
        set_error(
            error_buffer,
            error_buffer_size,
            "flock unlock(%s) failed: %s",
            handle->shm_name,
            strerror(errno)
        );
        atomic_store_explicit(pid_slot, 0, memory_order_release);
        (void)munlock(handle->shared, handle->mapping_size);
        (void)munmap(handle->shared, handle->mapping_size);
        close(handle->fd);
        free(handle);
        return -1;
    }

    int32_t peer_pid = atomic_load_explicit(
        &handle->shared->rank[1 - rank].pid, memory_order_acquire
    );
    if (peer_pid > 0) {
        uint32_t expected_unlinked = 0;
        if (atomic_compare_exchange_strong_explicit(
                &handle->shared->unlinked,
                &expected_unlinked,
                1U,
                memory_order_acq_rel,
                memory_order_acquire
            )) {
            if (shm_unlink(handle->shm_name) != 0) {
                set_error(
                    error_buffer,
                    error_buffer_size,
                    "shm_unlink(%s) failed: %s",
                    handle->shm_name,
                    strerror(errno)
                );
                publish_shared_error(handle, 1U);
                atomic_store_explicit(pid_slot, 0, memory_order_release);
                (void)munlock(handle->shared, handle->mapping_size);
                (void)munmap(handle->shared, handle->mapping_size);
                close(handle->fd);
                free(handle);
                return -1;
            }
        }
    }

    *output_handle = handle;
    return 0;
}

static int peer_is_dead(
    struct sglang_dp2_sync_handle *handle,
    int peer_rank
) {
    int32_t peer_pid = atomic_load_explicit(
        &handle->shared->rank[peer_rank].pid, memory_order_acquire
    );
    if (peer_pid <= 0) {
        return 0;
    }
    if (kill((pid_t)peer_pid, 0) == 0 || errno == EPERM) {
        return 0;
    }
    return errno == ESRCH;
}

int sglang_dp2_sync_exchange(
    void *opaque_handle,
    const int64_t *local_payload,
    int64_t *global_payload,
    struct sglang_dp2_sync_stats *stats,
    char *error_buffer,
    size_t error_buffer_size
) {
    if (error_buffer != NULL && error_buffer_size > 0) {
        error_buffer[0] = '\0';
    }
    if (
        opaque_handle == NULL ||
        local_payload == NULL ||
        global_payload == NULL ||
        stats == NULL ||
        error_buffer == NULL ||
        error_buffer_size < SGLANG_DP2_SYNC_ERROR_BUFFER_MIN
    ) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync received invalid exchange arguments"
        );
        return -1;
    }

    struct sglang_dp2_sync_handle *handle = opaque_handle;
    struct sglang_dp2_shared_state *shared = handle->shared;
    const int rank = handle->rank;
    const int peer_rank = 1 - rank;
    struct sglang_dp2_rank_state *local_state = &shared->rank[rank];
    struct sglang_dp2_rank_state *peer_state = &shared->rank[peer_rank];

    uint32_t shared_error = atomic_load_explicit(
        &shared->error_code, memory_order_acquire
    );
    if (shared_error != 0) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync peer previously published error %u",
            shared_error
        );
        return -1;
    }
    if (
        atomic_load_explicit(&local_state->pid, memory_order_acquire) !=
        (int32_t)handle->pid
    ) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync rank %d ownership changed",
            rank
        );
        publish_shared_error(handle, 2U);
        return -1;
    }

    const uint64_t started_ns = monotonic_ns();
    if (started_ns == 0) {
        set_error(
            error_buffer,
            error_buffer_size,
            "clock_gettime(CLOCK_MONOTONIC_RAW) failed"
        );
        publish_shared_error(handle, 3U);
        return -1;
    }
    const uint64_t sequence = ++handle->local_sequence;
    const uint64_t prior_sequence = atomic_load_explicit(
        &local_state->published_sequence, memory_order_acquire
    );
    if (prior_sequence != sequence - 1U) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync local sequence mismatch: expected %llu, "
            "found %llu",
            (unsigned long long)(sequence - 1U),
            (unsigned long long)prior_sequence
        );
        publish_shared_error(handle, 4U);
        return -1;
    }

    const size_t slot = (size_t)(sequence % SGLANG_DP2_SYNC_SLOTS);
    memcpy(
        local_state->payload[slot],
        local_payload,
        sizeof(local_state->payload[slot])
    );
    local_state->arrival_ns[slot] = started_ns;
    atomic_store_explicit(
        &local_state->published_sequence, sequence, memory_order_release
    );
    wake_rank(local_state);
    const uint64_t wait_started_ns = monotonic_ns();

    uint64_t peer_sequence = atomic_load_explicit(
        &peer_state->published_sequence, memory_order_acquire
    );
    while (peer_sequence < sequence) {
        shared_error = atomic_load_explicit(
            &shared->error_code, memory_order_acquire
        );
        if (shared_error != 0) {
            set_error(
                error_buffer,
                error_buffer_size,
                "DP2 shared-memory sync peer published error %u while "
                "waiting for sequence %llu",
                shared_error,
                (unsigned long long)sequence
            );
            return -1;
        }
        if (peer_is_dead(handle, peer_rank)) {
            set_error(
                error_buffer,
                error_buffer_size,
                "DP2 shared-memory sync peer rank %d died while waiting for "
                "sequence %llu",
                peer_rank,
                (unsigned long long)sequence
            );
            publish_shared_error(handle, 5U);
            return -1;
        }

        const uint64_t now_ns = monotonic_ns();
        const uint64_t elapsed_ns = now_ns - started_ns;
        if (elapsed_ns >= handle->timeout_ns) {
            set_error(
                error_buffer,
                error_buffer_size,
                "DP2 shared-memory sync timed out after %.3f ms waiting for "
                "rank %d sequence %llu (peer sequence %llu)",
                (double)elapsed_ns / 1000000.0,
                peer_rank,
                (unsigned long long)sequence,
                (unsigned long long)peer_sequence
            );
            publish_shared_error(handle, 6U);
            return -1;
        }

        const uint32_t epoch = atomic_load_explicit(
            &peer_state->futex_epoch, memory_order_acquire
        );
        peer_sequence = atomic_load_explicit(
            &peer_state->published_sequence, memory_order_acquire
        );
        if (peer_sequence >= sequence) {
            break;
        }
        const uint64_t remaining_ns = handle->timeout_ns - elapsed_ns;
        const uint64_t slice_ns =
            remaining_ns < SGLANG_DP2_SYNC_WAIT_SLICE_NS
            ? remaining_ns
            : SGLANG_DP2_SYNC_WAIT_SLICE_NS;
        if (futex_wait_shared(&peer_state->futex_epoch, epoch, slice_ns) != 0) {
            if (
                errno != EAGAIN &&
                errno != EINTR &&
                errno != ETIMEDOUT
            ) {
                set_error(
                    error_buffer,
                    error_buffer_size,
                    "DP2 shared-memory sync futex wait failed: %s",
                    strerror(errno)
                );
                publish_shared_error(handle, 7U);
                return -1;
            }
        }
        peer_sequence = atomic_load_explicit(
            &peer_state->published_sequence, memory_order_acquire
        );
    }

    if (peer_sequence > sequence + 1U) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync peer lapped local rank: local sequence "
            "%llu, peer sequence %llu",
            (unsigned long long)sequence,
            (unsigned long long)peer_sequence
        );
        publish_shared_error(handle, 8U);
        return -1;
    }

    const uint64_t peer_arrival_ns = peer_state->arrival_ns[slot];
    if (peer_arrival_ns == 0) {
        set_error(
            error_buffer,
            error_buffer_size,
            "DP2 shared-memory sync peer published sequence %llu without an "
            "arrival timestamp",
            (unsigned long long)sequence
        );
        publish_shared_error(handle, 9U);
        return -1;
    }

    memcpy(
        global_payload,
        shared->rank[0].payload[slot],
        sizeof(shared->rank[0].payload[slot])
    );
    memcpy(
        global_payload + SGLANG_DP2_SYNC_WIDTH,
        shared->rank[1].payload[slot],
        sizeof(shared->rank[1].payload[slot])
    );
    const uint64_t finished_ns = monotonic_ns();
    const uint64_t latest_arrival_ns =
        started_ns > peer_arrival_ns ? started_ns : peer_arrival_ns;
    const uint64_t earliest_arrival_ns =
        started_ns < peer_arrival_ns ? started_ns : peer_arrival_ns;

    stats->sequence = sequence;
    stats->total_ns = finished_ns - started_ns;
    stats->peer_wait_ns = finished_ns - wait_started_ns;
    stats->arrival_skew_ns = latest_arrival_ns - earliest_arrival_ns;
    stats->post_latest_arrival_ns = finished_ns - latest_arrival_ns;
    return 0;
}

int sglang_dp2_sync_exchange_values(
    void *opaque_handle,
    int64_t num_tokens,
    int64_t num_tokens_for_logprob,
    int64_t can_cuda_graph,
    int64_t is_extend_in_batch,
    int64_t local_can_run_tbo,
    int64_t local_forward_mode,
    int64_t can_run_breakable_cuda_graph,
    int64_t *global_payload,
    struct sglang_dp2_sync_stats *stats,
    char *error_buffer,
    size_t error_buffer_size
) {
    const int64_t local_payload[SGLANG_DP2_SYNC_WIDTH] = {
        num_tokens,
        num_tokens_for_logprob,
        can_cuda_graph,
        is_extend_in_batch,
        local_can_run_tbo,
        local_forward_mode,
        can_run_breakable_cuda_graph,
    };
    return sglang_dp2_sync_exchange(
        opaque_handle,
        local_payload,
        global_payload,
        stats,
        error_buffer,
        error_buffer_size
    );
}

void sglang_dp2_sync_close(void *opaque_handle) {
    if (opaque_handle == NULL) {
        return;
    }
    struct sglang_dp2_sync_handle *handle = opaque_handle;
    if (handle->shared != NULL) {
        _Atomic int32_t *pid_slot =
            &handle->shared->rank[handle->rank].pid;
        int32_t expected_pid = (int32_t)handle->pid;
        (void)atomic_compare_exchange_strong_explicit(
            pid_slot,
            &expected_pid,
            0,
            memory_order_acq_rel,
            memory_order_acquire
        );
        wake_rank(&handle->shared->rank[handle->rank]);
        const int peer_rank = 1 - handle->rank;
        const int32_t peer_pid = atomic_load_explicit(
            &handle->shared->rank[peer_rank].pid, memory_order_acquire
        );
        if (peer_pid <= 0) {
            uint32_t expected_unlinked = 0;
            if (atomic_compare_exchange_strong_explicit(
                    &handle->shared->unlinked,
                    &expected_unlinked,
                    1U,
                    memory_order_acq_rel,
                    memory_order_acquire
                )) {
                (void)shm_unlink(handle->shm_name);
            }
        }
        (void)munlock(handle->shared, handle->mapping_size);
        (void)munmap(handle->shared, handle->mapping_size);
    }
    if (handle->fd >= 0) {
        close(handle->fd);
    }
    free(handle);
}
