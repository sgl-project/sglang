#include "shm.h"
#if defined(__x86_64__)
#include "x86_64/shm.h"
#elif defined(__aarch64__)
#include "aarch64/shm.h"
#else
#error "unsupported architecture"
#endif

#include <ATen/ATen.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

// states for group collectives
enum group_coll_state {
  group_coll_begin = 0,

  group_coll_allreduce_naive__copy_in_done,
  group_coll_allreduce_naive__reduce_done,
  group_coll_alt1_allreduce_naive__copy_in_done,
  group_coll_alt2_allreduce_naive__copy_in_done,
  group_coll_alt1_allreduce_naive__reduce_done,

  group_coll_allgather_naive__copy_in_done,
  group_coll_alt1_allgather_naive__copy_in_done,
  group_coll_alt2_allgather_naive__copy_in_done,
};

// SHM building blocks
struct GroupSharedData {
  const char* name;
  int descriptor;
  void* bytes;
  size_t nbytes;
};

static void group_shared_open(
    GroupSharedData* data,
    const char* name,
    size_t nbytes) {
  int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    void* bytes = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
    data->name = name;
    data->descriptor = d;
    data->bytes = bytes;
    data->nbytes = nbytes;
  } else {
    if (errno != ENOENT) {
      // Do not print ENOENT because the caller keeps retrying until
      // other ranks create their SHM workspaces.
      printf("group_shared_open %s failed, errno=%d\n", name, errno);
    }

    data->descriptor = -1;
  }
}

static void group_shared_create(
    GroupSharedData* data,
    const char* name,
    void* bytes,
    size_t nbytes) {
  int d = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);

  if (d != -1) {
    nbytes = write(d, bytes, nbytes);
    if (nbytes > 0) {
      group_shared_open(data, name, nbytes);
    }
  } else {
    printf("group_shared_create %s failed\n", name);
  }
}

// SHM based group collective helper functions

#define GROUP_NAME_BUF_SIZE 1000
#define GROUP_MAX_BUF_SIZE (1048576 * 32)
#define GROUP_NAIVE_ALLREDUCE_THRESHOLD 1048576

#define GROUP_ALLGATHER_MAX_BUF_SIZE GROUP_MAX_BUF_SIZE
#define GROUP_ALLTOALL_MAX_BUF_SIZE GROUP_MAX_BUF_SIZE
#define GROUP_SYMMETRIC_ALLREDUCE_MAX_BUF_SIZE \
  GROUP_NAIVE_ALLREDUCE_THRESHOLD
#define GROUP_DISTRIBUTED_ALLREDUCE_MAX_BUF_SIZE GROUP_MAX_BUF_SIZE

#define GROUP_STATE_ALL_GATHER 0
#define GROUP_STATE_ALL_TO_ALL 1
#define GROUP_STATE_SYMMETRIC_ALLREDUCE 2
#define GROUP_STATE_DISTRIBUTED_ALLREDUCE 3
#define GROUP_STATE_NUM 4

struct group_workspace {
  enum group_coll_state states[GROUP_STATE_NUM];

  // double buffer for every collective
  char buffer[
      2 * GROUP_ALLGATHER_MAX_BUF_SIZE +
      2 * GROUP_ALLTOALL_MAX_BUF_SIZE +
      2 * GROUP_SYMMETRIC_ALLREDUCE_MAX_BUF_SIZE +
      2 * GROUP_DISTRIBUTED_ALLREDUCE_MAX_BUF_SIZE];
};

#define GROUP_ALLGATHER_BUFFER_OFFSET(current_buffer) \
  ((current_buffer) * GROUP_ALLGATHER_MAX_BUF_SIZE)

#define GROUP_ALLTOALL_BUFFER_OFFSET(current_buffer)  \
  (2 * GROUP_ALLGATHER_MAX_BUF_SIZE + (current_buffer) * GROUP_ALLTOALL_MAX_BUF_SIZE)

#define GROUP_SYMMETRIC_ALLREDUCE_BUFFER_OFFSET(current_buffer) \
  (2 * GROUP_ALLGATHER_MAX_BUF_SIZE + 2 * GROUP_ALLTOALL_MAX_BUF_SIZE +  \
   (current_buffer) * GROUP_SYMMETRIC_ALLREDUCE_MAX_BUF_SIZE)

#define GROUP_DISTRIBUTED_ALLREDUCE_BUFFER_OFFSET(current_buffer) \
  (2 * GROUP_ALLGATHER_MAX_BUF_SIZE + 2 * GROUP_ALLTOALL_MAX_BUF_SIZE + 2 * GROUP_SYMMETRIC_ALLREDUCE_MAX_BUF_SIZE + \
   (current_buffer) * GROUP_DISTRIBUTED_ALLREDUCE_MAX_BUF_SIZE)

struct group_shm_context {
  int group_size;
  int group_rank;

  // workspace[i] points to group-rank i's shared workspace.
  struct group_workspace** workspace;

  // buffer[double_buffer_idx][group_rank]
  char** allgather_buffer[2];
  char** alltoall_buffer[2];
  char** symmetric_allreduce_buffer[2];
  char** distributed_allreduce_buffer[2];

  // These states must be per ProcessGroup.
  // A single process can participate in multiple groups.
  int allgather_current_buffer;
  int allgather_state_idx;

  int alltoall_current_buffer;
  int alltoall_state_idx;

  int symmetric_allreduce_current_buffer;
  int symmetric_allreduce_state_idx;

  int distributed_allreduce_current_buffer;
  int distributed_allreduce_state_idx;
};

static std::vector<struct group_shm_context*> group_shm_contexts;
static struct group_shm_context* get_group_shm_context(int64_t handle) {
  TORCH_CHECK(handle >= 0 && handle < static_cast<int64_t>(group_shm_contexts.size()), "Invalid group SHM handle");
  auto* ctx = group_shm_contexts[handle];
  TORCH_CHECK(ctx != nullptr, "Group SHM context is null");
  return ctx;
}

static void wait_group_buffer_state_until_2(struct group_shm_context* ctx, int rank, enum group_coll_state state0, enum group_coll_state state1,
    int state_group) {
  volatile enum group_coll_state* state_ptr = &(ctx->workspace[rank]->states[state_group]);
  while (1) {
    volatile enum group_coll_state cur_state = *state_ptr;
    if (cur_state == state0 || cur_state == state1) {
      break;
    }
  }
}


static void get_group_copy_states(
    int state_idx,
    enum group_coll_state* copy_current,
    enum group_coll_state* copy_next) {
  switch (state_idx) {
    case 0:
      *copy_current = group_coll_allgather_naive__copy_in_done;
      *copy_next = group_coll_alt1_allgather_naive__copy_in_done;
      break;

    case 1:
      *copy_current = group_coll_alt1_allgather_naive__copy_in_done;
      *copy_next = group_coll_alt2_allgather_naive__copy_in_done;
      break;

    case 2:
      *copy_current = group_coll_alt2_allgather_naive__copy_in_done;
      *copy_next = group_coll_allgather_naive__copy_in_done;
      break;

    default:
      assert(!"Should not get here.");
  }
}


static size_t group_slice_size(size_t chunk_el, int slice_idx, int group_size) {
  size_t slice_size = chunk_el / static_cast<size_t>(group_size);
  return slice_idx == group_size - 1 ? slice_size + (chunk_el % static_cast<size_t>(group_size)) : slice_size;
}


static char* group_slice_data(char* data_ptr, size_t chunk_el, int element_size, int slice_idx, int group_size) {
  size_t slice_size = chunk_el / static_cast<size_t>(group_size);
  size_t el_offset = slice_size * static_cast<size_t>(slice_idx);
  return data_ptr + el_offset * element_size;
}

static size_t group_slice_el_start(size_t chunk_el, int slice_idx, int group_size) {
  size_t slice_size = chunk_el / static_cast<size_t>(group_size);
  return slice_size * static_cast<size_t>(slice_idx);
}


static void group_reduce_all_buffers(
    int start_elements,
    int num_elements,
    c10::ScalarType scalar_type,
    char* to_buffer,
    char** buffers,
    int group_size) {
  switch (scalar_type) {
    case c10::ScalarType::BFloat16:
      reduce_bf16_buffers(start_elements, num_elements, to_buffer, buffers, group_size);
      break;

    case c10::ScalarType::Half:
      reduce_fp16_buffers(start_elements, num_elements, to_buffer, buffers, group_size);
      break;

    case c10::ScalarType::Float:
      reduce_fp32_buffers(start_elements, num_elements, to_buffer, buffers, group_size);
      break;

    default:
      assert(!"Should not get here");
  }
}

int64_t shm_group_initialize(const std::string& group_name, int64_t group_size, int64_t group_rank) {
  TORCH_CHECK(group_size > 0, "group_size must be greater than 0");
  TORCH_CHECK(group_rank >= 0 && group_rank < group_size, "Invalid group_rank")
  auto* ctx =(struct group_shm_context*)malloc(sizeof(struct group_shm_context));
  ctx->group_size = static_cast<int>(group_size);
  ctx->group_rank = static_cast<int>(group_rank);

  ctx->allgather_current_buffer = 0;
  ctx->allgather_state_idx = 0;
  ctx->alltoall_current_buffer = 0;
  ctx->alltoall_state_idx = 0;
  ctx->symmetric_allreduce_current_buffer = 0;
  ctx->symmetric_allreduce_state_idx = 0;
  ctx->distributed_allreduce_current_buffer = 0;
  ctx->distributed_allreduce_state_idx = 0;

  ctx->workspace = (struct group_workspace**)malloc(group_size * sizeof(struct group_workspace*));
  ctx->allgather_buffer[0] = (char**)malloc(group_size * sizeof(char*));

  ctx->allgather_buffer[1] = (char**)malloc(group_size * sizeof(char*));

  ctx->alltoall_buffer[0] = (char**)malloc(group_size * sizeof(char*));

  ctx->alltoall_buffer[1] = (char**)malloc(group_size * sizeof(char*));

  ctx->symmetric_allreduce_buffer[0] = (char**)malloc(group_size * sizeof(char*));

  ctx->symmetric_allreduce_buffer[1] = (char**)malloc(group_size * sizeof(char*));

  ctx->distributed_allreduce_buffer[0] = (char**)malloc(group_size * sizeof(char*));

  ctx->distributed_allreduce_buffer[1] = (char**)malloc(group_size * sizeof(char*));

  char shm_name[GROUP_NAME_BUF_SIZE];

  // Create current group-rank's workspace.
  auto* workspace_buf =(struct group_workspace*)calloc(1, sizeof(struct group_workspace));
  workspace_buf->states[GROUP_STATE_ALL_GATHER] = group_coll_alt2_allgather_naive__copy_in_done;
  workspace_buf->states[GROUP_STATE_ALL_TO_ALL] = group_coll_alt2_allgather_naive__copy_in_done;
  workspace_buf->states[GROUP_STATE_SYMMETRIC_ALLREDUCE] = group_coll_alt2_allreduce_naive__copy_in_done;
  workspace_buf->states[GROUP_STATE_DISTRIBUTED_ALLREDUCE] = group_coll_begin;
  snprintf(shm_name, GROUP_NAME_BUF_SIZE, "%.900s_%d", group_name.c_str(), static_cast<int>(group_rank));
  GroupSharedData local_shm = {};
  local_shm.descriptor = -1;
  group_shared_create(&local_shm, shm_name, workspace_buf, sizeof(struct group_workspace));
  free(workspace_buf);

  TORCH_CHECK(local_shm.descriptor != -1, "Failed to create group SHM workspace");

  auto* local_workspace =(struct group_workspace*)local_shm.bytes;

  // Map SHM workspace of every group rank.
  for (int i = 0; i < group_size; ++i) {
    if (i == group_rank) {
      ctx->workspace[i] = local_workspace;
    } else {
      snprintf(shm_name, GROUP_NAME_BUF_SIZE, "%.900s_%d", group_name.c_str(), i);
      GroupSharedData peer_shm = {};
      peer_shm.descriptor = -1;
      do {
        group_shared_open(&peer_shm, shm_name, sizeof(struct group_workspace));
      } while (peer_shm.descriptor == -1 && errno == ENOENT);
      TORCH_CHECK(peer_shm.descriptor != -1, "Failed to open group SHM workspace");
      ctx->workspace[i] = (struct group_workspace*)peer_shm.bytes;
    }

    ctx->allgather_buffer[0][i] = ctx->workspace[i]->buffer + GROUP_ALLGATHER_BUFFER_OFFSET(0);

    ctx->allgather_buffer[1][i] = ctx->workspace[i]->buffer + GROUP_ALLGATHER_BUFFER_OFFSET(1);

    ctx->alltoall_buffer[0][i] = ctx->workspace[i]->buffer + GROUP_ALLTOALL_BUFFER_OFFSET(0);

    ctx->alltoall_buffer[1][i] = ctx->workspace[i]->buffer + GROUP_ALLTOALL_BUFFER_OFFSET(1);

    ctx->symmetric_allreduce_buffer[0][i] = ctx->workspace[i]->buffer + GROUP_SYMMETRIC_ALLREDUCE_BUFFER_OFFSET(0);

    ctx->symmetric_allreduce_buffer[1][i] = ctx->workspace[i]->buffer + GROUP_SYMMETRIC_ALLREDUCE_BUFFER_OFFSET(1);

    ctx->distributed_allreduce_buffer[0][i] = ctx->workspace[i]->buffer + GROUP_DISTRIBUTED_ALLREDUCE_BUFFER_OFFSET(0);

    ctx->distributed_allreduce_buffer[1][i] = ctx->workspace[i]->buffer + GROUP_DISTRIBUTED_ALLREDUCE_BUFFER_OFFSET(1);
  }
    const int64_t handle = static_cast<int64_t>(group_shm_contexts.size());
    group_shm_contexts.push_back(ctx);
    return handle;
}

void group_all_gather(int64_t handle, char* output_ptr, char* input_ptr, size_t data_size) {
  auto* ctx = get_group_shm_context(handle);
  TORCH_CHECK(data_size <= GROUP_ALLGATHER_MAX_BUF_SIZE, "group SHM all-gather input size exceeds maximum buffer size.");
  enum group_coll_state copy_current = group_coll_allgather_naive__copy_in_done;
  enum group_coll_state copy_next = group_coll_alt1_allgather_naive__copy_in_done;
  get_group_copy_states(ctx->allgather_state_idx, &copy_current, &copy_next);
  ctx->allgather_state_idx = (ctx->allgather_state_idx + 1) % 3;

  const int current_buffer = ctx->allgather_current_buffer;

  // Step 1:
  // Copy local input to this group rank's SHM workspace.
  parallel_memcpy(ctx->allgather_buffer[current_buffer][ctx->group_rank], input_ptr, data_size);
  std::atomic_thread_fence(std::memory_order_release);
  ctx->workspace[ctx->group_rank]->states[GROUP_STATE_ALL_GATHER] = copy_current;

  // Step 2:
  // Wait until all ranks in this ProcessGroup have copied input.
  for (int i = 0; i < ctx->group_size; ++i) {
    if (i != ctx->group_rank) {
      wait_group_buffer_state_until_2(ctx, i, copy_current, copy_next, GROUP_STATE_ALL_GATHER);
    }
  }
  // Step 3:
  // Gather buffers in group-rank order.
  for (int i = 0; i < ctx->group_size; ++i) {
    parallel_memcpy(output_ptr + static_cast<size_t>(i) * data_size, ctx->allgather_buffer[current_buffer][i], data_size);
  }
  // switch buffer
  ctx->allgather_current_buffer = 1 - current_buffer;
}

void group_all_to_all(int64_t handle, char* output_ptr, char* input_ptr, size_t data_size) {
  auto* ctx = get_group_shm_context(handle);
  TORCH_CHECK(data_size <= GROUP_ALLTOALL_MAX_BUF_SIZE, "group SHM all-to-all input size exceeds maximum buffer size.");
  TORCH_CHECK(data_size % static_cast<size_t>(ctx->group_size) == 0, "group SHM all-to-all input size must be divisible by group size.");
  enum group_coll_state copy_current = group_coll_allgather_naive__copy_in_done;
  enum group_coll_state copy_next = group_coll_alt1_allgather_naive__copy_in_done;
  get_group_copy_states(ctx->alltoall_state_idx, &copy_current, &copy_next);
  ctx->alltoall_state_idx = (ctx->alltoall_state_idx + 1) % 3;
  const int current_buffer = ctx->alltoall_current_buffer;

   // Step 1:
  // Copy complete destination-major input to SHM.
  parallel_memcpy(ctx->alltoall_buffer[current_buffer][ctx->group_rank], input_ptr, data_size);

  std::atomic_thread_fence(std::memory_order_release);

  ctx->workspace[ctx->group_rank]->states[GROUP_STATE_ALL_TO_ALL] = copy_current;

   // Step 2:
  // Wait until all ranks in this ProcessGroup have copied input.
  for (int i = 0; i < ctx->group_size; ++i) {
    if (i != ctx->group_rank) {
      wait_group_buffer_state_until_2(ctx, i, copy_current, copy_next, GROUP_STATE_ALL_TO_ALL);
    }
  }
  const size_t peer_chunk_size = data_size / static_cast<size_t>(ctx->group_size);

  // Step 3:
  // Read the chunk addressed to this group rank
  // from every source group rank.
  for (int src = 0; src < ctx->group_size; ++src) {
    char* src_ptr = ctx->alltoall_buffer[current_buffer][src] + static_cast<size_t>(ctx->group_rank) * peer_chunk_size;
    char* dst_ptr = output_ptr + static_cast<size_t>(src) * peer_chunk_size;
    parallel_memcpy(dst_ptr, src_ptr, peer_chunk_size);
  }
  ctx->alltoall_current_buffer = 1 - current_buffer;
}

static void group_symmetric_naive_all_reduce(struct group_shm_context* ctx, char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el) {
  const int state_group = GROUP_STATE_SYMMETRIC_ALLREDUCE;
  enum group_coll_state copy_current = group_coll_allreduce_naive__copy_in_done;
  enum group_coll_state copy_next = group_coll_alt1_allreduce_naive__copy_in_done;

  switch (ctx->symmetric_allreduce_state_idx) {
    case 0:
      copy_current = group_coll_allreduce_naive__copy_in_done;
      copy_next = group_coll_alt1_allreduce_naive__copy_in_done;
      break;

    case 1:
      copy_current = group_coll_alt1_allreduce_naive__copy_in_done;
      copy_next = group_coll_alt2_allreduce_naive__copy_in_done;
      break;

    case 2:
      copy_current = group_coll_alt2_allreduce_naive__copy_in_done;
      copy_next = group_coll_allreduce_naive__copy_in_done;
      break;

    default:
      assert(!"Should not get here.");
  }

  ctx->symmetric_allreduce_state_idx = (ctx->symmetric_allreduce_state_idx + 1) % 3;
  const int current_buffer = ctx->symmetric_allreduce_current_buffer;

  parallel_memcpy(ctx->symmetric_allreduce_buffer[current_buffer][ctx->group_rank], data_ptr, chunk_size);

  std::atomic_thread_fence(std::memory_order_release);

  ctx->workspace[ctx->group_rank]->states[state_group] = copy_current;

  for (int i = 0; i < ctx->group_size; ++i) {
    if (i != ctx->group_rank) {
      wait_group_buffer_state_until_2(ctx, i, copy_current, copy_next, state_group);
    }
  }

  // Each rank reduces all group buffers independently,
  // so no second synchronization is required.
  group_reduce_all_buffers(0, chunk_el, scalar_type, data_ptr, ctx->symmetric_allreduce_buffer[current_buffer], ctx->group_size);

  // switch buffer
  ctx->symmetric_allreduce_current_buffer = 1 - current_buffer;
}

// naive group allreduce distributed:
// each rank reduces its own slice.
static void group_distributed_naive_reduce(struct group_shm_context* ctx, char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el) {
  const int state_group = GROUP_STATE_DISTRIBUTED_ALLREDUCE;
  enum group_coll_state copy_current = group_coll_allreduce_naive__copy_in_done;

  enum group_coll_state reduce_current = group_coll_allreduce_naive__reduce_done;

  enum group_coll_state copy_next = group_coll_alt1_allreduce_naive__copy_in_done;

  // Distributed reduce has two barriers, therefore only two
  // state generations are required.
  switch (ctx->distributed_allreduce_state_idx) {
    case 0:
      copy_current = group_coll_allreduce_naive__copy_in_done;
      reduce_current = group_coll_allreduce_naive__reduce_done;
      copy_next = group_coll_alt1_allreduce_naive__copy_in_done;
      break;

    case 1:
      copy_current = group_coll_alt1_allreduce_naive__copy_in_done;
      reduce_current = group_coll_alt1_allreduce_naive__reduce_done;
      copy_next = group_coll_allreduce_naive__copy_in_done;
      break;

    default:
      assert(!"Should not get here.");
  }

  ctx->distributed_allreduce_state_idx = (ctx->distributed_allreduce_state_idx + 1) % 2;
  const int current_buffer = ctx->distributed_allreduce_current_buffer;
  const int element_size = static_cast<int>(chunk_size / chunk_el);

  // Step 1:
  // Every group rank publishes its complete input.
  parallel_memcpy(ctx->distributed_allreduce_buffer[current_buffer][ctx->group_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  ctx->workspace[ctx->group_rank] ->states[state_group] = copy_current;

  // Step 2:
  // Wait until all group ranks have copied input.
  for (int i = 0; i < ctx->group_size; ++i) {
    if (i != ctx->group_rank) {
      wait_group_buffer_state_until_2(ctx, i, copy_current, reduce_current, state_group);
    }
  }

  // Step 3:
  // Each rank reduces its own slice.
  const size_t start_el = group_slice_el_start(chunk_el, ctx->group_rank, ctx->group_size);
  const size_t local_el = group_slice_size(chunk_el, ctx->group_rank, ctx->group_size);
  group_reduce_all_buffers(static_cast<int>(start_el), static_cast<int>(local_el), scalar_type, ctx->distributed_allreduce_buffer[current_buffer][ctx->group_rank],
      ctx->distributed_allreduce_buffer[current_buffer], ctx->group_size);
  std::atomic_thread_fence(std::memory_order_release);
  ctx->workspace[ctx->group_rank]->states[state_group] = reduce_current;

  // Step 4:
  // Wait until all ranks finish reducing their slice.
  for (int i = 0; i < ctx->group_size; ++i) {
    if (i != ctx->group_rank) {
      wait_group_buffer_state_until_2(ctx, i, reduce_current, copy_next, state_group);
    }
  }

  // Step 5:
  // Gather reduced slices back into local tensor.
  for (int i = 0; i < ctx->group_size; ++i) {
    const int rank =(i + ctx->group_rank) % ctx->group_size;
    parallel_memcpy(group_slice_data(data_ptr, chunk_el, element_size, rank, ctx->group_size),
        group_slice_data(ctx->distributed_allreduce_buffer[current_buffer][rank], chunk_el, element_size, rank, ctx->group_size),
        group_slice_size(chunk_el, rank, ctx->group_size) * element_size);
  }

  // switch buffer
  ctx->distributed_allreduce_current_buffer = 1 - current_buffer;
}

void group_all_reduce(int64_t handle, char* data_ptr, c10::ScalarType scalar_type, size_t data_size, size_t numel) {
  auto* ctx = get_group_shm_context(handle);
  if (numel == 0) {
    return;
  }

  const size_t element_size = data_size / numel;

  for (size_t offset = 0; offset < data_size; offset += GROUP_MAX_BUF_SIZE) {
    char* chunk_ptr = data_ptr + offset;
    size_t chunk_size = std::min(static_cast<size_t>(GROUP_MAX_BUF_SIZE), data_size - offset);
    size_t chunk_el = chunk_size / element_size;

    if (chunk_size < GROUP_NAIVE_ALLREDUCE_THRESHOLD) {
      group_symmetric_naive_all_reduce(ctx, chunk_ptr, scalar_type, chunk_size, chunk_el);
    } else {
      group_distributed_naive_reduce(ctx, chunk_ptr, scalar_type, chunk_size, chunk_el);
    }
  }
}



