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

// states for collectives
enum coll_state {
  coll_begin = 0,
  coll_allreduce_naive__copy_in_done,
  coll_allreduce_naive__reduce_done,
  // alternative state when allreduce is working on alternative buffer
  // of the double buffer.
  coll_alt1_allreduce_naive__copy_in_done,
  coll_alt2_allreduce_naive__copy_in_done,
  coll_alt1_allreduce_naive__reduce_done,
  coll_allgather_naive__copy_in_done,
  coll_alt1_allgather_naive__copy_in_done,
  coll_alt2_allgather_naive__copy_in_done,
  coll_reduce_scatter_naive__copy_in_done,
  coll_reduce_scatter_naive__reduce_done,
  coll_alt1_reduce_scatter_naive__copy_in_done,
  coll_alt2_reduce_scatter_naive__copy_in_done,
};

// SHM building blocks
struct SharedData {
  const char* name;
  int descriptor;
  void* bytes;
  size_t nbytes;
};

void shared_open(SharedData* data, const char* name, size_t nbytes) {
  int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    void* bytes = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
    data->name = name;
    data->descriptor = d;
    data->bytes = bytes;
    data->nbytes = nbytes;
  } else {
    if (errno != ENOENT) {
      // don't print if shm can not be found because we want to loop over from
      // caller again until the other ranks created the shm
      printf("shared_open %s failed, errno=%d\n", name, errno);
    }
    data->descriptor = -1;
  }
}

void shared_create(SharedData* data, const char* name, void* bytes, size_t nbytes) {
  int d = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    nbytes = write(d, bytes, nbytes);
    if (nbytes > 0) {
      shared_open(data, name, nbytes);
    }
  } else {
    printf("shared_create %s failed\n", name);
  }
}

// SHM based allreduce helper functions
// buffer that holds shm name
#define NAME_BUF_SIZE 1000
#define MAX_BUF_SIZE 1048576 * 32
#define NAIVE_ALLREDUCE_THRESHOLD 1048576
#define SHM_BUFFER_NAME "deepspeed_allreduce_buffer"
struct allreduce_workspace {
  enum coll_state states[6];  // idx=0 -- state for symmetric_naive_all_reduce
                              // idx=1 -- state for distributed_naive_all_reduce
                              // idx=2 -- state for all_gather
                              // idx=3 -- state for all_gather_into_tensor
                              // idx=4 -- state for reduce_scatter
                              // idx=5 -- state for all_to_all
  // double buffer to avoid syncing between rounds
  // offset=0 -- 2*NAIVE_ALLREDUCE_THRESHOLD : buffer for
  // symmetric_naive_all_reduce after that : buffer for
  // distributed_naive_all_reduce
  char buffer
      [2 * NAIVE_ALLREDUCE_THRESHOLD +  // symmetric allreduce
       2 * MAX_BUF_SIZE +               // distributed naive reduce
       2 * MAX_BUF_SIZE +               // allgather
       2 * MAX_BUF_SIZE +               // allgather_into_tensor
       2 * MAX_BUF_SIZE +               // reduce_scatter
       2 * MAX_BUF_SIZE                 // all_to_all

  ];
};

#define BUFFER0_OFFSET(current_buffer) current_buffer* NAIVE_ALLREDUCE_THRESHOLD
#define BUFFER1_OFFSET(current_buffer) 2 * NAIVE_ALLREDUCE_THRESHOLD + current_buffer* MAX_BUF_SIZE
#define BUFFER2_OFFSET(current_buffer) \
  (2 * NAIVE_ALLREDUCE_THRESHOLD + 2 * MAX_BUF_SIZE + current_buffer * MAX_BUF_SIZE)  // allgather
#define BUFFER3_OFFSET(current_buffer) \
  (2 * NAIVE_ALLREDUCE_THRESHOLD + 4 * MAX_BUF_SIZE + current_buffer * MAX_BUF_SIZE)  // allgather_into_tensor
#define BUFFER4_OFFSET(current_buffer) \
  (2 * NAIVE_ALLREDUCE_THRESHOLD + 6 * MAX_BUF_SIZE + current_buffer * MAX_BUF_SIZE)  // reduce_scatter
#define BUFFER5_OFFSET(current_buffer) \
  (2 * NAIVE_ALLREDUCE_THRESHOLD + 8 * MAX_BUF_SIZE + current_buffer * MAX_BUF_SIZE)  // all_to_all

struct shm_context {
  int group_size;
  int group_rank;

  struct allreduce_workspace** workspace;

  char** symmetric_buffer[2];
  char** distributed_buffer[2];
  char** allgather_buffer[2];
  char** allgather_into_tensor_buffer[2];
  char** reduce_scatter_buffer[2];
  char** alltoall_buffer[2];

  int symmetric_current_buffer;
  int symmetric_state_idx;

  int distributed_current_buffer;
  int distributed_state_idx;

  int allgather_current_buffer;
  int allgather_state_idx;

  int allgather_into_tensor_current_buffer;
  int allgather_into_tensor_state_idx;

  int reduce_scatter_current_buffer;
  int reduce_scatter_state_idx;

  int alltoall_current_buffer;
  int alltoall_state_idx;
};

static struct shm_context* default_shm_context = nullptr;
static std::vector<struct shm_context*> shm_contexts;

static struct shm_context* get_shm_context(int64_t handle) {
  if (handle == -1) {
    TORCH_CHECK(default_shm_context != nullptr, "Default SHM context is not initialized");
    return default_shm_context;
  }
  TORCH_CHECK(handle >= 0 && handle < static_cast<int64_t>(shm_contexts.size()), "Invalid SHM context handle: ");

  auto* ctx = shm_contexts[handle];
  TORCH_CHECK(ctx != nullptr, "SHM context is null");
  return ctx;
}

int64_t shm_get_group_size(int64_t handle) {
  return get_shm_context(handle)->group_size;
}

void wait_buffer_state_until_2(
    struct shm_context* ctx, int index, enum coll_state state0, enum coll_state state1, int state_group) {
  volatile enum coll_state* state_ptr = &(ctx->workspace[index]->states[state_group]);

  while (1) {
    volatile enum coll_state cur_state = *state_ptr;
    if (cur_state == state0 || cur_state == state1) break;
  }
}

void reduce_all_buffers(
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

int64_t shm_initialize(int size, int rank, const char* addr_string, const char* port_string, const char* group_name) {
  auto* ctx = (struct shm_context*)malloc(sizeof(struct shm_context));
  ctx->group_size = size;
  ctx->group_rank = rank;
  ctx->symmetric_current_buffer = 0;
  ctx->symmetric_state_idx = 0;
  ctx->distributed_current_buffer = 0;
  ctx->distributed_state_idx = 0;
  ctx->allgather_current_buffer = 0;
  ctx->allgather_state_idx = 0;
  ctx->allgather_into_tensor_current_buffer = 0;
  ctx->allgather_into_tensor_state_idx = 0;
  ctx->reduce_scatter_current_buffer = 0;
  ctx->reduce_scatter_state_idx = 0;
  ctx->alltoall_current_buffer = 0;
  ctx->alltoall_state_idx = 0;

  char shm_name_prefix[NAME_BUF_SIZE];
  char shm_name[NAME_BUF_SIZE];
  snprintf(
      shm_name_prefix,
      NAME_BUF_SIZE,
      "%s_%d_%s_%s_%s",
      SHM_BUFFER_NAME,
      getuid(),
      addr_string,
      port_string,
      group_name);
  // create shared workspace for SHM based collectives
  SharedData allreduce_buffer;
  // allocate workspace_buf for current rank
  struct allreduce_workspace* workspace_buf;
  struct allreduce_workspace* workspace_buf_other;
  workspace_buf = (struct allreduce_workspace*)malloc(sizeof(struct allreduce_workspace));
  snprintf(shm_name, NAME_BUF_SIZE, "%.900s_%d", shm_name_prefix, rank);
  shared_create(&allreduce_buffer, shm_name, workspace_buf, sizeof(struct allreduce_workspace));
  free(workspace_buf);
  workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes;
  workspace_buf->states[STATE_GROUP_SYMMETRIC_ALLREDUCE] =
      coll_alt2_allreduce_naive__copy_in_done;                            // symmetric_naive_all_reduce
  workspace_buf->states[STATE_GROUP_DISTRIBUTED_ALLREDUCE] = coll_begin;  // distributed_naive_reduce
  workspace_buf->states[STATE_GROUP_ALL_GATHER] = coll_alt2_allgather_naive__copy_in_done;  // all_gather
  workspace_buf->states[STATE_GROUP_ALL_GATHER_INTO_TENSOR] =
      coll_alt2_allgather_naive__copy_in_done;                                              // all_gather_into_tensor
  workspace_buf->states[STATE_GROUP_REDUCE_SCATTER] = coll_begin;                           // reduce_scatter
  workspace_buf->states[STATE_GROUP_ALL_TO_ALL] = coll_alt2_allgather_naive__copy_in_done;  // all_to_all

  // create the workspace pointer list
  ctx->workspace = (struct allreduce_workspace**)malloc(size * sizeof(struct allreduce_workspace*));
  ctx->symmetric_buffer[0] = (char**)malloc(size * sizeof(char*));
  ctx->symmetric_buffer[1] = (char**)malloc(size * sizeof(char*));
  ctx->distributed_buffer[0] = (char**)malloc(size * sizeof(char*));
  ctx->distributed_buffer[1] = (char**)malloc(size * sizeof(char*));
  ctx->allgather_buffer[0] = (char**)malloc(size * sizeof(char*));
  ctx->allgather_buffer[1] = (char**)malloc(size * sizeof(char*));

  ctx->allgather_into_tensor_buffer[0] = (char**)malloc(size * sizeof(char*));
  ctx->allgather_into_tensor_buffer[1] = (char**)malloc(size * sizeof(char*));

  ctx->reduce_scatter_buffer[0] = (char**)malloc(size * sizeof(char*));
  ctx->reduce_scatter_buffer[1] = (char**)malloc(size * sizeof(char*));

  ctx->alltoall_buffer[0] = (char**)malloc(size * sizeof(char*));
  ctx->alltoall_buffer[1] = (char**)malloc(size * sizeof(char*));

  // map shm of all ranks
  for (int i = 0; i < size; i++) {
    if (i != rank) {
      snprintf(shm_name, NAME_BUF_SIZE, "%.900s_%d", shm_name_prefix, i);
      // printf("open %s, %d\n", shm_name, rank);
      do {
        shared_open(&allreduce_buffer, shm_name, sizeof(struct allreduce_workspace));
      } while (allreduce_buffer.descriptor == -1 && errno == ENOENT);
      workspace_buf_other = (struct allreduce_workspace*)allreduce_buffer.bytes;
      ctx->workspace[i] = workspace_buf_other;
    } else {
      ctx->workspace[i] = workspace_buf;
    }

    ctx->symmetric_buffer[0][i] = ctx->workspace[i]->buffer + BUFFER0_OFFSET(0);
    ctx->symmetric_buffer[1][i] = ctx->workspace[i]->buffer + BUFFER0_OFFSET(1);

    ctx->distributed_buffer[0][i] = ctx->workspace[i]->buffer + BUFFER1_OFFSET(0);
    ctx->distributed_buffer[1][i] = ctx->workspace[i]->buffer + BUFFER1_OFFSET(1);

    ctx->allgather_buffer[0][i] = ctx->workspace[i]->buffer + BUFFER2_OFFSET(0);
    ctx->allgather_buffer[1][i] = ctx->workspace[i]->buffer + BUFFER2_OFFSET(1);

    ctx->allgather_into_tensor_buffer[0][i] = ctx->workspace[i]->buffer + BUFFER3_OFFSET(0);
    ctx->allgather_into_tensor_buffer[1][i] = ctx->workspace[i]->buffer + BUFFER3_OFFSET(1);

    ctx->reduce_scatter_buffer[0][i] = ctx->workspace[i]->buffer + BUFFER4_OFFSET(0);
    ctx->reduce_scatter_buffer[1][i] = ctx->workspace[i]->buffer + BUFFER4_OFFSET(1);

    ctx->alltoall_buffer[0][i] = ctx->workspace[i]->buffer + BUFFER5_OFFSET(0);
    ctx->alltoall_buffer[1][i] = ctx->workspace[i]->buffer + BUFFER5_OFFSET(1);
  }

  const int64_t handle = static_cast<int64_t>(shm_contexts.size());
  shm_contexts.push_back(ctx);
  return handle;
}
void shm_default_initialize(int size, int rank, const char* addr_string, const char* port_string) {
  int64_t handle = shm_initialize(size, rank, addr_string, port_string, "default");

  default_shm_context = shm_contexts[handle];
}

int64_t shm_group_initialize(const std::string& group_name, int64_t group_size, int64_t group_rank) {
  const char* addr_string = getenv("MASTER_ADDR");
  const char* port_string = getenv("MASTER_PORT");

  TORCH_CHECK(addr_string != nullptr, "MASTER_ADDR is not set");
  TORCH_CHECK(port_string != nullptr, "MASTER_PORT is not set");

  return shm_initialize(group_size, group_rank, addr_string, port_string, group_name.c_str());
}

size_t slice_size(size_t chunk_el, int slice_idx, int group_size) {
  size_t slice_size = chunk_el / group_size;
  return slice_idx == group_size - 1 ? slice_size + (chunk_el % group_size) : slice_size;
}

char* slice_data(char* data_ptr, size_t chunk_el, int el_size, int slice_idx, int group_size) {
  size_t slice_size = chunk_el / group_size;
  size_t el_offset = slice_size * slice_idx;
  return data_ptr + el_offset * el_size;
}

size_t slice_el_start(size_t chunk_el, int slice_idx, int group_size) {
  size_t slice_size = chunk_el / group_size;
  return slice_size * slice_idx;
}

void symmetric_naive_all_reduce(
    struct shm_context* ctx, char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el) {
  const int state_group = STATE_GROUP_SYMMETRIC_ALLREDUCE;
  int& current_buffer = ctx->symmetric_current_buffer;
  int& state_idx = ctx->symmetric_state_idx;

  // init states to case 0 to get rid of "maybe-uninitialized" warning.
  enum coll_state copy_current = coll_allreduce_naive__copy_in_done;
  enum coll_state copy_next = coll_alt1_allreduce_naive__copy_in_done;

  switch (state_idx) {
    case 0:
      copy_current = coll_allreduce_naive__copy_in_done;
      copy_next = coll_alt1_allreduce_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_allreduce_naive__copy_in_done;
      copy_next = coll_alt2_allreduce_naive__copy_in_done;
      break;
    case 2:
      copy_current = coll_alt2_allreduce_naive__copy_in_done;
      copy_next = coll_allreduce_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }
  state_idx = (state_idx + 1) % 3;

  parallel_memcpy(ctx->symmetric_buffer[current_buffer][ctx->group_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  ctx->workspace[ctx->group_rank]->states[state_group] = copy_current;

  for (int i = 0; i < ctx->group_size; i++) {
    // wait until the other rank copy the buffer
    if (i != ctx->group_rank) {
      wait_buffer_state_until_2(ctx, i, copy_current, copy_next, state_group);
    }
  }

  // each rank reduce the buffer independently so there is no need for
  // synchronization afterward
  reduce_all_buffers(0, chunk_el, scalar_type, data_ptr, ctx->symmetric_buffer[current_buffer], ctx->group_size);

  // switch buffer
  current_buffer = 1 - current_buffer;
}

void distributed_naive_reduce(
    struct shm_context* ctx, char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el) {
  const int state_group = STATE_GROUP_DISTRIBUTED_ALLREDUCE;
  int& current_buffer = ctx->distributed_current_buffer;
  int& state_idx = ctx->distributed_state_idx;
  // init states to case 0 to get rid of "maybe-uninitialized" warning.
  enum coll_state copy_current = coll_allreduce_naive__copy_in_done;
  enum coll_state reduce_current = coll_allreduce_naive__reduce_done;
  enum coll_state copy_next = coll_alt1_allreduce_naive__copy_in_done;

  // similar to symmetric_naive_allreduce, but here we only need two sets of
  // states, because distributed naive reduce has two barriers in the algorithm
  switch (state_idx) {
    case 0:
      copy_current = coll_allreduce_naive__copy_in_done;
      reduce_current = coll_allreduce_naive__reduce_done;
      copy_next = coll_alt1_allreduce_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_allreduce_naive__copy_in_done;
      reduce_current = coll_alt1_allreduce_naive__reduce_done;
      copy_next = coll_allreduce_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }
  state_idx = (state_idx + 1) % 2;

  int data_size = chunk_size / chunk_el;
  parallel_memcpy(ctx->distributed_buffer[current_buffer][ctx->group_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  ctx->workspace[ctx->group_rank]->states[state_group] = copy_current;

  for (int i = 0; i < ctx->group_size; i++) {
    // wait until all the other ranks copy the buffer
    if (i != ctx->group_rank) {
      wait_buffer_state_until_2(ctx, i, copy_current, reduce_current, state_group);
    }
  }

  // reduce scatter
  reduce_all_buffers(
      slice_el_start(chunk_el, ctx->group_rank, ctx->group_size),
      slice_size(chunk_el, ctx->group_rank, ctx->group_size),
      scalar_type,
      ctx->distributed_buffer[current_buffer][ctx->group_rank],
      ctx->distributed_buffer[current_buffer],
      ctx->group_size);

  std::atomic_thread_fence(std::memory_order_release);
  ctx->workspace[ctx->group_rank]->states[state_group] = reduce_current;

  for (int i = 0; i < ctx->group_size; i++) {
    // wait until all the other ranks reduce the buffer
    if (i != ctx->group_rank) {
      wait_buffer_state_until_2(ctx, i, reduce_current, copy_next, state_group);
    }
  }

  for (int i = 0; i < ctx->group_size; i++) {
    int rank = (i + ctx->group_rank) % ctx->group_size;
    parallel_memcpy(
        slice_data(data_ptr, chunk_el, data_size, rank, ctx->group_size),
        slice_data(
            ctx->distributed_buffer[current_buffer][rank], chunk_el, chunk_size / chunk_el, rank, ctx->group_size),
        slice_size(chunk_el, rank, ctx->group_size) * data_size);
  }

  current_buffer = 1 - current_buffer;
}

void all_reduce_outer_loop(torch::Tensor& data, size_t numel, int data_size, int64_t handle) {
  auto* ctx = get_shm_context(handle);
  for (int offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
    auto data_ptr = ((char*)(data.data_ptr()) + offset);
    size_t chunk_size = data_size - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : data_size - offset;
    size_t chunk_el = chunk_size / (data_size / numel);
    if (chunk_size < NAIVE_ALLREDUCE_THRESHOLD) {
      symmetric_naive_all_reduce(ctx, data_ptr, data.scalar_type(), chunk_size, chunk_el);
    } else {
      distributed_naive_reduce(ctx, data_ptr, data.scalar_type(), chunk_size, chunk_el);
    }
  }
}

template <int STATE_GROUP>
void naive_all_gather(
    struct shm_context* ctx, char* result_ptr, char* data_ptr, size_t res_stride, size_t chunk_size, size_t chunk_el) {
  int& current_buffer =
      STATE_GROUP == STATE_GROUP_ALL_GATHER ? ctx->allgather_current_buffer : ctx->allgather_into_tensor_current_buffer;
  int& state_idx =
      STATE_GROUP == STATE_GROUP_ALL_GATHER ? ctx->allgather_state_idx : ctx->allgather_into_tensor_state_idx;

  char*** buffer = nullptr;
  if constexpr (STATE_GROUP == STATE_GROUP_ALL_GATHER) {
    buffer = ctx->allgather_buffer;
  } else if constexpr (STATE_GROUP == STATE_GROUP_ALL_GATHER_INTO_TENSOR) {
    buffer = ctx->allgather_into_tensor_buffer;
  } else {
    static_assert(
        STATE_GROUP == STATE_GROUP_ALL_GATHER || STATE_GROUP == STATE_GROUP_ALL_GATHER_INTO_TENSOR,
        "Unsupported STATE_GROUP");
  }

  // init states to case 0 to get rid of "maybe-uninitialized" warning.
  enum coll_state copy_current = coll_allgather_naive__copy_in_done;
  enum coll_state copy_next = coll_alt1_allgather_naive__copy_in_done;

  switch (state_idx) {
    case 0:
      copy_current = coll_allgather_naive__copy_in_done;
      copy_next = coll_alt1_allgather_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_allgather_naive__copy_in_done;
      copy_next = coll_alt2_allgather_naive__copy_in_done;
      break;
    case 2:
      copy_current = coll_alt2_allgather_naive__copy_in_done;
      copy_next = coll_allgather_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }
  state_idx = (state_idx + 1) % 3;

  parallel_memcpy(buffer[current_buffer][ctx->group_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  ctx->workspace[ctx->group_rank]->states[STATE_GROUP] = copy_current;

  for (int i = 0; i < ctx->group_size; i++) {
    // wait until all the other ranks copy the buffer
    if (i != ctx->group_rank) {
      wait_buffer_state_until_2(ctx, i, copy_current, copy_next, STATE_GROUP);
    }
  }
  for (int i = 0; i < ctx->group_size; i++) {
    parallel_memcpy(result_ptr + i * res_stride, buffer[current_buffer][i], chunk_size);
  }
  current_buffer = 1 - current_buffer;
}

template <int STATE_GROUP>
torch::Tensor&
all_gather(torch::Tensor& result, torch::Tensor& data, int dim, size_t numel, int data_size, int64_t handle) {
  auto* ctx = get_shm_context(handle);
  size_t dim_el = data.stride(dim) * data.size(dim);
  int dtype_size = data_size / numel;
  size_t dim_size = dim_el * dtype_size;
  int dim_count = data_size / dim_size;
  auto data_ptr = (char*)(data.data_ptr());
  auto result_ptr = (char*)(result.data_ptr());
  for (int i = 0; i < dim_count; i++) {
    for (size_t offset = 0; offset < dim_size; offset += MAX_BUF_SIZE) {
      size_t chunk_size = dim_size - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : dim_size - offset;
      size_t chunk_el = chunk_size / dtype_size;
      naive_all_gather<STATE_GROUP>(
          ctx,
          result_ptr + i * dim_size * ctx->group_size + offset,
          data_ptr + i * dim_size + offset,
          dim_size,
          chunk_size,
          chunk_el);
    }
  }
  return result;
}
template torch::Tensor& all_gather<STATE_GROUP_ALL_GATHER>(torch::Tensor&, torch::Tensor&, int, size_t, int, int64_t);
template torch::Tensor&
all_gather<STATE_GROUP_ALL_GATHER_INTO_TENSOR>(torch::Tensor&, torch::Tensor&, int, size_t, int, int64_t);
void naive_reduce_scatter(
    struct shm_context* ctx,
    char* output_ptr,
    char* data_ptr,
    c10::ScalarType scalar_type,
    size_t chunk_size,
    size_t chunk_el,
    int element_size) {
  const int state_group = STATE_GROUP_REDUCE_SCATTER;
  int& current_buffer = ctx->reduce_scatter_current_buffer;
  int& state_idx = ctx->reduce_scatter_state_idx;

  enum coll_state copy_current = coll_reduce_scatter_naive__copy_in_done;
  enum coll_state copy_next = coll_alt1_reduce_scatter_naive__copy_in_done;

  switch (state_idx) {
    case 0:
      copy_current = coll_reduce_scatter_naive__copy_in_done;
      copy_next = coll_alt1_reduce_scatter_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_reduce_scatter_naive__copy_in_done;
      copy_next = coll_alt2_reduce_scatter_naive__copy_in_done;
      break;
    case 2:
      copy_current = coll_alt2_reduce_scatter_naive__copy_in_done;
      copy_next = coll_reduce_scatter_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }
  state_idx = (state_idx + 1) % 3;

  // Step 1: copy local data to shared buffer
  parallel_memcpy(ctx->reduce_scatter_buffer[current_buffer][ctx->group_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  ctx->workspace[ctx->group_rank]->states[state_group] = copy_current;

  // Step 2: wait for all ranks to copy in
  for (int i = 0; i < ctx->group_size; i++) {
    if (i != ctx->group_rank) {
      wait_buffer_state_until_2(ctx, i, copy_current, copy_next, state_group);
    }
  }

  // Step 3: do local reduce on this rank's slice only
  int start_el = slice_el_start(chunk_el, ctx->group_rank, ctx->group_size);

  // each rank reduce its slice of buffer independently so there is no need for
  // synchronization afterward
  reduce_all_buffers(
      start_el,
      slice_size(chunk_el, ctx->group_rank, ctx->group_size),
      scalar_type,
      output_ptr - start_el * element_size,
      ctx->reduce_scatter_buffer[current_buffer],
      ctx->group_size);

  // done
  current_buffer = 1 - current_buffer;
}

void reduce_scatter_outer_loop(
    torch::Tensor& output, torch::Tensor& data, size_t numel, int data_size, int64_t handle) {
  auto* ctx = get_shm_context(handle);
  for (int offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
    auto data_ptr = ((char*)(data.data_ptr()) + offset);
    auto output_ptr = ((char*)(output.data_ptr()) + offset);
    size_t chunk_size = std::min((size_t)MAX_BUF_SIZE, (size_t)(data_size - offset));
    size_t chunk_el = chunk_size / (data_size / numel);

    naive_reduce_scatter(ctx, output_ptr, data_ptr, data.scalar_type(), chunk_size, chunk_el, data.element_size());
  }
}

void all_to_all(char* output_ptr, char* input_ptr, size_t data_size, int64_t handle) {
  auto* ctx = get_shm_context(handle);

  TORCH_CHECK(data_size <= MAX_BUF_SIZE, "SHM all-to-all input size exceeds maximum buffer size");
  TORCH_CHECK(data_size % ctx->group_size == 0, "SHM all-to-all input size must be divisible by group size");

  int& current_buffer = ctx->alltoall_current_buffer;
  int& state_idx = ctx->alltoall_state_idx;

  enum coll_state copy_current = coll_allgather_naive__copy_in_done;
  enum coll_state copy_next = coll_alt1_allgather_naive__copy_in_done;

  switch (state_idx) {
    case 0:
      copy_current = coll_allgather_naive__copy_in_done;
      copy_next = coll_alt1_allgather_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_allgather_naive__copy_in_done;
      copy_next = coll_alt2_allgather_naive__copy_in_done;
      break;
    case 2:
      copy_current = coll_alt2_allgather_naive__copy_in_done;
      copy_next = coll_allgather_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }

  state_idx = (state_idx + 1) % 3;
  // copy local input to shared buffer
  parallel_memcpy(ctx->alltoall_buffer[current_buffer][ctx->group_rank], input_ptr, data_size);
  std::atomic_thread_fence(std::memory_order_release);
  ctx->workspace[ctx->group_rank]->states[STATE_GROUP_ALL_TO_ALL] = copy_current;

  // wait until all ranks copy the buffer
  for (int i = 0; i < ctx->group_size; i++) {
    if (i != ctx->group_rank) {
      wait_buffer_state_until_2(ctx, i, copy_current, copy_next, STATE_GROUP_ALL_TO_ALL);
    }
  }

  size_t chunk_size = data_size / ctx->group_size;

  // read the chunk addressed to this rank from every source rank
  for (int i = 0; i < ctx->group_size; i++) {
    parallel_memcpy(
        output_ptr + i * chunk_size,
        ctx->alltoall_buffer[current_buffer][i] + ctx->group_rank * chunk_size,
        chunk_size);
  }
  current_buffer = 1 - current_buffer;
}
