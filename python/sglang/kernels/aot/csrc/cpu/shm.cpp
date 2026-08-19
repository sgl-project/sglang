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

static int world_size;

// SHM based allreduce helper functions
// buffer that holds shm name
#define NAME_BUF_SIZE 1000
#define MAX_BUF_SIZE 1048576 * 32
#define NAIVE_ALLREDUCE_THRESHOLD 1048576
#define SHM_BUFFER_NAME "deepspeed_allreduce_buffer"
struct allreduce_workspace {
  enum coll_state states[5];  // idx=0 -- state for symmetric_naive_all_reduce
                              // idx=1 -- state for distributed_naive_all_reduce
                              // idx=2 -- state for all_gather
                              // idx=3 -- state for all_gather_into_tensor
                              // idx=4 -- state for reduce_scatter
  // double buffer to avoid syncing between rounds
  // offset=0 -- 2*NAIVE_ALLREDUCE_THRESHOLD : buffer for
  // symmetric_naive_all_reduce after that : buffer for
  // distributed_naive_all_reduce
  char buffer
      [2 * NAIVE_ALLREDUCE_THRESHOLD +  // symmetric allreduce
       2 * MAX_BUF_SIZE +               // distributed naive reduce
       2 * MAX_BUF_SIZE +               // allgather
       2 * MAX_BUF_SIZE +               // allgather_into_tensor
       2 * MAX_BUF_SIZE                 // reduce_scatter
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

struct allreduce_workspace** workspace;

// buffer for small messages, double buffer
char** symmetric_buffer[2];
// buffer for large messages, double buffer
char** distributed_buffer[2];

char** allgather_buffer[2];
char** allgather_into_tensor_buffer[2];
char** reduce_scatter_buffer[2];

void wait_buffer_state_until_2(int index, enum coll_state state0, enum coll_state state1, int state_group) {
  volatile enum coll_state* state_ptr = &(workspace[index]->states[state_group]);

  while (1) {
    volatile enum coll_state cur_state = *state_ptr;
    if (cur_state == state0 || cur_state == state1) break;
  }
}

void reduce_all_buffers(
    int start_elements,
    int num_elements,
    c10::ScalarType scalar_type,
    int to_buffer_idx,
    char* to_buffer,
    char** buffers,
    int reduce_world_size) {
  switch (scalar_type) {
    case c10::ScalarType::BFloat16:
      reduce_bf16_buffers(start_elements, num_elements, to_buffer, buffers, reduce_world_size);
      break;
    case c10::ScalarType::Half:
      reduce_fp16_buffers(start_elements, num_elements, to_buffer, buffers, reduce_world_size);
      break;
    case c10::ScalarType::Float:
      reduce_fp32_buffers(start_elements, num_elements, to_buffer, buffers, reduce_world_size);
      break;
    default:
      assert(!"Should not get here");
  }
}

static bool is_initialized = false;
static int world_rank;

void shm_initialize(int size, int rank, const char* addr_string, const char* port_string) {
  if (is_initialized) {
    return;
  }
  is_initialized = true;

  world_size = size;
  world_rank = rank;

  char shm_name_prefix[NAME_BUF_SIZE];
  char shm_name[NAME_BUF_SIZE];
  snprintf(shm_name_prefix, NAME_BUF_SIZE, "%s_%d_%s_%s", SHM_BUFFER_NAME, getuid(), addr_string, port_string);
  // create shared workspace for SHM based allreduce
  SharedData allreduce_buffer;
  // allocate workspace_buf for current rank
  struct allreduce_workspace* workspace_buf;
  struct allreduce_workspace* workspace_buf_other;
  workspace_buf = (struct allreduce_workspace*)malloc(sizeof(struct allreduce_workspace));
  snprintf(shm_name, NAME_BUF_SIZE, "%.900s_%d", shm_name_prefix, rank);
  shared_create(&allreduce_buffer, shm_name, workspace_buf, sizeof(struct allreduce_workspace));
  workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes;
  workspace_buf->states[STATE_GROUP_SYMMETRIC_ALLREDUCE] =
      coll_alt2_allreduce_naive__copy_in_done;                            // symmetric_naive_all_reduce
  workspace_buf->states[STATE_GROUP_DISTRIBUTED_ALLREDUCE] = coll_begin;  // distributed_naive_reduce
  workspace_buf->states[STATE_GROUP_ALL_GATHER] = coll_alt2_allgather_naive__copy_in_done;  // all_gather
  workspace_buf->states[STATE_GROUP_ALL_GATHER_INTO_TENSOR] =
      coll_alt2_allgather_naive__copy_in_done;                     // all_gather_into_tensor
  workspace_buf->states[STATE_GROUP_REDUCE_SCATTER] = coll_begin;  // reduce_scatter

  // create the workspace pointer list
  workspace = (struct allreduce_workspace**)malloc(size * sizeof(struct allreduce_workspace*));
  symmetric_buffer[0] = (char**)malloc(size * sizeof(char**));
  symmetric_buffer[1] = (char**)malloc(size * sizeof(char**));
  distributed_buffer[0] = (char**)malloc(size * sizeof(char**));
  distributed_buffer[1] = (char**)malloc(size * sizeof(char**));

  allgather_buffer[0] = (char**)malloc(size * sizeof(char*));
  allgather_buffer[1] = (char**)malloc(size * sizeof(char*));

  allgather_into_tensor_buffer[0] = (char**)malloc(size * sizeof(char*));
  allgather_into_tensor_buffer[1] = (char**)malloc(size * sizeof(char*));

  reduce_scatter_buffer[0] = (char**)malloc(size * sizeof(char*));
  reduce_scatter_buffer[1] = (char**)malloc(size * sizeof(char*));

  // map shm of all ranks
  for (int i = 0; i < size; i++) {
    if (i != rank) {
      snprintf(shm_name, NAME_BUF_SIZE, "%.900s_%d", shm_name_prefix, i);
      // printf("open %s, %d\n", shm_name, rank);
      do {
        shared_open(&allreduce_buffer, shm_name, sizeof(struct allreduce_workspace));
      } while (allreduce_buffer.descriptor == -1 && errno == ENOENT);
      workspace_buf_other = (struct allreduce_workspace*)allreduce_buffer.bytes;
      workspace[i] = workspace_buf_other;
    } else {
      workspace[i] = workspace_buf;
    }
    symmetric_buffer[0][i] = workspace[i]->buffer + BUFFER0_OFFSET(0);
    symmetric_buffer[1][i] = workspace[i]->buffer + BUFFER0_OFFSET(1);
    distributed_buffer[0][i] = workspace[i]->buffer + BUFFER1_OFFSET(0);
    distributed_buffer[1][i] = workspace[i]->buffer + BUFFER1_OFFSET(1);

    allgather_buffer[0][i] = workspace[i]->buffer + BUFFER2_OFFSET(0);
    allgather_buffer[1][i] = workspace[i]->buffer + BUFFER2_OFFSET(1);

    allgather_into_tensor_buffer[0][i] = workspace[i]->buffer + BUFFER3_OFFSET(0);
    allgather_into_tensor_buffer[1][i] = workspace[i]->buffer + BUFFER3_OFFSET(1);

    reduce_scatter_buffer[0][i] = workspace[i]->buffer + BUFFER4_OFFSET(0);
    reduce_scatter_buffer[1][i] = workspace[i]->buffer + BUFFER4_OFFSET(1);
  }
}

#define positive_mod(num, mod) ((((num) % (mod)) + (mod)) % (mod))
#define rank_mod(rank) positive_mod(rank, world_size)
size_t slice_size(size_t chunk_el, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  return slice_idx == world_size - 1 ? slice_size + (chunk_el % world_size) : slice_size;
}

char* slice_data(char* data_ptr, size_t chunk_el, int el_size, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  size_t el_offset = slice_size * slice_idx;
  return data_ptr + el_offset * el_size;
}

size_t slice_el_start(size_t chunk_el, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  return slice_size * slice_idx;
}

size_t group_slice_el_start(size_t chunk_el, int slice_idx, int group_size) {
  const size_t size = chunk_el / group_size;
  return size * slice_idx;
}
size_t group_slice_size(size_t chunk_el, int slice_idx, int group_size) {
  size_t size = chunk_el / group_size;
  if (slice_idx == group_size - 1) {
    size += chunk_el % group_size;
  }
  return size;
}
char* group_slice_data(char* data_ptr, size_t chunk_el, int el_size, int slice_idx, int group_size) {
  size_t slice_size = chunk_el / group_size;
  size_t el_offset = slice_size * slice_idx;
  return data_ptr + el_offset * el_size;
}
void symmetric_naive_all_reduce(char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el) {
  const int state_group = STATE_GROUP_SYMMETRIC_ALLREDUCE;
  static int current_buffer = 0;
  static int state_idx = 0;

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

  parallel_memcpy(symmetric_buffer[current_buffer][world_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = copy_current;

  for (int i = 0; i < world_size; i++) {
    // wait until the other rank copy the buffer
    if (i != world_rank) {
      wait_buffer_state_until_2(i, copy_current, copy_next, state_group);
    }
  }

  // each rank reduce the buffer independently so therre is no need for
  // synchronization afterward
  reduce_all_buffers(0, chunk_el, scalar_type, world_rank, data_ptr, symmetric_buffer[current_buffer], world_size);

  // switch buffer
  current_buffer = 1 - current_buffer;
}

// naive allreduce distributed, each rank do naive reduce on its slice
void distributed_naive_reduce(char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el) {
  const int state_group = STATE_GROUP_DISTRIBUTED_ALLREDUCE;
  static int current_buffer = 0;
  static int state_idx = 0;

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
  parallel_memcpy(distributed_buffer[current_buffer][world_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = copy_current;

  for (int i = 0; i < world_size; i++) {
    // wait until all the other ranks copy the buffer
    if (i != world_rank) wait_buffer_state_until_2(i, copy_current, reduce_current, state_group);
  }

  // reduce scatter
  reduce_all_buffers(
      slice_el_start(chunk_el, world_rank),
      slice_size(chunk_el, world_rank),
      scalar_type,
      world_rank,
      distributed_buffer[current_buffer][world_rank],
      distributed_buffer[current_buffer],
      world_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = reduce_current;

  for (int i = 0; i < world_size; i++) {
    // wait until all the other ranks reduce the buffer
    if (i != world_rank) wait_buffer_state_until_2(i, reduce_current, copy_next, state_group);
  }

  for (int i = 0; i < world_size; i++) {
    int rank = (i + world_rank) % world_size;
    parallel_memcpy(
        slice_data(data_ptr, chunk_el, data_size, rank),
        slice_data(distributed_buffer[current_buffer][rank], chunk_el, chunk_size / chunk_el, rank),
        slice_size(chunk_el, rank) * data_size);
  }

  current_buffer = 1 - current_buffer;
}

void all_reduce_outer_loop(torch::Tensor& data, size_t numel, int data_size) {
  for (int offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
    auto data_ptr = ((char*)(data.data_ptr()) + offset);
    size_t chunk_size = data_size - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : data_size - offset;
    size_t chunk_el = chunk_size / (data_size / numel);
    if (chunk_size < NAIVE_ALLREDUCE_THRESHOLD) {
      symmetric_naive_all_reduce(data_ptr, data.scalar_type(), chunk_size, chunk_el);
    } else {
      distributed_naive_reduce(data_ptr, data.scalar_type(), chunk_size, chunk_el);
    }
  }
}

template <int STATE_GROUP>
void naive_all_gather(char* result_ptr, char* data_ptr, size_t res_stride, size_t chunk_size, size_t chunk_el) {
  static int current_buffer = 0;
  static int state_idx = 0;

  char*** buffer = nullptr;
  if constexpr (STATE_GROUP == STATE_GROUP_ALL_GATHER) {
    buffer = allgather_buffer;
  } else if constexpr (STATE_GROUP == STATE_GROUP_ALL_GATHER_INTO_TENSOR) {
    buffer = allgather_into_tensor_buffer;
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

  parallel_memcpy(buffer[current_buffer][world_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[STATE_GROUP] = copy_current;

  for (int i = 0; i < world_size; i++) {
    // wait until all the other ranks copy the buffer
    if (i != world_rank) wait_buffer_state_until_2(i, copy_current, copy_next, STATE_GROUP);
  }
  for (int i = 0; i < world_size; i++) {
    parallel_memcpy(result_ptr + i * res_stride, buffer[current_buffer][i], chunk_size);
  }
  current_buffer = 1 - current_buffer;
}

template <int STATE_GROUP>
torch::Tensor& all_gather(torch::Tensor& result, torch::Tensor& data, int dim, size_t numel, int data_size) {
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
          result_ptr + i * dim_size * world_size + offset,
          data_ptr + i * dim_size + offset,
          dim_size,
          chunk_size,
          chunk_el);
    }
  }
  return result;
}

template torch::Tensor& all_gather<STATE_GROUP_ALL_GATHER>(torch::Tensor&, torch::Tensor&, int, size_t, int);
template torch::Tensor&
all_gather<STATE_GROUP_ALL_GATHER_INTO_TENSOR>(torch::Tensor&, torch::Tensor&, int, size_t, int);

void naive_reduce_scatter(
    char* output_ptr,
    char* data_ptr,
    c10::ScalarType scalar_type,
    size_t chunk_size,
    size_t chunk_el,
    int element_size) {
  const int state_group = STATE_GROUP_REDUCE_SCATTER;
  static int current_buffer = 0;
  static int state_idx = 0;

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
  parallel_memcpy(reduce_scatter_buffer[current_buffer][world_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = copy_current;

  // Step 2: wait for all ranks to copy in
  for (int i = 0; i < world_size; i++) {
    if (i != world_rank) wait_buffer_state_until_2(i, copy_current, copy_next, state_group);
  }

  // // Step 3: do local reduce on this rank’s slice only
  int start_el = slice_el_start(chunk_el, world_rank);
  // each rank reduce its slice of buffer independently so therre is no need for
  // synchronization afterward
  reduce_all_buffers(
      start_el,
      slice_size(chunk_el, world_rank),
      scalar_type,
      world_rank,
      output_ptr -
          start_el * element_size,  // in reduce_all_buffers, the output_ptr is the buffer for all ranks, but here
                                    // output_ptr is already the local buffer for one rank. Adjust it here.
      reduce_scatter_buffer[current_buffer],
      world_size);

  // done
  current_buffer = 1 - current_buffer;
}

void reduce_scatter_outer_loop(torch::Tensor& output, torch::Tensor& data, size_t numel, int data_size) {
  for (int offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
    auto data_ptr = ((char*)(data.data_ptr()) + offset);
    auto output_ptr = ((char*)(output.data_ptr()) + offset);
    size_t chunk_size = std::min((size_t)MAX_BUF_SIZE, (size_t)(data_size - offset));
    size_t chunk_el = chunk_size / (data_size / numel);

    naive_reduce_scatter(output_ptr, data_ptr, data.scalar_type(), chunk_size, chunk_el, data.element_size());
  }
}

#define GROUP_ALLGATHER_MAX_BUF_SIZE MAX_BUF_SIZE
#define GROUP_ALLTOALL_MAX_BUF_SIZE (64 * 1024 * 1024)
#define GROUP_SYMMETRIC_ALLREDUCE_MAX_BUF_SIZE MAX_BUF_SIZE
#define GROUP_DISTRIBUTED_ALLREDUCE_MAX_BUF_SIZE MAX_BUF_SIZE
#define GROUP_STATE_ALL_GATHER 0
#define GROUP_STATE_ALL_TO_ALL 1
#define GROUP_STATE_SYMMETRIC_ALL_REDUCE 2
#define GROUP_STATE_DISTRIBUTED_ALL_REDUCE 3
struct group_workspace {
  enum coll_state states[2];
  char buffer
      [2 * GROUP_ALLGATHER_MAX_BUF_SIZE + 2 * GROUP_ALLTOALL_MAX_BUF_SIZE + 2 * GROUP_SYMMETRIC_ALLREDUCE_MAX_BUF_SIZE +
       2 * GROUP_DISTRIBUTED_ALLREDUCE_MAX_BUF_SIZE];
};
#define GROUP_ALLGATHER_BUFFER_OFFSET(current_buffer) current_buffer* GROUP_ALLGATHER_MAX_BUF_SIZE
#define GROUP_ALLTOALL_BUFFER_OFFSET(current_buffer) \
  2 * GROUP_ALLGATHER_MAX_BUF_SIZE + current_buffer* GROUP_ALLTOALL_MAX_BUF_SIZE
#define GROUP_SYMMETRIC_ALLREDUCE_BUFFER_OFFSET(current_buffer)        \
  2 * GROUP_ALLGATHER_MAX_BUF_SIZE + 2 * GROUP_ALLTOALL_MAX_BUF_SIZE + \
      current_buffer* GROUP_SYMMETRIC_ALLREDUCE_MAX_BUF_SIZE
#define GROUP_DISTRIBUTED_ALLREDUCE_BUFFER_OFFSET(current_buffer)                                                   \
  2 * GROUP_ALLGATHER_MAX_BUF_SIZE + 2 * GROUP_ALLTOALL_MAX_BUF_SIZE + 2 * GROUP_SYMMETRIC_ALLREDUCE_MAX_BUF_SIZE + \
      current_buffer* GROUP_DISTRIBUTED_ALLREDUCE_MAX_BUF_SIZE
struct group_shm_context {
  int group_size;
  int group_rank;

  // workspace[i] points to group-rank i's shared workspace.
  struct group_workspace** workspace;

  // buffer[double_buffer_idx][source_group_rank]
  char** allgather_buffer[2];
  char** alltoall_buffer[2];
  char** symmetric_allreduce_buffer[2];
  char** distributed_allreduce_buffer[2];

  // State must be per context instead of function-static because a process
  // can participate in multiple ProcessGroups.
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

static void get_group_copy_states(int state_idx, enum coll_state* copy_current, enum coll_state* copy_next) {
  switch (state_idx) {
    case 0:
      *copy_current = coll_allgather_naive__copy_in_done;
      *copy_next = coll_alt1_allgather_naive__copy_in_done;
      break;

    case 1:
      *copy_current = coll_alt1_allgather_naive__copy_in_done;
      *copy_next = coll_alt2_allgather_naive__copy_in_done;
      break;

    case 2:
      *copy_current = coll_alt2_allgather_naive__copy_in_done;
      *copy_next = coll_allgather_naive__copy_in_done;
      break;

    default:
      assert(!"Invalid group SHM state index.");
  }
}

static void wait_group_buffer_state_until_2(
    struct group_shm_context* ctx, int rank, enum coll_state state0, enum coll_state state1, int state_group) {
  volatile enum coll_state* state_ptr = &(ctx->workspace[rank]->states[state_group]);

  while (1) {
    volatile enum coll_state cur_state = *state_ptr;

    if (cur_state == state0 || cur_state == state1) {
      break;
    }
  }
}

int64_t shm_group_initialize(const std::string& group_name, int64_t group_size, int64_t group_rank) {
  auto* ctx = (struct group_shm_context*)malloc(sizeof(struct group_shm_context));
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

  char shm_name[NAME_BUF_SIZE];

  auto* workspace_buf = (struct group_workspace*)calloc(1, sizeof(struct group_workspace));
  workspace_buf->states[GROUP_STATE_ALL_GATHER] = coll_alt2_allgather_naive__copy_in_done;

  workspace_buf->states[GROUP_STATE_ALL_TO_ALL] = coll_alt2_allgather_naive__copy_in_done;
  workspace_buf->states[GROUP_STATE_SYMMETRIC_ALL_REDUCE] = coll_alt2_allreduce_naive__copy_in_done;

  workspace_buf->states[GROUP_STATE_DISTRIBUTED_ALL_REDUCE] = coll_begin;

  snprintf(shm_name, NAME_BUF_SIZE, "%.900s_%d", group_name.c_str(), group_rank);

  SharedData local_shm = {};
  local_shm.descriptor = -1;

  shared_create(&local_shm, shm_name, workspace_buf, sizeof(struct group_workspace));
  free(workspace_buf);

  auto* local_workspace = (struct group_workspace*)local_shm.bytes;

  for (int i = 0; i < group_size; ++i) {
    if (i == group_rank) {
      ctx->workspace[i] = local_workspace;
    } else {
      snprintf(shm_name, NAME_BUF_SIZE, "%.900s_%d", group_name.c_str(), i);

      SharedData peer_shm = {};
      peer_shm.descriptor = -1;

      do {
        shared_open(&peer_shm, shm_name, sizeof(struct group_workspace));
      } while (peer_shm.descriptor == -1 && errno == ENOENT);

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
  enum coll_state copy_current = coll_allgather_naive__copy_in_done;

  enum coll_state copy_next = coll_alt1_allgather_naive__copy_in_done;
  get_group_copy_states(ctx->allgather_state_idx, &copy_current, &copy_next);

  ctx->allgather_state_idx = (ctx->allgather_state_idx + 1) % 3;

  const int current_buffer = ctx->allgather_current_buffer;

  // Step 1:
  // publish my complete input into my SHM workspace.
  parallel_memcpy(ctx->allgather_buffer[current_buffer][ctx->group_rank], input_ptr, data_size);

  std::atomic_thread_fence(std::memory_order_release);

  ctx->workspace[ctx->group_rank]->states[GROUP_STATE_ALL_GATHER] = copy_current;

  // Step 2:
  // wait for all members of THIS ProcessGroup.
  for (int i = 0; i < ctx->group_size; ++i) {
    if (i != ctx->group_rank) {
      wait_group_buffer_state_until_2(ctx, i, copy_current, copy_next, GROUP_STATE_ALL_GATHER);
    }
  }

  // Step 3:
  // gather every source's complete buffer.
  for (int src = 0; src < ctx->group_size; ++src) {
    parallel_memcpy(
        output_ptr + static_cast<size_t>(src) * data_size, ctx->allgather_buffer[current_buffer][src], data_size);
  }

  ctx->allgather_current_buffer = 1 - current_buffer;
}

void group_all_to_all(int64_t handle, char* output_ptr, char* input_ptr, size_t data_size) {
  auto* ctx = get_group_shm_context(handle);
  enum coll_state copy_current = coll_allgather_naive__copy_in_done;

  enum coll_state copy_next = coll_alt1_allgather_naive__copy_in_done;

  get_group_copy_states(ctx->alltoall_state_idx, &copy_current, &copy_next);

  ctx->alltoall_state_idx = (ctx->alltoall_state_idx + 1) % 3;

  const int current_buffer = ctx->alltoall_current_buffer;

  // Step 1:
  // Publish the complete destination-major input:
  parallel_memcpy(ctx->alltoall_buffer[current_buffer][ctx->group_rank], input_ptr, data_size);

  std::atomic_thread_fence(std::memory_order_release);

  ctx->workspace[ctx->group_rank]->states[GROUP_STATE_ALL_TO_ALL] = copy_current;

  // Step 2:
  // wait for all members of this actual ProcessGroup.
  for (int i = 0; i < ctx->group_size; ++i) {
    if (i != ctx->group_rank) {
      wait_group_buffer_state_until_2(ctx, i, copy_current, copy_next, GROUP_STATE_ALL_TO_ALL);
    }
  }
  const size_t peer_chunk_size = data_size / static_cast<size_t>(ctx->group_size);

  // Step 3:
  // Only read the chunk intended for this destination from each source.
  // group_rank r reads:
  //     source_buffer + r * peer_chunk_size
  for (int src = 0; src < ctx->group_size; ++src) {
    char* src_ptr = ctx->alltoall_buffer[current_buffer][src] + static_cast<size_t>(ctx->group_rank) * peer_chunk_size;

    char* dst_ptr = output_ptr + static_cast<size_t>(src) * peer_chunk_size;

    parallel_memcpy(dst_ptr, src_ptr, peer_chunk_size);
  }
  ctx->alltoall_current_buffer = 1 - current_buffer;
}

void group_symmetric_all_reduce(
    struct group_shm_context* ctx, char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el) {
  enum coll_state copy_current = coll_allreduce_naive__copy_in_done;
  enum coll_state copy_next = coll_alt1_allreduce_naive__copy_in_done;

  get_group_copy_states(ctx->symmetric_allreduce_state_idx, &copy_current, &copy_next);

  ctx->symmetric_allreduce_state_idx = (ctx->symmetric_allreduce_state_idx + 1) % 3;

  const int current_buffer = ctx->symmetric_allreduce_current_buffer;

  // Step 1:
  // Copy local input into this rank's SHM buffer.
  parallel_memcpy(ctx->symmetric_allreduce_buffer[current_buffer][ctx->group_rank], data_ptr, chunk_size);

  std::atomic_thread_fence(std::memory_order_release);

  ctx->workspace[ctx->group_rank]->states[GROUP_STATE_SYMMETRIC_ALL_REDUCE] = copy_current;

  // Step 2:
  // Wait until every rank in THIS ProcessGroup
  // has published its input.
  for (int i = 0; i < ctx->group_size; ++i) {
    if (i != ctx->group_rank) {
      wait_group_buffer_state_until_2(ctx, i, copy_current, copy_next, GROUP_STATE_SYMMETRIC_ALL_REDUCE);
    }
  }

  // Step 3:
  // Each rank independently reduces all group buffers.
  reduce_all_buffers(
      0,
      chunk_el,
      scalar_type,
      ctx->group_rank,
      data_ptr,
      ctx->symmetric_allreduce_buffer[current_buffer],
      ctx->group_size);

  // Step 4:
  // Switch double buffer.
  ctx->symmetric_allreduce_current_buffer = 1 - current_buffer;
}

void group_distributed_reduce(
    struct group_shm_context* ctx, char* data_ptr, c10::ScalarType scalar_type, size_t chunk_size, size_t chunk_el) {
  const int state_group = GROUP_STATE_DISTRIBUTED_ALL_REDUCE;

  const int current_buffer = ctx->distributed_allreduce_current_buffer;

  enum coll_state copy_current = coll_allreduce_naive__copy_in_done;

  enum coll_state reduce_current = coll_allreduce_naive__reduce_done;

  enum coll_state copy_next = coll_alt1_allreduce_naive__copy_in_done;

  switch (ctx->distributed_allreduce_state_idx) {
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

  ctx->distributed_allreduce_state_idx = (ctx->distributed_allreduce_state_idx + 1) % 2;

  const int element_size = static_cast<int>(chunk_size / chunk_el);

  // Step 1:
  // Every rank publishes the complete input.
  parallel_memcpy(ctx->distributed_allreduce_buffer[current_buffer][ctx->group_rank], data_ptr, chunk_size);

  std::atomic_thread_fence(std::memory_order_release);

  ctx->workspace[ctx->group_rank]->states[state_group] = copy_current;

  // Step 2:
  // Wait until all group members have published input.
  for (int i = 0; i < ctx->group_size; ++i) {
    if (i != ctx->group_rank) {
      wait_group_buffer_state_until_2(ctx, i, copy_current, reduce_current, state_group);
    }
  }

  // Step 3:
  // Reduce only this rank's slice.
  const size_t start_el = group_slice_el_start(chunk_el, ctx->group_rank, ctx->group_size);

  const size_t local_el = group_slice_size(chunk_el, ctx->group_rank, ctx->group_size);

  reduce_all_buffers(
      start_el,
      local_el,
      scalar_type,
      ctx->group_rank,
      ctx->distributed_allreduce_buffer[current_buffer][ctx->group_rank],
      ctx->distributed_allreduce_buffer[current_buffer],
      ctx->group_size);

  std::atomic_thread_fence(std::memory_order_release);
  ctx->workspace[ctx->group_rank]->states[state_group] = reduce_current;

  // Step 4:
  // Wait until every rank has reduced its own slice.
  for (int i = 0; i < ctx->group_size; ++i) {
    if (i != ctx->group_rank) {
      wait_group_buffer_state_until_2(ctx, i, reduce_current, copy_next, state_group);
    }
  }

  // Step 5:
  // Gather reduced slices from every group rank
  // back into the local output tensor.
  for (int i = 0; i < ctx->group_size; ++i) {
    const int rank = (i + ctx->group_rank) % ctx->group_size;
    parallel_memcpy(
        group_slice_data(data_ptr, chunk_el, element_size, rank, ctx->group_size),
        group_slice_data(
            ctx->distributed_allreduce_buffer[current_buffer][rank], chunk_el, element_size, rank, ctx->group_size),
        group_slice_size(chunk_el, rank, ctx->group_size) * element_size);
  }

  ctx->distributed_allreduce_current_buffer = 1 - current_buffer;
}

void group_all_reduce(int64_t handle, char* data_ptr, c10::ScalarType scalar_type, size_t data_size, size_t numel) {
  auto* ctx = get_group_shm_context(handle);

  const size_t element_size = data_size / numel;

  for (size_t offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
    char* chunk_ptr = data_ptr + offset;
    const size_t chunk_size = std::min(static_cast<size_t>(MAX_BUF_SIZE), data_size - offset);
    const size_t chunk_el = chunk_size / element_size;

    if (chunk_size < NAIVE_ALLREDUCE_THRESHOLD) {
      group_symmetric_all_reduce(ctx, chunk_ptr, scalar_type, chunk_size, chunk_el);
    } else {
      group_distributed_reduce(ctx, chunk_ptr, scalar_type, chunk_size, chunk_el);
    }
  }
}
