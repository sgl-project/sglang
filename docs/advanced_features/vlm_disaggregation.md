# Encode/Language Disaggregation Architecture

> **Version**: 1.0
> **Date**: 2025-10-28

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Highlights](#implementation-highlights)
4. [Key Capabilities](#key-capabilities)
5. [Demo](#demo)
6. [Performance Metrics](#performance-metrics)
7. [Future Work](#future-work)

---

## Overview

### Context

Large multimodal models handled by a single inference cluster face several challenges:
1. **Uneven resource utilization**: the vision encoder and the language model place very different demands on compute and memory.
2. **Limited scalability**: the encoding and generation stages cannot be scaled independently.
3. **Latency bottlenecks**: the vision encoder and language model execute serially, capping overall throughput.

### Design Goals

This design introduces an **Encode/Language disaggregated architecture** that splits multimodal inference into two independent services:
- **Encode**: handles image or video encoding and produces embeddings.
- **Language**: consumes the embeddings and performs text generation.


## Architecture

### High-Level Design


![EPD Framework](./EPD_Framework.svg)


### Execution Flow


```
1. Both the language and encode instances receive requests:
   Language instances read only the text component
      ├─ Extract text inputs
      └─ Enqueue into MultimodalLanguagePreallocQueue

   Encode instances receive the full multimodal payload
      ├─ Run multimodal preprocessing
      └─ Enqueue into MultimodalEmbeddingBootstrapQueue

2. The language instance establishes a connection
   ├─ Query the bootstrap server for the encode instance address
   ├─ Create a MooncakeEmbeddingReceiver
   ├─ Pre-allocate a receive buffer (16,384 tokens). Without multimodal hints, allocation follows default_allocated_tokens
   └─ Send an init message to the encode plane

3. The encode instance processes the request
   ├─ Receive the init message
   ├─ Run the vision encoder to produce embedding tokens
   ├─ Allocate block buffers based on the actual token length (e.g., two blocks for 8,000 tokens)
   └─ Fill input_embeddings, fill_ids, mrope_positions, and aux_datas

4. Encode → language data transfer
   ├─ Send data through Mooncake/RDMA
   ├─ Return Success when the transfer completes; if the language cache is insufficient, transfer a partial chunk
   └─ On Success, release encode buffers; otherwise, wait for resume-transfer and repeat

5. Language processing
   5.1 Full transfer detected (Success)
       ├─ Read embeddings from the buffer (e.g., 8,000 tokens)
       ├─ Assemble input_embeds and multimodal_inputs
       ├─ Run LLM prefill + decode
       └─ Stream responses back to the client

   5.2 Partial transfer detected (Transferring)
       ├─ Move received embeddings into a temporary buffer
       └─ Request additional cache, trigger resume-transfer, and loop back to 5.1

6. Resource cleanup
   ├─ Release the language-side receive buffers
   └─ Finalize the request lifecycle
```

### Core Components

#### 1. Encode Instance

```python
# File: python/sglang/srt/disaggregation/multimodal_embedding.py

class MultimodalEmbeddingBootstrapQueue:
    """
    Coordinate connection setup and block allocation for the encode instance
    - Handle bootstrap handshakes
    - Allocate blocks based on the actual sequence length
    - Initialize the embedding sender
    """

class MooncakeEmbeddingSender:
    """
    Stream embedding payloads to the language instance
    - Support block-level transfer
    - Retry on transient failures
    - Manage states (Bootstrapping → WaitingForInput → Transferring → Success)
    """
```

**Workflow**:
1. Receive requests that contain multimodal data.
2. Execute the vision encode on images or video frames.
3. Populate block-level buffers with embedding tensors.
4. Transfer the buffers via MooncakeSender.
5. Monitor transfer states and recover from errors.

#### 2. Language Instance

```python
# File: python/sglang/srt/disaggregation/multimodal_language.py

class MultimodalLanguagePreallocQueue:
    """
    Queue for the preallocation stage
    - Manage connection handshakes
    - Allocate receive buffer blocks
    - Respect the default_allocate_tokens configuration
    """

class MultimodalLanguageTransferQueue:
    """
    Queue for the transfer stage
    - Track transfer states
    - Handle partial transfers and resume-transfer requests
    - Merge embedding blocks
    - Construct MultimodalInputs
    """

class MultimodalLanguageRequest:
    """
    Request state tracker on the language instance
    - Cache intermediate data for partial transfers
    - Support resume-transfer workflows
    """
```

**Workflow**:
1. Dequeue requests from PreallocQueue.
2. Allocate block-aligned buffers according to default_allocate_tokens.
3. Receive embeddings through MooncakeReceiver.
4. Detect partial transfers and issue resume requests when needed.
5. Merge the completed embedding buffers.
6. Feed embeddings into the language model for generation.


## Implementation Highlights

### 1. Block-Based Allocation

#### Approach

```python
# File: python/sglang/srt/disaggregation/utils.py

class ReqToMetadataBlockAllocator:
    """Block allocator"""

    def __init__(self, size: int, block_size: int):
        self.total_blocks = size
        self.block_size = block_size  # Tokens per block
        self.free_blocks = deque(list(range(size)))
        self.req_to_blocks = {}  # req_id -> [block_indices]

    def alloc(self, num_tokens: int, req_id: str) -> Optional[List[int]]:
        """Allocate blocks based on the actual token count"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            return None

        allocated_blocks = [
            self.free_blocks.popleft() for _ in range(num_blocks_needed)
        ]
        self.req_to_blocks[req_id] = allocated_blocks
        return allocated_blocks
```


#### Buffer Layout

```python
class MultimodalDataBuffers:
    def __init__(self, size: int, block_size: int, embedding_dim: int):
        # Each buffer has shape [num_blocks, block_size * item_size]
        self.input_embeddings = torch.zeros(
            (size, block_size * embedding_dim),
            dtype=torch.bfloat16,
            device="cpu"
        )
        self.fill_ids = torch.zeros(
            (size, block_size),
            dtype=torch.int32,
            device="cpu"
        )
        self.mrope_positions = torch.zeros(
            (size, 3 * block_size),
            dtype=torch.int32,
            device="cpu"
        )
        self.aux_datas = torch.zeros(
            (size, 16),
            dtype=torch.int32,
            device="cpu"
        )
        # Deepstack support (Qwen3-VL-MoE)
        if num_deepstack_embeddings > 0:
            self.deepstack_embeddings = torch.zeros(
                (size, block_size * embedding_dim * num_deepstack_embeddings),
                dtype=torch.bfloat16,
                device="cpu"
            )
```

**Buffer semantics**:
- `input_embeddings`: embeddings generated by the vision encoder.
- `fill_ids`: token ID sequences associated with the embeddings.
- `mrope_positions`: multimodal RoPE positional encodings (3D coordinates).
- `aux_datas`: auxiliary metadata such as `embedding_length` and `mrope_position_delta`.
- `deepstack_embeddings`: optional Deepstack embeddings for Qwen3-VL models.

### 2. Resume Transfer

#### Motivation

When the total embedding length exceeds the pre-allocated language buffer, the system falls back to staged transfers:

```
Encode plane:    [=========== 10,000 tokens ===========]
                    ↓
Language initial allocation: [==== 4,096 tokens ====]  (insufficient)
                    ↓
Detect partial transfer (KVPoll.Transferring)
                    ↓
Language extra allocation:   [==== 6,000 tokens ====]
                    ↓
Encode plane streams the remaining tokens
                    ↓
Transfer completes (KVPoll.Success)
```

#### Mechanism

**Language-side detection**:

```python
# File: python/sglang/srt/disaggregation/multimodal_language.py

def pop_transferred(self):
    polls = poll_and_all_reduce([req.embedding_receiver for req in self.queue])

    for language_req, poll in zip(self.queue, polls):
        if poll == KVPoll.Transferring:
            # Retrieve the data received so far
            (embedding_data, fill_ids, mrope_positions,
             aux_datas, deepstack_embedding) = self.metadata_buffers.get_buf(
                block_indices=language_req.embedding_indices
            )

            actual_total_length = int(aux_datas[0])  # Total token length
            sent_tokens = len(fill_ids)  # Tokens already transmitted

            if actual_total_length > sent_tokens:
                # Resume transfer is required
                remaining_tokens = actual_total_length - sent_tokens

                # Cache partial data
                language_req.partial_input_embeds = embedding_data
                language_req.partial_fill_ids = fill_ids.tolist()
                language_req.partial_mrope_positions = mrope_positions
                language_req.partial_aux_datas = aux_datas
                language_req.partial_sent_tokens = sent_tokens
                language_req.partial_deepstack_embedding = deepstack_embedding

                # Allocate space for the remaining tokens
                new_allocation = self.req_to_metadata_buffer_idx_allocator.alloc(
                    num_tokens=remaining_tokens,
                    req_id=language_req.req.rid
                )

                # Release the old allocation
                self.req_to_metadata_buffer_idx_allocator.free(
                    block_indices=language_req.embedding_indices,
                    req_id=language_req.req.rid
                )

                # Switch to the new buffers
                language_req.embedding_indices = new_allocation
                allocated_tokens = len(new_allocation) * block_size

                # Notify the encode plane
                language_req.embedding_receiver.resume_transfer(
                    embedding_indices=new_allocation,
                    sent_tokens=sent_tokens,
                    allocated_tokens=allocated_tokens
                )
```

**Encode-side handling**:

```python
# File: python/sglang/srt/disaggregation/mooncake/conn_multimodal.py

def embedding_thread():
    """Handle resume requests from the language plane"""
    while True:
        waiting_req_bytes = self.server_socket.recv_multipart()

        # Resume requests carry eight fields
        is_resume = len(waiting_req_bytes) >= 8

        if is_resume:
            transfer_info = TransferEmbeddingInfo.from_zmq(waiting_req_bytes)
            req = self.transfer_infos[room][mooncake_session_id]

            # Update transfer progress
            req.sent_tokens = transfer_info.sent_tokens
            req.allocated_tokens = transfer_info.allocated_tokens
            req.dst_embedding_indices = transfer_info.dst_embedding_indices
            req.resume_ready = True

            # Resume when all destination ranks are ready
            if all(dst_req.resume_ready for dst_req in self.transfer_infos[room].values()):
                self.transfer_queues[shard_idx].put(
                    TransferEmbeddingChunk(
                        room=room,
                        embedding_indices=req.src_embedding_indices,
                        is_last=True,
                        total_tokens=req.total_tokens
                    )
                )
```

**Partial transfer support in the sender**:

```python
def send_embedding(
    self,
    mooncake_session_id: str,
    embedding_indices: List[int],
    dst_embedding_ptrs: List[int],
    dst_embedding_indices: List[int],
    total_tokens: int,
    block_size: int,
    sent_tokens: int = 0,
    allocated_tokens: int = None
) -> Tuple[int, bool]:
    """
    Returns:
        (ret, is_partial)
        - ret: 0 for success, 1 for failure
        - is_partial: True if additional data remains to be sent
    """
    remaining_tokens = total_tokens - sent_tokens

    if remaining_tokens > allocated_tokens:
        tokens_to_send = allocated_tokens
        is_partial = True
    else:
        tokens_to_send = remaining_tokens
        is_partial = False

    # Identify the starting block
    start_block = sent_tokens // block_size
    dst_blocks_needed = (tokens_to_send + block_size - 1) // block_size

    # Transfer via Mooncake/RDMA
    ret = self.engine.batch_transfer_sync(
        mooncake_session_id, src_addrs, dst_addrs, lengths
    )

    return ret, is_partial
```

## Demo

### Scenario: Qwen3-VL-MoE Multimodal Inference

#### 1. Launch the Bootstrap Server

```bash
# Launch the bootstrap server on a control node
python3 -m sglang.srt.disaggregation.mini_lb --host $HOST_IP \
    --port $SERVER_PORT --vision http://${EMBEDDING_IP}:${EMBEDDING_PORT} \
    --prefill http://${LANGUAGE_IP_LIST[0]}:${LANGUAGE_PORT} \
    --enable-multimodal-disagg
```

#### 2. Launch the Encode Service

```bash
export PORT=8001
export SGLANG_VLM_CACHE_SIZE_MB=40960
export TENSOR_PARALLEL_SIZE=2
export CHUNKED_PREFILL_SIZE=81920
export MAX_RUNNING_REQUESTS=128
export MEM_FRACTION_STATIC=0.85
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=128
export SGLANG_EMBEDDING_CACHE_BLOCK_SIZE=16384
# Encode: vision encoding
python3 -m sglang.launch_server --model-path ${MODEL_PATH} --enable-torch-compile --max-prefill-tokens $CHUNKED_PREFILL_SIZE \
        --host $HOST_IP --port $PORT --trust-remote-code --tp-size ${TENSOR_PARALLEL_SIZE} --mem-fraction-static ${MEM_FRACTION_STATIC} \
        --enable-cache-report --log-level info --max-running-requests ${MAX_RUNNING_REQUESTS} \
        --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} --attention-backend fa3 --json-model-override-args '{"is_multimodal_embedding": true}' \
        --mm-attention-backend fa3 --disaggregation-mode encode
```


#### 3. Launch the Language Service

```bash
export PORT=8002
export TENSOR_PARALLEL_SIZE=8
export MAX_RUNNING_REQUESTS=128
export SGLANG_EMBEDDING_CACHE_BUFFER_SIZE=128
export SGLANG_EMBEDDING_CACHE_BLOCK_SIZE=16384
export MEM_FRACTION_STATIC=0.85
export CHUNKED_PREFILL_SIZE=8192
# Configure the default buffer allocation
export SGLANG_EMBEDDING_DEFAULT_ALLOCATE_BUFFER_SIZE=16384

# Language: text generation
# Qwen2.5-VL: "architectures": ["Qwen2ForCausalLM"]
python3 -m sglang.launch_server --model-path ${MODEL_PATH} --enable-torch-compile --disable-radix-cache \
        --host $HOST_IP --port $PORT --trust-remote-code --tp-size ${TENSOR_PARALLEL_SIZE} --served-model-name "qwen3-vl" \
        --enable-cache-report --log-level info --max-running-requests ${MAX_RUNNING_REQUESTS} --json-model-override-args '{"architectures": ["Qwen3MoeForCausalLM"]}' \
        --mem-fraction-static ${MEM_FRACTION_STATIC} --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} --attention-backend fa3 \
        --disaggregation-mode language
```

#### 4. Benchmark

```
python3 -m sglang.bench_serving \
                    --host ${HOST_IP} \
                    --port ${PORT} \
                    --model $MODEL_PATH \
                    --backend sglang-oai-chat \
                    --dataset-name "image" \
                    --random-input-len $input_len \
                    --random-output-len $output_len \
                    --random-range-ratio 1 \
                    --num-prompt $num_prompt \
                    --warmup-requests 0 \
                    --flush-cache \
                    --image-count 1 \
                    --image-resolution $image_size \
                    --image-format "jpeg" \
                    --image-content "random" \
                    --request-rate $qps \
                    --output-file $result_file \
                    --max-concurrency 128
```

---
## Performance Metrics

SLA: mean TTFT < 4 s; mean TPOT < 100 ms.


### Single Node (TP=8)

| Model               | Scenario                                        | qps/gpu           | TTFT (Mean) | TPOT (Mean) |
|---------------------|-------------------------------------------------|-------------------|-------------|-------------|
| qwen2.5-vl-72B      | Single image + text (2000×2000 + 1k), output 300, qps: 0.52 | 0.06625 req/s/gpu | 3826.07 ms  | 78.63 ms    |
| qwen3-vl-235B-A22B  | Single image + text (2000×2000 + 1k), output 300, qps: 0.85 | 0.1075 req/s/gpu  | 3738.24 ms  | 91.24 ms    |

### Disaggregated Deployment

| Configuration | Encode Plane       | Language Plane     |
|---------------|--------------------|--------------------|
| GPU model     | NVIDIA H20 96GB    | NVIDIA H20 96GB    |
| GPU count     | 2 (TP = 2)         | 8 (TP = 8)         |

| Model               | Scenario                                        | qps/gpu                  | TTFT (Mean)  | TPOT (Mean) |
|---------------------|-------------------------------------------------|--------------------------|--------------|-------------|
| qwen2.5-vl-72B      | Single image + text (2000×2000 + 1k), output 300, qps: 0.78 | 0.07420 (+12%) req/s/gpu | 3632.70 ms   | 95.61 ms    |
| qwen3-vl-235B-A22B  | Single image + text (2000×2000 + 1k), output 300, qps: 1.40 | 0.141 (+31.2%) req/s/gpu | 3831.58 ms   | 95.34 ms    |


## Future Work

### Encode/Prefill/Decode Disaggregation

The current implementation disaggregates for encode and language. The language can be extended to integrate with the prefill, enabling Encode/Prefill/Decode separation.

### CachePool Integration

`MultimodalDataBuffers` will integrate with CachePool to enable direct device-to-device transfers.

### Chunk-Prefill and Embedding Transfer Overlap

Support overlapping chunked prefill execution with embedding transfers.

### Migration to mini-lb Router

Port the current mini load-balancer router implementation to production.
