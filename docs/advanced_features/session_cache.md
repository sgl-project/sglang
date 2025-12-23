# Session level Key-Value Cache Design

SGLang provides session-level KV cache management, enabling user-defined business logic to explicitly control advanced features such as the lifecycle of the KV cache, hierarchical caching, and replica count management. In large-scale LLM service scenarios within data centers, SGLang serves as a high-performance data processing node, better supporting business logic and achieving superior results in performance, cost efficiency, and user experience.

## How to

SGLang provides the `--enable-session-cache` parameter (which requires combination with `--enable-hierarchical-cache`) to enable session KV cache. Additionally, the `generate` method has been extended with the `stored_kv_cache` and `fresh_kv_cache` fields to describe the KV cache information of the request session. For example:

```json
{
  "model": "default",
  "text": "...",
  "sampling_params": {
    "temperature": 0,
    "max_new_tokens": 1000
  },
  "session_params": {
    "id": "session-97f1226c-e752-4961-a858-432814b5dce0",
    "stored_kv_cache": [
      {
        "token_start": 0,
        "token_length": 113,
        "kv_uri": "file:///session-97f1226c-e752-4961-a858-432814b5dce0",
        "kv_start": 0
      },
      {
        "token_start": 113,
        "token_length": 100,
        "kv_uri": "file:///session-97f1226c-e752-4961-a858-432814b5dce0",
        "kv_start": 16662528
      }
    ],
    "fresh_kv_cache": [
      {
        "token_start": 213,
        "token_length": 1213,
        "kv_uri": "file:///session-97f1226c-e752-4961-a858-432814b5dce0",
        "kv_start": 31408128
      }
    ]
  }
}
```

- The first turn covers tokens [0, 113), and its KV cache is stored locally at `/session-97f1226c-e752-4961-a858-432814b5dce0`.
- The second turn covers tokens [113, 213).
- The upcoming turn will generate tokens [213, 1213) (up to 1213), and the associated KV cache will be saved locally at `/session-97f1226c-e752-4961-a858-432814b5dce0`.

### Supported storage backend

- **Local file**: A simple file-based storage backend for demonstration purposes.

### Advanced features

- **predictable performance and cost**: By leveraging explicit KV cache information, the number of tokens that can be served from the cache can be estimated prior to session launch, enabling accurate prediction of execution time and cost.
- **lifecycle of KV cache**: When a new session begins, relevant metadata is created, which includes the URI of the KV cache; when the session is terminated, the corresponding KV cache is deleted. Furthermore, the retention duration of the KV cache can be managed with different policies for users of varying priority levels.
- **explicit sharing**: When multiple sessions have identical segments in their old KV cache, they can be explicitly configured to share the KV cache. For example, if different users ask different questions about the same paper, the majority of the KV cache can be shared to maximize efficiency.
- **readahead hierarchical caching**: When a session request arrives, the system can proactively load the user's KV cache in advance (e.g., from disk to memory). Once the prefetching is completed, the session request is then routed to the GPU node for processing. This approach significantly reduces GPU latency caused by waiting for I/O operations and improves GPU utilization.
- **replica count management**: The number of stored replicas is determined based on the popularity of the session content. For example, popular papers can have an increased number of replicas to prevent storage node performance from becoming a bottleneck.
- **stateless inference**: With the support of high-performance distributed storage, SGLang will no longer rely (or will rely minimally) on in-process memory for caching. Consequently, user sessions can be processed symmetrically across any SGLang instance within the cluster. In large-scale deployments, stateless inference significantly improves the average utilization of computational resources across the cluster.
