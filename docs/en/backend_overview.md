# Backend Overview

This document offers an overview of the backend/runtime framework implemented by SGLang, focusing on its key features and advantages in the field of Language Model (LM) programming. **Understanding these fundamentals can enhance your programming practices, helping you leverage your devices to their full potential.**

## The SGLang Advantage

SGLang offers an extensible and efficient framework for LM Programming, addressing critical challenges in LLM generation and management. Its backend is specifically designed to enhance the performance and reliability of LLM operations.

### **Understanding LM Programs**

**An LM Program is a structured approach that uses programmatic control to orchestrate and manage the generation processes of LLMs.** This method allows for more precise and efficient handling of complex language tasks. A notable example is the  [Tree of Thought Planning](https://github.com/princeton-nlp/tree-of-thought-llm), which employs tree structures and state machines to facilitate step-by-step processing, rollback capabilities, and pruning mechanisms in language models.

### **Key Challenges in LM Programming**

LM Programs face two significant challenges during the inference phase:

1. **Output Instability in Language Models**: Language models often exhibit high sensitivity to minor input changes. For instance, the addition of a single newline character (`\n`) at the end of a prompt can dramatically alter the output of models like Vicuna-7b. This instability necessitates the implementation of more robust, rule-based generation techniques, such as enforcing specific output formats (e.g., JSON).
2. **Inefficiencies in KV Cache Reuse Methods**: Current LLM serving frameworks, such as [vLLM](https://github.com/vllm-project/vllm), [TGI](https://github.com/huggingface/text-generation-inference), and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) are pure Request-In-Response-Out frameworks, i.e., without any direct knowledge of the workload. At least, there are numerous opportunities to reuse the KV Cache across multiple requests that share a common prefix.

To address these challenges, SGLang has implemented improvements in both its frontend and backend. The first challenge is primarily mitigated through the use of a Compressed Finite State Machine, while the second challenge is addressed via backend runtime optimization, employing a method based on **RadixAttention**, a form of prefix tree management.

## RadixAttention

A Radix Tree, also known as a Patricia tree or compressed prefix tree, is a tree-based data structure commonly used for efficient storage and lookup of strings or key-value pairs. It is a space-optimized trie that reduces storage requirements by merging nodes with common prefixes. Each node represents a string prefix, with nodes along a path collectively forming a complete string or key.

RadixAttention can be broadly described as Radix Tree-based KV Cache management. Tree structures naturally accommodate insertion, deletion, modification, and lookup operations, with negligible computational overhead, as demonstrated in the [ablation study](https://arxiv.org/html/2312.07104v2) of the original paper.

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="./_static/image/radix.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">An illustration of RadixAttention from the SGLang paper.</div> </center>

It is noteworthy that Node deletion in the RadixAttention employs the Least Recently Used (LRU) method. During batch inference operations, reference counting is applied to active nodes to prevent inadvertent deletion resulting from non-utilization by concurrent query requests.

**SGLang's architecture is specifically optimized for inference on prompts that share common prefixes, such as system prompts. This specialization enables enhanced performance in certain use cases. To maximize efficiency, it is advisable to increase the length of common prefixes between requests.** RadixAttention employs a unified, dynamically managed memory allocation system for both running requests and KV Cache operations. This design allows for a direct correlation between cache hit rates and processing speeds: **higher cache hit rates result in accelerated processing of corresponding running requests and enable larger batch sizes.** In scenarios involving extended system prompts, cache hit rates can reach up to 99.94%, leading to exceptionally efficient request processing. This high level of optimization demonstrates the potential for significant performance gains in specific application contexts.

## Request Scheduling

The backend uses the Radix Tree for KV Cache reuse for a given sequence of requests. Naturally, the frontend can schedule the order in which requests are sent to the backend, **increasing the prefix overlap rate between adjacent requests, and further improving the cache hit rate.** The request order given by this scheduling algorithm is called the **longest-shared-prefix-first order.**

We prove in our paper that the longest-shared prefix-first order is near-optimal in both offline and online modes. In distributed conditions, since the insertion, deletion, lookup, and modification of the same tree are deterministic, the same tree is initialized on multiple GPUs, and subsequent operations will be completely consistent, eliminating the need for inter-GPU communication for KV Cache.

**It's worth emphasizing that this request scheduling method might lead to the starvation of some requests** that have almost no shared prefix with other requests when the number of requests is very large. This issue is currently unresolved, but in practice, it's rarely encountered because these requests will eventually be sent to the backend when the queue is empty.