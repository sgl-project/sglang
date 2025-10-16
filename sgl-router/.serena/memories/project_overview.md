# Project Overview
- **Name**: SGLang Router
- **Purpose**: High-performance routing control and data plane for large-scale LLM deployments. Manages worker registration, load balancing, caching, tokenization, and exposes OpenAI-compatible APIs (HTTP & gRPC) tightly integrated with the SGLang runtime.
- **Primary Tech Stack**: Rust (main router, tokenization, gRPC pipeline), Python packaging/launcher utilities, Hugging Face tokenizers, tokio async runtime.
- **Key Components**:
  - Control plane: worker registry, health monitoring, job queue, discovery.
  - Data plane: HTTP, gRPC, and OpenAI proxy routers with resilience (retries, rate limiting, circuit breakers).
  - Tokenization subsystem with Hugging Face & Tiktoken backends, chat templating.
  - Optional multi-model inference gateway mode.
- **Notable Integrations**: Prometheus metrics, MCP tooling, Hugging Face Hub downloads, OpenAI-compatible APIs.