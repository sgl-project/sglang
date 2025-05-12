# SGLang Load Balancer

A load balancer server for SGLang prefill and decode servers, built with Rust and Axum.

## Features

- Distributes requests between multiple prefill and decode servers.
- Health check and cache flush endpoints.
- Dynamic server registration.
- Batch request support.

## Requirements

- Rust (edition 2021)
- Cargo


## Usage

### Build

```sh
cargo build --release

cargo run --release -- \
  --prefill <PREFILL_SERVER_URL> \
  --decode <DECODE_SERVER_URL> \
  --prefill-bootstrap-ports <BOOTSTRAP_PORT>
```

You can specify multiple `--prefill`, `--decode`, and `--prefill-bootstrap-ports` arguments.
Example

```sh
cargo run --release -- \
  --prefill http://127.0.0.1:30001 \
  --decode http://127.0.0.1:30001 \
  --prefill-bootstrap-ports 8998
```
