# Catalog Producer Capability Handshake

The SGLang producer now treats the SpecForge Catalog capability response as a
required startup contract. Every active owner calls `GET /capabilities` before
creating the selected-layer exporter, connecting Mooncake, allocating registered
Host storage or opening the publication journal. Inactive ranks make no Catalog
or Store call. A failing active rank enters the existing distributed resource
failure vote, so prepared peers close before capture activation.

The strict response is defined by
[`catalog-capabilities.schema.json`](../training-data-contract/catalog-capabilities.schema.json)
and its [example](../training-data-contract/catalog-capabilities.example.json).
It advertises producer protocol version, complete contract records, accepted
Store protocols, hard-pin and checkpoint-retention semantics, and the maximum
HTTP metadata request size. A record matches only when its contract ID, schema
version, payload format and KV codec all match. The producer does not combine
fields across records. Missing fields, unknown fields, wrong scalar types,
unsupported transports, retention downgrade and a limit below 8 MiB all reject
startup.

The GET uses the same optional Bearer credential, proxy isolation, redirect
rejection, one-MiB response limit and bounded retry policy as the existing
producer API. It has no request body and does not create a capture lease. The
local Mooncake adapter still independently requires registered-buffer methods
and forces `ReplicateConfig.with_hard_pin`; a Catalog declaration does not make
an incompatible SDK usable.

## Validation

The focused suite covers the strict schema/example, complete-record matching,
malformed values, missing and unknown fields, authentication headers, empty GET
body, bounded timeout retry, HTTP errors, response size and redirect rejection.
It also proves an incompatible response prevents exporter, Mooncake, Host pool
and journal construction, and that an injected connected Store is closed before
Host allocation. The four-process startup suite injects one incompatible active
rank and verifies a common resource-phase failure with peer cleanup.

The complete capture unit directory passes **345 tests in 288.732 seconds**;
the eight focused methods pass in **4.721 seconds**. A real Qwen3-0.6B runtime
test with native Mooncake TCP passes in **565.995 seconds** from the same frozen
source. It records exactly one successful capability request before each eager
and decode-CUDA-graph service startup. The final lifecycle case publishes five
complete samples, and the runtime matrix verifies complete post-exit snapshot
readback. Its final result and artifact binding are recorded in
[`catalog-capabilities.json`](catalog-capabilities.json).

The runtime Catalog remains an HTTP test double implementing the producer
contract. This work does not implement SpecForge's production persistence,
consumer claim handshake, checkpoint watermark, retention GC or HA database,
and adds no new RDMA or trained-model acceptance evidence.
