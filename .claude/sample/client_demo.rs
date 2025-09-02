use sglang_router_rs::grpc::{proto, SglangSchedulerClient};
use tokio_stream::StreamExt;
use tracing::{error, info, warn};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("Starting SGLang gRPC client demo");

    // Configuration - change this to your SGLang scheduler endpoint
    let endpoint = "http://127.0.0.1:50051";
    let client_id = "rust-demo-client".to_string();

    // Test basic network connectivity first
    info!("Testing network connectivity to {}", endpoint);
    let tcp_start = std::time::Instant::now();
    match tokio::net::TcpStream::connect("127.0.0.1:50051").await {
        Ok(stream) => {
            let tcp_duration = tcp_start.elapsed();
            info!("TCP connection successful in {:?}", tcp_duration);
            drop(stream); // Close the TCP connection
        }
        Err(e) => {
            error!("TCP connection failed: {}", e);
            error!("Make sure the Python demo server is running: cd examples && python3 demo_server.py");
            return Err(e.into());
        }
    }

    // Create and connect the client
    info!("Connecting to SGLang scheduler at {}", endpoint);
    let grpc_start = std::time::Instant::now();
    let mut client = match SglangSchedulerClient::connect(endpoint).await {
        Ok(client) => {
            let grpc_duration = grpc_start.elapsed();
            info!("Successfully connected to scheduler in {:?}", grpc_duration);
            client
        }
        Err(e) => {
            let grpc_duration = grpc_start.elapsed();
            error!(
                "Failed to connect to scheduler after {:?}: {}",
                grpc_duration, e
            );
            warn!("Make sure SGLang scheduler is running at {}", endpoint);
            return Err(e);
        }
    };

    // Initialize the client
    info!("Initializing client with ID: {}", client_id);
    info!("Sending initialize request...");
    let init_start = std::time::Instant::now();
    match client.initialize(client_id.clone()).await {
        Ok(response) => {
            let init_duration = init_start.elapsed();
            info!("Client initialized successfully in {:?}!", init_duration);
            info!("Scheduler version: {}", response.scheduler_version);
            if let Some(model_info) = response.model_info {
                info!(
                    "Model: {} (vocab size: {}, context length: {})",
                    model_info.model_name, model_info.vocab_size, model_info.max_context_length
                );
            }
            if let Some(capabilities) = response.capabilities {
                info!(
                    "Server capabilities: continuous_batching={}, max_batch_size={}",
                    capabilities.continuous_batching, capabilities.max_batch_size
                );
            }
        }
        Err(e) => {
            let init_duration = init_start.elapsed();
            error!(
                "Failed to initialize client after {:?}: {}",
                init_duration, e
            );
            return Err(e);
        }
    }

    // Perform health check
    info!("Performing health check...");
    match client.health_check().await {
        Ok(health) => {
            info!("Health check successful: healthy={}", health.healthy);
            info!(
                "Running requests: {}, waiting: {}",
                health.num_requests_running, health.num_requests_waiting
            );
            info!(
                "GPU cache usage: {:.1}%, memory usage: {:.1}%",
                health.gpu_cache_usage * 100.0,
                health.gpu_memory_usage * 100.0
            );
        }
        Err(e) => {
            warn!("Health check failed: {}", e);
        }
    }

    // Demo 1: Simple generation request
    info!("\n=== Demo 1: Simple Text Generation ===");
    let sampling_params = proto::SamplingParams {
        temperature: 0.7,
        max_new_tokens: 50,
        top_p: 0.9,
        top_k: 50,
        stop: vec!["</s>".to_string(), "\n\n".to_string()],
        ..Default::default()
    };

    let generate_request = proto::GenerateRequest {
        request_id: "demo-req-1".to_string(),
        input: Some(proto::generate_request::Input::Text(
            "The capital of France is".to_string(),
        )),
        sampling_params: Some(sampling_params.clone()),
        return_logprob: true,
        top_logprobs_num: 3,
        ..Default::default()
    };

    match client.generate_stream(generate_request).await {
        Ok(mut stream) => {
            info!("Generation request sent, receiving tokens...");
            let mut full_text = String::new();
            let mut token_count = 0;

            while let Some(response) = stream.next().await {
                match response {
                    Ok(gen_response) => match gen_response.response {
                        Some(proto::generate_response::Response::Chunk(chunk)) => {
                            print!("{}", chunk.text);
                            full_text.push_str(&chunk.text);
                            token_count += 1;

                            if token_count % 10 == 0 {
                                info!(
                                    "\nGenerated {} tokens, queue_time: {}ms",
                                    chunk.completion_tokens, chunk.queue_time
                                );
                            }
                        }
                        Some(proto::generate_response::Response::Complete(complete)) => {
                            println!("\n");
                            info!("Generation completed!");
                            info!("Finish reason: {:?}", complete.finish_reason);
                            info!(
                                "Total tokens: prompt={}, completion={}, cached={}",
                                complete.prompt_tokens,
                                complete.completion_tokens,
                                complete.cached_tokens
                            );
                            info!(
                                "Generation time: {:.2}s, tokens/sec: {:.2}",
                                complete.total_generation_time, complete.tokens_per_second
                            );
                            break;
                        }
                        Some(proto::generate_response::Response::Error(error)) => {
                            error!("Generation error: {}", error.message);
                            break;
                        }
                        None => {
                            warn!("Received empty response");
                        }
                    },
                    Err(e) => {
                        error!("Stream error: {}", e);
                        break;
                    }
                }
            }

            info!("Final generated text: '{}'", full_text);
        }
        Err(e) => {
            error!("Failed to start generation: {}", e);
        }
    }

    // Demo 2: Generation with structured output (JSON)
    info!("\n=== Demo 2: Structured JSON Generation ===");
    let json_schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "city": {"type": "string"}
        },
        "required": ["name", "age", "city"]
    }"#;

    let structured_sampling = proto::SamplingParams {
        temperature: 0.3,
        max_new_tokens: 100,
        constraint: Some(proto::sampling_params::Constraint::JsonSchema(
            json_schema.to_string(),
        )),
        ..Default::default()
    };

    let structured_request = proto::GenerateRequest {
        request_id: "demo-req-2".to_string(),
        input: Some(proto::generate_request::Input::Text(
            "Generate a person's information in JSON format:".to_string(),
        )),
        sampling_params: Some(structured_sampling),
        return_logprob: false,
        ..Default::default()
    };

    match client.generate_stream(structured_request).await {
        Ok(mut stream) => {
            info!("Structured generation request sent...");
            let mut json_output = String::new();

            while let Some(response) = stream.next().await {
                match response {
                    Ok(gen_response) => match gen_response.response {
                        Some(proto::generate_response::Response::Chunk(chunk)) => {
                            print!("{}", chunk.text);
                            json_output.push_str(&chunk.text);
                        }
                        Some(proto::generate_response::Response::Complete(complete)) => {
                            println!("\n");
                            info!("Structured generation completed!");
                            info!("Generated JSON: {}", json_output);
                            break;
                        }
                        Some(proto::generate_response::Response::Error(error)) => {
                            error!("Structured generation error: {}", error.message);
                            break;
                        }
                        None => {}
                    },
                    Err(e) => {
                        error!("Stream error: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            error!("Failed to start structured generation: {}", e);
        }
    }

    // Demo 3: Embedding request
    info!("\n=== Demo 3: Text Embeddings ===");
    let embed_request = proto::EmbedRequest {
        request_id: "demo-embed-1".to_string(),
        input: Some(proto::embed_request::Input::Text(
            "This is a sample text for embedding generation.".to_string(),
        )),
        log_metrics: true,
        ..Default::default()
    };

    // Note: This demo assumes the scheduler has an embedding model loaded
    // If not, this will fail gracefully
    info!("Sending embedding request...");
    // Uncomment the following lines if you have an embedding model loaded:
    /*
    match client.embed(embed_request).await {
        Ok(embed_response) => {
            match embed_response.response {
                Some(proto::embed_response::Response::Complete(complete)) => {
                    info!("Embedding generated successfully!");
                    info!("Embedding dimension: {}", complete.embedding_dim);
                    info!("First 5 values: {:?}", &complete.embedding[..5.min(complete.embedding.len())]);
                    info!("Prompt tokens: {}", complete.prompt_tokens);
                },
                Some(proto::embed_response::Response::Error(error)) => {
                    error!("Embedding error: {}", error.message);
                },
                None => {
                    warn!("Empty embedding response");
                }
            }
        },
        Err(e) => {
            warn!("Embedding request failed (this is expected if no embedding model is loaded): {}", e);
        }
    }
    */
    info!("Skipping embedding demo (uncomment code if embedding model is available)");

    // Demo 4: Cache management
    info!("\n=== Demo 4: Cache Management ===");

    // Flush cache
    match client.flush_cache(false).await {
        Ok(flush_response) => {
            info!("Cache flush completed successfully!");
            info!(
                "Entries flushed: {}, memory freed: {} bytes",
                flush_response.num_entries_flushed, flush_response.memory_freed
            );
        }
        Err(e) => {
            warn!("Cache flush failed: {}", e);
        }
    }

    info!("\nDemo completed successfully!");
    Ok(())
}
