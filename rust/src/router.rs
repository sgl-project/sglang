use crate::tree::Tree;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use bytes::Bytes;
use futures_util::{Stream, StreamExt, TryStreamExt};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Debug)]
pub enum Router {
    RoundRobin {
        worker_urls: Vec<String>,
        current_index: AtomicUsize,
    },
    Random {
        worker_urls: Vec<String>,
    },
    CacheAware {
        /*
        Cache-Aware Load Balancing Router

        This router combines two strategies to optimize both cache utilization and request distribution:

        1. Cache-Aware Routing (Approximate Tree)
        2. Load Balancing (Shortest Queue)

        For each incoming request, the router chooses between these strategies:
        - With probability P: Uses cache-aware routing
        - With probability (1-P): Uses load balancing
        where P is configured via `cache_routing_prob`

        Strategy Details:

        1. Cache-Aware Routing (Approximate Tree)
        -------------------------------------------
        This strategy maintains an approximate radix tree for each worker based on request history,
        eliminating the need for direct cache state queries. The tree stores raw text characters
        instead of token IDs to avoid tokenization overhead.

        Process:
        a. For each request, find the worker with the highest prefix match
        b. If match rate > cache_threshold:
        Route to the worker with highest match (likely has relevant data cached)
        c. If match rate ≤ cache_threshold:
        Route to the worker with smallest tree size (most available cache capacity)
        d. Background maintenance:
        Periodically evict least recently used leaf nodes to prevent memory overflow

        2. Load Balancing (Shortest Queue)
        -------------------------------------------
        This strategy tracks pending request counts per worker and routes new requests
        to the least busy worker for optimal load distribution.

        Configuration Parameters:
        ------------------------
        1. cache_routing_prob: (float, 0.0 to 1.0)
        - 0.0: Exclusively use load balancing
        - 1.0: Exclusively use cache-aware routing
        - Between 0-1: Probability of using cache-aware routing vs load balancing

        2. cache_threshold: (float, 0.0 to 1.0)
        Minimum prefix match ratio to use highest-match routing.
        Below this threshold, routes to worker with most available cache space.

        3. eviction_interval_secs: (integer)
        Interval between LRU eviction cycles for the approximate trees.

        4. max_tree_size: (integer)
        Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
        during the next eviction cycle.
        */
        worker_urls: Vec<String>,
        tree: Arc<Mutex<Tree>>,
        running_queue: Arc<Mutex<HashMap<String, usize>>>,
        processed_queue: Arc<Mutex<HashMap<String, usize>>>,
        cache_threshold: f32,
        cache_routing_prob: f32,
        _eviction_thread: Option<thread::JoinHandle<()>>, // Store thread handle
    },
}

#[derive(Debug)]
pub enum PolicyConfig {
    RandomConfig,
    RoundRobinConfig,
    CacheAwareConfig {
        cache_threshold: f32,
        cache_routing_prob: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
    },
}

fn get_text_from_request(body: &Bytes) -> String {
    // 1. convert body to json
    let json = serde_json::from_slice::<serde_json::Value>(body).unwrap();
    // 2. get the text field
    let text = json.get("text").and_then(|t| t.as_str()).unwrap_or("");
    return text.to_string();
}

impl Router {
    pub fn new(worker_urls: Vec<String>, policy_config: PolicyConfig) -> Self {
        match policy_config {
            PolicyConfig::RandomConfig => Router::Random { worker_urls },
            PolicyConfig::RoundRobinConfig => Router::RoundRobin {
                worker_urls,
                current_index: std::sync::atomic::AtomicUsize::new(0),
            },
            PolicyConfig::CacheAwareConfig {
                cache_threshold,
                cache_routing_prob,
                eviction_interval_secs,
                max_tree_size,
            } => {
                let mut running_queue = HashMap::new();
                for url in &worker_urls {
                    running_queue.insert(url.clone(), 0);
                }

                let mut processed_queue = HashMap::new();
                for url in &worker_urls {
                    processed_queue.insert(url.clone(), 0);
                }

                let tree = Arc::new(Mutex::new(Tree::new()));
                let running_queue = Arc::new(Mutex::new(running_queue));
                let processed_queue = Arc::new(Mutex::new(processed_queue));

                // Create background eviction thread
                let tree_clone = Arc::clone(&tree);
                let processed_queue_clone = Arc::clone(&processed_queue);
                let eviction_thread = thread::spawn(move || {
                    loop {
                        // Sleep for the specified interval
                        thread::sleep(Duration::from_secs(eviction_interval_secs));

                        let locked_tree_clone = tree_clone.lock().unwrap();
                        // Run eviction
                        locked_tree_clone.evict_tenant_data(max_tree_size);

                        // Print the process queue
                        let locked_processed_queue = processed_queue_clone.lock().unwrap();
                        println!("Processed Queue: {:?}", locked_processed_queue);
                    }
                });

                for url in &worker_urls {
                    tree.lock().unwrap().insert(&"".to_string(), url);
                }

                Router::CacheAware {
                    worker_urls,
                    tree,
                    running_queue,
                    processed_queue,
                    cache_threshold,
                    cache_routing_prob,
                    _eviction_thread: Some(eviction_thread),
                }
            }
        }
    }

    pub fn get_first(&self) -> Option<String> {
        match self {
            Router::RoundRobin { worker_urls, .. }
            | Router::Random { worker_urls }
            | Router::CacheAware { worker_urls, .. } => {
                if worker_urls.is_empty() {
                    None
                } else {
                    Some(worker_urls[0].clone())
                }
            }
        }
    }

    pub async fn dispatch(
        &self,
        client: &reqwest::Client,
        req: HttpRequest,
        body: Bytes,
    ) -> HttpResponse {
        let text = get_text_from_request(&body);

        let worker_url = match self {
            Router::RoundRobin {
                worker_urls,
                current_index,
            } => {
                let idx = current_index
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |x| Some((x + 1) % worker_urls.len()),
                    )
                    .unwrap();

                worker_urls[idx].clone()
            }

            Router::Random { worker_urls } => {
                worker_urls[rand::random::<usize>() % worker_urls.len()].clone()
            }

            Router::CacheAware {
                worker_urls,
                tree,
                running_queue,
                processed_queue,
                cache_threshold,
                cache_routing_prob,
                ..
            } => {
                // even though the tree is thread-safe, we still put a lock to ensure the whole op (tree read + queue read + tree write + queue write) is atomic to handle some edge cases (e.g. multiple requests with long prefix entering at the same time)

                let mut tree = tree.lock().unwrap();
                let mut running_queue = running_queue.lock().unwrap();

                // Generate a random float between 0 and 1 for probability check
                let sampled_p: f32 = rand::random();

                let selected_url = if sampled_p < *cache_routing_prob {
                    // Cache-aware routing logic
                    let (matched_text, matched_worker) = tree.prefix_match(&text);
                    let matched_rate =
                        matched_text.chars().count() as f32 / text.chars().count() as f32;

                    if matched_rate > *cache_threshold {
                        matched_worker.to_string()
                    } else {
                        let m_map: HashMap<String, usize> = tree
                            .tenant_char_count
                            .iter()
                            .map(|entry| (entry.key().clone(), *entry.value()))
                            .collect();

                        println!("map: {:?}, mmap: {:?}", tree.get_tenant_char_count(), m_map);

                        tree.get_smallest_tenant()
                    }
                } else {
                    // Shortest queue routing logic
                    running_queue
                        .iter()
                        .min_by_key(|(_url, &count)| count)
                        .map(|(url, _)| url.clone())
                        .unwrap_or_else(|| worker_urls[0].clone())
                };

                // Update running queue
                let count = running_queue.get_mut(&selected_url).unwrap();
                *count += 1;

                // Update processed queue
                let mut locked_processed_queue = processed_queue.lock().unwrap();
                let count = locked_processed_queue.get_mut(&selected_url).unwrap();
                *count += 1;

                // Update tree with the new request
                tree.insert(&text, &selected_url);

                selected_url
            }
        };

        let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
            .map(|v| v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false))
            .unwrap_or(false);

        let res = match client
            .post(format!("{}/generate", worker_url.clone()))
            .header(
                "Content-Type",
                req.headers()
                    .get("Content-Type")
                    .and_then(|h| h.to_str().ok())
                    .unwrap_or("application/json"),
            )
            .body(body.to_vec())
            .send()
            .await
        {
            Ok(res) => res,
            Err(_) => return HttpResponse::InternalServerError().finish(),
        };

        let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, get response first
            let response = match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(_) => HttpResponse::InternalServerError().finish(),
            };

            // Then decrement running queue counter if using CacheAware
            if let Router::CacheAware { running_queue, .. } = self {
                if let Ok(mut queue) = running_queue.lock() {
                    if let Some(count) = queue.get_mut(&worker_url) {
                        *count = count.saturating_sub(1);
                    }
                }
            }

            response
        } else if let Router::CacheAware { running_queue, .. } = self {
            let running_queue = Arc::clone(running_queue);
            let worker_url = worker_url.clone();

            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(
                    res.bytes_stream()
                        .map_err(|_| {
                            actix_web::error::ErrorInternalServerError("Failed to read stream")
                        })
                        .inspect(move |bytes| {
                            let bytes = bytes.as_ref().unwrap();
                            if bytes
                                .as_ref()
                                .windows(12)
                                .any(|window| window == b"data: [DONE]")
                            {
                                let mut locked_queue = running_queue.lock().unwrap();
                                let count = locked_queue.get_mut(&worker_url).unwrap();
                                *count = count.saturating_sub(1);
                                // print
                                // println!("streaming is done!!")
                            }
                        }),
                )
        } else {
            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(res.bytes_stream().map_err(|_| {
                    actix_web::error::ErrorInternalServerError("Failed to read stream")
                }))
        }
    }
}
