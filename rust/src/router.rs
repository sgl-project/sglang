use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use bytes::Bytes;
use futures_util::TryStreamExt;
use std::fmt::Debug;

#[derive(Debug)]
pub enum Router {
    RoundRobin {
        worker_urls: Vec<String>,
        current_index: std::sync::atomic::AtomicUsize,
    },
    Random {
        worker_urls: Vec<String>,
    },
}

impl Router {
    pub fn new(worker_urls: Vec<String>, policy: String) -> Self {
        match policy.to_lowercase().as_str() {
            "random" => Router::Random { worker_urls },
            "round_robin" => Router::RoundRobin {
                worker_urls,
                current_index: std::sync::atomic::AtomicUsize::new(0),
            },
            _ => panic!(
                "Unknown routing policy: {}. The available policies are 'random' and 'round_robin'",
                policy
            ),
        }
    }

    pub fn get_first(&self) -> Option<String> {
        match self {
            Router::RoundRobin { worker_urls, .. } | Router::Random { worker_urls } => {
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
        let worker_url = match self {
            Router::RoundRobin {
                worker_urls,
                current_index,
            } => {
                current_index
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |x| Some((x + 1) % worker_urls.len()),
                    )
                    .expect_err("Error updating index in round robin");

                &worker_urls[current_index.load(std::sync::atomic::Ordering::SeqCst)]
            }
            Router::Random { worker_urls } => {
                &worker_urls[rand::random::<usize>() % worker_urls.len()]
            }
        };

        // Check if client requested streaming
        let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
            .map(|v| v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false))
            .unwrap_or(false);

        let res = match client
            .post(format!("{}/generate", worker_url))
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
            match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(_) => HttpResponse::InternalServerError().finish(),
            }
        } else {
            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(res.bytes_stream().map_err(|_| {
                    actix_web::error::ErrorInternalServerError("Failed to read string")
                }))
        }
    }
}
