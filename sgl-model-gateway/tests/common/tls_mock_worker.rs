// TLS-enabled mock worker for mTLS integration tests
#![allow(dead_code)]

use std::{
    net::SocketAddr,
    path::Path,
    sync::{Arc, Once},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use axum_server::tls_rustls::RustlsConfig;
use rustls::{server::WebPkiClientVerifier, RootCertStore};
use serde_json::json;
use tokio::sync::RwLock;

// Ensure crypto provider is installed exactly once
static CRYPTO_PROVIDER_INIT: Once = Once::new();

fn ensure_crypto_provider() {
    CRYPTO_PROVIDER_INIT.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Configuration for TLS mock worker behavior
#[derive(Clone)]
pub struct TlsMockWorkerConfig {
    pub port: u16,
    /// Require client certificate (mTLS) or just server TLS
    pub require_client_cert: bool,
    /// Response delay in milliseconds
    pub response_delay_ms: u64,
    /// Fail rate (0.0 - 1.0)
    pub fail_rate: f32,
}

impl Default for TlsMockWorkerConfig {
    fn default() -> Self {
        Self {
            port: 0,
            require_client_cert: true,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }
}

/// TLS-enabled mock worker server for mTLS testing
pub struct TlsMockWorker {
    config: Arc<RwLock<TlsMockWorkerConfig>>,
    shutdown_handle: Option<tokio::task::JoinHandle<()>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl TlsMockWorker {
    pub fn new(config: TlsMockWorkerConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            shutdown_handle: None,
            shutdown_tx: None,
        }
    }

    /// Start the TLS mock worker server
    ///
    /// # Arguments
    /// * `server_cert_path` - Path to server certificate PEM file
    /// * `server_key_path` - Path to server private key PEM file
    /// * `ca_cert_path` - Path to CA certificate for client verification (optional for TLS-only mode)
    pub async fn start(
        &mut self,
        server_cert_path: &Path,
        server_key_path: &Path,
        ca_cert_path: Option<&Path>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Ensure crypto provider is installed before using rustls
        ensure_crypto_provider();

        let config = self.config.clone();
        let port = config.read().await.port;
        let require_client_cert = config.read().await.require_client_cert;

        // If port is 0, find an available port
        let port = if port == 0 {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            drop(listener);
            config.write().await.port = port;
            port
        } else {
            port
        };

        let app = Router::new()
            .route("/health", get(health_handler))
            .route("/health_generate", get(health_generate_handler))
            .route("/get_server_info", get(server_info_handler))
            .route("/generate", post(generate_handler))
            .route("/v1/chat/completions", post(chat_completions_handler))
            .with_state(config);

        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        self.shutdown_tx = Some(shutdown_tx);

        // Build TLS configuration
        let rustls_config = if require_client_cert {
            // mTLS: require client certificate
            let ca_cert_path = ca_cert_path.ok_or("CA cert path required for mTLS")?;
            build_mtls_config(server_cert_path, server_key_path, ca_cert_path).await?
        } else {
            // TLS only: no client cert required
            build_tls_config(server_cert_path, server_key_path).await?
        };

        let addr = SocketAddr::from(([127, 0, 0, 1], port));

        // Spawn the server in a separate task
        let handle = tokio::spawn(async move {
            let server =
                axum_server::bind_rustls(addr, rustls_config).serve(app.into_make_service());

            tokio::select! {
                result = server => {
                    if let Err(e) = result {
                        eprintln!("TLS Server error: {}", e);
                    }
                }
                _ = &mut shutdown_rx => {
                    // Graceful shutdown
                }
            }
        });

        self.shutdown_handle = Some(handle);

        // Wait for the server to start
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let url = format!("https://127.0.0.1:{}", port);
        Ok(url)
    }

    /// Stop the TLS mock worker server
    pub async fn stop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }

        if let Some(handle) = self.shutdown_handle.take() {
            let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), handle).await;
        }
    }
}

impl Drop for TlsMockWorker {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
    }
}

/// Build TLS config for server-only TLS (no client cert required)
async fn build_tls_config(
    cert_path: &Path,
    key_path: &Path,
) -> Result<RustlsConfig, Box<dyn std::error::Error + Send + Sync>> {
    let config = RustlsConfig::from_pem_file(cert_path, key_path).await?;
    Ok(config)
}

/// Build mTLS config requiring client certificate
async fn build_mtls_config(
    cert_path: &Path,
    key_path: &Path,
    ca_cert_path: &Path,
) -> Result<RustlsConfig, Box<dyn std::error::Error + Send + Sync>> {
    use std::io::BufReader;

    use rustls_pemfile::{certs, pkcs8_private_keys};

    // Read server certificate
    let cert_file = std::fs::File::open(cert_path)?;
    let mut reader = BufReader::new(cert_file);
    let cert_chain: Vec<_> = certs(&mut reader).filter_map(|r| r.ok()).collect();

    // Read server private key
    let key_file = std::fs::File::open(key_path)?;
    let mut reader = BufReader::new(key_file);
    let private_key = pkcs8_private_keys(&mut reader)
        .next()
        .ok_or("No private key found")??;

    // Read CA certificate for client verification
    let ca_file = std::fs::File::open(ca_cert_path)?;
    let mut reader = BufReader::new(ca_file);
    let ca_certs: Vec<_> = certs(&mut reader).filter_map(|r| r.ok()).collect();

    // Build root certificate store for client verification
    let mut root_store = RootCertStore::empty();
    for cert in ca_certs {
        root_store.add(cert)?;
    }

    // Create client certificate verifier
    let client_verifier = WebPkiClientVerifier::builder(Arc::new(root_store))
        .build()
        .map_err(|e| format!("Failed to build client verifier: {}", e))?;

    // Build server config with client verification
    let server_config = rustls::ServerConfig::builder()
        .with_client_cert_verifier(client_verifier)
        .with_single_cert(cert_chain, private_key.into())
        .map_err(|e| format!("Failed to build server config: {}", e))?;

    Ok(RustlsConfig::from_config(Arc::new(server_config)))
}

// Handler implementations (simplified versions of mock_worker handlers)

async fn should_fail(config: &TlsMockWorkerConfig) -> bool {
    rand::random::<f32>() < config.fail_rate
}

async fn health_handler(State(config): State<Arc<RwLock<TlsMockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Random failure" })),
        )
            .into_response();
    }

    Json(json!({
        "status": "healthy",
        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        "tls_enabled": true
    }))
    .into_response()
}

async fn health_generate_handler(
    State(config): State<Arc<RwLock<TlsMockWorkerConfig>>>,
) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Random failure" })),
        )
            .into_response();
    }

    Json(json!({
        "status": "ok",
        "queue_length": 0,
        "processing_time_ms": config.response_delay_ms
    }))
    .into_response()
}

async fn server_info_handler(State(config): State<Arc<RwLock<TlsMockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Random failure" })),
        )
            .into_response();
    }

    Json(json!({
        "model_path": "mock-tls-model",
        "port": config.port,
        "host": "127.0.0.1",
        "tls_enabled": true,
        "version": "0.3.0"
    }))
    .into_response()
}

async fn generate_handler(
    State(config): State<Arc<RwLock<TlsMockWorkerConfig>>>,
    Json(_payload): Json<serde_json::Value>,
) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Random failure" })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    Json(json!({
        "text": "This is a mock TLS response.",
        "meta_info": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "tls_verified": true
        }
    }))
    .into_response()
}

async fn chat_completions_handler(
    State(config): State<Arc<RwLock<TlsMockWorkerConfig>>>,
    Json(_payload): Json<serde_json::Value>,
) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure",
                    "type": "internal_error"
                }
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": timestamp,
        "model": "mock-tls-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a mock TLS chat response."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }))
    .into_response()
}
