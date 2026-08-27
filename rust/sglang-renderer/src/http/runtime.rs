//! Standalone renderer HTTP runtime.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::{
    DynamoTokenizer, OpenAIRequestLowerer, PooledTokenizer, RendererConfig, RendererService,
    TextTokenizer, load_tokenizer,
};

pub struct RendererRuntimeConfig {
    pub http_addr: SocketAddr,
    pub http_workers: usize,
    pub tokenizer_workers: usize,
    pub queue_capacity: usize,
    pub renderer: RendererConfig,
}

pub struct RendererRuntime {
    thread: Mutex<Option<JoinHandle<()>>>,
    shutdown: Mutex<Option<flume::Sender<()>>>,
}

impl RendererRuntime {
    pub fn start(config: RendererRuntimeConfig) -> Result<Self, String> {
        if config.renderer.skip_tokenizer_init {
            return Err("standalone rendering requires a tokenizer".into());
        }
        let tokenizer = load_tokenizer(
            (!config.renderer.tokenizer_path.is_empty())
                .then_some(config.renderer.tokenizer_path.as_str()),
            config.renderer.revision.as_deref(),
            false,
        )?
        .ok_or_else(|| "standalone rendering requires a tokenizer".to_owned())?;
        let tokenizer: Arc<dyn TextTokenizer> = Arc::new(DynamoTokenizer::new(tokenizer));
        let backend = Arc::new(PooledTokenizer::new(
            tokenizer,
            config.tokenizer_workers,
            config.queue_capacity,
        ));
        let renderer = Arc::new(RendererService::new(
            OpenAIRequestLowerer::new(config.renderer),
            backend,
        ));
        let listener = std::net::TcpListener::bind(config.http_addr)
            .map_err(|error| format!("binding renderer on {} failed: {error}", config.http_addr))?;
        listener
            .set_nonblocking(true)
            .map_err(|error| format!("configuring renderer listener failed: {error}"))?;
        let (shutdown_tx, shutdown_rx) = flume::bounded(1);
        let thread = std::thread::Builder::new()
            .name("renderer-http".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(config.http_workers.max(1))
                    .enable_all()
                    .build()
                    .expect("build renderer runtime");
                runtime.block_on(async move {
                    let listener = match tokio::net::TcpListener::from_std(listener) {
                        Ok(listener) => listener,
                        Err(error) => {
                            tracing::error!(%error, "failed to adopt renderer listener");
                            return;
                        }
                    };
                    let server =
                        axum::serve(listener, super::render_routes(renderer).into_make_service());
                    tokio::select! {
                        result = server => {
                            if let Err(error) = result {
                                tracing::error!(%error, "renderer HTTP server exited");
                            }
                        }
                        _ = shutdown_rx.recv_async() => {}
                    }
                });
            })
            .map_err(|error| format!("spawning renderer runtime failed: {error}"))?;
        Ok(Self {
            thread: Mutex::new(Some(thread)),
            shutdown: Mutex::new(Some(shutdown_tx)),
        })
    }

    pub fn request_shutdown(&self) {
        self.shutdown
            .lock()
            .expect("renderer shutdown mutex")
            .take();
        if let Some(thread) = self.thread.lock().expect("renderer thread mutex").take() {
            let _ = thread.join();
        }
    }
}

impl Drop for RendererRuntime {
    fn drop(&mut self) {
        self.request_shutdown();
    }
}
