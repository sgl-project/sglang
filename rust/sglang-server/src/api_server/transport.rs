//! HTTP connection ownership and HTTP/2 settings for the native listener.

use axum::extract::ConnectInfo;
use axum::{Extension, Router};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use hyper_util::service::TowerToHyperService;
use tokio::task::JoinSet;

#[derive(Clone, Copy)]
pub(super) struct Http2Settings {
    pub max_concurrent_streams: u32,
    pub initial_connection_window_size: u32,
}

pub(super) async fn serve_listener(
    listener: std::net::TcpListener,
    app: Router,
    http2: Option<Http2Settings>,
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::from_std(listener)?;
    // Dropping this future closes the listener and aborts every connection.
    // Request guards then cancel unfinished scheduler work on shutdown.
    let mut connections = JoinSet::new();
    loop {
        let accepted = tokio::select! {
            accepted = listener.accept() => accepted,
            Some(result) = connections.join_next(), if !connections.is_empty() => {
                if let Err(error) = result {
                    tracing::warn!(%error, "HTTP connection task failed");
                }
                continue;
            }
        };
        let (stream, peer) = match accepted {
            Ok(connection) => connection,
            Err(error) => {
                tracing::warn!(%error, "HTTP accept failed");
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        let service = TowerToHyperService::new(app.clone().layer(Extension(ConnectInfo(peer))));
        connections.spawn(async move {
            let mut builder = Builder::new(TokioExecutor::new());
            if let Some(settings) = http2 {
                builder
                    .http2()
                    .max_concurrent_streams(settings.max_concurrent_streams)
                    .initial_connection_window_size(settings.initial_connection_window_size);
            } else {
                builder = builder.http1_only();
            }
            if let Err(error) = builder
                .serve_connection(TokioIo::new(stream), service)
                .await
            {
                tracing::debug!(%peer, %error, "HTTP connection closed with an error");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use axum::routing::get;
    use tokio::sync::Notify;

    use super::*;

    fn listener() -> (std::net::TcpListener, String) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        (listener, url)
    }

    struct Active(Arc<AtomicUsize>);
    impl Drop for Active {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn protocols_stream_limit_peer_address_and_shutdown_match_configuration() {
        for enable_http2 in [false, true] {
            let active = Arc::new(AtomicUsize::new(0));
            let peak = Arc::new(AtomicUsize::new(0));
            let started = Arc::new(Notify::new());
            let work = {
                let active = active.clone();
                let peak = peak.clone();
                move |ConnectInfo(peer): ConnectInfo<SocketAddr>| {
                    let active = active.clone();
                    let peak = peak.clone();
                    async move {
                        let running = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(running, Ordering::SeqCst);
                        let _guard = Active(active);
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        peer.ip().to_string()
                    }
                }
            };
            let hold = {
                let active = active.clone();
                let started = started.clone();
                move || {
                    let active = active.clone();
                    let started = started.clone();
                    async move {
                        active.fetch_add(1, Ordering::SeqCst);
                        let _guard = Active(active);
                        started.notify_one();
                        std::future::pending::<&'static str>().await
                    }
                }
            };
            let app = Router::new()
                .route("/work", get(work))
                .route("/hold", get(hold));
            let (listener, url) = listener();
            let server = tokio::spawn(serve_listener(
                listener,
                app,
                enable_http2.then_some(Http2Settings {
                    max_concurrent_streams: 1,
                    initial_connection_window_size: 2 * 1024 * 1024,
                }),
            ));
            let http1 = reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(5))
                .http1_only()
                .build()
                .unwrap();
            let response = http1.get(format!("{url}/work")).send().await.unwrap();
            assert_eq!(response.version(), reqwest::Version::HTTP_11);
            assert_eq!(response.text().await.unwrap(), "127.0.0.1");

            let http2 = reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(5))
                .http2_prior_knowledge()
                .build()
                .unwrap();
            if enable_http2 {
                let response = http2.get(format!("{url}/work")).send().await.unwrap();
                assert_eq!(response.version(), reqwest::Version::HTTP_2);
                assert_eq!(response.text().await.unwrap(), "127.0.0.1");
                let responses = futures::future::join_all(
                    (0..4).map(|_| http2.get(format!("{url}/work")).send()),
                )
                .await;
                assert!(
                    responses
                        .into_iter()
                        .all(|response| response.unwrap().status().is_success())
                );
                assert_eq!(
                    peak.load(Ordering::SeqCst),
                    1,
                    "HTTP/2 SETTINGS must enforce the configured stream limit"
                );
            } else {
                assert!(http2.get(format!("{url}/work")).send().await.is_err());
            }
            let client = if enable_http2 { http2 } else { http1 };
            let held = tokio::spawn(async move { client.get(format!("{url}/hold")).send().await });
            tokio::time::timeout(Duration::from_secs(2), started.notified())
                .await
                .unwrap();
            server.abort();
            assert!(server.await.unwrap_err().is_cancelled());
            tokio::time::timeout(Duration::from_secs(2), async {
                while active.load(Ordering::SeqCst) != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert!(
                tokio::time::timeout(Duration::from_secs(2), held)
                    .await
                    .unwrap()
                    .unwrap()
                    .is_err()
            );
        }
    }
}
