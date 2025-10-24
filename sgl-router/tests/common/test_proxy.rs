// Simple HTTP proxy for testing proxy configuration
use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use tokio::sync::oneshot;

pub struct TestProxy {
    addr: SocketAddr,
    hits: Arc<AtomicUsize>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl TestProxy {
    pub async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("Failed to bind proxy");
        let addr = listener.local_addr().expect("Failed to get proxy addr");

        let hits = Arc::new(AtomicUsize::new(0));
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();

        let hits_clone = hits.clone();
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    result = listener.accept() => {
                        let Ok((stream, _)) = result else { break };
                        let hits = hits_clone.clone();
                        tokio::spawn(async move {
                            let _ = Self::handle_connection(stream, hits).await;
                        });
                    }
                }
            }
        });

        Self {
            addr,
            hits,
            shutdown_tx: Some(shutdown_tx),
            handle: Some(handle),
        }
    }

    async fn handle_connection(
        stream: tokio::net::TcpStream,
        hits: Arc<AtomicUsize>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use tokio::io::{AsyncBufReadExt, BufReader};

        hits.fetch_add(1, Ordering::SeqCst);

        let mut reader = BufReader::new(stream);
        let mut request_line = String::new();
        reader.read_line(&mut request_line).await?;

        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Ok(());
        }

        let method = parts[0];
        let target = parts[1];

        if method == "CONNECT" {
            Self::handle_connect(reader, target).await
        } else {
            Self::handle_http(reader, method, target).await
        }
    }

    async fn handle_connect(
        mut reader: tokio::io::BufReader<tokio::net::TcpStream>,
        target: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

        let (host, port) = target
            .split_once(':')
            .map(|(h, p)| (h, p.parse().unwrap_or(443)))
            .unwrap_or((target, 443));

        loop {
            let mut line = String::new();
            reader.read_line(&mut line).await?;
            if line == "\r\n" || line == "\n" {
                break;
            }
        }

        let mut client = reader.into_inner();
        let mut server = tokio::net::TcpStream::connect((host, port)).await?;
        client
            .write_all(b"HTTP/1.1 200 Connection established\r\n\r\n")
            .await?;

        let (mut cr, mut cw) = client.split();
        let (mut sr, mut sw) = server.split();
        tokio::try_join!(
            tokio::io::copy(&mut cr, &mut sw),
            tokio::io::copy(&mut sr, &mut cw)
        )?;
        Ok(())
    }

    async fn handle_http(
        mut reader: tokio::io::BufReader<tokio::net::TcpStream>,
        method: &str,
        target: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};

        let mut headers = Vec::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).await?;
            if line == "\r\n" || line == "\n" {
                break;
            }
            headers.push(line);
        }

        let (host, port, path) = if target.starts_with("http://") {
            let url = target.strip_prefix("http://").unwrap();
            let (authority, path) = url.split_once('/').unwrap_or((url, "/"));
            let (h, p) = authority
                .split_once(':')
                .map(|(h, p)| (h.to_string(), p.parse().unwrap_or(80)))
                .unwrap_or((authority.to_string(), 80));
            (h, p, path.to_string())
        } else {
            let host_header = headers
                .iter()
                .find(|h| h.to_lowercase().starts_with("host:"))
                .ok_or("Missing Host header")?;
            let host_value = host_header.split_once(':').unwrap().1.trim();
            let (h, p) = host_value
                .split_once(':')
                .map(|(h, p)| (h.to_string(), p.parse().unwrap_or(80)))
                .unwrap_or((host_value.to_string(), 80));
            (h, p, target.to_string())
        };

        let mut server = tokio::net::TcpStream::connect((host.as_str(), port)).await?;

        server
            .write_all(format!("{} {} HTTP/1.1\r\n", method, path).as_bytes())
            .await?;
        for header in &headers {
            server.write_all(header.as_bytes()).await?;
        }
        server.write_all(b"\r\n").await?;

        if let Some(len) = headers
            .iter()
            .find(|h| h.to_lowercase().starts_with("content-length:"))
            .and_then(|h| h.split_once(':'))
            .and_then(|(_, v)| v.trim().parse::<usize>().ok())
        {
            let mut buf = vec![0u8; len];
            reader.read_exact(&mut buf).await?;
            server.write_all(&buf).await?;
        }
        server.flush().await?;

        let mut client = reader.into_inner();
        let (mut cr, mut cw) = client.split();
        let (mut sr, mut sw) = server.split();

        tokio::try_join!(
            tokio::io::copy(&mut cr, &mut sw),
            tokio::io::copy(&mut sr, &mut cw)
        )?;

        Ok(())
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn hit_count(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }

    pub async fn wait_for_hits(&self, expected: usize, timeout: Duration) -> bool {
        let hits = self.hits.clone();
        tokio::time::timeout(timeout, async move {
            while hits.load(Ordering::SeqCst) < expected {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .is_ok()
    }

    pub async fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

impl Drop for TestProxy {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}
