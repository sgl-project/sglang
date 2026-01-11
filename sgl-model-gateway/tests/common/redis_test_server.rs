use std::{
    process::{Child, Command, Stdio},
    sync::OnceLock,
    time::Duration,
};

use tokio::net::TcpStream;
use tracing::warn;

// ============================================================================
// Shared Redis Server (singleton for all tests)
// ============================================================================

static SHARED_SERVER: OnceLock<SharedRedisServer> = OnceLock::new();

pub struct SharedRedisServer {
    _process: Child,
    pub port: u16,
}

impl SharedRedisServer {
    fn start() -> Self {
        let port = portpicker::pick_unused_port().expect("No available port");
        let process = Command::new("redis-server")
            .args([
                "--port",
                &port.to_string(),
                "--save",
                "",
                "--appendonly",
                "no",
                "--daemonize",
                "no",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to start redis-server. Is redis-server installed?");

        std::thread::sleep(Duration::from_millis(500));
        Self {
            _process: process,
            port,
        }
    }

    pub fn url(&self) -> String {
        format!("redis://127.0.0.1:{}", self.port)
    }
}

/// Get or create a shared Redis server for tests.
/// Multiple tests can use this server with different key prefixes.
pub fn get_shared_server() -> &'static SharedRedisServer {
    SHARED_SERVER.get_or_init(SharedRedisServer::start)
}

// ============================================================================
// Per-test Redis Server (for tests that need isolated instances)
// ============================================================================

pub struct RedisTestServer {
    process: Option<Child>,
    port: u16,
    pub url: String,
}

impl RedisTestServer {
    pub async fn start() -> Result<Self, String> {
        let port = portpicker::pick_unused_port()
            .ok_or_else(|| "Failed to find available port".to_string())?;
        let url = format!("redis://127.0.0.1:{}", port);

        let process = Command::new("redis-server")
            .args([
                "--port",
                &port.to_string(),
                "--save",
                "",
                "--appendonly",
                "no",
                "--daemonize",
                "no",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                format!(
                    "Failed to start redis-server: {}. Is redis-server installed?",
                    e
                )
            })?;

        let server = Self {
            process: Some(process),
            port,
            url,
        };

        server.wait_ready().await?;
        Ok(server)
    }

    async fn wait_ready(&self) -> Result<(), String> {
        let addr = format!("127.0.0.1:{}", self.port);
        for _ in 0..50 {
            if TcpStream::connect(&addr).await.is_ok() {
                tokio::time::sleep(Duration::from_millis(50)).await;
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Err(format!(
            "Redis server failed to start on port {}",
            self.port
        ))
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub async fn stop(&mut self) {
        self.stop_inner();
    }

    fn stop_inner(&mut self) {
        if let Some(mut process) = self.process.take() {
            if let Err(e) = process.kill() {
                warn!("Failed to kill redis-server process: {}", e);
            }
            if let Err(e) = process.wait() {
                warn!("Failed to wait for redis-server process: {}", e);
            }
        }
    }

    pub async fn flush_all(&self) -> Result<(), String> {
        let client = redis::Client::open(self.url.as_str())
            .map_err(|e| format!("Failed to connect to Redis: {}", e))?;
        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| format!("Failed to get connection: {}", e))?;
        redis::cmd("FLUSHALL")
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| format!("Failed to flush: {}", e))?;
        Ok(())
    }
}

impl Drop for RedisTestServer {
    fn drop(&mut self) {
        self.stop_inner();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_redis_server_start_stop() {
        let server = RedisTestServer::start().await.unwrap();
        assert!(server.url().starts_with("redis://"));

        let client = redis::Client::open(server.url()).unwrap();
        let mut conn = client.get_multiplexed_async_connection().await.unwrap();

        let _: () = redis::cmd("SET")
            .arg("test_key")
            .arg("test_value")
            .query_async(&mut conn)
            .await
            .unwrap();

        let value: String = redis::cmd("GET")
            .arg("test_key")
            .query_async(&mut conn)
            .await
            .unwrap();
        assert_eq!(value, "test_value");
    }
}
