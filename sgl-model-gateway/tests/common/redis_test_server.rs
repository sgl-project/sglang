use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::time::timeout;

pub struct RedisTestServer {
    process: Option<Child>,
    port: u16,
    pub url: String,
}

impl RedisTestServer {
    pub async fn start() -> Result<Self, String> {
        let port = find_available_port().await?;
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
            .map_err(|e| format!("Failed to start redis-server: {}. Is redis-server installed?", e))?;

        let mut server = Self {
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
        Err(format!("Redis server failed to start on port {}", self.port))
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub async fn stop(&mut self) {
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
            let _ = process.wait();
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
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
}

async fn find_available_port() -> Result<u16, String> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| format!("Failed to bind: {}", e))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("Failed to get local addr: {}", e))?
        .port();
    drop(listener);
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
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
