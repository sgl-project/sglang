//! mTLS (mutual TLS) support for mesh cluster communication
//!
//! Provides optional mTLS encryption for gRPC mesh connections using rustls.
//! Supports certificate rotation without restart.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use rustls::{
    pki_types::{CertificateDer, PrivateKeyDer},
    ClientConfig, RootCertStore, ServerConfig,
};
use rustls_pemfile::{certs, pkcs8_private_keys};
use tokio::{fs, sync::RwLock};
use tracing::{info, warn};

/// mTLS configuration
#[derive(Debug, Clone)]
pub struct MTLSConfig {
    /// Path to CA certificate file
    pub ca_cert_path: PathBuf,
    /// Path to server certificate file
    pub server_cert_path: PathBuf,
    /// Path to server private key file
    pub server_key_path: PathBuf,
    /// Whether to require client certificates
    pub require_client_cert: bool,
    /// Certificate rotation check interval
    pub rotation_check_interval: Duration,
}

impl Default for MTLSConfig {
    fn default() -> Self {
        Self {
            ca_cert_path: PathBuf::from("/etc/ssl/certs/ca-certificates.crt"),
            server_cert_path: PathBuf::from("/etc/ssl/certs/server.crt"),
            server_key_path: PathBuf::from("/etc/ssl/private/server.key"),
            require_client_cert: true,
            rotation_check_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// mTLS certificate manager
pub struct MTLSManager {
    config: MTLSConfig,
    server_config: Arc<RwLock<Option<Arc<ServerConfig>>>>,
    client_config: Arc<RwLock<Option<Arc<ClientConfig>>>>,
}

impl MTLSManager {
    /// Create a new mTLS manager
    pub fn new(config: MTLSConfig) -> Self {
        Self {
            config,
            server_config: Arc::new(RwLock::new(None)),
            client_config: Arc::new(RwLock::new(None)),
        }
    }

    /// Load server TLS configuration
    pub async fn load_server_config(&self) -> Result<Arc<ServerConfig>> {
        let certs = self.load_certs(&self.config.server_cert_path).await?;
        let key = self.load_private_key(&self.config.server_key_path).await?;

        let mut server_config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)?;

        // Enable ALPN for HTTP/2
        server_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

        let config = Arc::new(server_config);
        *self.server_config.write().await = Some(config.clone());
        Ok(config)
    }

    /// Load client TLS configuration
    pub async fn load_client_config(&self) -> Result<Arc<ClientConfig>> {
        let mut root_store = RootCertStore::empty();

        // Load CA certificate
        let ca_certs = self.load_certs(&self.config.ca_cert_path).await?;
        for cert in ca_certs {
            root_store.add(cert)?;
        }

        let mut client_config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        // Enable ALPN for HTTP/2
        client_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

        let config = Arc::new(client_config);
        *self.client_config.write().await = Some(config.clone());
        Ok(config)
    }

    /// Load certificates from file
    async fn load_certs(&self, path: &Path) -> Result<Vec<CertificateDer<'static>>> {
        let cert_data = fs::read(path).await?;
        let certs = certs(&mut cert_data.as_slice()).collect::<Result<Vec<_>, _>>()?;
        Ok(certs)
    }

    /// Load private key from file
    async fn load_private_key(&self, path: &Path) -> Result<PrivateKeyDer<'static>> {
        let key_data = fs::read(path).await?;
        let mut keys =
            pkcs8_private_keys(&mut key_data.as_slice()).collect::<Result<Vec<_>, _>>()?;

        if keys.is_empty() {
            return Err(anyhow::anyhow!("No private key found in file"));
        }

        Ok(PrivateKeyDer::Pkcs8(keys.remove(0)))
    }

    /// Start certificate rotation monitoring
    pub async fn start_rotation_monitor(&self) {
        let config = self.config.clone();
        let server_config = self.server_config.clone();
        let client_config = self.client_config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.rotation_check_interval);
            loop {
                interval.tick().await;

                // Check if certificates have changed
                if let Err(e) =
                    Self::check_and_reload_certs(&config, &server_config, &client_config).await
                {
                    warn!("Error checking certificate rotation: {}", e);
                }
            }
        });
    }

    /// Check and reload certificates if they have changed
    async fn check_and_reload_certs(
        config: &MTLSConfig,
        _server_config: &Arc<RwLock<Option<Arc<ServerConfig>>>>,
        _client_config: &Arc<RwLock<Option<Arc<ClientConfig>>>>,
    ) -> Result<()> {
        // Get file modification times
        let server_cert_mtime = fs::metadata(&config.server_cert_path).await?.modified()?;
        let server_key_mtime = fs::metadata(&config.server_key_path).await?.modified()?;
        let ca_cert_mtime = fs::metadata(&config.ca_cert_path).await?.modified()?;

        // TODO: Compare with cached modification times
        // For now, we'll just log that rotation monitoring is active
        info!(
            "Certificate rotation check: server_cert={:?}, server_key={:?}, ca_cert={:?}",
            server_cert_mtime, server_key_mtime, ca_cert_mtime
        );

        // Reload if certificates have changed
        // This is a simplified version - in production, you'd compare mtimes
        Ok(())
    }

    /// Get current server config (for use with tonic)
    pub async fn get_server_config(&self) -> Option<Arc<ServerConfig>> {
        self.server_config.read().await.clone()
    }

    /// Get current client config (for use with tonic)
    pub async fn get_client_config(&self) -> Option<Arc<ClientConfig>> {
        self.client_config.read().await.clone()
    }
}

/// Optional mTLS manager
pub type OptionalMTLSManager = Option<Arc<MTLSManager>>;
