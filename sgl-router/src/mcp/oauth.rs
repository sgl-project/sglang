// OAuth authentication support for MCP servers

use std::{net::SocketAddr, sync::Arc};

use axum::{
    extract::{Query, State},
    response::Html,
    routing::get,
    Router,
};
use rmcp::transport::auth::OAuthState;
use serde::Deserialize;
use tokio::sync::{oneshot, Mutex};

use crate::mcp::error::{McpError, McpResult};

/// OAuth callback parameters
#[derive(Debug, Deserialize)]
struct CallbackParams {
    code: String,
    #[allow(dead_code)]
    state: Option<String>,
}

/// State for the callback server
#[derive(Clone)]
struct CallbackState {
    code_receiver: Arc<Mutex<Option<oneshot::Sender<String>>>>,
}

/// HTML page returned after successful OAuth callback
const CALLBACK_HTML: &str = r#"
<!DOCTYPE html>
<html>
<head>
    <title>OAuth Success</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }
        h1 { color: #333; }
        p { color: #666; margin: 20px 0; }
        .success { color: #4CAF50; font-size: 48px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="success">✓</div>
        <h1>Authentication Successful!</h1>
        <p>You can now close this window and return to your application.</p>
    </div>
</body>
</html>
"#;

/// OAuth authentication helper for MCP servers
pub struct OAuthHelper {
    server_url: String,
    redirect_uri: String,
    callback_port: u16,
}

impl OAuthHelper {
    /// Create a new OAuth helper
    pub fn new(server_url: String, redirect_uri: String, callback_port: u16) -> Self {
        Self {
            server_url,
            redirect_uri,
            callback_port,
        }
    }

    /// Perform OAuth authentication flow
    pub async fn authenticate(
        &self,
        scopes: &[&str],
    ) -> McpResult<rmcp::transport::auth::AuthorizationManager> {
        // Initialize OAuth state machine
        let mut oauth_state = OAuthState::new(&self.server_url, None)
            .await
            .map_err(|e| McpError::Auth(format!("Failed to initialize OAuth: {}", e)))?;

        oauth_state
            .start_authorization(scopes, &self.redirect_uri)
            .await
            .map_err(|e| McpError::Auth(format!("Failed to start authorization: {}", e)))?;

        // Get authorization URL
        let auth_url = oauth_state
            .get_authorization_url()
            .await
            .map_err(|e| McpError::Auth(format!("Failed to get authorization URL: {}", e)))?;

        tracing::info!("OAuth authorization URL: {}", auth_url);

        // Start callback server and wait for code
        let auth_code = self.start_callback_server().await?;

        // Exchange code for token
        oauth_state
            .handle_callback(&auth_code)
            .await
            .map_err(|e| McpError::Auth(format!("Failed to handle OAuth callback: {}", e)))?;

        // Get authorization manager
        oauth_state
            .into_authorization_manager()
            .ok_or_else(|| McpError::Auth("Failed to get authorization manager".to_string()))
    }

    /// Start a local HTTP server to receive the OAuth callback
    async fn start_callback_server(&self) -> McpResult<String> {
        let (code_sender, code_receiver) = oneshot::channel::<String>();

        let state = CallbackState {
            code_receiver: Arc::new(Mutex::new(Some(code_sender))),
        };

        // Create router for callback
        let app = Router::new()
            .route("/callback", get(Self::callback_handler))
            .with_state(state);

        let addr = SocketAddr::from(([127, 0, 0, 1], self.callback_port));

        // Start server in background
        let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
            McpError::Auth(format!(
                "Failed to bind to callback port {}: {}",
                self.callback_port, e
            ))
        })?;

        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        tracing::info!(
            "OAuth callback server started on port {}",
            self.callback_port
        );

        // Wait for authorization code
        code_receiver
            .await
            .map_err(|_| McpError::Auth("Failed to receive authorization code".to_string()))
    }

    /// Handle OAuth callback
    async fn callback_handler(
        Query(params): Query<CallbackParams>,
        State(state): State<CallbackState>,
    ) -> Html<String> {
        tracing::debug!("Received OAuth callback with code");

        // Send code to waiting task
        if let Some(sender) = state.code_receiver.lock().await.take() {
            let _ = sender.send(params.code);
        }

        Html(CALLBACK_HTML.to_string())
    }
}

/// Create an OAuth-authenticated client
pub async fn create_oauth_client(
    server_url: String,
    _sse_url: String,
    redirect_uri: String,
    callback_port: u16,
    scopes: &[&str],
) -> McpResult<rmcp::transport::auth::AuthClient<reqwest::Client>> {
    let helper = OAuthHelper::new(server_url, redirect_uri, callback_port);
    let auth_manager = helper.authenticate(scopes).await?;

    let client = rmcp::transport::auth::AuthClient::new(reqwest::Client::default(), auth_manager);

    Ok(client)
}
