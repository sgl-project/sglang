use std::{path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::Utc;
use deadpool::managed::{Manager, Metrics, Pool, PoolError, RecycleError, RecycleResult};
use oracle::{sql_type::OracleType, Connection};
use serde_json::Value;

use crate::{
    config::OracleConfig,
    data_connector::conversations::{
        Conversation, ConversationId, ConversationMetadata, ConversationStorage,
        ConversationStorageError, NewConversation, Result,
    },
};

#[derive(Clone)]
pub struct OracleConversationStorage {
    pool: Pool<ConversationOracleConnectionManager>,
}

impl OracleConversationStorage {
    pub fn new(config: OracleConfig) -> Result<Self> {
        configure_oracle_client(&config)?;
        initialize_schema(&config)?;

        let config = Arc::new(config);
        let manager = ConversationOracleConnectionManager::new(config.clone());

        let mut builder = Pool::builder(manager)
            .max_size(config.pool_max)
            .runtime(deadpool::Runtime::Tokio1);

        if config.pool_timeout_secs > 0 {
            builder = builder.wait_timeout(Some(Duration::from_secs(config.pool_timeout_secs)));
        }

        let pool = builder.build().map_err(|err| {
            ConversationStorageError::StorageError(format!(
                "failed to build Oracle pool for conversations: {err}"
            ))
        })?;

        Ok(Self { pool })
    }

    async fn with_connection<F, T>(&self, func: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let connection = self.pool.get().await.map_err(map_pool_error)?;
        tokio::task::spawn_blocking(move || {
            let result = func(&connection);
            drop(connection);
            result
        })
        .await
        .map_err(|err| {
            ConversationStorageError::StorageError(format!(
                "failed to execute Oracle conversation task: {err}"
            ))
        })?
    }

    fn parse_metadata(raw: Option<String>) -> Result<Option<ConversationMetadata>> {
        match raw {
            Some(json) if !json.is_empty() => {
                let value: Value = serde_json::from_str(&json)?;
                match value {
                    Value::Object(map) => Ok(Some(map)),
                    Value::Null => Ok(None),
                    other => Err(ConversationStorageError::StorageError(format!(
                        "conversation metadata expected object, got {other}"
                    ))),
                }
            }
            _ => Ok(None),
        }
    }
}

#[async_trait]
impl ConversationStorage for OracleConversationStorage {
    async fn create_conversation(&self, input: NewConversation) -> Result<Conversation> {
        let conversation = Conversation::new(input);
        let id_str = conversation.id.0.clone();
        let created_at = conversation.created_at;
        let metadata_json = conversation
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;

        self.with_connection(move |conn| {
            conn.execute(
                "INSERT INTO conversations (id, created_at, metadata) VALUES (:1, :2, :3)",
                &[&id_str, &created_at, &metadata_json],
            )
            .map(|_| ())
            .map_err(map_oracle_error)
        })
        .await?;

        Ok(conversation)
    }

    async fn get_conversation(&self, id: &ConversationId) -> Result<Option<Conversation>> {
        let lookup = id.0.clone();
        self.with_connection(move |conn| {
            let mut stmt = conn
                .statement("SELECT id, created_at, metadata FROM conversations WHERE id = :1")
                .build()
                .map_err(map_oracle_error)?;
            let mut rows = stmt.query(&[&lookup]).map_err(map_oracle_error)?;

            if let Some(row_res) = rows.next() {
                let row = row_res.map_err(map_oracle_error)?;
                let id: String = row.get(0).map_err(map_oracle_error)?;
                let created_at: chrono::DateTime<Utc> = row.get(1).map_err(map_oracle_error)?;
                let metadata_raw: Option<String> = row.get(2).map_err(map_oracle_error)?;
                let metadata = Self::parse_metadata(metadata_raw)?;
                Ok(Some(Conversation::with_parts(
                    ConversationId(id),
                    created_at,
                    metadata,
                )))
            } else {
                Ok(None)
            }
        })
        .await
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>> {
        let id_str = id.0.clone();
        let metadata_json = metadata.as_ref().map(serde_json::to_string).transpose()?;
        let conversation_id = id.clone();

        self.with_connection(move |conn| {
            let mut stmt = conn
                .statement(
                    "UPDATE conversations \
                         SET metadata = :1 \
                         WHERE id = :2 \
                         RETURNING created_at INTO :3",
                )
                .build()
                .map_err(map_oracle_error)?;

            stmt.bind(3, &OracleType::TimestampTZ(6))
                .map_err(map_oracle_error)?;
            stmt.execute(&[&metadata_json, &id_str])
                .map_err(map_oracle_error)?;

            if stmt.row_count().map_err(map_oracle_error)? == 0 {
                return Ok(None);
            }

            let mut created_at: Vec<chrono::DateTime<Utc>> =
                stmt.returned_values(3).map_err(map_oracle_error)?;
            let created_at = created_at.pop().ok_or_else(|| {
                ConversationStorageError::StorageError(
                    "Oracle update did not return created_at".to_string(),
                )
            })?;

            Ok(Some(Conversation::with_parts(
                conversation_id,
                created_at,
                metadata,
            )))
        })
        .await
    }

    async fn delete_conversation(&self, id: &ConversationId) -> Result<bool> {
        let id_str = id.0.clone();
        let res = self
            .with_connection(move |conn| {
                conn.execute("DELETE FROM conversations WHERE id = :1", &[&id_str])
                    .map_err(map_oracle_error)
            })
            .await?;

        Ok(res.row_count().map_err(map_oracle_error)? > 0)
    }
}

#[derive(Clone)]
struct ConversationOracleConnectionManager {
    params: Arc<OracleConnectParams>,
}

impl ConversationOracleConnectionManager {
    fn new(config: Arc<OracleConfig>) -> Self {
        let params = OracleConnectParams {
            username: config.username.clone(),
            password: config.password.clone(),
            connect_descriptor: config.connect_descriptor.clone(),
        };

        Self {
            params: Arc::new(params),
        }
    }
}

impl std::fmt::Debug for ConversationOracleConnectionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationOracleConnectionManager")
            .field("username", &self.params.username)
            .field("connect_descriptor", &self.params.connect_descriptor)
            .finish()
    }
}

#[derive(Clone)]
struct OracleConnectParams {
    username: String,
    password: String,
    connect_descriptor: String,
}

impl std::fmt::Debug for OracleConnectParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleConnectParams")
            .field("username", &self.username)
            .field("connect_descriptor", &self.connect_descriptor)
            .finish()
    }
}

#[async_trait]
impl Manager for ConversationOracleConnectionManager {
    type Type = Connection;
    type Error = oracle::Error;

    fn create(
        &self,
    ) -> impl std::future::Future<Output = std::result::Result<Connection, oracle::Error>> + Send
    {
        let params = self.params.clone();
        async move {
            let mut conn = Connection::connect(
                &params.username,
                &params.password,
                &params.connect_descriptor,
            )?;
            conn.set_autocommit(true);
            Ok(conn)
        }
    }

    #[allow(clippy::manual_async_fn)]
    fn recycle(
        &self,
        conn: &mut Connection,
        _: &Metrics,
    ) -> impl std::future::Future<Output = RecycleResult<Self::Error>> + Send {
        async move { conn.ping().map_err(RecycleError::Backend) }
    }
}

fn configure_oracle_client(config: &OracleConfig) -> Result<()> {
    if let Some(wallet_path) = &config.wallet_path {
        let wallet_path = Path::new(wallet_path);
        if !wallet_path.is_dir() {
            return Err(ConversationStorageError::StorageError(format!(
                "Oracle wallet/config path '{}' is not a directory",
                wallet_path.display()
            )));
        }

        if !wallet_path.join("tnsnames.ora").exists() && !wallet_path.join("sqlnet.ora").exists() {
            return Err(ConversationStorageError::StorageError(format!(
                "Oracle wallet/config path '{}' is missing tnsnames.ora or sqlnet.ora",
                wallet_path.display()
            )));
        }

        std::env::set_var("TNS_ADMIN", wallet_path);
    }
    Ok(())
}

fn initialize_schema(config: &OracleConfig) -> Result<()> {
    let conn = Connection::connect(
        &config.username,
        &config.password,
        &config.connect_descriptor,
    )
    .map_err(map_oracle_error)?;

    let exists: i64 = conn
        .query_row_as(
            "SELECT COUNT(*) FROM user_tables WHERE table_name = 'CONVERSATIONS'",
            &[],
        )
        .map_err(map_oracle_error)?;

    if exists == 0 {
        conn.execute(
            "CREATE TABLE conversations (
                id VARCHAR2(64) PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE,
                metadata CLOB
            )",
            &[],
        )
        .map_err(map_oracle_error)?;
    }

    Ok(())
}

fn map_pool_error(err: PoolError<oracle::Error>) -> ConversationStorageError {
    match err {
        PoolError::Backend(e) => map_oracle_error(e),
        other => ConversationStorageError::StorageError(format!(
            "failed to obtain Oracle conversation connection: {other}"
        )),
    }
}

fn map_oracle_error(err: oracle::Error) -> ConversationStorageError {
    if let Some(db_err) = err.db_error() {
        ConversationStorageError::StorageError(format!(
            "Oracle error (code {}): {}",
            db_err.code(),
            db_err.message()
        ))
    } else {
        ConversationStorageError::StorageError(err.to_string())
    }
}
