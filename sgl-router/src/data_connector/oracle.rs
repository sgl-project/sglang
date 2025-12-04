//! Oracle storage implementation using OracleStore helper.
//!
//! Structure:
//! 1. OracleStore helper and common utilities
//! 2. OracleConversationStorage
//! 3. OracleConversationItemStorage
//! 4. OracleResponseStorage

use std::{path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool::managed::{Manager, Metrics, Pool, RecycleError, RecycleResult};
use oracle::{
    sql_type::{OracleType, ToSql},
    Connection, Row,
};
use serde_json::Value;

use super::core::{
    make_item_id, Conversation, ConversationId, ConversationItem, ConversationItemId,
    ConversationItemStorage, ConversationItemStorageError, ConversationMetadata,
    ConversationStorage, ConversationStorageError, ListParams, NewConversation,
    NewConversationItem, ResponseChain, ResponseId, ResponseStorage, ResponseStorageError,
    SortOrder, StoredResponse,
};
use crate::{
    config::OracleConfig,
    data_connector::common::{
        parse_json_value, parse_metadata, parse_raw_response, parse_tool_calls,
    },
};
// ============================================================================
// PART 1: OracleStore Helper + Common Utilities
// ============================================================================

/// Shared Oracle connection pool infrastructure
///
/// This helper eliminates ~540 LOC of duplication across storage implementations.
/// It handles connection pooling, error mapping, and client configuration.
pub(crate) struct OracleStore {
    pool: Pool<OracleConnectionManager>,
}

impl OracleStore {
    /// Create pool with custom schema initialization
    ///
    /// The `init_schema` function receives a connection and should:
    /// - Check if tables/indexes exist
    /// - Create them if needed
    /// - Return Ok(()) on success or Err(message) on failure
    pub fn new(
        config: &OracleConfig,
        init_schema: impl FnOnce(&Connection) -> Result<(), String>,
    ) -> Result<Self, String> {
        // Configure Oracle client (wallet, etc.)
        configure_oracle_client(config)?;

        // Initialize schema using the provided function
        let conn = Connection::connect(
            &config.username,
            &config.password,
            &config.connect_descriptor,
        )
        .map_err(map_oracle_error)?;

        init_schema(&conn)?;
        drop(conn);

        // Create connection pool
        let config_arc = Arc::new(config.clone());
        let manager = OracleConnectionManager {
            params: Arc::new(OracleConnectParams::from_config(&config_arc)),
        };

        let mut builder = Pool::builder(manager)
            .max_size(config.pool_max)
            .runtime(deadpool::Runtime::Tokio1);

        if config.pool_timeout_secs > 0 {
            builder = builder.wait_timeout(Some(Duration::from_secs(config.pool_timeout_secs)));
        }

        let pool = builder
            .build()
            .map_err(|e| format!("Failed to build Oracle pool: {e}"))?;

        Ok(Self { pool })
    }

    /// Execute function with a connection from the pool
    pub async fn execute<F, T>(&self, func: F) -> Result<T, String>
    where
        F: FnOnce(&Connection) -> Result<T, String> + Send + 'static,
        T: Send + 'static,
    {
        let connection = self
            .pool
            .get()
            .await
            .map_err(|e| format!("Failed to get Oracle connection: {e}"))?;

        tokio::task::spawn_blocking(move || {
            let result = func(&connection);
            drop(connection);
            result
        })
        .await
        .map_err(|e| format!("Task execution failed: {e}"))?
    }
}

impl Clone for OracleStore {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

// Error mapping helper
pub(crate) fn map_oracle_error(err: oracle::Error) -> String {
    if let Some(db_err) = err.db_error() {
        format!(
            "Oracle error (code {}): {}",
            db_err.code(),
            db_err.message()
        )
    } else {
        err.to_string()
    }
}

// Client configuration helper
fn configure_oracle_client(config: &OracleConfig) -> Result<(), String> {
    if let Some(wallet_path) = &config.wallet_path {
        let path = Path::new(wallet_path);

        if !path.is_dir() {
            return Err(format!(
                "Oracle wallet path '{}' is not a directory",
                wallet_path
            ));
        }

        if !path.join("tnsnames.ora").exists() && !path.join("sqlnet.ora").exists() {
            return Err(format!(
                "Oracle wallet path '{}' is missing tnsnames.ora or sqlnet.ora",
                wallet_path
            ));
        }

        std::env::set_var("TNS_ADMIN", wallet_path);
    }
    Ok(())
}

// Connection parameters
#[derive(Clone)]
pub(crate) struct OracleConnectParams {
    pub username: String,
    pub password: String,
    pub connect_descriptor: String,
}

impl OracleConnectParams {
    pub fn from_config(config: &OracleConfig) -> Self {
        Self {
            username: config.username.clone(),
            password: config.password.clone(),
            connect_descriptor: config.connect_descriptor.clone(),
        }
    }
}

impl std::fmt::Debug for OracleConnectParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleConnectParams")
            .field("username", &self.username)
            .field("connect_descriptor", &self.connect_descriptor)
            .finish()
    }
}

// Connection manager (same for all stores)
#[derive(Clone)]
struct OracleConnectionManager {
    params: Arc<OracleConnectParams>,
}

impl std::fmt::Debug for OracleConnectionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleConnectionManager")
            .field("username", &self.params.username)
            .field("connect_descriptor", &self.params.connect_descriptor)
            .finish()
    }
}

#[async_trait]
impl Manager for OracleConnectionManager {
    type Type = Connection;
    type Error = oracle::Error;

    fn create(
        &self,
    ) -> impl std::future::Future<Output = Result<Connection, oracle::Error>> + Send {
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

// ============================================================================
// PART 2: OracleConversationStorage
// ============================================================================

#[derive(Clone)]
pub struct OracleConversationStorage {
    store: OracleStore,
}

impl OracleConversationStorage {
    pub fn new(config: OracleConfig) -> Result<Self, ConversationStorageError> {
        let store = OracleStore::new(&config, |conn| {
            // Check if table exists
            let exists: i64 = conn
                .query_row_as(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name = 'CONVERSATIONS'",
                    &[],
                )
                .map_err(map_oracle_error)?;

            // Create table if missing
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
        })
        .map_err(ConversationStorageError::StorageError)?;

        Ok(Self { store })
    }

    fn parse_metadata(
        raw: Option<String>,
    ) -> Result<Option<ConversationMetadata>, ConversationStorageError> {
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
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> Result<Conversation, ConversationStorageError> {
        let conversation = Conversation::new(input);
        let id_str = conversation.id.0.clone();
        let created_at = conversation.created_at;
        let metadata_json = conversation
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;

        self.store
            .execute(move |conn| {
                conn.execute(
                    "INSERT INTO conversations (id, created_at, metadata) VALUES (:1, :2, :3)",
                    &[&id_str, &created_at, &metadata_json],
                )
                .map(|_| ())
                .map_err(map_oracle_error)
            })
            .await
            .map_err(ConversationStorageError::StorageError)?;

        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let lookup = id.0.clone();
        self.store
            .execute(move |conn| {
                let mut stmt = conn
                    .statement("SELECT id, created_at, metadata FROM conversations WHERE id = :1")
                    .build()
                    .map_err(map_oracle_error)?;
                let mut rows = stmt.query(&[&lookup]).map_err(map_oracle_error)?;

                if let Some(row_res) = rows.next() {
                    let row = row_res.map_err(map_oracle_error)?;
                    let id: String = row.get(0).map_err(map_oracle_error)?;
                    let created_at: DateTime<Utc> = row.get(1).map_err(map_oracle_error)?;
                    let metadata_raw: Option<String> = row.get(2).map_err(map_oracle_error)?;
                    let metadata = Self::parse_metadata(metadata_raw).map_err(|e| e.to_string())?;
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
            .map_err(ConversationStorageError::StorageError)
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let id_str = id.0.clone();
        let metadata_json = metadata.as_ref().map(serde_json::to_string).transpose()?;
        let conversation_id = id.clone();

        self.store
            .execute(move |conn| {
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

                let mut created_at: Vec<DateTime<Utc>> =
                    stmt.returned_values(3).map_err(map_oracle_error)?;
                let created_at = created_at
                    .pop()
                    .ok_or_else(|| "Oracle update did not return created_at".to_string())?;

                Ok(Some(Conversation::with_parts(
                    conversation_id,
                    created_at,
                    metadata,
                )))
            })
            .await
            .map_err(ConversationStorageError::StorageError)
    }

    async fn delete_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<bool, ConversationStorageError> {
        let id_str = id.0.clone();
        let res = self
            .store
            .execute(move |conn| {
                conn.execute("DELETE FROM conversations WHERE id = :1", &[&id_str])
                    .map_err(map_oracle_error)
            })
            .await
            .map_err(ConversationStorageError::StorageError)?;

        Ok(res
            .row_count()
            .map_err(|e| ConversationStorageError::StorageError(map_oracle_error(e)))?
            > 0)
    }
}

// ============================================================================
// PART 3: OracleConversationItemStorage
// ============================================================================

#[derive(Clone)]
pub struct OracleConversationItemStorage {
    store: OracleStore,
}

impl OracleConversationItemStorage {
    pub fn new(config: OracleConfig) -> Result<Self, ConversationItemStorageError> {
        let store = OracleStore::new(&config, |conn| {
            // Create conversation_items table
            let exists_items: i64 = conn
                .query_row_as(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name = 'CONVERSATION_ITEMS'",
                    &[],
                )
                .map_err(map_oracle_error)?;

            if exists_items == 0 {
                conn.execute(
                    "CREATE TABLE conversation_items (
                        id VARCHAR2(64) PRIMARY KEY,
                        response_id VARCHAR2(64),
                        item_type VARCHAR2(32) NOT NULL,
                        role VARCHAR2(32),
                        content CLOB,
                        status VARCHAR2(32),
                        created_at TIMESTAMP WITH TIME ZONE
                    )",
                    &[],
                )
                .map_err(map_oracle_error)?;
            }

            // Create conversation_item_links table
            let exists_links: i64 = conn
                .query_row_as(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name = 'CONVERSATION_ITEM_LINKS'",
                    &[],
                )
                .map_err(map_oracle_error)?;

            if exists_links == 0 {
                conn.execute(
                    "CREATE TABLE conversation_item_links (
                        conversation_id VARCHAR2(64) NOT NULL,
                        item_id VARCHAR2(64) NOT NULL,
                        added_at TIMESTAMP WITH TIME ZONE,
                        CONSTRAINT pk_conv_item_link PRIMARY KEY (conversation_id, item_id)
                    )",
                    &[],
                )
                .map_err(map_oracle_error)?;

                conn.execute(
                    "CREATE INDEX conv_item_links_conv_idx ON conversation_item_links (conversation_id, added_at)",
                    &[],
                )
                .map_err(map_oracle_error)?;
            }

            Ok(())
        })
        .map_err(ConversationItemStorageError::StorageError)?;

        Ok(Self { store })
    }
}

#[async_trait]
impl ConversationItemStorage for OracleConversationItemStorage {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> Result<ConversationItem, ConversationItemStorageError> {
        let id = item
            .id
            .clone()
            .unwrap_or_else(|| make_item_id(&item.item_type));
        let created_at = Utc::now();
        let content_json = serde_json::to_string(&item.content)?;

        let conversation_item = ConversationItem {
            id: id.clone(),
            response_id: item.response_id.clone(),
            item_type: item.item_type.clone(),
            role: item.role.clone(),
            content: item.content,
            status: item.status.clone(),
            created_at,
        };

        let id_str = conversation_item.id.0.clone();
        let response_id = conversation_item.response_id.clone();
        let item_type = conversation_item.item_type.clone();
        let role = conversation_item.role.clone();
        let status = conversation_item.status.clone();

        self.store
            .execute(move |conn| {
                conn.execute(
                    "INSERT INTO conversation_items (id, response_id, item_type, role, content, status, created_at) \
                     VALUES (:1, :2, :3, :4, :5, :6, :7)",
                    &[&id_str, &response_id, &item_type, &role, &content_json, &status, &created_at],
                )
                .map_err(map_oracle_error)?;
                Ok(())
            })
            .await
            .map_err(ConversationItemStorageError::StorageError)?;

        Ok(conversation_item)
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> Result<(), ConversationItemStorageError> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();
        self.store
            .execute(move |conn| {
                conn.execute(
                    "INSERT INTO conversation_item_links (conversation_id, item_id, added_at) VALUES (:1, :2, :3)",
                    &[&cid, &iid, &added_at],
                )
                .map_err(map_oracle_error)?;
                Ok(())
            })
            .await
            .map_err(ConversationItemStorageError::StorageError)
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> Result<Vec<ConversationItem>, ConversationItemStorageError> {
        let cid = conversation_id.0.clone();
        let limit: i64 = params.limit as i64;
        let order_desc = matches!(params.order, SortOrder::Desc);
        let after_id = params.after.clone();

        // Resolve the added_at of the after cursor if provided
        let after_key: Option<(DateTime<Utc>, String)> = if let Some(ref aid) = after_id {
            self.store
                .execute({
                    let cid = cid.clone();
                    let aid = aid.clone();
                    move |conn| {
                        let mut stmt = conn
                            .statement(
                                "SELECT added_at FROM conversation_item_links WHERE conversation_id = :1 AND item_id = :2",
                            )
                            .build()
                            .map_err(map_oracle_error)?;
                        let mut rows = stmt.query(&[&cid, &aid]).map_err(map_oracle_error)?;
                        if let Some(row_res) = rows.next() {
                            let row = row_res.map_err(map_oracle_error)?;
                            let ts: DateTime<Utc> = row.get(0).map_err(map_oracle_error)?;
                            Ok(Some((ts, aid)))
                        } else {
                            Ok(None)
                        }
                    }
                })
                .await
                .map_err(ConversationItemStorageError::StorageError)?
        } else {
            None
        };

        // Build the main list query
        let rows: Vec<(String, Option<String>, String, Option<String>, Option<String>, Option<String>, DateTime<Utc>)> =
            self.store
                .execute({
                    let cid = cid.clone();
                    move |conn| {
                        let mut sql = String::from(
                            "SELECT i.id, i.response_id, i.item_type, i.role, i.content, i.status, i.created_at \
                             FROM conversation_item_links l \
                             JOIN conversation_items i ON i.id = l.item_id \
                             WHERE l.conversation_id = :cid",
                        );

                        // Cursor predicate
                        if let Some((_ts, _iid)) = &after_key {
                            if order_desc {
                                sql.push_str(" AND (l.added_at < :ats OR (l.added_at = :ats AND l.item_id < :iid))");
                            } else {
                                sql.push_str(" AND (l.added_at > :ats OR (l.added_at = :ats AND l.item_id > :iid))");
                            }
                        }

                        // Order and limit
                        if order_desc {
                            sql.push_str(" ORDER BY l.added_at DESC, l.item_id DESC");
                        } else {
                            sql.push_str(" ORDER BY l.added_at ASC, l.item_id ASC");
                        }
                        sql.push_str(" FETCH NEXT :limit ROWS ONLY");

                        // Build params and perform a named SELECT query
                        let mut params_vec: Vec<(&str, &dyn ToSql)> = vec![("cid", &cid)];
                        if let Some((ts, iid)) = &after_key {
                            params_vec.push(("ats", ts));
                            params_vec.push(("iid", iid));
                        }
                        params_vec.push(("limit", &limit));

                        let rows_iter = conn.query_named(&sql, &params_vec).map_err(map_oracle_error)?;

                        let mut out = Vec::new();
                        for row_res in rows_iter {
                            let row = row_res.map_err(map_oracle_error)?;
                            let id: String = row.get(0).map_err(map_oracle_error)?;
                            let resp_id: Option<String> = row.get(1).map_err(map_oracle_error)?;
                            let item_type: String = row.get(2).map_err(map_oracle_error)?;
                            let role: Option<String> = row.get(3).map_err(map_oracle_error)?;
                            let content_raw: Option<String> = row.get(4).map_err(map_oracle_error)?;
                            let status: Option<String> = row.get(5).map_err(map_oracle_error)?;
                            let created_at: DateTime<Utc> = row.get(6).map_err(map_oracle_error)?;
                            out.push((id, resp_id, item_type, role, content_raw, status, created_at));
                        }
                        Ok(out)
                    }
                })
                .await
                .map_err(ConversationItemStorageError::StorageError)?;

        // Map rows to ConversationItem
        rows.into_iter()
            .map(
                |(id, resp_id, item_type, role, content_raw, status, created_at)| {
                    let content = match content_raw {
                        Some(s) => {
                            serde_json::from_str(&s).map_err(ConversationItemStorageError::from)?
                        }
                        None => Value::Null,
                    };
                    Ok(ConversationItem {
                        id: ConversationItemId(id),
                        response_id: resp_id,
                        item_type,
                        role,
                        content,
                        status,
                        created_at,
                    })
                },
            )
            .collect()
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> Result<Option<ConversationItem>, ConversationItemStorageError> {
        let iid = item_id.0.clone();

        self.store
            .execute(move |conn| {
                let mut stmt = conn
                    .statement(
                        "SELECT id, response_id, item_type, role, content, status, created_at \
                         FROM conversation_items WHERE id = :1",
                    )
                    .build()
                    .map_err(map_oracle_error)?;

                let mut rows = stmt.query(&[&iid]).map_err(map_oracle_error)?;

                if let Some(row_res) = rows.next() {
                    let row = row_res.map_err(map_oracle_error)?;
                    let id: String = row.get(0).map_err(map_oracle_error)?;
                    let response_id: Option<String> = row.get(1).map_err(map_oracle_error)?;
                    let item_type: String = row.get(2).map_err(map_oracle_error)?;
                    let role: Option<String> = row.get(3).map_err(map_oracle_error)?;
                    let content_raw: Option<String> = row.get(4).map_err(map_oracle_error)?;
                    let status: Option<String> = row.get(5).map_err(map_oracle_error)?;
                    let created_at: DateTime<Utc> = row.get(6).map_err(map_oracle_error)?;

                    let content = match content_raw {
                        Some(s) => serde_json::from_str(&s).map_err(|e| e.to_string())?,
                        None => Value::Null,
                    };

                    Ok(Some(ConversationItem {
                        id: ConversationItemId(id),
                        response_id,
                        item_type,
                        role,
                        content,
                        status,
                        created_at,
                    }))
                } else {
                    Ok(None)
                }
            })
            .await
            .map_err(ConversationItemStorageError::StorageError)
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> Result<bool, ConversationItemStorageError> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();

        self.store
            .execute(move |conn| {
                let count: i64 = conn
                    .query_row_as(
                        "SELECT COUNT(*) FROM conversation_item_links WHERE conversation_id = :1 AND item_id = :2",
                        &[&cid, &iid],
                    )
                    .map_err(map_oracle_error)?;
                Ok(count > 0)
            })
            .await
            .map_err(ConversationItemStorageError::StorageError)
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> Result<(), ConversationItemStorageError> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();

        self.store
            .execute(move |conn| {
                conn.execute(
                    "DELETE FROM conversation_item_links WHERE conversation_id = :1 AND item_id = :2",
                    &[&cid, &iid],
                )
                .map_err(map_oracle_error)?;
                Ok(())
            })
            .await
            .map_err(ConversationItemStorageError::StorageError)
    }
}

// ============================================================================
// PART 4: OracleResponseStorage
// ============================================================================

const SELECT_BASE: &str = "SELECT id, previous_response_id, input, instructions, output, \
    tool_calls, metadata, created_at, safety_identifier, model, conversation_id, raw_response FROM responses";

#[derive(Clone)]
pub struct OracleResponseStorage {
    store: OracleStore,
}

impl OracleResponseStorage {
    pub fn new(config: OracleConfig) -> Result<Self, ResponseStorageError> {
        let store = OracleStore::new(&config, |conn| {
            // Create responses table
            let exists: i64 = conn
                .query_row_as(
                    "SELECT COUNT(*) FROM user_tables WHERE table_name = 'RESPONSES'",
                    &[],
                )
                .map_err(map_oracle_error)?;

            if exists == 0 {
                conn.execute(
                    "CREATE TABLE responses (
                        id VARCHAR2(64) PRIMARY KEY,
                        conversation_id VARCHAR2(64),
                        previous_response_id VARCHAR2(64),
                        input CLOB,
                        instructions CLOB,
                        output CLOB,
                        tool_calls CLOB,
                        metadata CLOB,
                        created_at TIMESTAMP WITH TIME ZONE,
                        safety_identifier VARCHAR2(128),
                        model VARCHAR2(128),
                        raw_response CLOB
                    )",
                    &[],
                )
                .map_err(map_oracle_error)?;
            } else {
                Self::alter_safety_identifier_column(conn)?;
                Self::remove_user_id_column_if_exists(conn)?;
            }

            // Create indexes
            create_index_if_missing(
                conn,
                "RESPONSES_PREV_IDX",
                "CREATE INDEX responses_prev_idx ON responses(previous_response_id)",
            )?;
            create_index_if_missing(
                conn,
                "RESPONSES_USER_IDX",
                "CREATE INDEX responses_user_idx ON responses(safety_identifier)",
            )?;

            Ok(())
        })
        .map_err(ResponseStorageError::StorageError)?;

        Ok(Self { store })
    }

    // Alter safety_identifier column if missing
    fn alter_safety_identifier_column(conn: &Connection) -> Result<(), String> {
        let present: i64 = conn
            .query_row_as(
                "SELECT COUNT(*) FROM user_tab_columns WHERE table_name = 'RESPONSES' AND column_name = 'SAFETY_IDENTIFIER'",
                &[],
            )
            .map_err(map_oracle_error)?;

        if present == 0 {
            if let Err(err) = conn.execute(
                "ALTER TABLE responses ADD (safety_identifier VARCHAR2(128))",
                &[],
            ) {
                let present_after: i64 = conn
                    .query_row_as(
                        "SELECT COUNT(*) FROM user_tab_columns WHERE table_name = 'RESPONSES' AND column_name = 'SAFETY_IDENTIFIER'",
                        &[],
                    )
                    .map_err(map_oracle_error)?;
                if present_after == 0 {
                    return Err(map_oracle_error(err));
                }
            }
        }

        Ok(())
    }

    // Remove user_id column if exists
    fn remove_user_id_column_if_exists(conn: &Connection) -> Result<(), String> {
        let present: i64 = conn
            .query_row_as(
                "SELECT COUNT(*) FROM user_tab_columns WHERE table_name = 'RESPONSES' AND column_name = 'USER_ID'",
                &[],
            )
            .map_err(map_oracle_error)?;

        if present > 0 {
            if let Err(err) = conn.execute("ALTER TABLE responses DROP COLUMN USER_ID", &[]) {
                let present_after: i64 = conn
                    .query_row_as(
                        "SELECT COUNT(*) FROM user_tab_columns WHERE table_name = 'RESPONSES' AND column_name = 'USER_ID'",
                        &[],
                    )
                    .map_err(map_oracle_error)?;
                if present_after > 0 {
                    return Err(map_oracle_error(err));
                }
            }
        }

        Ok(())
    }

    fn build_response_from_row(row: &Row) -> Result<StoredResponse, String> {
        let id: String = row.get(0).map_err(map_oracle_error)?;
        let previous: Option<String> = row.get(1).map_err(map_oracle_error)?;
        let input_json: Option<String> = row.get(2).map_err(map_oracle_error)?;
        let instructions: Option<String> = row.get(3).map_err(map_oracle_error)?;
        let output_json: Option<String> = row.get(4).map_err(map_oracle_error)?;
        let tool_calls_json: Option<String> = row.get(5).map_err(map_oracle_error)?;
        let metadata_json: Option<String> = row.get(6).map_err(map_oracle_error)?;
        let created_at: DateTime<Utc> = row.get(7).map_err(map_oracle_error)?;
        let safety_identifier: Option<String> = row.get(8).map_err(map_oracle_error)?;
        let model: Option<String> = row.get(9).map_err(map_oracle_error)?;
        let conversation_id: Option<String> = row.get(10).map_err(map_oracle_error)?;
        let raw_response_json: Option<String> = row.get(11).map_err(map_oracle_error)?;

        let previous_response_id = previous.map(ResponseId);
        let tool_calls = parse_tool_calls(tool_calls_json)?;
        let metadata = parse_metadata(metadata_json)?;
        let raw_response = parse_raw_response(raw_response_json)?;
        let input = parse_json_value(input_json)?;
        let output = parse_json_value(output_json)?;

        Ok(StoredResponse {
            id: ResponseId(id),
            previous_response_id,
            input,
            instructions,
            output,
            tool_calls,
            metadata,
            created_at,
            safety_identifier,
            model,
            conversation_id,
            raw_response,
        })
    }
}

#[async_trait]
impl ResponseStorage for OracleResponseStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        let response_id = response.id.clone();
        let response_id_str = response_id.0.clone();
        let previous_id = response.previous_response_id.map(|r| r.0);
        let json_input = serde_json::to_string(&response.input)?;
        let json_output = serde_json::to_string(&response.output)?;
        let json_tool_calls = serde_json::to_string(&response.tool_calls)?;
        let json_metadata = serde_json::to_string(&response.metadata)?;
        let json_raw_response = serde_json::to_string(&response.raw_response)?;
        let instructions = response.instructions.clone();
        let created_at = response.created_at;
        let safety_identifier = response.safety_identifier.clone();
        let model = response.model.clone();
        let conversation_id = response.conversation_id.clone();

        self.store
            .execute(move |conn| {
                conn.execute(
                    "INSERT INTO responses (id, previous_response_id, input, instructions, output, \
                        tool_calls, metadata, created_at, safety_identifier, model, conversation_id, raw_response) \
                     VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12)",
                    &[
                        &response_id_str,
                        &previous_id,
                        &json_input,
                        &instructions,
                        &json_output,
                        &json_tool_calls,
                        &json_metadata,
                        &created_at,
                        &safety_identifier,
                        &model,
                        &conversation_id,
                        &json_raw_response,
                    ],
                )
                .map(|_| ())
                .map_err(map_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::StorageError)?;

        Ok(response_id)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        let id = response_id.0.clone();
        self.store
            .execute(move |conn| {
                let mut stmt = conn
                    .statement(&format!("{} WHERE id = :1", SELECT_BASE))
                    .build()
                    .map_err(map_oracle_error)?;
                let mut rows = stmt.query(&[&id]).map_err(map_oracle_error)?;
                match rows.next() {
                    Some(row) => {
                        let row = row.map_err(map_oracle_error)?;
                        Self::build_response_from_row(&row).map(Some)
                    }
                    None => Ok(None),
                }
            })
            .await
            .map_err(ResponseStorageError::StorageError)
    }

    async fn delete_response(&self, response_id: &ResponseId) -> Result<(), ResponseStorageError> {
        let id = response_id.0.clone();
        self.store
            .execute(move |conn| {
                conn.execute("DELETE FROM responses WHERE id = :1", &[&id])
                    .map(|_| ())
                    .map_err(map_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::StorageError)
    }

    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> Result<ResponseChain, ResponseStorageError> {
        let mut chain = ResponseChain::new();
        let mut current_id = Some(response_id.clone());
        let mut visited = 0usize;

        while let Some(ref lookup_id) = current_id {
            if let Some(limit) = max_depth {
                if visited >= limit {
                    break;
                }
            }

            let fetched = self.get_response(lookup_id).await?;
            match fetched {
                Some(response) => {
                    current_id = response.previous_response_id.clone();
                    chain.responses.push(response);
                    visited += 1;
                }
                None => break,
            }
        }

        chain.responses.reverse();
        Ok(chain)
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>, ResponseStorageError> {
        let identifier = identifier.to_string();

        self.store
            .execute(move |conn| {
                let sql = if let Some(limit) = limit {
                    format!(
                        "SELECT * FROM ({} WHERE safety_identifier = :1 ORDER BY created_at DESC) WHERE ROWNUM <= {}",
                        SELECT_BASE, limit
                    )
                } else {
                    format!("{} WHERE safety_identifier = :1 ORDER BY created_at DESC", SELECT_BASE)
                };

                let mut stmt = conn.statement(&sql).build().map_err(map_oracle_error)?;
                let mut rows = stmt.query(&[&identifier]).map_err(map_oracle_error)?;
                let mut results = Vec::new();

                for row in &mut rows {
                    let row = row.map_err(map_oracle_error)?;
                    results.push(Self::build_response_from_row(&row)?);
                }

                Ok(results)
            })
            .await
            .map_err(ResponseStorageError::StorageError)
    }

    async fn delete_identifier_responses(
        &self,
        identifier: &str,
    ) -> Result<usize, ResponseStorageError> {
        let identifier = identifier.to_string();
        let affected = self
            .store
            .execute(move |conn| {
                conn.execute(
                    "DELETE FROM responses WHERE safety_identifier = :1",
                    &[&identifier],
                )
                .map_err(map_oracle_error)
            })
            .await
            .map_err(ResponseStorageError::StorageError)?;

        let deleted = affected
            .row_count()
            .map_err(|e| ResponseStorageError::StorageError(map_oracle_error(e)))?
            as usize;
        Ok(deleted)
    }
}

// Helper functions for response parsing

fn create_index_if_missing(conn: &Connection, index_name: &str, ddl: &str) -> Result<(), String> {
    let count: i64 = conn
        .query_row_as(
            "SELECT COUNT(*) FROM user_indexes WHERE table_name = 'RESPONSES' AND index_name = :1",
            &[&index_name],
        )
        .map_err(map_oracle_error)?;

    if count == 0 {
        if let Err(err) = conn.execute(ddl, &[]) {
            if let Some(db_err) = err.db_error() {
                // ORA-00955: name is already used by an existing object
                // ORA-01408: such column list already indexed
                // Both errors indicate the index already exists (race condition)
                if db_err.code() != 955 && db_err.code() != 1408 {
                    return Err(map_oracle_error(err));
                }
            } else {
                return Err(map_oracle_error(err));
            }
        }
    }

    Ok(())
}
