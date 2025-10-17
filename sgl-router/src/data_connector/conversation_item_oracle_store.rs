use std::{path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool::managed::{Manager, Metrics, Pool, PoolError, RecycleError, RecycleResult};
use oracle::{sql_type::ToSql, Connection};
use serde_json::Value;

use crate::{
    config::OracleConfig,
    data_connector::{
        conversation_items::{
            make_item_id, ConversationItem, ConversationItemId, ConversationItemStorage,
            ConversationItemStorageError, ListParams, Result as ItemResult, SortOrder,
        },
        conversations::ConversationId,
    },
};

#[derive(Clone)]
pub struct OracleConversationItemStorage {
    pool: Pool<ConversationItemOracleConnectionManager>,
}

impl OracleConversationItemStorage {
    pub fn new(config: OracleConfig) -> ItemResult<Self> {
        configure_oracle_client(&config)?;
        initialize_schema(&config)?;

        let config = Arc::new(config);
        let manager = ConversationItemOracleConnectionManager::new(config.clone());
        let mut builder = Pool::builder(manager)
            .max_size(config.pool_max)
            .runtime(deadpool::Runtime::Tokio1);
        if config.pool_timeout_secs > 0 {
            builder = builder.wait_timeout(Some(Duration::from_secs(config.pool_timeout_secs)));
        }
        let pool = builder.build().map_err(|err| {
            ConversationItemStorageError::StorageError(format!(
                "failed to build Oracle pool for conversation items: {err}"
            ))
        })?;
        Ok(Self { pool })
    }

    async fn with_connection<F, T>(&self, func: F) -> ItemResult<T>
    where
        F: FnOnce(&Connection) -> ItemResult<T> + Send + 'static,
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
            ConversationItemStorageError::StorageError(format!(
                "failed to execute Oracle conversation item task: {err}"
            ))
        })?
    }

    // reserved for future use when parsing JSON columns directly into Value
    // fn parse_json(raw: Option<String>) -> ItemResult<Value> { ... }
}

#[async_trait]
impl ConversationItemStorage for OracleConversationItemStorage {
    async fn create_item(
        &self,
        item: crate::data_connector::conversation_items::NewConversationItem,
    ) -> ItemResult<ConversationItem> {
        let id = item
            .id
            .clone()
            .unwrap_or_else(|| make_item_id(&item.item_type));
        let created_at = Utc::now();
        let content_json = serde_json::to_string(&item.content)?;

        // Build the return value up-front; move inexpensive clones as needed for SQL
        let conversation_item = ConversationItem {
            id: id.clone(),
            response_id: item.response_id.clone(),
            item_type: item.item_type.clone(),
            role: item.role.clone(),
            content: item.content,
            status: item.status.clone(),
            created_at,
        };

        // Prepare values for SQL insertion
        let id_str = conversation_item.id.0.clone();
        let response_id = conversation_item.response_id.clone();
        let item_type = conversation_item.item_type.clone();
        let role = conversation_item.role.clone();
        let status = conversation_item.status.clone();

        self.with_connection(move |conn| {
            conn.execute(
                "INSERT INTO conversation_items (id, response_id, item_type, role, content, status, created_at) \
                 VALUES (:1, :2, :3, :4, :5, :6, :7)",
                &[&id_str, &response_id, &item_type, &role, &content_json, &status, &created_at],
            )
            .map_err(map_oracle_error)?;
            Ok(())
        })
        .await?;

        Ok(conversation_item)
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ItemResult<()> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();
        self.with_connection(move |conn| {
            conn.execute(
                "INSERT INTO conversation_item_links (conversation_id, item_id, added_at) VALUES (:1, :2, :3)",
                &[&cid, &iid, &added_at],
            )
            .map_err(map_oracle_error)?;
            Ok(())
        })
        .await
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ItemResult<Vec<ConversationItem>> {
        let cid = conversation_id.0.clone();
        let limit: i64 = params.limit as i64;
        let order_desc = matches!(params.order, SortOrder::Desc);
        let after_id = params.after.clone();

        // Resolve the added_at of the after cursor if provided
        let after_key: Option<(DateTime<Utc>, String)> = if let Some(ref aid) = after_id {
            self.with_connection({
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
            .await?
        } else {
            None
        };

        // Build the main list query
        let rows: Vec<(String, Option<String>, String, Option<String>, Option<String>, Option<String>, DateTime<Utc>)> =
            self.with_connection({
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
            .await?;

        // Map rows to ConversationItem; propagate JSON parse errors instead of swallowing
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

    async fn get_item(&self, item_id: &ConversationItemId) -> ItemResult<Option<ConversationItem>> {
        let iid = item_id.0.clone();

        self.with_connection(move |conn| {
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
                    Some(s) => serde_json::from_str(&s)?,
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
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ItemResult<bool> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();

        self.with_connection(move |conn| {
            let count: i64 = conn
                .query_row_as(
                    "SELECT COUNT(*) FROM conversation_item_links WHERE conversation_id = :1 AND item_id = :2",
                    &[&cid, &iid],
                )
                .map_err(map_oracle_error)?;
            Ok(count > 0)
        })
        .await
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ItemResult<()> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();

        self.with_connection(move |conn| {
            // Delete ONLY the link (do not delete the item itself)
            conn.execute(
                "DELETE FROM conversation_item_links WHERE conversation_id = :1 AND item_id = :2",
                &[&cid, &iid],
            )
            .map_err(map_oracle_error)?;

            Ok(())
        })
        .await
    }
}

#[derive(Clone)]
struct ConversationItemOracleConnectionManager {
    params: Arc<OracleConnectParams>,
}

#[derive(Clone)]
struct OracleConnectParams {
    username: String,
    password: String,
    connect_descriptor: String,
}

impl ConversationItemOracleConnectionManager {
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

impl std::fmt::Debug for ConversationItemOracleConnectionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationItemOracleConnectionManager")
            .field("username", &self.params.username)
            .field("connect_descriptor", &self.params.connect_descriptor)
            .finish()
    }
}

#[async_trait]
impl Manager for ConversationItemOracleConnectionManager {
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

fn configure_oracle_client(config: &OracleConfig) -> ItemResult<()> {
    if let Some(wallet_path) = &config.wallet_path {
        let wallet_path = Path::new(wallet_path);
        if !wallet_path.is_dir() {
            return Err(ConversationItemStorageError::StorageError(format!(
                "Oracle wallet/config path '{}' is not a directory",
                wallet_path.display()
            )));
        }
        if !wallet_path.join("tnsnames.ora").exists() && !wallet_path.join("sqlnet.ora").exists() {
            return Err(ConversationItemStorageError::StorageError(format!(
                "Oracle wallet/config path '{}' is missing tnsnames.ora or sqlnet.ora",
                wallet_path.display()
            )));
        }
        std::env::set_var("TNS_ADMIN", wallet_path);
    }
    Ok(())
}

fn initialize_schema(config: &OracleConfig) -> ItemResult<()> {
    let conn = Connection::connect(
        &config.username,
        &config.password,
        &config.connect_descriptor,
    )
    .map_err(map_oracle_error)?;

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
}

fn map_pool_error(err: PoolError<oracle::Error>) -> ConversationItemStorageError {
    match err {
        PoolError::Backend(e) => map_oracle_error(e),
        other => ConversationItemStorageError::StorageError(format!(
            "failed to obtain Oracle conversation item connection: {other}"
        )),
    }
}

fn map_oracle_error(err: oracle::Error) -> ConversationItemStorageError {
    if let Some(db_err) = err.db_error() {
        ConversationItemStorageError::StorageError(format!(
            "Oracle error (code {}): {}",
            db_err.code(),
            db_err.message()
        ))
    } else {
        ConversationItemStorageError::StorageError(err.to_string())
    }
}
