use std::{collections::HashMap, path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use deadpool::managed::{Manager, Metrics, Pool, PoolError, RecycleError, RecycleResult};
use oracle::{Connection, Row};
use serde_json::Value;

use crate::{
    config::OracleConfig,
    data_connector::responses::{
        ResponseChain, ResponseId, ResponseStorage, ResponseStorageError, Result as StorageResult,
        StoredResponse,
    },
};

const SELECT_BASE: &str = "SELECT id, previous_response_id, input, instructions, output, \
    tool_calls, metadata, created_at, user_id, model, conversation_id, raw_response FROM responses";

#[derive(Clone)]
pub struct OracleResponseStorage {
    pool: Pool<OracleConnectionManager>,
}

impl OracleResponseStorage {
    pub fn new(config: OracleConfig) -> StorageResult<Self> {
        let config = Arc::new(config);
        configure_oracle_client(&config)?;
        initialize_schema(&config)?;

        let manager = OracleConnectionManager::new(config.clone());
        let mut builder = Pool::builder(manager)
            .max_size(config.pool_max)
            .runtime(deadpool::Runtime::Tokio1);

        if config.pool_timeout_secs > 0 {
            builder = builder.wait_timeout(Some(Duration::from_secs(config.pool_timeout_secs)));
        }

        let pool = builder.build().map_err(|err| {
            ResponseStorageError::StorageError(format!(
                "failed to build Oracle connection pool: {err}"
            ))
        })?;

        Ok(Self { pool })
    }

    async fn with_connection<F, T>(&self, func: F) -> StorageResult<T>
    where
        F: FnOnce(&Connection) -> StorageResult<T> + Send + 'static,
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
            ResponseStorageError::StorageError(format!(
                "failed to execute Oracle query task: {err}"
            ))
        })?
    }

    fn build_response_from_row(row: &Row) -> StorageResult<StoredResponse> {
        let id: String = row
            .get(0)
            .map_err(|err| map_oracle_error(err).into_storage_error("fetch id"))?;
        let previous: Option<String> = row.get(1).map_err(|err| {
            map_oracle_error(err).into_storage_error("fetch previous_response_id")
        })?;
        let input: String = row
            .get(2)
            .map_err(|err| map_oracle_error(err).into_storage_error("fetch input"))?;
        let instructions: Option<String> = row
            .get(3)
            .map_err(|err| map_oracle_error(err).into_storage_error("fetch instructions"))?;
        let output: String = row
            .get(4)
            .map_err(|err| map_oracle_error(err).into_storage_error("fetch output"))?;
        let tool_calls_json: Option<String> = row
            .get(5)
            .map_err(|err| map_oracle_error(err).into_storage_error("fetch tool_calls"))?;
        let metadata_json: Option<String> = row
            .get(6)
            .map_err(|err| map_oracle_error(err).into_storage_error("fetch metadata"))?;
        let created_at: chrono::DateTime<chrono::Utc> = row
            .get(7)
            .map_err(|err| map_oracle_error(err).into_storage_error("fetch created_at"))?;
        let user_id: Option<String> = row
            .get(8)
            .map_err(|err| map_oracle_error(err).into_storage_error("fetch user_id"))?;
        let model: Option<String> = row
            .get(9)
            .map_err(|err| map_oracle_error(err).into_storage_error("fetch model"))?;
        let conversation_id: Option<String> = row
            .get(10)
            .map_err(|err| map_oracle_error(err).into_storage_error("fetch conversation_id"))?;
        let raw_response_json: Option<String> = row
            .get(11)
            .map_err(|err| map_oracle_error(err).into_storage_error("fetch raw_response"))?;

        let previous_response_id = previous.map(ResponseId);
        let tool_calls = parse_tool_calls(tool_calls_json)?;
        let metadata = parse_metadata(metadata_json)?;
        let raw_response = parse_raw_response(raw_response_json)?;

        Ok(StoredResponse {
            id: ResponseId(id),
            previous_response_id,
            input,
            instructions,
            output,
            tool_calls,
            metadata,
            created_at,
            user: user_id,
            model,
            conversation_id,
            raw_response,
        })
    }
}

#[async_trait]
impl ResponseStorage for OracleResponseStorage {
    async fn store_response(&self, response: StoredResponse) -> StorageResult<ResponseId> {
        let StoredResponse {
            id,
            previous_response_id,
            input,
            instructions,
            output,
            tool_calls,
            metadata,
            created_at,
            user,
            model,
            conversation_id,
            raw_response,
        } = response;

        let response_id = id.clone();
        let response_id_str = response_id.0.clone();
        let previous_id = previous_response_id.map(|r| r.0);
        let json_tool_calls = serde_json::to_string(&tool_calls)?;
        let json_metadata = serde_json::to_string(&metadata)?;
        let json_raw_response = serde_json::to_string(&raw_response)?;

        self.with_connection(move |conn| {
            conn.execute(
                "INSERT INTO responses (id, previous_response_id, input, instructions, output, \
                    tool_calls, metadata, created_at, user_id, model, conversation_id, raw_response) \
                 VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12)",
                &[
                    &response_id_str,
                    &previous_id,
                    &input,
                    &instructions,
                    &output,
                    &json_tool_calls,
                    &json_metadata,
                    &created_at,
                    &user,
                    &model,
                    &conversation_id,
                    &json_raw_response,
                ],
            )
            .map(|_| ())
            .map_err(map_oracle_error)
        })
        .await?;

        Ok(response_id)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> StorageResult<Option<StoredResponse>> {
        let id = response_id.0.clone();
        self.with_connection(move |conn| {
            let mut stmt = conn
                .statement(&format!("{} WHERE id = :1", SELECT_BASE))
                .build()
                .map_err(map_oracle_error)?;
            let mut rows = stmt.query(&[&id]).map_err(map_oracle_error)?;
            match rows.next() {
                Some(row) => {
                    let row = row.map_err(map_oracle_error)?;
                    OracleResponseStorage::build_response_from_row(&row).map(Some)
                }
                None => Ok(None),
            }
        })
        .await
    }

    async fn delete_response(&self, response_id: &ResponseId) -> StorageResult<()> {
        let id = response_id.0.clone();
        self.with_connection(move |conn| {
            conn.execute("DELETE FROM responses WHERE id = :1", &[&id])
                .map(|_| ())
                .map_err(map_oracle_error)
        })
        .await
    }

    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> StorageResult<ResponseChain> {
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

    async fn list_user_responses(
        &self,
        user: &str,
        limit: Option<usize>,
    ) -> StorageResult<Vec<StoredResponse>> {
        let user = user.to_string();

        self.with_connection(move |conn| {
            let sql = if let Some(limit) = limit {
                format!(
                    "SELECT * FROM ({} WHERE user_id = :1 ORDER BY created_at DESC) WHERE ROWNUM <= {}",
                    SELECT_BASE, limit
                )
            } else {
                format!("{} WHERE user_id = :1 ORDER BY created_at DESC", SELECT_BASE)
            };

            let mut stmt = conn.statement(&sql).build().map_err(map_oracle_error)?;
            let mut rows = stmt.query(&[&user]).map_err(map_oracle_error)?;
            let mut results = Vec::new();

            for row in &mut rows {
                let row = row.map_err(map_oracle_error)?;
                results.push(OracleResponseStorage::build_response_from_row(&row)?);
            }

            Ok(results)
        })
        .await
    }

    async fn delete_user_responses(&self, user: &str) -> StorageResult<usize> {
        let user = user.to_string();
        let affected = self
            .with_connection(move |conn| {
                conn.execute("DELETE FROM responses WHERE user_id = :1", &[&user])
                    .map_err(map_oracle_error)
            })
            .await?;

        let deleted = affected.row_count().map_err(map_oracle_error)? as usize;
        Ok(deleted)
    }
}

#[derive(Clone)]
struct OracleConnectionManager {
    params: Arc<OracleConnectParams>,
}

impl OracleConnectionManager {
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

impl std::fmt::Debug for OracleConnectionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleConnectionManager")
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

fn configure_oracle_client(config: &OracleConfig) -> StorageResult<()> {
    if let Some(wallet_path) = &config.wallet_path {
        let wallet_path = Path::new(wallet_path);
        if !wallet_path.is_dir() {
            return Err(ResponseStorageError::StorageError(format!(
                "Oracle wallet/config path '{}' is not a directory",
                wallet_path.display()
            )));
        }

        if !wallet_path.join("tnsnames.ora").exists() && !wallet_path.join("sqlnet.ora").exists() {
            return Err(ResponseStorageError::StorageError(format!(
                "Oracle wallet/config path '{}' is missing tnsnames.ora or sqlnet.ora",
                wallet_path.display()
            )));
        }

        std::env::set_var("TNS_ADMIN", wallet_path);
    }
    Ok(())
}

fn initialize_schema(config: &OracleConfig) -> StorageResult<()> {
    let conn = Connection::connect(
        &config.username,
        &config.password,
        &config.connect_descriptor,
    )
    .map_err(map_oracle_error)?;

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
                user_id VARCHAR2(128),
                model VARCHAR2(128),
                raw_response CLOB
            )",
            &[],
        )
        .map_err(map_oracle_error)?;
    }

    create_index_if_missing(
        &conn,
        "RESPONSES_PREV_IDX",
        "CREATE INDEX responses_prev_idx ON responses(previous_response_id)",
    )?;
    create_index_if_missing(
        &conn,
        "RESPONSES_USER_IDX",
        "CREATE INDEX responses_user_idx ON responses(user_id)",
    )?;

    Ok(())
}

fn create_index_if_missing(conn: &Connection, index_name: &str, ddl: &str) -> StorageResult<()> {
    let count: i64 = conn
        .query_row_as(
            "SELECT COUNT(*) FROM user_indexes WHERE table_name = 'RESPONSES' AND index_name = :1",
            &[&index_name],
        )
        .map_err(map_oracle_error)?;

    if count == 0 {
        if let Err(err) = conn.execute(ddl, &[]) {
            if err.db_error().map(|db| db.code()) != Some(1408) {
                return Err(map_oracle_error(err));
            }
        }
    }

    Ok(())
}

fn parse_tool_calls(raw: Option<String>) -> StorageResult<Vec<Value>> {
    match raw {
        Some(s) if !s.is_empty() => {
            serde_json::from_str(&s).map_err(ResponseStorageError::SerializationError)
        }
        _ => Ok(Vec::new()),
    }
}

fn parse_metadata(raw: Option<String>) -> StorageResult<HashMap<String, Value>> {
    match raw {
        Some(s) if !s.is_empty() => {
            serde_json::from_str(&s).map_err(ResponseStorageError::SerializationError)
        }
        _ => Ok(HashMap::new()),
    }
}

fn parse_raw_response(raw: Option<String>) -> StorageResult<Value> {
    match raw {
        Some(s) if !s.is_empty() => {
            serde_json::from_str(&s).map_err(ResponseStorageError::SerializationError)
        }
        _ => Ok(Value::Null),
    }
}

fn map_pool_error(err: PoolError<oracle::Error>) -> ResponseStorageError {
    match err {
        PoolError::Backend(e) => map_oracle_error(e),
        other => ResponseStorageError::StorageError(format!(
            "failed to obtain Oracle connection: {other}"
        )),
    }
}

fn map_oracle_error(err: oracle::Error) -> ResponseStorageError {
    if let Some(db_err) = err.db_error() {
        ResponseStorageError::StorageError(format!(
            "Oracle error (code {}): {}",
            db_err.code(),
            db_err.message()
        ))
    } else {
        ResponseStorageError::StorageError(err.to_string())
    }
}

trait OracleErrorExt {
    fn into_storage_error(self, context: &str) -> ResponseStorageError;
}

impl OracleErrorExt for ResponseStorageError {
    fn into_storage_error(self, context: &str) -> ResponseStorageError {
        ResponseStorageError::StorageError(format!("{context}: {self}"))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parse_tool_calls_handles_empty_input() {
        assert!(parse_tool_calls(None).unwrap().is_empty());
        assert!(parse_tool_calls(Some(String::new())).unwrap().is_empty());
    }

    #[test]
    fn parse_tool_calls_round_trips() {
        let payload = json!([{ "type": "test", "value": 1 }]).to_string();
        let parsed = parse_tool_calls(Some(payload)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["type"], "test");
        assert_eq!(parsed[0]["value"], 1);
    }

    #[test]
    fn parse_metadata_defaults_to_empty_map() {
        assert!(parse_metadata(None).unwrap().is_empty());
    }

    #[test]
    fn parse_metadata_round_trips() {
        let payload = json!({"key": "value", "nested": {"bool": true}}).to_string();
        let parsed = parse_metadata(Some(payload)).unwrap();
        assert_eq!(parsed.get("key").unwrap(), "value");
        assert_eq!(parsed["nested"]["bool"], true);
    }

    #[test]
    fn parse_raw_response_handles_null() {
        assert_eq!(parse_raw_response(None).unwrap(), Value::Null);
    }

    #[test]
    fn parse_raw_response_round_trips() {
        let payload = json!({"id": "abc"}).to_string();
        let parsed = parse_raw_response(Some(payload)).unwrap();
        assert_eq!(parsed["id"], "abc");
    }
}
