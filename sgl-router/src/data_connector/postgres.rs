//! Postgres storage implementation using PostgresStore helper
//!
//! Structure:
//! 1. PostgresStore helper and common utilities
//! 2. PostgresConversationStorage
//! 3. PostgresConversationItemStorage
//! 4. PostgresResponseStorage

use std::str::FromStr;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use serde_json::Value;
use tokio_postgres::{NoTls, Row};

use crate::{
    config::PostgresConfig,
    data_connector::{
        common::{parse_json_value, parse_metadata, parse_raw_response, parse_tool_calls},
        core::{
            make_item_id, ConversationItemResult, ConversationItemStorageError,
            ConversationMetadata, ConversationResult, ConversationStorageError, ResponseChain,
            ResponseResult, ResponseStorageError,
        },
        Conversation, ConversationId, ConversationItem, ConversationItemId,
        ConversationItemStorage, ConversationStorage, ListParams, NewConversation,
        NewConversationItem, ResponseId, ResponseStorage, SortOrder, StoredResponse,
    },
};

pub(crate) struct PostgresStore {
    pool: Pool,
}

impl PostgresStore {
    pub fn new(config: PostgresConfig) -> Result<Self, String> {
        let pg_config = tokio_postgres::Config::from_str(config.db_url.as_str()).unwrap();
        let mgr_config = ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        };
        let mgr = Manager::from_config(pg_config, NoTls, mgr_config);
        let pool = Pool::builder(mgr)
            .max_size(config.pool_max)
            .build()
            .unwrap();

        Ok(Self { pool })
    }
}

impl Clone for PostgresStore {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

pub struct PostgresConversationStorage {
    store: PostgresStore,
}

impl PostgresConversationStorage {
    pub fn new(store: PostgresStore) -> Result<Self, ConversationStorageError> {
        futures::executor::block_on(Self::initialize_schema(store.clone()))
            .expect("Failed to initialize conversations schema");
        Ok(Self { store })
    }

    async fn initialize_schema(store: PostgresStore) -> Result<(), ConversationStorageError> {
        let client = store.pool.get().await.unwrap();
        client
            .batch_execute(
                "
            CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR(64) PRIMARY KEY,
                created_at TIMESTAMPTZ,
                metadata JSON
            );",
            )
            .await
            .unwrap();

        Ok(())
    }

    fn parse_metadata(
        metadata: Option<String>,
    ) -> Result<Option<ConversationMetadata>, ConversationStorageError> {
        match metadata {
            None => Ok(None),
            Some(s) => {
                let s = s.trim();
                if s.is_empty() || s.eq_ignore_ascii_case("null") {
                    return Ok(None);
                }
                serde_json::from_str::<ConversationMetadata>(s)
                    .map(Some)
                    .map_err(|e| ConversationStorageError::StorageError(e.to_string()))
            }
        }
    }
}

#[async_trait]
impl ConversationStorage for PostgresConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> Result<Conversation, ConversationStorageError> {
        let conversation = Conversation::new(input);
        let id_str = conversation.id.0.as_str();
        let created_at: DateTime<Utc> = conversation.created_at;
        let metadata_json = conversation
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let client = self.store.pool.get().await.unwrap();
        client
            .execute(
                "INSERT INTO conversations (id, created_at, metadata) VALUES ($1, $2, $3)",
                &[&id_str, &created_at, &metadata_json],
            )
            .await
            .unwrap();
        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let conversation_id = id.0.clone();
        let client = self.store.pool.get().await.unwrap();
        let rows = client
            .query(
                "SELECT id, created_at, metadata FROM conversations WHERE id = $1",
                &[&conversation_id],
            )
            .await
            .unwrap();
        if rows.is_empty() {
            return Ok(None);
        }
        let row = &rows[0];
        let id_str: String = row.get(0);
        let created_at: DateTime<Utc> = row.get(1);
        let metadata_json: Option<String> = row.get(2);
        let metadata = Self::parse_metadata(metadata_json)?;
        Ok(Some(Conversation::with_parts(
            ConversationId(id_str),
            created_at,
            metadata,
        )))
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let conversation_id = id.0.clone();
        let metadata_json = metadata.as_ref().map(serde_json::to_string).transpose()?;
        let client = self.store.pool.get().await.unwrap();
        let rows = client
            .query(
                "UPDATE conversations SET metadata = $1 WHERE id = $2 RETURNING created_at",
                &[&metadata_json, &conversation_id],
            )
            .await
            .unwrap();
        if rows.is_empty() {
            return Ok(None);
        }
        let row = &rows[0];
        let created_at: DateTime<Utc> = row.get(0);
        Ok(Some(Conversation::with_parts(
            ConversationId(conversation_id),
            created_at,
            metadata,
        )))
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        let conversation_id = id.0.clone();
        let client = self.store.pool.get().await.unwrap();
        let rows_deleted = client
            .execute(
                "DELETE FROM conversations WHERE id = $1",
                &[&conversation_id],
            )
            .await
            .unwrap();
        Ok(rows_deleted > 0)
    }
}

pub struct PostgresConversationItemStorage {
    store: PostgresStore,
}

impl PostgresConversationItemStorage {
    pub fn new(store: PostgresStore) -> Result<Self, ConversationItemStorageError> {
        futures::executor::block_on(Self::initialize_schema(store.clone()))
            .expect("Failed to initialize conversation_items or conversation_item_links schema");
        Ok(Self { store })
    }

    async fn initialize_schema(store: PostgresStore) -> Result<(), ConversationItemStorageError> {
        let client = store.pool.get().await.unwrap();
        client
            .batch_execute(
                "
            CREATE TABLE IF NOT EXISTS conversation_items (
                id SERIAL PRIMARY KEY,
                response_id VARCHAR(64),
                item_type VARCHAR(32) NOT NULL,
                role VARCHAR(32),
                content JSON,
                status VARCHAR(32),
                created_at TIMESTAMPTZ
            );",
            )
            .await
            .unwrap();

        // Create conversation_item_links table
        client
            .batch_execute(
                "
            CREATE TABLE IF NOT EXISTS conversation_item_links (
                conversation_id VARCHAR(64),
                item_id VARCHAR(64) NOT NULL,
                added_at TIMESTAMPTZ,
                CONSTRAINT pk_conv_item_link PRIMARY KEY (conversation_id, item_id)
            );
            CREATE INDEX IF NOT EXISTS conv_item_links_conv_idx ON conversation_item_links (conversation_id, added_at);",
            )
            .await
            .unwrap();
        Ok(())
    }
}

#[async_trait]
impl ConversationItemStorage for PostgresConversationItemStorage {
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
            content: item.content.clone(),
            status: item.status.clone(),
            created_at,
        };
        let id_str = conversation_item.id.0.clone();
        let response_id = conversation_item.response_id.clone();
        let item_type = conversation_item.item_type.clone();
        let role = conversation_item.role.clone();
        let status = conversation_item.status.clone();

        let client = self.store.pool.get().await.unwrap();
        client.execute("INSERT INTO conversation_items (id, response_id, item_type, role, content, status, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)",
            &[&id_str, &response_id, &item_type, &role, &content_json, &status, &created_at])
            .await
            .unwrap();
        Ok(conversation_item)
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();
        let client = self.store.pool.get().await.unwrap();
        client.execute("INSERT INTO conversation_item_links (conversation_id, item_id, added_at) VALUES ($1, $2, $3)",
            &[&cid, &iid, &added_at])
            .await
            .unwrap();
        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        let cid = conversation_id.0.clone();
        let limit: i64 = params.limit as i64;
        let order_desc = matches!(params.order, SortOrder::Desc);
        let after_id = params.after.clone();

        let after_key: Option<(DateTime<Utc>, String)> = if let Some(ref aid) = after_id {
            let cid = cid.clone();
            let aid = aid.clone();
            let client = self.store.pool.get().await.unwrap();
            let rows = client
                .query(
                    "SELECT added_at FROM conversation_item_links WHERE conversation_id = $1 AND item_id = $2",
                    &[&cid, &aid],
                )
                .await
                .unwrap();
            if !rows.is_empty() {
                let row = &rows[0];
                let ts: DateTime<Utc> = row.get(0);
                Some((ts, aid))
            } else {
                None
            }
        } else {
            None
        };

        let cid = cid.clone();
        let mut sql = String::from(
            "SELECT i.id, i.response_id, i.item_type, i.role, i.content, i.status, i.created_at \
                             FROM conversation_item_links l \
                             JOIN conversation_items i ON i.id = l.item_id \
                             WHERE l.conversation_id = $1",
        );
        // If cursor provided, append predicate using $2/$3
        if let Some((_ts, _iid)) = &after_key {
            if order_desc {
                sql.push_str(" AND (l.added_at < $2 OR (l.added_at = $2 AND l.item_id < $3))");
            } else {
                sql.push_str(" AND (l.added_at > $2 OR (l.added_at = $2 AND l.item_id > $3))");
            }
        }
        // Order and limit
        if order_desc {
            sql.push_str(" ORDER BY l.added_at DESC, l.item_id DESC");
        } else {
            sql.push_str(" ORDER BY l.added_at ASC, l.item_id ASC");
        }
        // PostgreSQL LIMIT
        if after_key.is_some() {
            sql.push_str(" LIMIT $4");
        } else {
            sql.push_str(" LIMIT $2");
        }
        let client = self.store.pool.get().await.unwrap();
        let rows = if let Some((ts, iid)) = &after_key {
            client.query(&sql, &[&cid, ts, iid, &limit]).await.unwrap()
        } else {
            client.query(&sql, &[&cid, &limit]).await.unwrap()
        };
        let mut out = Vec::new();
        for row in rows {
            let id = row.get(0);
            let resp_id: Option<String> = row.get(1);
            let item_type: String = row.get(2);
            let role: Option<String> = row.get(3);
            let content_raw: Option<String> = row.get(4);
            let status: Option<String> = row.get(5);
            let created_at: DateTime<Utc> = row.get(6);
            out.push((
                id,
                resp_id,
                item_type,
                role,
                content_raw,
                status,
                created_at,
            ));
        }
        out.into_iter()
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
        let client = self.store.pool.get().await.unwrap();
        let row = client.query_one("SELECT id, response_id, item_type, role, content, status, created_at FROM converstation_items WHERE id = $1", &[&iid]).await.unwrap();
        if row.is_empty() {
            Ok(None)
        } else {
            let id: String = row.get(0);
            let response_id: Option<String> = row.get(1);
            let item_type: String = row.get(2);
            let role: Option<String> = row.get(3);
            let content_raw: Option<String> = row.get(4);
            let status: Option<String> = row.get(5);
            let created_at: DateTime<Utc> = row.get(6);

            let content = match content_raw {
                Some(s) => serde_json::from_str(&s)
                    .map_err(ConversationItemStorageError::SerializationError)?,
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
        }
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();
        let client = self.store.pool.get().await.unwrap();
        let row = client
            .query_one(
                "SELECT COUNT(*) FROM conversation_item_links WHERE conversation_id = $1 AND item_id = $2",
                &[&cid, &iid],
            )
            .await
            .unwrap();
        let count: i64 = row.get(0);
        Ok(count > 0)
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        let cid = conversation_id.0.clone();
        let iid = item_id.0.clone();
        let client = self.store.pool.get().await.unwrap();
        client
            .execute(
                "DELETE FROM conversation_item_links WHERE conversation_id = $1 AND item_id = $2",
                &[&cid, &iid],
            )
            .await
            .unwrap();
        Ok(())
    }
}

pub struct PostgresResponseStorage {
    store: PostgresStore,
}

impl PostgresResponseStorage {
    pub fn new(store: PostgresStore) -> Result<Self, ResponseStorageError> {
        futures::executor::block_on(Self::initialize_schema(store.clone()))
            .expect("Failed to initialize responses schema");
        Ok(Self { store })
    }

    async fn initialize_schema(store: PostgresStore) -> Result<(), ResponseStorageError> {
        let client = store.pool.get().await.unwrap();
        client
            .batch_execute(
                "
            CREATE TABLE IF NOT EXISTS responses (
                id VARCHAR(64) PRIMARY KEY,
                conversation_id VARCHAR(64),
                previous_response_id VARCHAR(64),
                input JSON,
                instructions TEXT,
                output JSON,
                tool_calls JSON,
                metadata JSON,
                created_at TIMESTAMPTZ,
                safety_identifier VARCHAR(128),
                model VARCHAR(128),
                raw_response JSON
            );",
            )
            .await
            .unwrap();
        Ok(())
    }

    pub fn build_response_from_now(row: &Row) -> Result<StoredResponse, String> {
        let id: String = row.get("id");
        let conversation_id: Option<String> = row.get("conversation_id");
        let previous: Option<String> = row.get("previous_response_id");
        let input_json: Option<String> = row.get("input");
        let instructions: Option<String> = row.get("instructions");
        let output_json: Option<String> = row.get("output");
        let tool_calls_json: Option<String> = row.get("tool_calls");
        let metadata_json: Option<String> = row.get("metadata");
        let created_at: DateTime<Utc> = row.get("created_at");
        let safety_identifier: Option<String> = row.get("safety_identifier");
        let model: Option<String> = row.get("model");
        let raw_response_json: Option<String> = row.get("raw_response");

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
impl ResponseStorage for PostgresResponseStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        let response_id = response.id.clone();
        let response_id_str = response_id.0.clone();
        let previous_id = response.previous_response_id.map(|r| r.0);
        let json_input = &response.input;
        let json_output = &response.output;
        let json_tool_calls = serde_json::to_string(&response.tool_calls)?;
        let json_metadata = serde_json::to_string(&response.metadata)?;
        let json_raw_response = &response.raw_response;
        let instructions = response.instructions.clone();
        let created_at = response.created_at;
        let safety_identifier = response.safety_identifier.clone();
        let model = response.model.clone();
        let conversation_id = response.conversation_id.clone();
        let client = self.store.pool.get().await.unwrap();
        let insert_count = client.execute(
            "INSERT INTO responses (id, previous_response_id, input, instructions, output, \
                    tool_calls, metadata, created_at, safety_identifier, model, conversation_id, raw_response) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
            &[
                &response_id_str,
                &previous_id,
                &json_input,
                &instructions,
                &json_output,
                &serde_json::json!(&json_tool_calls),
                &serde_json::json!(&json_metadata),
                &created_at,
                &safety_identifier,
                &model,
                &conversation_id,
                &json_raw_response,
            ]).await.unwrap();
        println!("INSERT INTO responses VALUES {insert_count}");
        Ok(response_id)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        let id = response_id.0.clone();
        let client = self.store.pool.get().await.unwrap();
        let row = client
            .query_one("SELECT * FROM responses WHERE id = $1", &[&id])
            .await
            .unwrap();
        Self::build_response_from_now(&row)
            .map(Some)
            .map_err(|err| ResponseStorageError::StorageError(err.to_string()))
    }

    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()> {
        let id = response_id.0.clone();
        let client = self.store.pool.get().await.unwrap();
        client
            .execute("DELETE FROM responses WHERE id = $1", &[&id])
            .await
            .unwrap();
        Ok(())
    }

    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> ResponseResult<ResponseChain> {
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
    ) -> ResponseResult<Vec<StoredResponse>> {
        let identifier = identifier.to_string();
        let client = self.store.pool.get().await.unwrap();
        let rows = if let Some(l) = limit {
            let l_i64: i64 = l as i64;
            client
                .query(
                    "SELECT * FROM responses WHERE safety_identifier = $1 ORDER BY created_at DESC LIMIT $2",
                    &[&identifier, &l_i64],
                )
                .await
                .unwrap()
        } else {
            client
                .query(
                    "SELECT * FROM responses WHERE safety_identifier = $1 ORDER BY created_at DESC",
                    &[&identifier],
                )
                .await
                .unwrap()
        };

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let resp =
                Self::build_response_from_now(&row).map_err(ResponseStorageError::StorageError)?;
            out.push(resp);
        }

        Ok(out)
    }

    async fn delete_identifier_responses(&self, identifier: &str) -> ResponseResult<usize> {
        let identifier = identifier.to_string();
        let client = self.store.pool.get().await.unwrap();
        let rows_deleted = client
            .execute(
                "DELETE FROM responses WHERE safety_identifier = $1",
                &[&identifier],
            )
            .await
            .unwrap();
        Ok(rows_deleted as usize)
    }
}
