//! Redis storage implementation using RedisStore helper
//!
//! Structure:
//! 1. RedisStore helper and common utilities
//! 2. RedisConversationStorage
//! 3. RedisConversationItemStorage
//! 4. RedisResponseStorage

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool_redis::{Config, Pool, Runtime};
use redis::AsyncCommands;
use serde_json::Value;

use crate::{
    config::RedisConfig,
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

pub(crate) struct RedisStore {
    pool: Pool,
    retention_days: Option<u64>,
}

impl RedisStore {
    pub fn new(config: RedisConfig) -> Result<Self, String> {
        let mut cfg = Config::from_url(config.url);
        cfg.pool = Some(deadpool_redis::PoolConfig::new(config.pool_max));
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            pool,
            retention_days: config.retention_days,
        })
    }
}

impl Clone for RedisStore {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            retention_days: self.retention_days,
        }
    }
}

pub struct RedisConversationStorage {
    store: RedisStore,
}

impl RedisConversationStorage {
    pub fn new(store: RedisStore) -> Self {
        Self { store }
    }

    fn conversation_key(id: &str) -> String {
        format!("conversation:{}", id)
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
impl ConversationStorage for RedisConversationStorage {
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

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        let key = Self::conversation_key(id_str);

        let mut pipe = redis::pipe();
        pipe.hset(&key, "id", id_str);
        pipe.hset(&key, "created_at", created_at.to_rfc3339());
        if let Some(meta) = metadata_json {
            pipe.hset(&key, "metadata", meta);
        }

        // Expire after configured retention days (optional)
        if let Some(days) = self.store.retention_days {
            pipe.expire(&key, (days * 24 * 60 * 60) as i64);
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let id_str = id.0.as_str();
        let key = Self::conversation_key(id_str);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        let exists: bool = conn
            .exists(&key)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        if !exists {
            return Ok(None);
        }

        let (created_at_str, metadata_json): (String, Option<String>) = redis::pipe()
            .hget(&key, "created_at")
            .hget(&key, "metadata")
            .query_async(&mut conn)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?
            .with_timezone(&Utc);

        let metadata = Self::parse_metadata(metadata_json)?;

        Ok(Some(Conversation::with_parts(
            id.clone(),
            created_at,
            metadata,
        )))
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>, ConversationStorageError> {
        let id_str = id.0.as_str();
        let key = Self::conversation_key(id_str);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        let exists: bool = conn
            .exists(&key)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        if !exists {
            return Ok(None);
        }

        let metadata_json = metadata.as_ref().map(serde_json::to_string).transpose()?;

        if let Some(meta) = metadata_json {
            conn.hset::<_, _, _, ()>(&key, "metadata", meta)
                .await
                .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        } else {
            conn.hdel::<_, _, ()>(&key, "metadata")
                .await
                .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        }

        // We need to fetch created_at to return the full object
        let created_at_str: String = conn
            .hget(&key, "created_at")
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;
        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?
            .with_timezone(&Utc);

        Ok(Some(Conversation::with_parts(
            id.clone(),
            created_at,
            metadata,
        )))
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        let id_str = id.0.as_str();
        let key = Self::conversation_key(id_str);
        // Also delete the items list for this conversation
        let items_key = format!("{}:items", key);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        let count: usize = redis::pipe()
            .del(&key)
            .del(&items_key)
            .query_async(&mut conn)
            .await
            .map_err(|e| ConversationStorageError::StorageError(e.to_string()))?;

        Ok(count > 0)
    }
}

pub struct RedisConversationItemStorage {
    store: RedisStore,
}

impl RedisConversationItemStorage {
    pub fn new(store: RedisStore) -> Self {
        Self { store }
    }

    fn item_key(id: &str) -> String {
        format!("item:{}", id)
    }

    fn conv_items_key(conv_id: &str) -> String {
        format!("conversation:{}:items", conv_id)
    }
}

#[async_trait]
impl ConversationItemStorage for RedisConversationItemStorage {
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

        let id_str = conversation_item.id.0.as_str();
        let key = Self::item_key(id_str);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        let mut pipe = redis::pipe();

        pipe.hset(&key, "id", id_str);
        if let Some(rid) = &conversation_item.response_id {
            pipe.hset(&key, "response_id", rid);
        }
        pipe.hset(&key, "item_type", &conversation_item.item_type);
        if let Some(r) = &conversation_item.role {
            pipe.hset(&key, "role", r);
        }
        pipe.hset(&key, "content", content_json);
        if let Some(s) = &conversation_item.status {
            pipe.hset(&key, "status", s);
        }
        pipe.hset(&key, "created_at", created_at.to_rfc3339());

        // Expire after configured retention days
        if let Some(days) = self.store.retention_days {
            pipe.expire(&key, (days * 24 * 60 * 60) as i64);
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        Ok(conversation_item)
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        let cid = conversation_id.0.as_str();
        let iid = item_id.0.as_str();
        let key = Self::conv_items_key(cid);

        let score = added_at.timestamp_millis() as f64;

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        conn.zadd::<_, _, _, ()>(&key, iid, score)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        let cid = conversation_id.0.as_str();
        let key = Self::conv_items_key(cid);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        let mut min = "-inf".to_string();
        let mut max = "+inf".to_string();

        if let Some(after_id) = &params.after {
            let score: Option<f64> = conn
                .zscore(&key, after_id)
                .await
                .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
            if let Some(s) = score {
                match params.order {
                    SortOrder::Asc => min = format!("({}", s),
                    SortOrder::Desc => max = format!("({}", s),
                }
            }
        }

        let item_ids: Vec<String> = match params.order {
            SortOrder::Asc => {
                // ZRANGEBYSCORE key min max LIMIT offset count
                conn.zrangebyscore_limit(&key, min, max, 0, params.limit as isize)
                    .await
                    .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?
            }
            SortOrder::Desc => {
                // ZREVRANGEBYSCORE key max min LIMIT offset count
                conn.zrevrangebyscore_limit(&key, max, min, 0, params.limit as isize)
                    .await
                    .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?
            }
        };

        if item_ids.is_empty() {
            return Ok(Vec::<ConversationItem>::new());
        }

        // Fetch all items in pipeline
        let mut pipe = redis::pipe();
        for iid in &item_ids {
            pipe.hgetall(Self::item_key(iid));
        }

        let results: Vec<std::collections::HashMap<String, String>> = pipe
            .query_async(&mut conn)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        let mut items: Vec<ConversationItem> = Vec::new();
        for (i, map) in results.into_iter().enumerate() {
            if map.is_empty() {
                // Item might have been deleted or expired, skip
                continue;
            }

            let id = ConversationItemId(
                map.get("id")
                    .cloned()
                    .unwrap_or_else(|| item_ids[i].clone()),
            );
            let response_id = map.get("response_id").cloned();
            let item_type = map.get("item_type").cloned().unwrap_or_default();
            let role = map.get("role").cloned();
            let status = map.get("status").cloned();

            let content_raw = map.get("content");
            let content = match content_raw {
                Some(s) => serde_json::from_str(s).unwrap_or(Value::Null),
                None => Value::Null,
            };

            let created_at_str = map
                .get("created_at")
                .cloned()
                .unwrap_or_else(|| Utc::now().to_rfc3339());
            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            items.push(ConversationItem {
                id,
                response_id,
                item_type,
                role,
                content,
                status,
                created_at,
            });
        }

        Ok(items)
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>> {
        let iid = item_id.0.as_str();
        let key = Self::item_key(iid);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        let map: std::collections::HashMap<String, String> = conn
            .hgetall(&key)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        if map.is_empty() {
            return Ok(None);
        }

        let id = ConversationItemId(map.get("id").cloned().unwrap_or_else(|| iid.to_string()));
        let response_id = map.get("response_id").cloned();
        let item_type = map.get("item_type").cloned().unwrap_or_default();
        let role = map.get("role").cloned();
        let status = map.get("status").cloned();

        let content_raw = map.get("content");
        let content = match content_raw {
            Some(s) => serde_json::from_str(s).unwrap_or(Value::Null),
            None => Value::Null,
        };

        let created_at_str = map
            .get("created_at")
            .cloned()
            .unwrap_or_else(|| Utc::now().to_rfc3339());
        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Ok(Some(ConversationItem {
            id,
            response_id,
            item_type,
            role,
            content,
            status,
            created_at,
        }))
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        let cid = conversation_id.0.as_str();
        let iid = item_id.0.as_str();
        let key = Self::conv_items_key(cid);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        let score: Option<f64> = conn
            .zscore(&key, iid)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        Ok(score.is_some())
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        let cid = conversation_id.0.as_str();
        let iid = item_id.0.as_str();
        let key = Self::conv_items_key(cid);

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;
        conn.zrem::<_, _, ()>(&key, iid)
            .await
            .map_err(|e| ConversationItemStorageError::StorageError(e.to_string()))?;

        Ok(())
    }
}

pub struct RedisResponseStorage {
    store: RedisStore,
}

impl RedisResponseStorage {
    pub fn new(store: RedisStore) -> Self {
        Self { store }
    }

    fn response_key(id: &str) -> String {
        format!("response:{}", id)
    }

    fn safety_key(identifier: &str) -> String {
        format!("safety:{}:responses", identifier)
    }
}

#[async_trait]
impl ResponseStorage for RedisResponseStorage {
    async fn store_response(
        &self,
        response: StoredResponse,
    ) -> Result<ResponseId, ResponseStorageError> {
        let response_id = response.id.clone();
        let response_id_str = response_id.0.as_str();
        let key = Self::response_key(response_id_str);

        let json_input = serde_json::to_string(&response.input)?;
        let json_output = serde_json::to_string(&response.output)?;
        let json_tool_calls = serde_json::to_string(&response.tool_calls)?;
        let json_metadata = serde_json::to_string(&response.metadata)?;
        let json_raw_response = serde_json::to_string(&response.raw_response)?;

        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        let mut pipe = redis::pipe();

        pipe.hset(&key, "id", response_id_str);
        if let Some(prev) = &response.previous_response_id {
            pipe.hset(&key, "previous_response_id", &prev.0);
        }
        pipe.hset(&key, "input", json_input);
        if let Some(inst) = &response.instructions {
            pipe.hset(&key, "instructions", inst);
        }
        pipe.hset(&key, "output", json_output);
        pipe.hset(&key, "tool_calls", json_tool_calls);
        pipe.hset(&key, "metadata", json_metadata);
        pipe.hset(&key, "created_at", response.created_at.to_rfc3339());
        if let Some(safety) = &response.safety_identifier {
            pipe.hset(&key, "safety_identifier", safety);
        }
        if let Some(model) = &response.model {
            pipe.hset(&key, "model", model);
        }
        if let Some(cid) = &response.conversation_id {
            pipe.hset(&key, "conversation_id", cid);
        }
        pipe.hset(&key, "raw_response", json_raw_response);

        // Expire after configured retention days
        if let Some(days) = self.store.retention_days {
            pipe.expire(&key, (days * 24 * 60 * 60) as i64);
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        // Index by safety identifier if present
        if let Some(safety) = &response.safety_identifier {
            let safety_key = Self::safety_key(safety);
            let score = response.created_at.timestamp_millis() as f64;
            conn.zadd::<_, _, _, ()>(safety_key, response_id_str, score)
                .await
                .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        }

        Ok(response_id)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> Result<Option<StoredResponse>, ResponseStorageError> {
        let id = response_id.0.as_str();
        let key = Self::response_key(id);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        let map: std::collections::HashMap<String, String> = conn
            .hgetall(&key)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        if map.is_empty() {
            return Ok(None);
        }

        let id = ResponseId(map.get("id").cloned().unwrap_or_else(|| id.to_string()));
        let previous_response_id = map
            .get("previous_response_id")
            .map(|s| ResponseId(s.clone()));
        let conversation_id = map.get("conversation_id").cloned();

        let input = match parse_json_value(map.get("input").cloned()) {
            Ok(v) => v,
            Err(e) => return Err(ResponseStorageError::StorageError(e)),
        };
        let instructions = map.get("instructions").cloned();
        let output = match parse_json_value(map.get("output").cloned()) {
            Ok(v) => v,
            Err(e) => return Err(ResponseStorageError::StorageError(e)),
        };
        let tool_calls = match parse_tool_calls(map.get("tool_calls").cloned()) {
            Ok(v) => v,
            Err(e) => return Err(ResponseStorageError::StorageError(e)),
        };
        let metadata = match parse_metadata(map.get("metadata").cloned()) {
            Ok(v) => v,
            Err(e) => return Err(ResponseStorageError::StorageError(e)),
        };

        let created_at_str = map
            .get("created_at")
            .cloned()
            .unwrap_or_else(|| Utc::now().to_rfc3339());
        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        let safety_identifier = map.get("safety_identifier").cloned();
        let model = map.get("model").cloned();
        let raw_response = match parse_raw_response(map.get("raw_response").cloned()) {
            Ok(v) => v,
            Err(e) => return Err(ResponseStorageError::StorageError(e)),
        };

        Ok(Some(StoredResponse {
            id,
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
        }))
    }

    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()> {
        let id = response_id.0.as_str();
        let key = Self::response_key(id);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        // First check if it has a safety identifier to remove from index
        let safety: Option<String> = conn.hget(&key, "safety_identifier").await.ok();

        conn.del::<_, ()>(&key)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        if let Some(s) = safety {
            conn.zrem::<_, _, ()>(Self::safety_key(&s), id)
                .await
                .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        }

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
        let key = Self::safety_key(identifier);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        // ZREVRANGE key 0 limit-1
        let stop = match limit {
            Some(l) => (l as isize) - 1,
            None => -1,
        };

        let response_ids: Vec<String> = conn
            .zrevrange(&key, 0, stop)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        if response_ids.is_empty() {
            return Ok(Vec::<StoredResponse>::new());
        }

        let mut pipe = redis::pipe();
        for id in &response_ids {
            pipe.hgetall(Self::response_key(id));
        }

        let results: Vec<std::collections::HashMap<String, String>> = pipe
            .query_async(&mut conn)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        let mut out: Vec<StoredResponse> = Vec::with_capacity(results.len());
        for (i, map) in results.into_iter().enumerate() {
            if map.is_empty() {
                continue;
            }

            let id = ResponseId(
                map.get("id")
                    .cloned()
                    .unwrap_or_else(|| response_ids[i].clone()),
            );
            let previous_response_id = map
                .get("previous_response_id")
                .map(|s| ResponseId(s.clone()));
            let conversation_id = map.get("conversation_id").cloned();

            let input = match parse_json_value(map.get("input").cloned()) {
                Ok(v) => v,
                Err(e) => return Err(ResponseStorageError::StorageError(e)),
            };
            let instructions = map.get("instructions").cloned();
            let output = match parse_json_value(map.get("output").cloned()) {
                Ok(v) => v,
                Err(e) => return Err(ResponseStorageError::StorageError(e)),
            };
            let tool_calls = match parse_tool_calls(map.get("tool_calls").cloned()) {
                Ok(v) => v,
                Err(e) => return Err(ResponseStorageError::StorageError(e)),
            };
            let metadata = match parse_metadata(map.get("metadata").cloned()) {
                Ok(v) => v,
                Err(e) => return Err(ResponseStorageError::StorageError(e)),
            };

            let created_at_str = map
                .get("created_at")
                .cloned()
                .unwrap_or_else(|| Utc::now().to_rfc3339());
            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let safety_identifier = map.get("safety_identifier").cloned();
            let model = map.get("model").cloned();
            let raw_response = match parse_raw_response(map.get("raw_response").cloned()) {
                Ok(v) => v,
                Err(e) => return Err(ResponseStorageError::StorageError(e)),
            };

            out.push(StoredResponse {
                id,
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
            });
        }

        Ok(out)
    }

    async fn delete_identifier_responses(&self, identifier: &str) -> ResponseResult<usize> {
        let key = Self::safety_key(identifier);
        let mut conn = self
            .store
            .pool
            .get()
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        // Get all IDs
        let response_ids: Vec<String> = conn
            .zrange(&key, 0, -1)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;
        let count = response_ids.len();

        if count == 0 {
            return Ok(0);
        }

        let mut pipe = redis::pipe();
        for id in response_ids {
            pipe.del(Self::response_key(&id));
        }
        pipe.del(&key);

        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(|e| ResponseStorageError::StorageError(e.to_string()))?;

        Ok(count)
    }
}
