use rand::Rng;
use serde_json::json;

#[derive(Debug, Clone)]
pub enum EngineType {
    Prefill,
    Decode,
}

#[derive(Debug, Clone)]
pub struct EngineInfo {
    pub engine_type: EngineType,
    pub url: String,
    pub bootstrap_port: Option<u16>,
}

impl EngineInfo {
    pub fn new_prefill(url: String, bootstrap_port: Option<u16>) -> Self {
        EngineInfo {
            engine_type: EngineType::Prefill,
            url,
            bootstrap_port,
        }
    }

    pub fn new_decode(url: String) -> Self {
        EngineInfo {
            engine_type: EngineType::Decode,
            url,
            bootstrap_port: None,
        }
    }

    pub fn api_path(&self, api_path: &str) -> String {
        if api_path.starts_with("/") {
            format!("{}{}", self.url, api_path)
        } else {
            format!("{}/{}", self.url, api_path)
        }
    }

    pub fn to_string(&self) -> String {
        format!("({:?}@{})", self.engine_type, self.url)
    }

    pub fn get_hostname(&self) -> String {
        let url = self
            .url
            .trim_start_matches("http://")
            .trim_start_matches("https://");
        url.split(':').next().unwrap().to_string()
    }
}

pub struct EngineLoad {
    pub engine_info: EngineInfo,
    pub load: isize,
}

impl EngineLoad {
    pub fn from_json(engine_info: &EngineInfo, json: &serde_json::Value) -> Self {
        let load = match json.get("load") {
            Some(load) => load.as_i64().unwrap_or(-1) as isize,
            None => -1,
        };
        EngineLoad {
            engine_info: engine_info.clone(),
            load,
        }
    }
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "engine": self.engine_info.to_string(),
            "load": self.load,
        })
    }

    pub fn to_string(&self) -> String {
        format!("{}: {}", self.engine_info.to_string(), self.load)
    }
}

#[derive(Debug, Clone)]
pub enum LBPolicy {
    Random,
    PowerOfTwo,
}

#[derive(Debug, Clone)]
pub struct StrategyLB {
    pub policy: LBPolicy,
    pub prefill_servers: Vec<EngineInfo>,
    pub decode_servers: Vec<EngineInfo>,
}

impl StrategyLB {
    pub fn new(
        policy: LBPolicy,
        prefill_servers: Vec<EngineInfo>,
        decode_servers: Vec<EngineInfo>,
    ) -> Self {
        StrategyLB {
            policy,
            prefill_servers,
            decode_servers,
        }
    }

    pub fn get_one_server(&self) -> EngineInfo {
        assert!(!self.prefill_servers.is_empty());
        assert!(!self.decode_servers.is_empty());
        self.prefill_servers[0].clone()
    }

    pub fn get_all_servers(&self) -> Vec<EngineInfo> {
        let mut all_servers = Vec::new();
        all_servers.extend(self.prefill_servers.clone());
        all_servers.extend(self.decode_servers.clone());
        all_servers
    }

    pub async fn select_pair(&self, client: &reqwest::Client) -> (EngineInfo, EngineInfo) {
        match self.policy {
            LBPolicy::Random => self.select_pd_pair_random(),
            LBPolicy::PowerOfTwo => self.select_pd_pair_po2(client).await,
        }
    }

    fn select_pd_pair_random(&self) -> (EngineInfo, EngineInfo) {
        let mut rng = rand::rng();
        let prefill_index = rng.random_range(0..self.prefill_servers.len());
        let decode_index = rng.random_range(0..self.decode_servers.len());

        (
            self.prefill_servers[prefill_index].clone(),
            self.decode_servers[decode_index].clone(),
        )
    }

    async fn get_load_from_engine(
        &self,
        client: &reqwest::Client,
        engine_info: &EngineInfo,
    ) -> Option<isize> {
        let url = engine_info.api_path("/get_load");
        let response = client.get(url).send().await.unwrap();
        match response.status() {
            reqwest::StatusCode::OK => {
                let data = response.json::<serde_json::Value>().await.unwrap();
                Some(data["load"].as_i64().unwrap() as isize)
            }
            _ => None,
        }
    }

    async fn select_pd_pair_po2(&self, client: &reqwest::Client) -> (EngineInfo, EngineInfo) {
        let mut rng = rand::rng();
        let prefill1 =
            self.prefill_servers[rng.random_range(0..self.prefill_servers.len())].clone();
        let prefill2 =
            self.prefill_servers[rng.random_range(0..self.prefill_servers.len())].clone();
        let decode1 = self.decode_servers[rng.random_range(0..self.decode_servers.len())].clone();
        let decode2 = self.decode_servers[rng.random_range(0..self.decode_servers.len())].clone();
        let prefill1_load = self.get_load_from_engine(client, &prefill1).await;
        let prefill2_load = self.get_load_from_engine(client, &prefill2).await;
        let decode1_load = self.get_load_from_engine(client, &decode1).await;
        let decode2_load = self.get_load_from_engine(client, &decode2).await;

        (
            if prefill1_load < prefill2_load {
                prefill1
            } else {
                prefill2
            },
            if decode1_load < decode2_load {
                decode1
            } else {
                decode2
            },
        )
    }
}
