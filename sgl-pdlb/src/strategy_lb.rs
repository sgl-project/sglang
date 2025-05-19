use rand::Rng;

#[derive(Debug, Clone)]
pub enum EngineType {
    Prefill,
    Decode,
}

#[derive(Debug, Clone)]
pub struct EngineInfo {
    #[allow(dead_code)]
    pub engine_type: EngineType,
    pub url: String,
    pub boostrap_port: Option<u16>,
}

impl EngineInfo {
    pub fn new_prefill(url: String, boostrap_port: Option<u16>) -> Self {
        EngineInfo {
            engine_type: EngineType::Prefill,
            url,
            boostrap_port,
        }
    }

    pub fn new_decode(url: String) -> Self {
        EngineInfo {
            engine_type: EngineType::Decode,
            url,
            boostrap_port: None,
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
        policy: String,
        prefill_infos: Vec<(String, Option<u16>)>,
        decode_infos: Vec<String>,
    ) -> Self {
        let policy = match policy.as_str() {
            "random" => LBPolicy::Random,
            "po2" => LBPolicy::PowerOfTwo,
            _ => panic!("Invalid policy"),
        };

        StrategyLB {
            policy,
            prefill_servers: prefill_infos
                .into_iter()
                .map(|(url, port)| EngineInfo::new_prefill(url, port))
                .collect(),
            decode_servers: decode_infos
                .into_iter()
                .map(|url| EngineInfo::new_decode(url))
                .collect(),
        }
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
