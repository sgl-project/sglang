mod controller;
pub mod service;
mod ping_server;

use service::{gossip, try_ping, ClusterState};
