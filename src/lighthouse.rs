// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use log::{error, info};
use structopt::StructOpt;
use tokio::sync::broadcast;
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio::time::sleep;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use crate::torchftpb::{
    lighthouse_service_server::{LighthouseService, LighthouseServiceServer},
    LighthouseQuorumRequest, LighthouseQuorumResponse, Quorum, QuorumMember,
};

struct QuorumMemberDetails {
    joined: Instant,
    member: QuorumMember,
}

struct State {
    channel: broadcast::Sender<Quorum>,
    participants: HashMap<String, QuorumMemberDetails>,
    prev_quorum: Option<Quorum>,
    quorum_id: i64,
}

pub struct Lighthouse {
    state: Mutex<State>,
    opt: LighthouseOpt,
}

#[derive(StructOpt, Debug)]
#[structopt()]
pub struct LighthouseOpt {
    // bind is the address to bind the server to.
    #[structopt(long = "bind", default_value = "[::]:29510")]
    bind: String,

    #[structopt(long = "join_timeout_ms", default_value = "60000")]
    join_timeout_ms: u64,

    #[structopt(long = "min_replicas")]
    min_replicas: u64,

    #[structopt(long = "quorum_tick_ms", default_value = "100")]
    quorum_tick_ms: u64,
}

fn quorum_changed(a: &Vec<QuorumMember>, b: &Vec<QuorumMember>) -> bool {
    let a_ids: Vec<&String> = a.iter().map(|p| &p.replica_id).collect();
    let b_ids: Vec<&String> = b.iter().map(|p| &p.replica_id).collect();

    return a_ids != b_ids;
}

impl Lighthouse {
    pub fn new(opt: LighthouseOpt) -> Arc<Self> {
        let (tx, _) = broadcast::channel(16);
        Arc::new(Self {
            state: Mutex::new(State {
                participants: HashMap::new(),
                channel: tx,
                prev_quorum: None,
                quorum_id: 0,
            }),
            opt: opt,
        })
    }

    async fn quorum_valid(&self) -> bool {
        let state = self.state.lock().await;

        let mut first_joined = Instant::now();

        for details in state.participants.values() {
            if details.joined < first_joined {
                first_joined = details.joined;
            }
        }

        if state.prev_quorum.is_some() {
            let mut is_fast_quorum = true;
            let prev_quorum = state.prev_quorum.as_ref().unwrap();

            for prev_member in prev_quorum.participants.iter() {
                if !state.participants.contains_key(&prev_member.replica_id) {
                    is_fast_quorum = false;
                }
            }

            if is_fast_quorum {
                info!("Fast quorum found!");
                return is_fast_quorum;
            }
        }

        if state.participants.len() < self.opt.min_replicas as usize {
            info!(
                "No quorum, only have {} participants, need {}",
                state.participants.len(),
                self.opt.min_replicas
            );
            return false;
        }

        // Quorum is valid at this point but lets wait for stragglers.

        if Instant::now().duration_since(first_joined)
            < Duration::from_millis(self.opt.join_timeout_ms)
        {
            info!(
                "Valid quorum with {} participants, waiting for stragglers due to join timeout",
                state.participants.len()
            );
            return false;
        }

        true
    }

    async fn _quorum_tick(self: Arc<Self>) -> Result<()> {
        // TODO: these should probably run under the same lock
        let quorum_met = self.quorum_valid().await;
        if quorum_met {
            let mut state = self.state.lock().await;
            let mut participants: Vec<QuorumMember> = state
                .participants
                .values()
                .map(|details| details.member.clone())
                .collect();

            // Sort by replica ID to get a consistent ordering across runs.
            participants.sort_by_key(|p| p.replica_id.clone());

            // only increment quorum ID if something about the quorum
            // changed (members/addresses/etc)
            if state.prev_quorum.is_none()
                || quorum_changed(
                    &participants,
                    &state.prev_quorum.as_ref().unwrap().participants,
                )
            {
                state.quorum_id += 1;
                info!(
                    "Detected quorum change, bumping quorum_id to {}",
                    state.quorum_id
                );
            }

            let quorum = Quorum {
                quorum_id: state.quorum_id,
                participants: participants,
            };

            info!("Quorum! {:?}", quorum);

            state.prev_quorum = Some(quorum.clone());
            state.participants.clear();
            match state.channel.send(quorum) {
                Ok(_) => (),
                Err(e) => error!("failed to send quorum {}", e),
            }
        }
        Ok(())
    }

    pub async fn _run_quorum(self: Arc<Self>) -> Result<()> {
        loop {
            self.clone()._quorum_tick().await?;

            sleep(Duration::from_millis(self.opt.quorum_tick_ms)).await;
        }
    }

    pub async fn _run_grpc(self: Arc<Self>) -> Result<()> {
        let bind = self.opt.bind.parse()?;
        info!("Lighthouse listening on {}", bind);

        Server::builder()
            .add_service(LighthouseServiceServer::new(self))
            .serve(bind)
            .await
            .map_err(|e| e.into())
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mut set = JoinSet::new();

        set.spawn(self.clone()._run_quorum());

        set.spawn(self.clone()._run_grpc());

        while let Some(res) = set.join_next().await {
            res??;
        }
        Ok(())
    }
}

#[tonic::async_trait]
impl LighthouseService for Arc<Lighthouse> {
    async fn quorum(
        &self,
        request: Request<LighthouseQuorumRequest>,
    ) -> Result<Response<LighthouseQuorumResponse>, Status> {
        let requester = request
            .into_inner()
            .requester
            .ok_or_else(|| return Status::invalid_argument("missing requester"))?;

        info!("got quorum request for replica {}", &requester.replica_id);

        let mut rx = {
            let mut state = self.state.lock().await;
            state.participants.insert(
                requester.replica_id.clone(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: requester,
                },
            );
            state.channel.subscribe()
        };

        // proactively run quorum tick
        self.clone()
            ._quorum_tick()
            .await
            .map_err(|e| Status::from_error(e.into()))?;

        let quorum = rx.recv().await.map_err(|e| Status::from_error(e.into()))?;

        let reply = LighthouseQuorumResponse {
            quorum: Some(quorum),
        };

        Ok(Response::new(reply))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Sub;

    use tonic::transport::{Channel, Endpoint};

    use crate::torchftpb::lighthouse_service_client::LighthouseServiceClient;

    fn lighthouse_test_new() -> Arc<Lighthouse> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "0.0.0.0:29510".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
        };
        Lighthouse::new(opt)
    }

    async fn lighthouse_client_new(addr: String) -> Result<LighthouseServiceClient<Channel>> {
        let conn = Endpoint::new(addr)?
            .connect_timeout(Duration::from_secs(10))
            .connect()
            .await?;
        Ok(LighthouseServiceClient::new(conn))
    }

    #[tokio::test]
    async fn test_quorum_join_timeout() {
        let lighthouse = lighthouse_test_new();
        assert!(!lighthouse.quorum_valid().await);

        {
            let mut state = lighthouse.state.lock().await;
            state.participants.insert(
                "a".to_string(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: QuorumMember {
                        replica_id: "a".to_string(),
                        address: "".to_string(),
                        store_address: "".to_string(),
                        step: 1,
                    },
                },
            );
        }

        assert!(!lighthouse.quorum_valid().await);

        {
            let mut state = lighthouse.state.lock().await;
            state.participants.get_mut("a").unwrap().joined =
                Instant::now().sub(Duration::from_secs(10 * 60 * 60));
        }

        assert!(lighthouse.quorum_valid().await);
    }

    #[tokio::test]
    async fn test_quorum_fast_prev_quorum() {
        let lighthouse = lighthouse_test_new();
        assert!(!lighthouse.quorum_valid().await);

        {
            let mut state = lighthouse.state.lock().await;
            state.participants.insert(
                "a".to_string(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: QuorumMember {
                        replica_id: "a".to_string(),
                        address: "".to_string(),
                        store_address: "".to_string(),
                        step: 1,
                    },
                },
            );
        }

        assert!(!lighthouse.quorum_valid().await);

        {
            let mut state = lighthouse.state.lock().await;
            state.prev_quorum = Some(Quorum {
                quorum_id: 1,
                participants: vec![QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                }],
            });
        }

        assert!(lighthouse.quorum_valid().await);
    }

    #[tokio::test]
    async fn test_lighthouse_e2e() {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "0.0.0.0:29510".to_string(),
            join_timeout_ms: 1,
            quorum_tick_ms: 10,
        };
        let lighthouse = Lighthouse::new(opt);

        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        let mut client = lighthouse_client_new("http://localhost:29510".to_string())
            .await
            .unwrap();

        let request = tonic::Request::new(LighthouseQuorumRequest {
            requester: Some(QuorumMember {
                replica_id: "foo".to_string(),
                address: "".to_string(),
                store_address: "".to_string(),
                step: 10,
            }),
        });

        let response = client.quorum(request).await.unwrap();
        let quorum = response.into_inner().quorum.unwrap();
        assert_eq!(quorum.participants.len(), 1);

        lighthouse_task.abort();
    }

    #[tokio::test]
    async fn test_quorum_changed() {
        let a = vec![QuorumMember {
            replica_id: "1".to_string(),
            address: "".to_string(),
            store_address: "".to_string(),
            step: 1,
        }];
        let b = vec![QuorumMember {
            replica_id: "1".to_string(),
            address: "changed".to_string(),
            store_address: "changed".to_string(),
            step: 1000,
        }];

        // replica_id is the same
        assert!(!quorum_changed(&a, &b));

        let c = vec![QuorumMember {
            replica_id: "2".to_string(),
            address: "".to_string(),
            store_address: "".to_string(),
            step: 1,
        }];
        // replica_id changed
        assert!(quorum_changed(&a, &c));
    }
}
