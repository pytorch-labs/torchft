// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use core::net::SocketAddr;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use gethostname::gethostname;
use tokio::sync::broadcast;
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio::time::sleep;
use tonic::transport::server::TcpIncoming;
use tonic::transport::Server;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Response, Status};

use crate::torchftpb::lighthouse_service_client::LighthouseServiceClient;
use crate::torchftpb::manager_service_client::ManagerServiceClient;
use crate::torchftpb::{
    manager_service_server::{ManagerService, ManagerServiceServer},
    CheckpointAddressRequest, CheckpointAddressResponse, KillRequest, KillResponse,
    LighthouseHeartbeatRequest, LighthouseQuorumRequest, ManagerQuorumRequest,
    ManagerQuorumResponse, Quorum, QuorumMember, ShouldCommitRequest, ShouldCommitResponse,
};

#[cfg(not(test))]
use log::{info, warn};

#[cfg(test)]
use std::{println as info, println as warn};

struct ManagerState {
    channel: broadcast::Sender<Quorum>,
    participants: u64,
    checkpoint_servers: HashMap<i64, String>,

    should_commit_channel: broadcast::Sender<bool>,
    should_commit_failures: HashSet<i64>,
    should_commit_count: HashSet<i64>,
}

pub struct Manager {
    replica_id: String,
    lighthouse_addr: String,
    address: String,
    store_address: String,
    world_size: u64,
    state: Mutex<ManagerState>,
    listener: Mutex<Option<tokio::net::TcpListener>>,
    local_addr: SocketAddr,
}

pub async fn manager_client_new(
    addr: String,
    timeout: Duration,
) -> Result<ManagerServiceClient<Channel>> {
    // TODO add retries + backoff so other nodes can start before the rank0 comes up

    info!("ManagerClient: establishing connection to {}", &addr);
    let conn = Endpoint::new(addr.clone())?
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(60))
        .connect()
        .await?;
    Ok(ManagerServiceClient::new(conn))
}

impl Manager {
    pub async fn new(
        replica_id: String,
        lighthouse_addr: String,
        address: String,
        bind: String,
        store_addr: String,
        world_size: u64,
    ) -> Result<Arc<Self>> {
        let (tx, _) = broadcast::channel(16);
        let (should_commit_tx, _) = broadcast::channel(16);

        let listener = tokio::net::TcpListener::bind(&bind).await?;

        Ok(Arc::new(Self {
            replica_id: replica_id,
            lighthouse_addr: lighthouse_addr,
            address: address,
            store_address: store_addr,
            world_size: world_size,
            state: Mutex::new(ManagerState {
                channel: tx,
                participants: 0,
                checkpoint_servers: HashMap::new(),

                should_commit_channel: should_commit_tx,
                should_commit_count: HashSet::new(),
                should_commit_failures: HashSet::new(),
            }),
            local_addr: listener.local_addr()?,
            listener: Mutex::new(Some(listener)),
        }))
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mut set = JoinSet::new();

        set.spawn(self.clone()._run_heartbeat());

        set.spawn(self.clone()._run_grpc());

        while let Some(res) = set.join_next().await {
            res??;
        }
        Ok(())
    }

    pub fn address(&self) -> String {
        format!(
            "http://{}:{}",
            gethostname().into_string().unwrap(),
            self.local_addr.port()
        )
    }

    async fn _run_grpc(self: Arc<Self>) -> Result<()> {
        info!(
            "Manager {} listening on {}",
            self.replica_id,
            self.address()
        );

        let listener = self.listener.lock().await.take().unwrap();
        let incoming =
            TcpIncoming::from_listener(listener, true, None).map_err(|e| anyhow::anyhow!(e))?;

        Server::builder()
            .add_service(ManagerServiceServer::new(self))
            .serve_with_incoming(incoming)
            .await
            .map_err(|e| e.into())
    }

    async fn _run_heartbeat(self: Arc<Self>) -> Result<()> {
        let mut client = self.lighthouse_client_new().await?;
        loop {
            let request = tonic::Request::new(LighthouseHeartbeatRequest {
                replica_id: self.replica_id.clone(),
            });

            let _response = client.heartbeat(request).await;

            sleep(Duration::from_millis(100)).await;
        }
    }

    async fn lighthouse_client_new(&self) -> Result<LighthouseServiceClient<Channel>> {
        info!(
            "Manager: connecting to lighthouse at {}",
            &self.lighthouse_addr
        );

        let conn = Endpoint::new(self.lighthouse_addr.clone())?
            .connect_timeout(Duration::from_secs(60))
            .connect()
            .await?;
        Ok(LighthouseServiceClient::new(conn))
    }
}

#[tonic::async_trait]
impl ManagerService for Arc<Manager> {
    async fn quorum(
        &self,
        request: Request<ManagerQuorumRequest>,
    ) -> Result<Response<ManagerQuorumResponse>, Status> {
        let req = request.into_inner();
        let rank = req.rank;

        info!("got quorum request for rank {}", rank);

        let mut rx = {
            let mut state = self.state.lock().await;

            // save checkpoint server info for healing process
            // TODO: make separate call to set?
            state
                .checkpoint_servers
                .insert(req.rank, req.checkpoint_server_addr.clone());

            // TODO check step
            state.participants += 1;
            let rx = state.channel.subscribe();

            if state.participants >= self.world_size {
                state.participants = 0;
                info!("all workers joined -- starting quorum");

                // TODO: don't hold the lock during quorum

                let mut client = self
                    .lighthouse_client_new()
                    .await
                    .map_err(|e| Status::from_error(e.into()))?;

                let request = tonic::Request::new(LighthouseQuorumRequest {
                    requester: Some(QuorumMember {
                        replica_id: self.replica_id.clone(),
                        address: self.address.clone(),
                        store_address: self.store_address.clone(),
                        step: req.step,
                        world_size: self.world_size,
                    }),
                });

                let response = client.quorum(request).await.unwrap();
                let resp = response.into_inner();

                info!("got lighthouse quorum {:?}", resp);

                state
                    .channel
                    .send(
                        resp.quorum
                            .ok_or_else(|| Status::internal("missing quorum"))?,
                    )
                    .map_err(|e| Status::from_error(e.into()))?;
            }

            rx
        };

        let quorum = rx
            .recv()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let participants = &quorum.participants;

        let mut replica_rank = 10000000000;
        for (i, p) in participants.iter().enumerate() {
            if p.replica_id == self.replica_id {
                replica_rank = i;
                break;
            }
        }

        let max_step = participants.iter().map(|p| p.step).max().unwrap();
        let max_participants: Vec<&QuorumMember> =
            participants.iter().filter(|p| p.step == max_step).collect();

        let primary = max_participants[rank as usize % max_participants.len()];

        let mut max_rank = None;
        for (i, p) in max_participants.iter().enumerate() {
            if p.replica_id == self.replica_id {
                max_rank = Some(i as i64);
                break;
            }
        }

        // Decide whether we should be healing:
        // 1. if we're not at the max step
        // 2. if everyone is at the first step and we're not the primary
        let heal = max_step != req.step || max_step == 1 && primary.replica_id != self.replica_id;
        if heal {
            info!(
                "healing is required step={}, max_step={}",
                req.step, max_step
            );
        }

        let reply = ManagerQuorumResponse {
            quorum_id: quorum.quorum_id,
            // address is used for looking up the checkpoint server address.
            address: primary.address.clone(),
            store_address: primary.store_address.clone(),
            max_step: max_step,
            max_rank: max_rank,
            max_world_size: max_participants.len() as i64,
            replica_rank: replica_rank as i64,
            replica_world_size: participants.len() as i64,
            heal: heal,
        };

        info!("returning quorum for rank {}", rank);

        Ok(Response::new(reply))
    }

    async fn checkpoint_address(
        &self,
        request: Request<CheckpointAddressRequest>,
    ) -> Result<Response<CheckpointAddressResponse>, Status> {
        let state = self.state.lock().await;

        let req = request.into_inner();

        let address = state
            .checkpoint_servers
            .get(&req.rank)
            .ok_or_else(|| Status::invalid_argument("rank not found"))?;

        let reply = CheckpointAddressResponse {
            checkpoint_server_address: address.clone(),
        };
        Ok(Response::new(reply))
    }

    async fn should_commit(
        &self,
        request: Request<ShouldCommitRequest>,
    ) -> Result<Response<ShouldCommitResponse>, Status> {
        let req = request.into_inner();
        let rank = req.rank;

        info!(
            "should_commit request from {} should_commit={}",
            rank, req.should_commit
        );

        // TODO: check step count

        let mut rx = {
            let mut state = self.state.lock().await;

            if !req.should_commit {
                state.should_commit_failures.insert(rank);
            }
            state.should_commit_count.insert(rank);

            let rx = state.should_commit_channel.subscribe();

            if state.should_commit_count.len() == self.world_size as usize {
                let decision = state.should_commit_failures.len() == 0;
                info!("should_commit completed should_commit={}", decision);

                state
                    .should_commit_channel
                    .send(decision)
                    .map_err(|e| Status::from_error(e.into()))?;

                // reset state
                state.should_commit_count.clear();
                state.should_commit_failures.clear();
                let (should_commit_tx, _) = broadcast::channel(16);
                state.should_commit_channel = should_commit_tx;
            }

            rx
        };

        let should_commit = rx
            .recv()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let reply = ShouldCommitResponse {
            should_commit: should_commit,
        };
        Ok(Response::new(reply))
    }

    async fn kill(&self, request: Request<KillRequest>) -> Result<Response<KillResponse>, Status> {
        let req = request.into_inner();

        warn!("got kill request: {}", req.msg);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lighthouse::{Lighthouse, LighthouseOpt};

    async fn should_commit(rank: i64, should_commit: bool) -> Result<ShouldCommitResponse> {
        let mut client = manager_client_new(
            "http://localhost:29531".to_string(),
            Duration::from_secs(10),
        )
        .await?;

        let request = tonic::Request::new(ShouldCommitRequest {
            rank: rank,
            step: 1,
            should_commit: should_commit,
        });
        let resp = client.should_commit(request).await?;

        Ok(resp.into_inner())
    }

    #[tokio::test]
    async fn test_should_commit() -> Result<()> {
        let manager = Manager::new(
            "rep_id".to_string(),
            "lighthouse".to_string(),
            "addr".to_string(),
            "[::]:29531".to_string(),
            "store_addr".to_string(),
            2,
        )
        .await?;
        let manager_fut = tokio::spawn(manager._run_grpc());

        let fut_a = tokio::spawn(should_commit(0, true));
        let fut_b = tokio::spawn(should_commit(1, true));
        let resp_a = fut_a.await??;
        let resp_b = fut_b.await??;

        assert!(resp_a.should_commit);
        assert!(resp_b.should_commit);

        let fut_a = tokio::spawn(should_commit(0, true));
        let fut_b = tokio::spawn(should_commit(1, false));
        let resp_a = fut_a.await??;
        let resp_b = fut_b.await??;

        assert!(!resp_a.should_commit);
        assert!(!resp_b.should_commit);

        manager_fut.abort();

        Ok(())
    }

    #[tokio::test]
    async fn test_get_quorum() -> Result<()> {
        let lighthouse = Lighthouse::new(LighthouseOpt {
            bind: "[::]:0".to_string(),
            join_timeout_ms: 100,
            min_replicas: 1,
            quorum_tick_ms: 100,
        })
        .await?;
        let lighthouse_fut = tokio::spawn(lighthouse.clone().run());

        let manager = Manager::new(
            "rep_id".to_string(),
            lighthouse.address(),
            "addr".to_string(),
            "[::]:0".to_string(),
            "store_addr".to_string(),
            1, // world size
        )
        .await?;
        let manager_fut = tokio::spawn(manager.clone().run());

        let mut client = manager_client_new(manager.address(), Duration::from_secs(10)).await?;

        let request = tonic::Request::new(ManagerQuorumRequest {
            rank: 0,
            step: 123,
            checkpoint_server_addr: "addr".to_string(),
        });
        let resp = client.quorum(request).await?.into_inner();

        manager_fut.abort();
        lighthouse_fut.abort();

        assert_eq!(resp.quorum_id, 1);
        assert_eq!(resp.address, "addr".to_string());
        assert_eq!(resp.store_address, "store_addr".to_string());
        assert_eq!(resp.max_step, 123);
        assert_eq!(resp.max_rank, Some(0));
        assert_eq!(resp.max_world_size, 1);
        assert_eq!(resp.replica_rank, 0);
        assert_eq!(resp.replica_world_size, 1);
        assert_eq!(resp.heal, false);

        Ok(())
    }
}
