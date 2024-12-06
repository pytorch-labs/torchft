// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use core::net::SocketAddr;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::time::{Instant, SystemTime};

use anyhow::{anyhow, Result};
use askama::Template;
use axum::{
    extract::Path,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Router,
};
use gethostname::gethostname;
use log::{error, info};
use structopt::StructOpt;
use tokio::sync::broadcast;
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio::time::sleep;
use tonic::service::Routes;
use tonic::transport::server::TcpIncoming;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use crate::manager::manager_client_new;
use crate::torchftpb::{
    lighthouse_service_server::{LighthouseService, LighthouseServiceServer},
    KillRequest, LighthouseHeartbeatRequest, LighthouseHeartbeatResponse, LighthouseQuorumRequest,
    LighthouseQuorumResponse, Quorum, QuorumMember,
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

    // heartbeat information
    // replica_id -> last heartbeat
    heartbeats: HashMap<String, Instant>,
}

pub struct Lighthouse {
    state: Mutex<State>,
    opt: LighthouseOpt,
    listener: Mutex<Option<tokio::net::TcpListener>>,
    local_addr: SocketAddr,
}

#[derive(StructOpt, Debug)]
#[structopt()]
pub struct LighthouseOpt {
    // bind is the address to bind the server to.
    #[structopt(long = "bind", default_value = "[::]:29510")]
    pub bind: String,

    #[structopt(long = "join_timeout_ms", default_value = "60000")]
    pub join_timeout_ms: u64,

    #[structopt(long = "min_replicas")]
    pub min_replicas: u64,

    #[structopt(long = "quorum_tick_ms", default_value = "100")]
    pub quorum_tick_ms: u64,
}

fn quorum_changed(a: &Vec<QuorumMember>, b: &Vec<QuorumMember>) -> bool {
    let a_ids: Vec<&String> = a.iter().map(|p| &p.replica_id).collect();
    let b_ids: Vec<&String> = b.iter().map(|p| &p.replica_id).collect();

    return a_ids != b_ids;
}

impl Lighthouse {
    pub async fn new(opt: LighthouseOpt) -> Result<Arc<Self>> {
        let (tx, _) = broadcast::channel(16);
        let listener = tokio::net::TcpListener::bind(&opt.bind).await?;
        Ok(Arc::new(Self {
            state: Mutex::new(State {
                participants: HashMap::new(),
                channel: tx,
                prev_quorum: None,
                quorum_id: 0,
                heartbeats: HashMap::new(),
            }),
            opt: opt,
            local_addr: listener.local_addr()?,
            listener: Mutex::new(Some(listener)),
        }))
    }

    // Checks whether the quorum is valid and an explanation for the state.
    async fn quorum_valid(&self) -> (bool, String) {
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
                return (is_fast_quorum, format!("Fast quorum found!"));
            }
        }

        if state.participants.len() < self.opt.min_replicas as usize {
            return (
                false,
                format!(
                    "No quorum, only have {} participants, need {}",
                    state.participants.len(),
                    self.opt.min_replicas
                ),
            );
        }

        // Quorum is valid at this point but lets wait for stragglers.

        if Instant::now().duration_since(first_joined)
            < Duration::from_millis(self.opt.join_timeout_ms)
        {
            return (
                false,
                format!(
                    "Valid quorum with {} participants, waiting for stragglers due to join timeout",
                    state.participants.len()
                ),
            );
        }

        (true, format!("Valid quorum found"))
    }

    async fn _quorum_tick(self: Arc<Self>) -> Result<()> {
        // TODO: these should probably run under the same lock
        let (quorum_met, reason) = self.quorum_valid().await;
        info!("{}", reason);

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
                created: Some(SystemTime::now().into()),
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

    async fn _run_quorum(self: Arc<Self>) -> Result<()> {
        loop {
            self.clone()._quorum_tick().await?;

            sleep(Duration::from_millis(self.opt.quorum_tick_ms)).await;
        }
    }

    pub fn address(&self) -> String {
        format!(
            "http://{}:{}",
            gethostname().into_string().unwrap(),
            self.local_addr.port()
        )
    }

    async fn _run_grpc(self: Arc<Self>) -> Result<()> {
        info!("Lighthouse listening on: {}", self.address());

        let listener = self.listener.lock().await.take().unwrap();
        let incoming =
            TcpIncoming::from_listener(listener, true, None).map_err(|e| anyhow::anyhow!(e))?;

        // Setup HTTP endpoints
        let app = Router::new()
            .route(
                "/",
                get(|| async { Html(IndexTemplate {}.render().unwrap()) }),
            )
            .route(
                "/status",
                get({
                    let self_clone = self.clone();
                    move || async { self_clone.get_status().await }
                }),
            )
            .route(
                "/replica/:replica_id/kill",
                post({
                    let self_clone = self.clone();
                    move |path| async { self_clone.kill(path).await }
                }),
            );

        // register the GRPC service
        let routes = Routes::from(app).add_service(LighthouseServiceServer::new(self));

        Server::builder()
            // allow non-GRPC connections
            .accept_http1(true)
            .add_routes(routes)
            .serve_with_incoming(incoming)
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

    async fn get_status(self: Arc<Self>) -> Html<String> {
        let (_, quorum_status) = self.quorum_valid().await;

        let template = {
            let state = self.state.lock().await;

            let max_step = {
                if let Some(quorum) = state.prev_quorum.clone() {
                    quorum
                        .participants
                        .iter()
                        .map(|p| p.step)
                        .max()
                        .unwrap_or(-1)
                } else {
                    -1
                }
            };

            StatusTemplate {
                quorum_id: state.quorum_id,
                prev_quorum: state.prev_quorum.clone(),
                heartbeats: state.heartbeats.clone(),
                quorum_status: quorum_status,
                old_age_threshold: Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .unwrap_or(Instant::now()),
                max_step: max_step,
            }
        };
        Html(template.render().unwrap())
    }

    async fn kill(self: Arc<Self>, Path(replica_id): Path<String>) -> Result<(), AppError> {
        let addr = 'addr: {
            let state = self.state.lock().await;
            if state.prev_quorum.is_none() {
                return Err(AppError(anyhow!("failed to find replica")));
            }

            for member in state.prev_quorum.clone().unwrap().participants {
                if member.replica_id == replica_id {
                    break 'addr member.address;
                }
            }
            return Err(AppError(anyhow!("failed to find replica")));
        };

        let mut client = manager_client_new(addr, Duration::from_secs(10)).await?;

        let request = tonic::Request::new(KillRequest {
            msg: "killed from dashboard".to_string(),
        });
        let _resp = client.kill(request).await?;

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

    async fn heartbeat(
        &self,
        request: Request<LighthouseHeartbeatRequest>,
    ) -> Result<Response<LighthouseHeartbeatResponse>, Status> {
        let replica_id = request.into_inner().replica_id;

        {
            let mut state = self.state.lock().await;
            state.heartbeats.insert(replica_id, Instant::now());
        }

        let reply = LighthouseHeartbeatResponse {};
        Ok(Response::new(reply))
    }
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {}

#[derive(Template)]
#[template(path = "status.html")]
struct StatusTemplate {
    prev_quorum: Option<Quorum>,
    quorum_id: i64,
    heartbeats: HashMap<String, Instant>,
    quorum_status: String,

    // visualization thresholds
    old_age_threshold: Instant,
    max_step: i64,
}

// Make our own error that wraps `anyhow::Error`.
struct AppError(anyhow::Error);

// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}

// This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
// `Result<_, AppError>`. That way you don't need to do that manually.
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Sub;

    use tonic::transport::{Channel, Endpoint};

    use crate::torchftpb::lighthouse_service_client::LighthouseServiceClient;

    async fn lighthouse_test_new() -> Result<Arc<Lighthouse>> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
        };
        Lighthouse::new(opt).await
    }

    async fn lighthouse_client_new(addr: String) -> Result<LighthouseServiceClient<Channel>> {
        let conn = Endpoint::new(addr)?
            .connect_timeout(Duration::from_secs(10))
            .connect()
            .await?;
        Ok(LighthouseServiceClient::new(conn))
    }

    #[tokio::test]
    async fn test_quorum_join_timeout() -> Result<()> {
        let lighthouse = lighthouse_test_new().await?;
        assert!(!lighthouse.quorum_valid().await.0);

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
                        world_size: 1,
                    },
                },
            );
        }

        assert!(!lighthouse.quorum_valid().await.0);

        {
            let mut state = lighthouse.state.lock().await;
            state.participants.get_mut("a").unwrap().joined =
                Instant::now().sub(Duration::from_secs(10 * 60 * 60));
        }

        assert!(lighthouse.quorum_valid().await.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_fast_prev_quorum() -> Result<()> {
        let lighthouse = lighthouse_test_new().await?;
        assert!(!lighthouse.quorum_valid().await.0);

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
                        world_size: 1,
                    },
                },
            );
        }

        assert!(!lighthouse.quorum_valid().await.0);

        {
            let mut state = lighthouse.state.lock().await;
            state.prev_quorum = Some(Quorum {
                quorum_id: 1,
                participants: vec![QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                }],
                created: Some(SystemTime::now().into()),
            });
        }

        assert!(lighthouse.quorum_valid().await.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_lighthouse_e2e() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 1,
            quorum_tick_ms: 10,
        };
        let lighthouse = Lighthouse::new(opt).await?;

        let lighthouse_task = tokio::spawn(lighthouse.clone().run());

        let mut client = lighthouse_client_new(lighthouse.address()).await.unwrap();

        {
            let request = tonic::Request::new(LighthouseHeartbeatRequest {
                replica_id: "foo".to_string(),
            });

            let _response = client.heartbeat(request).await.unwrap();
        }

        {
            let request = tonic::Request::new(LighthouseQuorumRequest {
                requester: Some(QuorumMember {
                    replica_id: "foo".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 10,
                    world_size: 1,
                }),
            });

            let response = client.quorum(request).await.unwrap();
            let quorum = response.into_inner().quorum.unwrap();
            assert_eq!(quorum.participants.len(), 1);
        }

        lighthouse_task.abort();
        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_changed() {
        let a = vec![QuorumMember {
            replica_id: "1".to_string(),
            address: "".to_string(),
            store_address: "".to_string(),
            step: 1,
            world_size: 1,
        }];
        let b = vec![QuorumMember {
            replica_id: "1".to_string(),
            address: "changed".to_string(),
            store_address: "changed".to_string(),
            step: 1000,
            world_size: 1,
        }];

        // replica_id is the same
        assert!(!quorum_changed(&a, &b));

        let c = vec![QuorumMember {
            replica_id: "2".to_string(),
            address: "".to_string(),
            store_address: "".to_string(),
            step: 1,
            world_size: 1,
        }];
        // replica_id changed
        assert!(quorum_changed(&a, &c));
    }
}
