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

#[derive(Clone)]
struct QuorumMemberDetails {
    joined: Instant,
    member: QuorumMember,
}

struct RoomState {
    room_id: String,
    channel: broadcast::Sender<Quorum>,
    participants: HashMap<String, QuorumMemberDetails>,
    prev_quorum: Option<Quorum>,
    quorum_id: i64,
}

struct State {
    rooms: HashMap<String, RoomState>,
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
    #[structopt(
        long = "bind",
        default_value = "[::]:29510",
        help = "Address to bind the server to"
    )]
    pub bind: String,

    #[structopt(
        long = "join_timeout_ms",
        default_value = "60000",
        help = "How long to wait for new replicas to join before considering a quorum"
    )]
    pub join_timeout_ms: u64,

    #[structopt(
        long = "min_replicas",
        help = "Minimum number of replicas to consider a quorum"
    )]
    pub min_replicas: u64,

    #[structopt(
        long = "quorum_tick_ms",
        default_value = "100",
        help = "How frequently to check for quorum when waiting for workers."
    )]
    pub quorum_tick_ms: u64,

    #[structopt(
        long = "heartbeat_timeout_ms",
        default_value = "5000",
        help = "how long to wait for a heartbeat before considering a replica dead."
    )]
    pub heartbeat_timeout_ms: u64,
}

fn quorum_changed(a: &Vec<QuorumMember>, b: &Vec<QuorumMember>) -> bool {
    let a_ids: Vec<&String> = a.iter().map(|p| &p.replica_id).collect();
    let b_ids: Vec<&String> = b.iter().map(|p| &p.replica_id).collect();

    return a_ids != b_ids;
}

// Checks whether the quorum is valid, the new quorum and an explanation for the state.
fn quorum_compute(
    now: Instant,
    heartbeats: &HashMap<String, Instant>,
    state: &RoomState,
    opt: &LighthouseOpt,
) -> (Option<Vec<QuorumMember>>, String) {
    let healthy_participants: HashMap<String, QuorumMemberDetails> = state
        .participants
        .clone()
        .into_iter()
        .filter(|(replica_id, _details)| {
            let last_heartbeat = heartbeats.get(replica_id);
            if last_heartbeat.is_none() {
                return false;
            }

            now.duration_since(*last_heartbeat.unwrap())
                < Duration::from_millis(opt.heartbeat_timeout_ms)
        })
        .collect();

    let mut candidate_participants: Vec<QuorumMember> = healthy_participants
        .values()
        .map(|details| details.member.clone())
        .collect();

    // Sort by replica ID to get a consistent ordering across runs.
    candidate_participants.sort_by_key(|p| p.replica_id.clone());

    let metadata = format!(
        "[{}/{} participants healthy]",
        healthy_participants.len(),
        state.participants.len()
    );

    // Check if we can use the previous quorum.
    if state.prev_quorum.is_some() {
        let prev_quorum = state.prev_quorum.as_ref().unwrap();

        // Fast quorum is when all previous participants are still in the quorum
        // and we have enough participants to form a quorum.
        let is_fast_quorum = prev_quorum
            .participants
            .iter()
            .all(|prev_member| healthy_participants.contains_key(&prev_member.replica_id));

        if is_fast_quorum {
            return (
                Some(candidate_participants),
                format!("Fast quorum found! {}", metadata),
            );
        }
    }

    if healthy_participants.len() < opt.min_replicas as usize {
        return (
            None,
            format!(
                "No quorum, only have {} participants, need {} {}",
                healthy_participants.len(),
                opt.min_replicas,
                metadata
            ),
        );
    }

    // Quorum is valid at this point but lets wait for stragglers.
    let first_joined = healthy_participants
        .values()
        .map(|details| details.joined)
        .min()
        .unwrap_or(now);
    if now.duration_since(first_joined) < Duration::from_millis(opt.join_timeout_ms) {
        return (
            None,
            format!(
                "Valid quorum with {} participants, waiting for stragglers due to join timeout {}",
                healthy_participants.len(),
                metadata
            ),
        );
    }

    (
        Some(candidate_participants),
        format!("Valid quorum found {}", metadata),
    )
}

impl Lighthouse {
    pub async fn new(opt: LighthouseOpt) -> Result<Arc<Self>> {
        let listener = tokio::net::TcpListener::bind(&opt.bind).await?;
        Ok(Arc::new(Self {
            state: Mutex::new(State {
                rooms: HashMap::new(),
                heartbeats: HashMap::new(),
            }),
            opt: opt,
            local_addr: listener.local_addr()?,
            listener: Mutex::new(Some(listener)),
        }))
    }

    fn _quorum_tick(
        self: Arc<Self>,
        heartbeats: &HashMap<String, Instant>,
        state: &mut RoomState,
    ) -> Result<()> {
        let (quorum_met, reason) = quorum_compute(Instant::now(), heartbeats, state, &self.opt);
        info!("{}: {}", state.room_id, reason);

        if quorum_met.is_some() {
            let participants = quorum_met.unwrap();

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
                    "{}: Detected quorum change, bumping quorum_id to {}",
                    state.room_id, state.quorum_id
                );
            }

            let quorum = Quorum {
                quorum_id: state.quorum_id,
                participants: participants,
                created: Some(SystemTime::now().into()),
            };

            info!("{}: Quorum! {:?}", state.room_id, quorum);

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
            {
                let mut state = self.state.lock().await;
                let heartbeats = state.heartbeats.clone();
                for (_room_id, room) in &mut state.rooms {
                    self.clone()._quorum_tick(&heartbeats, room)?;
                }
            }

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
        let template = {
            let state = self.state.lock().await;

            let rooms = state
                .rooms
                .iter()
                .map(|(room_id, room)| {
                    let (_, quorum_status) =
                        quorum_compute(Instant::now(), &state.heartbeats, &room, &self.opt);

                    let max_step = {
                        if let Some(quorum) = room.prev_quorum.clone() {
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

                    RoomStatus {
                        room_id: room_id.clone(),
                        quorum_id: room.quorum_id,
                        prev_quorum: room.prev_quorum.clone(),
                        quorum_status: quorum_status,

                        max_step: max_step,
                    }
                })
                .collect();

            StatusTemplate {
                rooms: rooms,
                heartbeats: state.heartbeats.clone(),
                old_age_threshold: Instant::now()
                    .checked_sub(Duration::from_millis(self.opt.heartbeat_timeout_ms))
                    .unwrap_or(Instant::now()),
            }
        };
        Html(template.render().unwrap())
    }

    async fn kill(self: Arc<Self>, Path(replica_id): Path<String>) -> Result<(), AppError> {
        let addr = 'addr: {
            let state = self.state.lock().await;

            for (_room_id, room) in &state.rooms {
                if room.prev_quorum.is_none() {
                    return Err(AppError(anyhow!("failed to find replica")));
                }

                for member in room.prev_quorum.clone().unwrap().participants {
                    if member.replica_id == replica_id {
                        break 'addr member.address;
                    }
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
        let req = request.into_inner();
        let room_id = req.room_id;
        let requester = req
            .requester
            .ok_or_else(|| return Status::invalid_argument("missing requester"))?;

        info!("got quorum request for replica {}", &requester.replica_id);

        let mut rx = {
            let mut state = self.state.lock().await;

            // implicit heartbeat
            state
                .heartbeats
                .insert(requester.replica_id.clone(), Instant::now());

            let heartbeats = state.heartbeats.clone();

            if !state.rooms.contains_key(&room_id) {
                let (tx, _) = broadcast::channel(16);

                state.rooms.insert(
                    room_id.clone(),
                    RoomState {
                        room_id: room_id.clone(),
                        participants: HashMap::new(),
                        channel: tx,
                        prev_quorum: None,
                        quorum_id: 0,
                    },
                );
            }

            let room = state.rooms.get_mut(&room_id).unwrap();

            room.participants.insert(
                requester.replica_id.clone(),
                QuorumMemberDetails {
                    joined: Instant::now(),
                    member: requester,
                },
            );
            let rx = room.channel.subscribe();

            // proactively run quorum tick
            self.clone()
                ._quorum_tick(&heartbeats, room)
                .map_err(|e| Status::from_error(e.into()))?;

            rx
        };

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
    rooms: Vec<RoomStatus>,
    heartbeats: HashMap<String, Instant>,

    // visualization thresholds
    old_age_threshold: Instant,
}

struct RoomStatus {
    room_id: String,
    prev_quorum: Option<Quorum>,
    quorum_id: i64,
    quorum_status: String,
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

    async fn lighthouse_client_new(addr: String) -> Result<LighthouseServiceClient<Channel>> {
        let conn = Endpoint::new(addr)?
            .connect_timeout(Duration::from_secs(10))
            .connect()
            .await?;
        Ok(LighthouseServiceClient::new(conn))
    }

    #[tokio::test]
    async fn test_quorum_join_timeout() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
        };

        let mut state = RoomState {
            room_id: "test".to_string(),
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
        };
        let mut heartbeats = HashMap::new();

        let now = Instant::now();

        assert!(!quorum_compute(now, &heartbeats, &state, &opt).0.is_some());

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                },
            },
        );
        heartbeats.insert("a".to_string(), now);

        assert!(!quorum_compute(now, &heartbeats, &state, &opt).0.is_some());

        state.participants.get_mut("a").unwrap().joined =
            now.sub(Duration::from_secs(10 * 60 * 60));

        assert!(quorum_compute(now, &heartbeats, &state, &opt).0.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_heartbeats() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 0,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
        };

        let mut state = RoomState {
            room_id: "test".to_string(),
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
        };
        let mut heartbeats = HashMap::new();

        let now = Instant::now();

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                },
            },
        );
        heartbeats.insert("a".to_string(), now);

        assert!(quorum_compute(now, &heartbeats, &state, &opt).0.is_some());

        // expired heartbeat
        heartbeats.insert("a".to_string(), now.sub(Duration::from_secs(10)));

        let (quorum_met, reason) = quorum_compute(now, &heartbeats, &state, &opt);
        assert!(quorum_met.is_none(), "{}", reason);

        // 1 healthy, 1 expired
        state.participants.insert(
            "b".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                },
            },
        );
        heartbeats.insert("b".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &heartbeats, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        let participants = quorum_met.unwrap();
        assert!(participants.len() == 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_quorum_fast_prev_quorum() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 60 * 60 * 1000, // 1hr
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
        };

        let mut state = RoomState {
            room_id: "test".to_string(),
            channel: broadcast::channel(16).0,
            participants: HashMap::new(),
            prev_quorum: None,
            quorum_id: 0,
        };
        let mut heartbeats = HashMap::new();

        let now = Instant::now();

        assert!(!quorum_compute(now, &heartbeats, &state, &opt).0.is_some());

        state.participants.insert(
            "a".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "a".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                },
            },
        );
        heartbeats.insert("a".to_string(), now);

        assert!(!quorum_compute(now, &heartbeats, &state, &opt).0.is_some());

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

        assert!(quorum_compute(now, &heartbeats, &state, &opt).0.is_some());

        // test expanding quorum w/ fast quorum
        state.participants.insert(
            "b".to_string(),
            QuorumMemberDetails {
                joined: now,
                member: QuorumMember {
                    replica_id: "b".to_string(),
                    address: "".to_string(),
                    store_address: "".to_string(),
                    step: 1,
                    world_size: 1,
                },
            },
        );
        heartbeats.insert("b".to_string(), now);

        let (quorum_met, reason) = quorum_compute(now, &heartbeats, &state, &opt);
        assert!(quorum_met.is_some(), "{}", reason);
        let participants = quorum_met.unwrap();
        assert!(participants.len() == 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_lighthouse_e2e() -> Result<()> {
        let opt = LighthouseOpt {
            min_replicas: 1,
            bind: "[::]:0".to_string(),
            join_timeout_ms: 1,
            quorum_tick_ms: 10,
            heartbeat_timeout_ms: 5000,
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
                room_id: "test".to_string(),
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
