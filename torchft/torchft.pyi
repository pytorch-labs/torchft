from datetime import timedelta
from typing import Optional, Tuple

class ManagerClient:
    def __init__(self, addr: str, connect_timeout: timedelta) -> None: ...
    def quorum(
        self,
        rank: int,
        step: int,
        checkpoint_server_addr: str,
        shrink_only: bool,
        timeout: timedelta,
    ) -> Tuple[int, int, int, str, str, int, Optional[int], int, bool]: ...
    def checkpoint_address(self, rank: int, timeout: timedelta) -> str: ...
    def should_commit(
        self,
        rank: int,
        step: int,
        should_commit: bool,
        timeout: timedelta,
    ) -> bool: ...

class Manager:
    def __init__(
        self,
        replica_id: str,
        lighthouse_addr: str,
        hostname: str,
        bind: str,
        store_addr: str,
        world_size: int,
        heartbeat_interval: timedelta,
        connect_timeout: timedelta,
    ) -> None: ...
    def address(self) -> str: ...
    def shutdown(self) -> None: ...

class Lighthouse:
    def __init__(
        self,
        bind: str,
        min_replicas: int,
        join_timeout_ms: Optional[int] = None,
        quorum_tick_ms: Optional[int] = None,
        heartbeat_timeout_ms: Optional[int] = None,
    ) -> None: ...
    def address(self) -> str: ...
    def shutdown(self) -> None: ...
