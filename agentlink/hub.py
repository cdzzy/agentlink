"""
AgentLink Hub — distributed agent registry (Roadmap).

A zero-dependency HTTP registry where agents across processes/networks
announce themselves and discover each other by capability. Liveness is
heartbeat-based: registrations expire after a TTL unless renewed.

Server::

    from agentlink.hub import HubServer

    hub = HubServer(host="0.0.0.0", port=7800, ttl_seconds=30)
    hub.start()
    ...
    hub.stop()

Client::

    from agentlink.hub import HubClient

    client = HubClient(
        hub_url="http://hub:7800",
        agent_id="researcher",
        capabilities=["web-search", "summarization"],
        endpoint="http://me:8000",
    )
    with client:
        peers = client.discover(capabilities=["web-search"])

HTTP API:
    POST /register                body: registration JSON  → {token}
    POST /heartbeat/<agent_id>    header: Authorization: Bearer <token>
    POST /deregister/<agent_id>   header: Authorization: Bearer <token>
    GET  /agents                  all live registrations
    GET  /discover?capability=a&capability=b&namespace=n
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


@dataclass
class Registration:
    agent_id: str
    namespace: str
    capabilities: List[str]
    endpoint: str
    metadata: Dict[str, Any]
    token: str
    registered_at: float
    last_heartbeat: float

    def public_dict(self) -> Dict[str, Any]:
        """Registration without the secret token (what discovery returns)."""
        return {
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "capabilities": self.capabilities,
            "endpoint": self.endpoint,
            "metadata": self.metadata,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
        }


class HubRegistry:
    """Thread-safe in-memory registration store with TTL expiry."""

    def __init__(self, ttl_seconds: float = 30.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._registrations: Dict[str, Registration] = {}

    def register(
        self,
        agent_id: str,
        namespace: str = "default",
        capabilities: Optional[List[str]] = None,
        endpoint: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Registration:
        import time

        with self._lock:
            token = uuid.uuid4().hex
            now = time.time()
            existing = self._registrations.get(agent_id)
            registration = Registration(
                agent_id=agent_id,
                namespace=namespace,
                capabilities=capabilities or [],
                endpoint=endpoint,
                metadata=metadata or {},
                token=token,
                registered_at=existing.registered_at if existing else now,
                last_heartbeat=now,
            )
            self._registrations[agent_id] = registration
            return registration

    def _purge_expired(self) -> None:
        import time

        now = time.time()
        expired = [
            agent_id
            for agent_id, reg in self._registrations.items()
            if now - reg.last_heartbeat > self.ttl_seconds
        ]
        for agent_id in expired:
            del self._registrations[agent_id]

    def heartbeat(self, agent_id: str, token: str) -> bool:
        """Renew a registration. Returns False if unknown/expired or token mismatch."""
        import time

        with self._lock:
            self._purge_expired()
            reg = self._registrations.get(agent_id)
            if reg is None or reg.token != token:
                return False
            reg.last_heartbeat = time.time()
            return True

    def deregister(self, agent_id: str, token: str) -> bool:
        with self._lock:
            reg = self._registrations.get(agent_id)
            if reg is None or reg.token != token:
                return False
            del self._registrations[agent_id]
            return True

    def all_live(self) -> List[Registration]:
        with self._lock:
            self._purge_expired()
            return [reg.public_dict() for reg in self._registrations.values()]

    def discover(
        self,
        capabilities: Optional[List[str]] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find live agents that have ALL the required capabilities."""
        required = capabilities or []
        return [
            reg
            for reg in self.all_live()
            if (namespace is None or reg["namespace"] == namespace)
            and all(cap in reg["capabilities"] for cap in required)
        ]

    def get(self, agent_id: str) -> Optional[Dict[str, Any]]:
        self._purge_expired()
        reg = self._registrations.get(agent_id)
        return reg.public_dict() if reg else None

    def __len__(self) -> int:
        return len(self.all_live())


# ── HTTP Server ──────────────────────────────────────────────────────────

def _make_handler(registry: HubRegistry):
    class HubHandler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # silence default stderr logging
            pass

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _bearer(self) -> str:
            auth = self.headers.get("Authorization", "")
            return auth[len("Bearer "):] if auth.startswith("Bearer ") else ""

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/agents":
                self._json(200, {"agents": registry.all_live()})
            elif parsed.path == "/discover":
                qs = parse_qs(parsed.query)
                capabilities = qs.get("capability", [])
                namespace = qs.get("namespace", [None])[0]
                self._json(200, {"agents": registry.discover(capabilities, namespace)})
            elif parsed.path.startswith("/agents/"):
                agent_id = parsed.path[len("/agents/"):]
                reg = registry.get(agent_id)
                self._json(200 if reg else 404, reg if reg else {"error": "not found"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            parsed = urlparse(self.path)

            if parsed.path == "/register":
                reg = registry.register(
                    agent_id=body.get("agent_id", ""),
                    namespace=body.get("namespace", "default"),
                    capabilities=body.get("capabilities"),
                    endpoint=body.get("endpoint", ""),
                    metadata=body.get("metadata"),
                )
                self._json(200, {"token": reg.token, "ttl_seconds": registry.ttl_seconds})
            elif parsed.path.startswith("/heartbeat/"):
                agent_id = parsed.path[len("/heartbeat/"):]
                ok = registry.heartbeat(agent_id, self._bearer())
                self._json(200 if ok else 404, {"ok": ok})
            elif parsed.path.startswith("/deregister/"):
                agent_id = parsed.path[len("/deregister/"):]
                ok = registry.deregister(agent_id, self._bearer())
                self._json(200 if ok else 404, {"ok": ok})
            else:
                self._json(404, {"error": "not found"})

    return HubHandler


class HubServer:
    """
    HTTP registry server. Runs in a daemon thread; ``port=0`` picks a free port.

    Usage::

        hub = HubServer(port=7800, ttl_seconds=30)
        hub.start()
        print(f"hub at {hub.url}")
        ...
        hub.stop()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0, ttl_seconds: float = 30.0) -> None:
        self.host = host
        self.registry = HubRegistry(ttl_seconds=ttl_seconds)
        self._server: Optional[ThreadingHTTPServer] = None
        self._requested_port = port

    def start(self) -> "HubServer":
        if self._server is not None:
            return self
        self._server = ThreadingHTTPServer((self.host, self._requested_port), _make_handler(self.registry))
        self._server.daemon_threads = True
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def port(self) -> int:
        if self._server is None:
            return self._requested_port
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> "HubServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ── Client ───────────────────────────────────────────────────────────────

class HubClient:
    """
    Registers an agent with a HubServer, heartbeats in the background, and
    discovers peers by capability.

    Usage::

        client = HubClient(
            hub_url="http://hub:7800",
            agent_id="researcher",
            capabilities=["web-search"],
            endpoint="http://me:8000",
            heartbeat_interval=10,
        )
        with client:
            peers = client.discover(capabilities=["web-search"])
    """

    def __init__(
        self,
        hub_url: str,
        agent_id: str,
        namespace: str = "default",
        capabilities: Optional[List[str]] = None,
        endpoint: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: float = 30.0,
        heartbeat_interval: Optional[float] = None,
    ) -> None:
        self.hub_url = hub_url.rstrip("/")
        self.agent_id = agent_id
        self.namespace = namespace
        self.capabilities = capabilities or []
        self.endpoint = endpoint
        self.metadata = metadata or {}
        self.ttl_seconds = ttl_seconds
        self.heartbeat_interval = heartbeat_interval
        self.token: Optional[str] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _post(self, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        data = json.dumps(body or {}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = Request(f"{self.hub_url}{path}", data=data, headers=headers, method="POST")
        return self._read_response(req)

    def _get(self, path: str) -> Any:
        req = Request(f"{self.hub_url}{path}", method="GET")
        return self._read_response(req)

    @staticmethod
    def _read_response(req: Request) -> Any:
        # The hub always returns a JSON body, even for 4xx — parse it so
        # callers can inspect error payloads (e.g. {"ok": false}).
        try:
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as err:
            body = getattr(err, "read", None)
            if body is not None:
                try:
                    return json.loads(err.read())
                except Exception:
                    pass
            raise

    def register(self) -> "HubClient":
        result = self._post("/register", {
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "capabilities": self.capabilities,
            "endpoint": self.endpoint,
            "metadata": self.metadata,
        })
        self.token = result["token"]
        return self

    def heartbeat(self) -> bool:
        result = self._post(f"/heartbeat/{self.agent_id}")
        return bool(result.get("ok"))

    def deregister(self) -> bool:
        try:
            result = self._post(f"/deregister/{self.agent_id}")
            return bool(result.get("ok"))
        except Exception:
            return False

    def agents(self) -> List[Dict[str, Any]]:
        return self._get("/agents")["agents"]

    def discover(
        self,
        capabilities: Optional[List[str]] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: List[str] = []
        for cap in capabilities or []:
            params.append(f"capability={cap}")
        if namespace:
            params.append(f"namespace={namespace}")
        query = ("?" + "&".join(params)) if params else ""
        return self._get(f"/discover{query}")["agents"]

    def _heartbeat_loop(self) -> None:
        interval = self.heartbeat_interval or max(1.0, self.ttl_seconds / 3)
        while not self._stop.wait(interval):
            try:
                self.heartbeat()
            except Exception:
                pass  # expired registrations are re-registered on next use

    def start(self) -> "HubClient":
        self.register()
        if self.heartbeat_interval is not False:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None
        self.deregister()

    def __enter__(self) -> "HubClient":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
