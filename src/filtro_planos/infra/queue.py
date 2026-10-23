"""Cola de trabajos entre la API y los workers.

- InlineQueue: ejecuta el trabajo en el momento (desarrollo y pruebas, sin Redis).
- RedisStreamQueue: Redis Streams con grupo de consumidores; varios workers comparten la cola,
  cada trabajo se confirma (XACK) al terminar y los que fallan van a <stream>:fallidos.
"""

from __future__ import annotations

import json
import socket
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Protocol

from filtro_planos.config import Settings

Handler = Callable[["Job"], None]


@dataclass
class Job:
    kind: str  # p. ej. "procesar_plano"
    payload: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> Job:
        return cls(**json.loads(raw))


class Queue(Protocol):
    def enqueue(self, job: Job) -> str: ...
    def ping(self) -> bool: ...


class InlineQueue:
    """Procesa al encolar. Sirve para desarrollo, pruebas y la CLI."""

    def __init__(self, handler: Handler | None = None):
        self.handler = handler

    def enqueue(self, job: Job) -> str:
        if self.handler is None:
            raise RuntimeError("InlineQueue sin handler: asigna queue.handler antes de encolar")
        self.handler(job)
        return job.id

    def ping(self) -> bool:
        return True


class RedisStreamQueue:
    def __init__(self, client, stream: str, group: str = "workers"):
        self.client = client
        self.stream = stream
        self.group = group
        self.failed_stream = f"{stream}:fallidos"

    @classmethod
    def from_url(cls, url: str, stream: str) -> RedisStreamQueue:
        import redis

        return cls(redis.Redis.from_url(url), stream)

    def enqueue(self, job: Job) -> str:
        self.client.xadd(self.stream, {"job": job.to_json()})
        return job.id

    def ping(self) -> bool:
        return bool(self.client.ping())

    def _ensure_group(self) -> None:
        import redis

        try:
            self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def consume(self, handler: Handler, consumer: str | None = None, block_ms: int = 5000, count: int = 1) -> int:
        """Lee y procesa hasta `count` trabajos; devuelve cuántos procesó (0 si no hubo)."""
        self._ensure_group()
        consumer = consumer or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        batches = self.client.xreadgroup(self.group, consumer, {self.stream: ">"}, count=count, block=block_ms)
        processed = 0
        for _, messages in batches or []:
            for message_id, fields in messages:
                raw = fields.get(b"job") or fields.get("job")
                try:
                    handler(Job.from_json(raw))
                except Exception as exc:  # el trabajo no se pierde: queda en la cola de fallidos
                    self.client.xadd(self.failed_stream, {"job": raw, "error": repr(exc)})
                finally:
                    self.client.xack(self.stream, self.group, message_id)
                processed += 1
        return processed


def build_queue(settings: Settings, stream: str | None = None) -> Queue:
    """Cola de ingesta por defecto; con stream=settings.training_stream, la de entrenamiento."""
    if settings.queue_backend == "inline":
        return InlineQueue()
    if settings.queue_backend == "redis":
        return RedisStreamQueue.from_url(settings.redis_url, stream or settings.queue_stream)
    raise ValueError(f"QUEUE_BACKEND desconocido: {settings.queue_backend!r} (inline | redis)")
