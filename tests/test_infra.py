import boto3
import fakeredis
import pytest
from moto import mock_aws

from filtro_planos.infra.queue import InlineQueue, Job, RedisStreamQueue
from filtro_planos.infra.repository import MemoryRepository
from filtro_planos.infra.storage import LocalStorage, S3Storage


def test_local_storage_round_trip_and_rejects_escaping_keys(tmp_path):
    storage = LocalStorage(tmp_path)

    storage.put("procesado/abc/p1/color.png", b"datos")

    assert storage.get("procesado/abc/p1/color.png") == b"datos"
    assert storage.exists("procesado/abc/p1/color.png")
    with pytest.raises(ValueError):
        storage.put("../fuera.txt", b"x")


def test_s3_storage_round_trip(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="planos")
        storage = S3Storage("planos")

        storage.put("originales/x/plano.pdf", b"%PDF-1.7", "application/pdf")

        assert storage.ping()
        assert storage.get("originales/x/plano.pdf") == b"%PDF-1.7"
        assert storage.exists("originales/x/plano.pdf")
        assert not storage.exists("originales/x/otro.pdf")


def test_inline_queue_runs_the_job_on_enqueue():
    seen = []
    queue = InlineQueue(seen.append)

    queue.enqueue(Job("procesar_plano", {"plan_id": "1"}))

    assert seen[0].payload == {"plan_id": "1"}


def test_redis_stream_queue_delivers_acks_and_parks_failures():
    client = fakeredis.FakeRedis()
    queue = RedisStreamQueue(client, "trabajos")
    queue.enqueue(Job("procesar_plano", {"plan_id": "ok"}))
    queue.enqueue(Job("procesar_plano", {"plan_id": "falla"}))
    seen = []

    def handler(job):
        if job.payload["plan_id"] == "falla":
            raise RuntimeError("plano roto")
        seen.append(job.payload["plan_id"])

    processed = queue.consume(handler, consumer="w1", block_ms=10, count=10)

    assert processed == 2
    assert seen == ["ok"]
    assert client.xlen("trabajos:fallidos") == 1  # el trabajo fallido no se pierde
    assert client.xpending("trabajos", "workers")["pending"] == 0  # ambos confirmados


def test_memory_repository_tracks_plan_and_pages():
    repo = MemoryRepository()
    plan_id = repo.create_plan("sha", "plano.pdf", "pdf", 10, "originales/sha/plano.pdf")

    repo.add_page(plan_id, 1, "ok", {"color": "k"}, {"wall_px": 5})
    repo.set_status(plan_id, "listo")

    plan = repo.get_plan(plan_id)
    assert repo.find_by_sha("sha") == plan_id
    assert plan["estado"] == "listo" and plan["paginas"][0]["metricas"]["wall_px"] == 5
