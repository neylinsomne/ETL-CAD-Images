import json
from dataclasses import asdict, replace

import pytest
from fastapi.testclient import TestClient

from filtro_planos.config import Settings
from filtro_planos.envelope import EnvelopeParams
from filtro_planos.service import trainer
from filtro_planos.service.api import create_app
from filtro_planos.service.worker import PARAMS_KIND, build_deps


@pytest.fixture
def deps(tmp_path):
    settings = replace(
        Settings.from_env(),
        storage_backend="local",
        storage_root=tmp_path,
        queue_backend="inline",
        database_backend="memory",
        envelope_params=tmp_path / "sin_parametros.json",
    )
    return build_deps(settings)


@pytest.fixture
def client(deps, monkeypatch):
    def fake_tuning(job, deps):
        """Igual que tune_params pero sin correr el ajuste: guarda una versión con bridge_factor 7."""
        params = replace(EnvelopeParams(), bridge_factor=7.0)
        uri = deps.storage.put("modelos/parametros_filtro/v-prueba.json", json.dumps(asdict(params)).encode())
        metrics = {"f1_balanceado_30": 0.8, "iteraciones": job.payload["parametros"]["iteraciones"]}
        model_id = deps.repo.register_model(PARAMS_KIND, "v-prueba", uri, metrics, active=job.payload["activar"])
        return metrics, {"modelo_id": model_id, "version": "v-prueba", "uri": uri}

    def broken(job, deps):
        raise RuntimeError("faltan datos")

    monkeypatch.setitem(trainer.TRAINERS, "ajuste_parametros", fake_tuning)
    monkeypatch.setitem(trainer.TRAINERS, "dataset_dfine", broken)
    return TestClient(create_app(deps))


def test_training_runs_registers_and_activates_a_version(client):
    started = client.post("/v1/entrenamientos", json={"tipo": "ajuste_parametros", "parametros": {"iteraciones": 3}})

    assert started.status_code == 202
    training = client.get(f"/v1/entrenamientos/{started.json()['id']}").json()
    assert training["estado"] == "listo"
    assert training["metricas"]["iteraciones"] == 3
    (model,) = client.get("/v1/modelos", params={"tipo": PARAMS_KIND}).json()
    assert model["id"] == training["artefactos"]["modelo_id"] and model["activo"] is False

    assert client.post(f"/v1/modelos/{model['id']}/activar").json()["activo"] is True
    active = client.get("/v1/parametros").json()
    assert active["version"] == "v-prueba"
    assert active["parametros"]["bridge_factor"] == 7.0  # los workers ya usan la versión activa


def test_failed_training_is_reported_not_lost(client):
    started = client.post("/v1/entrenamientos", json={"tipo": "dataset_dfine"})

    training = client.get(f"/v1/entrenamientos/{started.json()['id']}").json()
    assert training["estado"] == "error"
    assert "faltan datos" in training["error"]


def test_pending_and_unknown_training_types(client):
    assert client.post("/v1/entrenamientos", json={"tipo": "segmentador_trazos"}).status_code == 501
    assert client.post("/v1/entrenamientos", json={"tipo": "otra_cosa"}).status_code == 422
    assert client.get("/v1/entrenamientos/no-existe").status_code == 404
    assert client.post("/v1/modelos/no-existe/activar").status_code == 404
