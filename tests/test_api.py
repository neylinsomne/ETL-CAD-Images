from dataclasses import replace

import pytest
from conftest import encode
from fastapi.testclient import TestClient

from filtro_planos.config import Settings
from filtro_planos.service.api import create_app
from filtro_planos.service.worker import build_deps


@pytest.fixture
def client(tmp_path):
    settings = replace(
        Settings.from_env(),
        storage_backend="local",
        storage_root=tmp_path,
        queue_backend="inline",
        database_backend="memory",
        envelope_params=tmp_path / "sin_parametros.json",  # usa los parámetros por defecto
    )
    return TestClient(create_app(build_deps(settings)))


def test_upload_processes_plan_and_serves_colored_page(client, plan_png):
    response = client.post("/v1/planos", files={"file": ("plano.png", plan_png, "image/png")})

    assert response.status_code == 202
    plan_id = response.json()["id"]
    plan = client.get(f"/v1/planos/{plan_id}").json()
    assert plan["estado"] == "listo"
    assert plan["paginas"][0]["estado"] == "ok"
    assert plan["paginas"][0]["metricas"]["glass_px"] > 0
    image = client.get(f"/v1/planos/{plan_id}/paginas/1/color")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"


def test_pdf_with_two_pages_gives_two_processed_pages(client, plan_gray):
    pdf = encode([plan_gray, plan_gray], "PDF")
    response = client.post("/v1/planos", files={"file": ("lamina.pdf", pdf, "application/pdf")})

    plan = client.get(f"/v1/planos/{response.json()['id']}").json()
    assert [p["numero"] for p in plan["paginas"]] == [1, 2]


def test_same_file_twice_is_deduplicated(client, plan_png):
    first = client.post("/v1/planos", files={"file": ("a.png", plan_png, "image/png")}).json()
    second = client.post("/v1/planos", files={"file": ("b.png", plan_png, "image/png")})

    assert second.status_code == 200
    assert second.json() == {"id": first["id"], "estado": "listo", "duplicado": True}


def test_unsupported_file_returns_415(client):
    dwg = ("plano.dwg", b"AC1032 no soportado", "application/octet-stream")
    response = client.post("/v1/planos", files={"file": dwg})

    assert response.status_code == 415


def test_unknown_plan_and_output_return_404(client, plan_png):
    plan_id = client.post("/v1/planos", files={"file": ("plano.png", plan_png, "image/png")}).json()["id"]

    assert client.get("/v1/planos/no-existe").status_code == 404
    assert client.get(f"/v1/planos/{plan_id}/paginas/9/color").status_code == 404
    assert client.get(f"/v1/planos/{plan_id}/paginas/1/otra").status_code == 404


def test_health_and_ready(client):
    assert client.get("/health").json()["estado"] == "ok"
    expected = {"storage": True, "cola": True, "cola_entrenamiento": True, "repositorio": True}
    assert client.get("/ready").json() == expected
