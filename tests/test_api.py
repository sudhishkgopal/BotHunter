"""
Integration tests for the BotHunter FastAPI endpoints.

Uses httpx.AsyncClient to test the full request/response cycle
without spinning up a real server.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


class TestHealthEndpoint:
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_health_has_status_field(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    async def test_health_has_version(self, client):
        resp = await client.get("/health")
        assert resp.json()["version"] == "1.0.0"


class TestRootEndpoint:
    async def test_root_returns_200(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200

    async def test_root_has_endpoints_key(self, client):
        resp = await client.get("/")
        assert "endpoints" in resp.json()


class TestSimulateEndpoint:
    async def test_simulate_defaults(self, client):
        # Endpoint uses a Pydantic body — send empty JSON to use all defaults
        resp = await client.post("/simulate", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "detected_bots" in data
        assert "total_nodes" in data
        assert "job_id" in data

    async def test_simulate_custom_params(self, client):
        resp = await client.post(
            "/simulate",
            json={"num_humans": 50, "num_bots": 10, "k": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_nodes"] == 60   # 50 humans + 10 bots

    async def test_simulate_invalid_params(self, client):
        # num_humans too small (< 10)
        resp = await client.post("/simulate", json={"num_humans": 1})
        assert resp.status_code == 422   # FastAPI validation error

    async def test_simulate_returns_bot_ids(self, client):
        resp = await client.post("/simulate", json={"num_humans": 50, "num_bots": 8, "k": 5})
        assert "bot_ids" in resp.json()
        assert isinstance(resp.json()["bot_ids"], list)


class TestHistoryEndpoint:
    async def test_history_returns_200(self, client):
        resp = await client.get("/history")
        assert resp.status_code == 200

    async def test_history_returns_list(self, client):
        resp = await client.get("/history")
        assert isinstance(resp.json(), list)

    async def test_history_pagination(self, client):
        resp = await client.get("/history?limit=5&offset=0")
        assert resp.status_code == 200

    async def test_history_invalid_limit(self, client):
        resp = await client.get("/history?limit=999")
        assert resp.status_code == 422

    async def test_history_detail_404(self, client):
        resp = await client.get("/history/999999")
        assert resp.status_code == 404


class TestJobStatusEndpoint:
    async def test_status_unknown_job(self, client):
        resp = await client.get("/status/nonexistent-job-id")
        assert resp.status_code == 404

    async def test_status_after_simulate(self, client):
        sim = await client.post("/simulate", json={"num_humans": 30, "num_bots": 5, "k": 3})
        job_id = sim.json()["job_id"]
        resp = await client.get(f"/status/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("pending", "running", "done", "error")
        assert data["job_id"] == job_id


class TestDownloadEndpoint:
    async def test_download_nonexistent_file(self, client):
        resp = await client.get("/download/does_not_exist.json")
        assert resp.status_code == 404

    async def test_download_path_traversal_blocked(self, client):
        # Path traversal attempt — should 404, never serve ../etc/passwd
        resp = await client.get("/download/../../etc/passwd")
        assert resp.status_code in (404, 422)
