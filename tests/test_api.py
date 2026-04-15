"""
Integration tests for the BotHunter FastAPI endpoints.

Uses httpx.AsyncClient to test the full request/response cycle
without spinning up a real server.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from httpx import ASGITransport, AsyncClient

from main import app


async def _poll_done(client: AsyncClient, job_id: str, timeout: float = 30.0) -> dict:
    """Poll /status/{job_id} until status is 'done' or 'error', then return the job payload."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        resp = await client.get(f"/status/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("done", "error"):
            return data
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")
        await asyncio.sleep(0.1)


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
        # Endpoint now returns 202 immediately with a job_id
        resp = await client.post("/simulate", json={})
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "pending"
        assert "job_id" in data
        assert "poll_url" in data

    async def test_simulate_custom_params(self, client):
        resp = await client.post(
            "/simulate",
            json={"num_humans": 50, "num_bots": 10, "k": 5},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        # Poll until done to verify result correctness
        job = await _poll_done(client, job_id)
        assert job["status"] == "done"
        assert job["result"]["total_nodes"] == 60   # 50 humans + 10 bots

    async def test_simulate_invalid_params(self, client):
        # num_humans too small (< 10)
        resp = await client.post("/simulate", json={"num_humans": 1})
        assert resp.status_code == 422   # FastAPI validation error

    async def test_simulate_returns_bot_ids(self, client):
        resp = await client.post("/simulate", json={"num_humans": 50, "num_bots": 8, "k": 5})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        job = await _poll_done(client, job_id)
        assert job["status"] == "done"
        result = job["result"]
        assert "bot_ids" in result
        assert isinstance(result["bot_ids"], list)


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
