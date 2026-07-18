"""Tests for the FastAPI server — the platform validator's contract."""

import pytest
from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "healthy"}


def test_metadata_and_schema_shapes():
    meta = client.get("/metadata").json()
    assert meta["name"] == "BangaloreTrafficEnv"
    schema = client.get("/schema").json()
    assert schema["action"]["n"] == 2
    assert schema["observation"]["shape"] == [6]


def test_tasks_lists_three_with_grader_paths():
    tasks = client.get("/tasks").json()["tasks"]
    assert {t["id"] for t in tasks} == {
        "rush_hour_control", "off_peak_control", "sustained_flow"
    }
    for t in tasks:
        assert t["grader"].startswith("graders:grade_")


@pytest.mark.parametrize(
    "task_id", ["rush_hour_control", "off_peak_control", "sustained_flow"]
)
def test_reset_echoes_requested_task(task_id):
    res = client.post("/reset", json={"task": task_id})
    body = res.json()
    assert body["task"] == task_id
    assert len(body["state"]) == 6


def test_step_response_contract():
    client.post("/reset", json={"task": "off_peak_control"})
    body = client.post("/step", json={"action": 0}).json()
    assert set(body) == {"state", "reward", "done", "info", "score"}
    assert body["reward"] <= 0
    assert 0.001 <= body["score"] <= 0.999


def test_step_rejects_out_of_range_action():
    assert client.post("/step", json={"action": 5}).status_code == 422


def test_state_endpoint():
    assert len(client.get("/state").json()["state"]) == 6
