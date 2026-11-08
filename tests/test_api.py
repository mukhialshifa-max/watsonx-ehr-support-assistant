import requests
from fastapi.testclient import TestClient

from app.knowledge_base import KnowledgeBase
from app.llm import OfflineAnswerer
from app.main import create_app

client = TestClient(create_app(KnowledgeBase.from_file(), OfflineAnswerer()))


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["engine"] == "offline"
    assert body["articles"] == 15


def test_ask_returns_answer_and_sources():
    resp = client.post("/ask", json={"question": "How do I reschedule an appointment?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"][0]["id"] == "sched-01"
    assert "Reschedule" in body["answer"]


def test_ask_unknown_question_gives_fallback():
    body = client.post("/ask", json={"question": "best pizza in Halifax"}).json()
    assert body["sources"] == []
    assert "Service Desk" in body["answer"]


def test_ask_validates_input():
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_categories():
    cats = client.get("/categories").json()
    assert "Scheduling" in cats and cats == sorted(cats)


def test_index_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "EHR Support Assistant" in resp.text


class FailingAnswerer:
    name = "watsonx"

    def answer(self, question, context):
        raise requests.ConnectionError("watsonx unreachable")


def test_falls_back_when_watsonx_fails():
    c = TestClient(create_app(KnowledgeBase.from_file(), FailingAnswerer()))
    body = c.post("/ask", json={"question": "I forgot my password"}).json()
    assert body["engine"] == "offline-fallback"
    assert "password reset" in body["answer"]
