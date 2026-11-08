from unittest.mock import MagicMock

import pytest
import requests

from app.knowledge_base import FaqEntry, SearchResult
from app.llm import (
    FALLBACK_MESSAGE,
    OfflineAnswerer,
    WatsonxClient,
    answerer_from_env,
    build_prompt,
)

ENTRY = FaqEntry("login-01", "Login & Access", "I forgot my password", "Use the reset link.")
CONTEXT = [SearchResult(ENTRY, 0.9)]


def _response(json_body):
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status.return_value = None
    return resp


def test_prompt_is_grounded_in_the_articles():
    prompt = build_prompt("Locked out?", CONTEXT)
    assert "ONLY the support articles" in prompt
    assert "[login-01]" in prompt
    assert "Use the reset link." in prompt
    assert prompt.endswith("Question: Locked out?\nAnswer:")


def test_offline_answerer():
    assert OfflineAnswerer().answer("q", CONTEXT) == "Use the reset link."
    assert OfflineAnswerer().answer("q", []) == FALLBACK_MESSAGE


def test_watsonx_client_calls_iam_then_generation():
    session = MagicMock()
    session.post.side_effect = [
        _response({"access_token": "tok", "expiration": 9_999_999_999}),
        _response({"results": [{"generated_text": "  Click the reset link.  "}]}),
    ]
    client = WatsonxClient("key", "proj", url="https://example.ml.cloud.ibm.com/", session=session)

    assert client.answer("Locked out?", CONTEXT) == "Click the reset link."

    iam_call, gen_call = session.post.call_args_list
    assert iam_call.args[0] == "https://iam.cloud.ibm.com/identity/token"
    assert iam_call.kwargs["data"]["apikey"] == "key"
    assert gen_call.args[0] == "https://example.ml.cloud.ibm.com/ml/v1/text/generation"
    assert gen_call.kwargs["headers"]["Authorization"] == "Bearer tok"
    body = gen_call.kwargs["json"]
    assert body["project_id"] == "proj"
    assert body["model_id"] == "ibm/granite-3-8b-instruct"


def test_watsonx_token_is_reused():
    session = MagicMock()
    session.post.side_effect = [
        _response({"access_token": "tok", "expiration": 9_999_999_999}),
        _response({"results": [{"generated_text": "A"}]}),
        _response({"results": [{"generated_text": "B"}]}),
    ]
    client = WatsonxClient("key", "proj", session=session)
    client.answer("q1", CONTEXT)
    client.answer("q2", CONTEXT)
    assert session.post.call_count == 3  # one IAM call, two generation calls


def test_watsonx_skips_llm_when_nothing_retrieved():
    session = MagicMock()
    client = WatsonxClient("key", "proj", session=session)
    assert client.answer("q", []) == FALLBACK_MESSAGE
    session.post.assert_not_called()


def test_watsonx_raises_on_http_error():
    session = MagicMock()
    bad = MagicMock()
    bad.raise_for_status.side_effect = requests.HTTPError("401")
    session.post.return_value = bad
    with pytest.raises(requests.HTTPError):
        WatsonxClient("key", "proj", session=session).answer("q", CONTEXT)


def test_answerer_from_env(monkeypatch):
    monkeypatch.delenv("WATSONX_API_KEY", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    assert answerer_from_env().name == "offline"
    monkeypatch.setenv("WATSONX_API_KEY", "k")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "p")
    assert answerer_from_env().name == "watsonx"
