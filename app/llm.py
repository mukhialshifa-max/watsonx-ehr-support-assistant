"""Answer generation.

WatsonxClient calls IBM watsonx.ai text generation over its REST API:
  1. Exchange an IBM Cloud API key for an IAM access token.
  2. Call /ml/v1/text/generation with a grounded prompt.

If no watsonx credentials are set, OfflineAnswerer returns the best matching FAQ
answer directly, so the app still runs (and the tests pass) without an account.
"""

from __future__ import annotations

import os
import time
from typing import Protocol

import requests

from .knowledge_base import SearchResult

IAM_URL = "https://iam.cloud.ibm.com/identity/token"
API_VERSION = "2024-05-01"
DEFAULT_MODEL = "ibm/granite-3-8b-instruct"

FALLBACK_MESSAGE = (
    "I couldn't find this in the support knowledge base. Please contact the "
    "Service Desk or your at-the-elbow (ATE) support person, and include the screen "
    "name, the steps you took and any error message."
)


class Answerer(Protocol):
    name: str

    def answer(self, question: str, context: list[SearchResult]) -> str: ...


def build_prompt(question: str, context: list[SearchResult]) -> str:
    """Grounded prompt: the model may only use the FAQ context it is given."""
    sources = "\n\n".join(
        f"[{r.entry.id}] Q: {r.entry.question}\nA: {r.entry.answer}" for r in context
    )
    return (
        "You are a friendly EHR application support assistant for hospital staff.\n"
        "Answer the user's question using ONLY the support articles below.\n"
        "Write 2-4 short, plain-language sentences and give clear steps.\n"
        "If the articles do not answer the question, say you don't know and tell "
        "the user to contact the Service Desk.\n"
        "Never ask for or repeat passwords or patient information.\n\n"
        f"Support articles:\n{sources}\n\n"
        f"Question: {question}\nAnswer:"
    )


class OfflineAnswerer:
    """No LLM: return the top retrieved answer. Used for demos, tests and CI."""

    name = "offline"

    def answer(self, question: str, context: list[SearchResult]) -> str:
        if not context:
            return FALLBACK_MESSAGE
        return context[0].entry.answer


class WatsonxClient:
    name = "watsonx"

    def __init__(
        self,
        api_key: str,
        project_id: str,
        url: str = "https://us-south.ml.cloud.ibm.com",
        model_id: str = DEFAULT_MODEL,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.project_id = project_id
        self.url = url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self.session = session or requests.Session()
        self._token: str | None = None
        self._token_expiry = 0.0

    def _get_token(self) -> str:
        # Reuse the IAM token until 60 seconds before it expires.
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        resp = self.session.post(
            IAM_URL,
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": self.api_key},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        self._token_expiry = float(body.get("expiration", time.time() + 3600))
        return self._token

    def answer(self, question: str, context: list[SearchResult]) -> str:
        if not context:
            return FALLBACK_MESSAGE
        payload = {
            "model_id": self.model_id,
            "project_id": self.project_id,
            "input": build_prompt(question, context),
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 250,
                "repetition_penalty": 1.05,
                "stop_sequences": ["\n\nQuestion:"],
            },
        }
        resp = self.session.post(
            f"{self.url}/ml/v1/text/generation",
            params={"version": API_VERSION},
            json=payload,
            headers={"Authorization": f"Bearer {self._get_token()}", "Accept": "application/json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json()["results"][0]["generated_text"].strip()
        return text or context[0].entry.answer


def answerer_from_env() -> Answerer:
    """Use watsonx.ai when credentials are set, otherwise run offline."""
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    if api_key and project_id:
        return WatsonxClient(
            api_key=api_key,
            project_id=project_id,
            url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            model_id=os.getenv("WATSONX_MODEL_ID", DEFAULT_MODEL),
        )
    return OfflineAnswerer()
