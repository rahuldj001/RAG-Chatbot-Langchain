"""LLM Model Provider Factory using LangChain.

Dynamically initializes and switches between:
1. Cloud Groq API models (e.g. `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `llama-3.3-70b-versatile`)
2. Cloud OpenAI API models (e.g. `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`)
3. Local Ollama models (e.g. `llama3:8b-instruct-q8_0`)
"""

import os
import requests
from dotenv import load_dotenv
from ...setting import RAGSettings

# Reload .env on module import to pick up updated API keys
load_dotenv(override=True)

# List of models supported via Groq cloud API
GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# List of official OpenAI models
OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]


class LocalRAGModel:
    def __init__(self) -> None:
        pass

    @staticmethod
    def set(
        model_name: str = "openai/gpt-oss-20b",
        system_prompt: str | None = None,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None,
    ):
        """Factory method to construct the requested LangChain ChatModel instance based on provider rules."""
        load_dotenv(override=True)
        setting = setting or RAGSettings()
        model_name = model_name or "openai/gpt-oss-20b"

        # 1. Direct OpenAI API models
        if model_name in OPENAI_MODELS:
            from langchain_openai import ChatOpenAI
            api_key = os.environ.get("OPENAI_API_KEY") or "sk-dummy-key-for-init"
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=setting.ollama.temperature,
            )

        # 2. Cloud Groq API models (or custom organization/model endpoints)
        elif (
            model_name in GROQ_MODELS
            or "/" in model_name
            or model_name.startswith("llama")
            or model_name.startswith("mixtral")
            or model_name.startswith("qwen")
            or bool(os.environ.get("GROQ_API_KEY"))
        ):
            api_key = os.environ.get("GROQ_API_KEY") or "gsk_dummy_key_for_init"
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    model=model_name,
                    api_key=api_key,
                    temperature=setting.ollama.temperature,
                )
            except Exception:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=setting.ollama.temperature,
                )

        # 3. Fallback: Local Ollama daemon
        else:
            try:
                from langchain_ollama import ChatOllama
                return ChatOllama(
                    model=model_name,
                    base_url=f"http://{host}:{setting.ollama.port}",
                    temperature=setting.ollama.temperature,
                )
            except Exception:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=model_name,
                    base_url=f"http://{host}:{setting.ollama.port}",
                    temperature=setting.ollama.temperature,
                )

    @staticmethod
    def pull(host: str, model_name: str):
        """Instructs local Ollama daemon to download/pull a new model weights file.

        Skipped for cloud API models since they require no local downloads.
        """
        if model_name in GROQ_MODELS or model_name in OPENAI_MODELS or "/" in model_name:
            return None
        setting = RAGSettings()
        payload = {"name": model_name}
        try:
            return requests.post(
                f"http://{host}:{setting.ollama.port}/api/pull", json=payload, stream=True, timeout=5
            )
        except Exception:
            return None

    @staticmethod
    def check_model_exist(host: str, model_name: str) -> bool:
        """Verifies if the model is ready to use without requiring a download."""
        if model_name in GROQ_MODELS or model_name in OPENAI_MODELS or "/" in model_name:
            return True
        setting = RAGSettings()
        try:
            res = requests.get(f"http://{host}:{setting.ollama.port}/api/tags", timeout=2)
            if res.status_code != 200:
                return False
            data = res.json()
            if data.get("models") is None:
                return False
            list_model = [d["name"] for d in data["models"]]
            return model_name in list_model
        except Exception:
            return False
