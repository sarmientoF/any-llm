import os

from any_llm.providers.openai import BaseOpenAIProvider


class MzaiProvider(BaseOpenAIProvider):
    ANY_LLM_PLATFORM_URL = os.getenv("ANY_LLM_PLATFORM_URL", "https://platform-api.any-llm.ai")
    API_BASE = ANY_LLM_PLATFORM_URL + "/api/v1"

    ENV_API_KEY_NAME = "ANY_LLM_KEY"
    PROVIDER_NAME = "mzai"
    PROVIDER_DOCUMENTATION_URL = "https://any-llm.ai"
    SUPPORTS_RESPONSES = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = False
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
