import os
from openai import AsyncAzureOpenAI

def get_llm_client():
    """
    Returns a new initialized AsyncAzureOpenAI client.
    Note: Can't cache strict singleton for concurrent streaming due to potential async lock/pool issues.
    """
    return AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        max_retries=5,
        timeout=120.0,
    )


