import asyncio
import json
import os
import time
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, reasoning_effort: str = "low"):
        api = os.getenv("GROQ_API_KEY")
        if not api:
            raise ValueError("GROQ_API_KEY not found (put it in src/.env)")

        self.client = Groq(api_key=api)
        self.model = model
        self.reasoning_effort = reasoning_effort

    # ---- sync (kept for scripts / quick tests) ----
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Any,
        retries: int = 3,
    ) -> Any:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_schema.__name__,
                            "schema": response_schema.model_json_schema(),
                        },
                    },
                    reasoning_effort=self.reasoning_effort,
                )
                content = response.choices[0].message.content
                return response_schema.model_validate(json.loads(content))
            except Exception as e:  # rate limit, bad JSON, schema mismatch...
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err!r}")

    # ---- async wrapper: use THIS inside graph nodes ----
    async def agenerate_structured(
        self, system_prompt: str, user_prompt: str, response_schema: Any
    ) -> Any:
        # the Groq SDK call is blocking -> run it in a worker thread so the
        # FastAPI event loop (and parallel specialists) are not blocked
        return await asyncio.to_thread(
            self.generate_structured, system_prompt, user_prompt, response_schema
        )


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    """One shared client for the whole process. Tests can patch this."""
    return LLMClient()
