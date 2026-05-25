import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError

load_dotenv()

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds; sleep = BACKOFF_BASE ** attempt


def is_configured() -> bool:
    """Return True if GWDG credentials are present in the environment."""
    return bool(os.getenv("GWDG_API_KEY") and os.getenv("GWDG_API_BASE"))


def generate(prompt: str, max_tokens: int = 512) -> str:
    """Send prompt to the GWDG API and return the response text.

    Retries on rate-limit and connection errors with exponential backoff.
    Raises RuntimeError on auth failure, exhausted retries, or missing credentials.
    """
    api_key = os.getenv("GWDG_API_KEY")
    api_base = os.getenv("GWDG_API_BASE")
    model = os.getenv("GWDG_MODEL_NAME", "llama-3.3-70b-instruct")

    if not api_key or not api_base:
        raise RuntimeError(
            "GWDG credentials not configured. "
            "Set GWDG_API_KEY and GWDG_API_BASE in .env"
        )

    client = OpenAI(api_key=api_key, base_url=api_base)
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
                timeout=30,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError as exc:
            last_error = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE ** (attempt + 1))

        except APIConnectionError as exc:
            last_error = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE**attempt)

        except AuthenticationError as exc:
            raise RuntimeError(
                "GWDG authentication failed. Check your GWDG_API_KEY."
            ) from exc

        except APIError as exc:
            raise RuntimeError(f"GWDG API error: {exc}") from exc

    raise RuntimeError(
        f"GWDG API unavailable after {_MAX_RETRIES} attempts: {last_error}"
    ) from last_error
