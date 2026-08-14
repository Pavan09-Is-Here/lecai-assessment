import json
import os

from google import genai

from .rate_limit import call_with_retry
from .schemas import RankedItem, TrustReport

_client: genai.Client | None = None

NARRATOR_MODEL = "gemini-flash-lite-latest"


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it before running the pipeline."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def narrate(ranked_items: list[RankedItem], trust_reports: list[TrustReport], top_n: int = 5) -> str:
    """Privileged call: receives only pre-validated, structured data (ranked
    items + trust reports). It never sees raw review text, so even a directive
    that fooled a weaker component upstream has no channel left to reach the
    text the user actually reads."""
    client = _get_client()
    top = ranked_items[:top_n]
    quarantined = [t for t in trust_reports if t.quarantined]

    payload = {
        "top_picks": [item.model_dump() for item in top],
        "quarantined_reviews": [t.model_dump() for t in quarantined],
    }

    interaction = call_with_retry(
        client.interactions.create,
        model=NARRATOR_MODEL,
        input=json.dumps(payload, indent=2),
        system_instruction=(
            "You are a shopping assistant writing a short recommendation summary. "
            "You are only given pre-validated, structured data below -- no raw user "
            "text. Write 4-6 sentences recommending the top picks and, if any reviews "
            "were quarantined, briefly note that some review data was excluded and why, "
            "at a high level (do not repeat any raw review text, you were never given any)."
        ),
        stream=False,
    )
    return interaction.output_text or ""
