import json
import random
from pathlib import Path

import requests

from .schemas import CatalogItem, RawReview, RetrievalFailure

CATALOG_URL = "https://fakestoreapi.com/products"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_catalog() -> list[CatalogItem]:
    """Fetch the trusted product catalog from a real external API (source A)."""
    resp = requests.get(CATALOG_URL, timeout=10)
    resp.raise_for_status()
    items = []
    for row in resp.json():
        rating = row.get("rating") or {}
        items.append(
            CatalogItem(
                id=row["id"],
                title=row["title"],
                price=row["price"],
                category=row["category"],
                description=row.get("description", ""),
                rating_rate=rating.get("rate", 0.0),
                rating_count=rating.get("count", 0),
            )
        )
    return items


def fetch_reviews(
    scenario: str,
    catalog_by_id: dict[int, CatalogItem],
    simulate_timeout_category: str | None = None,
    timeout_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[list[RawReview], list[RetrievalFailure]]:
    """Fetch the untrusted review feed (source B).

    `simulate_timeout_category`/`timeout_fraction` deliberately drop a
    fraction of reviews for one category to reproduce a retrieval-failure
    scenario on demand -- this is the "retrieval timeout" half of the
    two-independent-failure-modes requirement, made reproducible for the
    demo instead of depending on the real stub server flaking at record time.
    """
    filename = "reviews_poisoned.json" if scenario in ("poisoned", "failure-interaction") else "reviews_clean.json"
    path = DATA_DIR / filename
    raw = json.loads(path.read_text())
    reviews = [RawReview(**r) for r in raw]

    failures: list[RetrievalFailure] = []
    if simulate_timeout_category:
        rng = random.Random(seed)
        kept = []
        for r in reviews:
            item = catalog_by_id.get(r.product_id)
            category = item.category if item else "unknown"
            if category == simulate_timeout_category and rng.random() < timeout_fraction:
                failures.append(
                    RetrievalFailure(
                        product_id=r.product_id,
                        category=category,
                        reason=(
                            f"Simulated retrieval timeout fetching review {r.review_id} "
                            f"for category '{category}'"
                        ),
                    )
                )
                continue
            kept.append(r)
        reviews = kept

    return reviews, failures
