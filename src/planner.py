from collections import defaultdict

from .extractor import extract_record
from .fetch import fetch_catalog, fetch_reviews
from .ranker import group_by_category, rank_price_only, rank_rating_weighted
from .schemas import CatalogItem, PlanEvent, RankedItem, TrustReport
from .validator import build_trust_reports

# Fraction of a category's expected review coverage that must be missing
# (retrieval failure) or quarantined (validation failure) before that
# category falls back from rating-weighted to price-only ranking. Using the
# *combined* fraction is what makes the two failure modes interact: neither
# alone may cross the threshold, but together they can.
UNRELIABLE_CATEGORY_THRESHOLD = 0.5


def run(
    scenario: str,
    simulate_timeout_category: str | None = None,
    timeout_fraction: float = 0.5,
) -> dict:
    events: list[PlanEvent] = []

    # Step 1: fetch source A -- trusted, structured, real external API.
    catalog = fetch_catalog()
    catalog_by_id = {c.id: c for c in catalog}
    events.append(PlanEvent(step="fetch_catalog", detail=f"Fetched {len(catalog)} products from fakestoreapi.com"))

    # Step 2: fetch source B -- untrusted, structured-with-a-free-text-field.
    reviews, retrieval_failures = fetch_reviews(
        scenario,
        catalog_by_id,
        simulate_timeout_category=simulate_timeout_category,
        timeout_fraction=timeout_fraction,
    )
    events.append(
        PlanEvent(
            step="fetch_reviews",
            detail=f"Fetched {len(reviews)} reviews; {len(retrieval_failures)} retrieval failures",
        )
    )

    # Step 3: quarantined extraction. This is the only place raw review text
    # is read by anything -- everything after this point works from
    # ExtractedRecord/TrustReport only.
    extractions = {r.review_id: extract_record(r.text) for r in reviews}
    events.append(PlanEvent(step="extract", detail=f"Extracted {len(extractions)} records via quarantined extractor"))

    # Step 4: validate -- contradiction / self-consistency / outlier checks.
    trust_reports = build_trust_reports(catalog_by_id, reviews, extractions)
    quarantined_count = sum(1 for t in trust_reports if t.quarantined)
    events.append(PlanEvent(step="validate", detail=f"{quarantined_count}/{len(trust_reports)} reviews quarantined"))
    for t in trust_reports:
        if t.quarantined:
            events.append(
                PlanEvent(
                    step="quarantine",
                    detail=f"review {t.review_id} (product {t.product_id}): {'; '.join(t.reasons)}",
                )
            )

    trust_by_product: dict[int, list[TrustReport]] = defaultdict(list)
    for t in trust_reports:
        trust_by_product[t.product_id].append(t)

    # Step 5: per-category reliability assessment -> strategy decision.
    # This is the re-plan point: a category doesn't just retry or skip when
    # its data looks bad, the planner picks a different ranking strategy for
    # it and explains why.
    category_items = group_by_category(catalog)

    expected_reviews_by_category: dict[str, int] = defaultdict(int)
    for r in reviews:
        item = catalog_by_id.get(r.product_id)
        if item:
            expected_reviews_by_category[item.category] += 1
    for f in retrieval_failures:
        expected_reviews_by_category[f.category] += 1  # failed fetches still count as expected coverage

    ranked_items: list[RankedItem] = []
    for category, items in category_items.items():
        expected = expected_reviews_by_category.get(category, 0)
        failed = sum(1 for f in retrieval_failures if f.category == category)
        quarantined_in_category = sum(
            1
            for t in trust_reports
            if t.quarantined and (ci := catalog_by_id.get(t.product_id)) and ci.category == category
        )
        unreliable_fraction = (failed + quarantined_in_category) / expected if expected else 0.0

        if expected > 0 and unreliable_fraction >= UNRELIABLE_CATEGORY_THRESHOLD:
            reason = (
                f"Category '{category}' review coverage is unreliable: {failed} retrieval "
                f"failures + {quarantined_in_category} quarantined records out of {expected} "
                f"expected ({unreliable_fraction:.0%}). Falling back from rating-weighted to "
                f"price-only ranking for this category instead of retrying or skipping it."
            )
            events.append(PlanEvent(step="replan", detail=reason))
            ranked_items.extend(rank_price_only(items, reason))
        else:
            ranked_items.extend(rank_rating_weighted(items, trust_by_product))

    ranked_items.sort(key=lambda r: r.adjusted_score, reverse=True)

    return {
        "catalog": catalog,
        "reviews": reviews,
        "retrieval_failures": retrieval_failures,
        "trust_reports": trust_reports,
        "ranked_items": ranked_items,
        "events": events,
    }
