from collections import defaultdict

from .schemas import CatalogItem, RankedItem, TrustReport


def rank_rating_weighted(
    items: list[CatalogItem],
    trust_by_product: dict[int, list[TrustReport]],
) -> list[RankedItem]:
    """Default strategy: blend catalog rating with review trust, normalized by price."""
    ranked = []
    for item in items:
        reports = trust_by_product.get(item.id, [])
        trusted = [r for r in reports if not r.quarantined]
        trust_bonus = (sum(r.trust_score for r in trusted) / len(trusted)) if trusted else 0.5
        adjusted = (item.rating_rate * (0.5 + 0.5 * trust_bonus)) / max(item.price, 1.0) ** 0.3
        notes = [f"{len(trusted)}/{len(reports)} reviews trusted"] if reports else ["no review data"]
        ranked.append(
            RankedItem(
                product_id=item.id,
                title=item.title,
                category=item.category,
                price=item.price,
                catalog_rating=item.rating_rate,
                strategy_used="rating_weighted",
                adjusted_score=adjusted,
                trust_notes=notes,
            )
        )
    return sorted(ranked, key=lambda r: r.adjusted_score, reverse=True)


def rank_price_only(items: list[CatalogItem], reason: str) -> list[RankedItem]:
    """Fallback strategy for categories where review data is too unreliable
    to use: rank on catalog price alone rather than trusting degraded data."""
    ranked = []
    for item in items:
        ranked.append(
            RankedItem(
                product_id=item.id,
                title=item.title,
                category=item.category,
                price=item.price,
                catalog_rating=item.rating_rate,
                strategy_used="price_only_fallback",
                adjusted_score=1.0 / max(item.price, 1.0),
                trust_notes=[reason],
            )
        )
    return sorted(ranked, key=lambda r: r.adjusted_score, reverse=True)


def group_by_category(items: list[CatalogItem]) -> dict[str, list[CatalogItem]]:
    grouped: dict[str, list[CatalogItem]] = defaultdict(list)
    for item in items:
        grouped[item.category].append(item)
    return grouped
