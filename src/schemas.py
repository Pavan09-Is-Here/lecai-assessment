from typing import Literal, Optional

from pydantic import BaseModel, Field


class CatalogItem(BaseModel):
    """A product record from the trusted external catalog (source A)."""

    id: int
    title: str
    price: float
    category: str
    description: str
    rating_rate: float
    rating_count: int


class RawReview(BaseModel):
    """A raw, untrusted review record from source B. `text` is the only
    untrusted surface -- everything downstream that touches `text` goes
    through the quarantined extractor first."""

    product_id: int
    review_id: str
    author: str
    text: str


class ExtractedRecord(BaseModel):
    """Strict, schema-constrained output of the quarantined extractor.
    Content-only: it deliberately carries no product_id/review_id so it
    can never be mistaken for an authoritative record on its own -- callers
    must pair it with the RawReview it came from."""

    summary: str
    claimed_price: Optional[float] = None
    claimed_priority: Optional[Literal["low", "normal", "high"]] = None
    sentiment: Literal["positive", "neutral", "negative"]
    contains_directive: bool
    directive_reasoning: Optional[str] = None
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class TrustReport(BaseModel):
    product_id: int
    review_id: str
    trust_score: float = Field(ge=0.0, le=1.0)
    quarantined: bool
    reasons: list[str]
    contradiction_score: float
    directive_flag: bool
    self_consistency_score: float
    outlier_score: float


class RetrievalFailure(BaseModel):
    product_id: int
    category: str
    reason: str


class RankedItem(BaseModel):
    product_id: int
    title: str
    category: str
    price: float
    catalog_rating: float
    strategy_used: str
    adjusted_score: float
    trust_notes: list[str]


class PlanEvent(BaseModel):
    step: str
    detail: str
