from __future__ import annotations

from .gene_expression_claim import GeneExpressionClaim
from .metadata_equals_claim import MetadataEqualsClaim
from .metadata_range_claim import MetadataRangeClaim

from ..base_claim import BaseClaim

_CLAIM_MAP = {
    GeneExpressionClaim.claim_type: GeneExpressionClaim,
    MetadataEqualsClaim.claim_type: MetadataEqualsClaim,
    MetadataRangeClaim.claim_type: MetadataRangeClaim,
}


def create_claim(claim_dict) -> BaseClaim:
    if not isinstance(claim_dict, dict):
        raise ValueError("claim_dict must be dict")
    ctype = claim_dict.get("type")
    if ctype not in _CLAIM_MAP:
        raise ValueError(f"Unknown claim type '{ctype}'")
    cls = _CLAIM_MAP[ctype]
    return cls.from_dict(claim_dict)

__all__ = [
    "GeneExpressionClaim",
    "MetadataEqualsClaim",
    "MetadataRangeClaim",
    "create_claim",
] 