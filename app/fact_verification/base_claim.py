from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import anndata as ad
import numpy as np

from .base_filter import BaseFilter
from .base_quantifier import BaseQuantifier, VerificationResult


class BaseClaim(ABC):
    """Abstract base for all claim types."""

    claim_type: str = "abstract"

    def __init__(self, filters: Optional[List[BaseFilter]], quantifier: BaseQuantifier, text: str):
        self.filters = filters or []
        self.quantifier = quantifier
        self.text = text

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict, filter_factory, quantifier_factory) -> Optional["BaseClaim"]:
        """Return concrete claim parsed from *data* or None if mismatched type.

        *filter_factory* – callable that converts filter dicts into BaseFilter objects.
        *quantifier_factory* – callable that produces BaseQuantifier from qualifier string.
        """

    # ------------------------------------------------------------------
    # Verification logic
    # ------------------------------------------------------------------
    def verify(self, adata: ad.AnnData) -> VerificationResult:
        """High-level verification flow described in spec."""
        try:
            # a) apply filters
            mask: Optional[np.ndarray] = None
            for flt in self.filters:
                mask = flt.apply(adata, mask)

            if mask is None:
                # No filters at all ⇒ consider all cells
                mask = np.ones(adata.n_obs, dtype=bool)

            filtered_total = int(mask.sum())
            if filtered_total == 0:
                return VerificationResult.not_verifiable("No cells left after filtering")

            # b) evaluate claim for filtered cells
            positives = self._evaluate_on_subset(adata, mask)

            # c) use quantifier
            return self.quantifier.verify(positives, filtered_total)
        except Exception as e:
            return VerificationResult.not_verifiable(f"Exception in verification: {e}")

    @abstractmethod
    def _evaluate_on_subset(self, adata: ad.AnnData, mask: np.ndarray) -> int:
        """Return number of cells (among *mask*) that satisfy the claim condition.""" 