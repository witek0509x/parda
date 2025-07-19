from __future__ import annotations

from typing import List, Dict

import numpy as np
import anndata as ad

from ..base_claim import BaseClaim
from ..filters import create_filter
from ..quantifiers import create_quantifier
from ..base_filter import BaseFilter
from ..base_quantifier import BaseQuantifier


class MetadataRangeClaim(BaseClaim):
    claim_type = "metadata_range"

    def __init__(
        self,
        key: str,
        low: float,
        high: float,
        filters: List[BaseFilter],
        quantifier: BaseQuantifier,
        text: str,
    ):
        super().__init__(filters, quantifier, text)
        self.key = key
        self.low = low
        self.high = high

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict) -> "MetadataRangeClaim":
        if data.get("type") != cls.claim_type:
            raise ValueError
        key = data.get("key")
        low = data.get("low")
        high = data.get("high")
        qualifier = data.get("qualifier")
        filters_data = data.get("filters", [])
        text = data.get("text", "")
        if any(x is None for x in [key, low, high, qualifier]):
            raise ValueError("Missing fields in metadata_range claim")
        filters = [create_filter(fd) for fd in filters_data]
        quant = create_quantifier(qualifier)
        return cls(key, low, high, filters, quant, text)

    # ------------------------------------------------------------------
    def _evaluate_on_subset(self, adata: ad.AnnData, mask: np.ndarray) -> int:
        if self.key not in adata.obs:
            return 0
        series = adata.obs[self.key].values
        if not np.issubdtype(series.dtype, np.number):
            return 0
        positives_mask = (series >= self.low) & (series <= self.high) & mask
        return int(positives_mask.sum()) 