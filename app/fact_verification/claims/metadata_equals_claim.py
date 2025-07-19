from __future__ import annotations

from typing import List, Dict

import numpy as np
import anndata as ad

from ..base_claim import BaseClaim
from ..filters import create_filter
from ..quantifiers import create_quantifier
from ..base_filter import BaseFilter
from ..base_quantifier import BaseQuantifier


class MetadataEqualsClaim(BaseClaim):
    claim_type = "metadata_equals"

    def __init__(
        self,
        key: str,
        value,
        filters: List[BaseFilter],
        quantifier: BaseQuantifier,
        text: str,
    ):
        super().__init__(filters, quantifier, text)
        self.key = key
        self.value = str(value)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict) -> "MetadataEqualsClaim":
        if data.get("type") != cls.claim_type:
            raise ValueError
        key = data.get("key")
        value = data.get("value")
        qualifier = data.get("qualifier")
        filters_data = data.get("filters", [])
        text = data.get("text", "")
        if any(x is None for x in [key, value, qualifier]):
            raise ValueError("Missing fields in metadata_equals claim")
        filters = [create_filter(fd) for fd in filters_data]
        quant = create_quantifier(qualifier)
        return cls(key, value, filters, quant, text)

    # ------------------------------------------------------------------
    def _evaluate_on_subset(self, adata: ad.AnnData, mask: np.ndarray) -> int:
        if self.key not in adata.obs:
            return 0
        series = adata.obs[self.key].astype(str).values
        positives = (series == self.value) & mask
        return int(positives.sum()) 