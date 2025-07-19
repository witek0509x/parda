from __future__ import annotations

from typing import Optional

import numpy as np
import anndata as ad

from ..base_filter import BaseFilter


class MetadataRangeFilter(BaseFilter):
    """Filter selecting cells where numeric obs[key] lies in [low, high] inclusive."""

    kind = "metadata_range"

    def __init__(self, key: str, low: float, high: float):
        super().__init__()
        self.key = key
        self.low = low
        self.high = high

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> Optional["MetadataRangeFilter"]:
        if data.get("kind") != cls.kind:
            return None
        key = data.get("key")
        if key is None:
            return None
        low = data.get("low")
        high = data.get("high")
        if low is None or high is None:
            return None
        return cls(key, low, high)

    # ------------------------------------------------------------------
    def apply(self, adata: ad.AnnData, base_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self.key not in adata.obs:
            raise ValueError(f"Metadata column '{self.key}' not found")
        series = adata.obs[self.key]
        if not np.issubdtype(series.dtype, np.number):
            raise ValueError(f"Metadata column '{self.key}' is not numeric")
        mask = (series.values >= self.low) & (series.values <= self.high)
        return self._combine_mask(mask, base_mask) 