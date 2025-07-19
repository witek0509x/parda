from __future__ import annotations

from typing import Optional

import numpy as np
import anndata as ad

from ..base_filter import BaseFilter


class KeywordFilter(BaseFilter):
    """Special keyword filters: NONE, SELECTED, HIGHLIGHTED."""

    kind = "keyword"

    _SUPPORTED = {"NONE", "SELECTED", "HIGHLIGHTED"}

    def __init__(self, value: str):
        super().__init__()
        v = value.upper()
        if v not in self._SUPPORTED:
            raise ValueError("Unsupported keyword filter value")
        self.value = v

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> Optional["KeywordFilter"]:
        if data.get("kind") != cls.kind:
            return None
        value = data.get("value")
        if value is None:
            return None
        return cls(value)

    # ------------------------------------------------------------------
    def apply(self, adata: ad.AnnData, base_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self.value == "NONE":
            mask = np.ones(adata.n_obs, dtype=bool)
        if self.value == "ALL":
            mask = np.zeros(adata.n_obs, dtype=bool)
        elif self.value == "SELECTED":
            if "selected" not in adata.obs:
                raise ValueError("'selected' column not found for SELECTED keyword filter")
            mask = adata.obs["selected"].astype(bool).values
            print(f"Applying SELECTED filter: {mask.sum()} cells selected out of {adata.n_obs}")
        elif self.value == "HIGHLIGHTED":
            # Use 'marked' if available else try 'queried' else error
            col = None
            if "marked" in adata.obs:
                col = "marked"
            elif "queried" in adata.obs:
                col = "queried"
            if col is None:
                raise ValueError("No highlighted column ('marked' or 'queried') found for HIGHLIGHTED filter")
            # For numeric columns consider >0 as highlighted
            series = adata.obs[col]
            mask = series.astype(float).values > np.quantile(series.astype(float), 0.95)
        else:
            raise ValueError("Unexpected keyword value")
        return self._combine_mask(mask, base_mask) 