from __future__ import annotations

from typing import Optional

import numpy as np
import anndata as ad

from ..base_filter import BaseFilter


class MetadataEqualsFilter(BaseFilter):
    """Filter selecting cells where `obs[key] == value`."""

    kind = "metadata_equals"

    def __init__(self, key: str, value):
        super().__init__()
        self.key = key
        self.value = value

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> Optional["MetadataEqualsFilter"]:
        if data.get("kind") != cls.kind:
            return None
        key = data.get("key")
        if key is None:
            return None
        value = data.get("value")
        return cls(key, value)

    # ------------------------------------------------------------------
    def apply(self, adata: ad.AnnData, base_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self.key not in adata.obs:
            raise ValueError(f"Metadata column '{self.key}' not found")
        series = adata.obs[self.key]
        mask = series.astype(str).values == str(self.value)
        return self._combine_mask(mask, base_mask) 