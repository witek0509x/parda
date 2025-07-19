from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
import anndata as ad


class BaseFilter(ABC):
    """Abstract filter.

    A filter selects a subset of cells given an AnnData object (and an optional pre-existing mask).
    Subclasses parse themselves from a JSON-serialisable dict produced by the claim extractor.
    """

    kind: str = "abstract"

    def __init__(self) -> None:
        super().__init__()

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> Optional["BaseFilter"]:
        """Return an instance if *data* matches this filter kind, else ``None``."""

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    @abstractmethod
    def apply(self, adata: ad.AnnData, base_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Return boolean mask of shape (n_obs,) selecting cells that satisfy the filter.

        *base_mask* (if given) is combined via logical *AND* with the filter’s own mask.
        """

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------
    def _combine_mask(self, own: np.ndarray, base: Optional[np.ndarray]) -> np.ndarray:
        if base is None:
            return own
        return np.logical_and(base, own) 