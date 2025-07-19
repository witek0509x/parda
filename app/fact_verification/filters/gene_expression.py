from __future__ import annotations

from typing import Optional, Callable

import numpy as np
import anndata as ad

from ..base_filter import BaseFilter


_OP_FUNCS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    ">": lambda arr, v: arr > v,
    "<": lambda arr, v: arr < v,
    ">=": lambda arr, v: arr >= v,
    "<=": lambda arr, v: arr <= v,
    "==": lambda arr, v: arr == v,
}


class GeneExpressionFilter(BaseFilter):
    """Filter selecting cells by gene expression condition."""

    kind = "gene_expression"

    def __init__(self, gene: str, op: str, value: float):
        super().__init__()
        if op not in _OP_FUNCS:
            raise ValueError(f"Unsupported operator '{op}'")
        self.gene = gene
        self.op = op
        self.value = value

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> Optional["GeneExpressionFilter"]:
        if data.get("kind") != cls.kind:
            return None
        gene = data.get("gene")
        op = data.get("op")
        value = data.get("value")
        if gene is None or op is None or value is None:
            return None
        return cls(gene, op, value)

    # ------------------------------------------------------------------
    def apply(self, adata: ad.AnnData, base_mask: Optional[np.ndarray] = None) -> np.ndarray:
        gene_idx: Optional[int] = None
        if "gene_name" in adata.var.columns:
            matches = np.where(adata.var["gene_name"].values == self.gene)[0]
            if matches.size:
                gene_idx = int(matches[0])
        if gene_idx is None:
            matches = np.where(adata.var_names == self.gene)[0]
            if matches.size:
                gene_idx = int(matches[0])
        if gene_idx is None:
            raise ValueError(f"Gene '{self.gene}' not found in AnnData")

        if hasattr(adata.X, "toarray"):
            expr_vector = adata.X.toarray()[:, gene_idx]
        else:
            expr_vector = np.asarray(adata.X[:, gene_idx]).flatten()

        mask = _OP_FUNCS[self.op](expr_vector, self.value)
        return self._combine_mask(mask, base_mask) 