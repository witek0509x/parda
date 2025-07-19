from __future__ import annotations

from typing import List, Dict, Optional, Callable

import numpy as np
import anndata as ad

from ..base_claim import BaseClaim
from ..base_filter import BaseFilter
from ..filters import create_filter
from ..quantifiers import create_quantifier
from ..base_quantifier import BaseQuantifier
from ..filters.gene_expression import _OP_FUNCS


class GeneExpressionClaim(BaseClaim):
    """Claim about gene expression threshold."""

    claim_type = "gene_expression"

    def __init__(
        self,
        gene: str,
        op: str,
        value: float,
        filters: List[BaseFilter],
        quantifier: BaseQuantifier,
        text: str,
    ):
        super().__init__(filters, quantifier, text)
        if op not in _OP_FUNCS:
            raise ValueError("Unsupported operator in gene_expression claim")
        self.gene = gene
        self.op = op
        self.value = value

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict) -> "GeneExpressionClaim":
        if data.get("type") != cls.claim_type:
            raise ValueError
        gene = data.get("gene")
        op = data.get("op")
        value = data.get("value")
        qualifier = data.get("qualifier")
        filters_data = data.get("filters", [])
        text = data.get("text", "")
        if any(x is None for x in [gene, op, value, qualifier]):
            raise ValueError("Missing keys in gene_expression claim")
        filters = [create_filter(fd) for fd in filters_data]
        quant = create_quantifier(qualifier)
        return cls(gene, op, value, filters, quant, text)

    # ------------------------------------------------------------------
    def _evaluate_on_subset(self, adata: ad.AnnData, mask: np.ndarray) -> int:
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
            return 0  # gene not present – treat as zero expression

        if hasattr(adata.X, "toarray"):
            expr = adata.X.toarray()[:, gene_idx]
        else:
            expr = np.asarray(adata.X[:, gene_idx]).flatten()

        expr_in_subset = expr[mask]
        positives_mask = _OP_FUNCS[self.op](expr_in_subset, self.value)
        return int(positives_mask.sum()) 