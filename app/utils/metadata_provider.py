from __future__ import annotations

"""Utility class that provides cluster-level and selection-level summaries of
AnnData metadata and gene expression.  Only clustering logic is implemented
now (chunks 0-2).

The class depends on an existing `AnnDataModel` instance so that it can reuse
all data-loading, embedding and helper functionality already present in the
codebase.
"""

from typing import Optional

import scanpy as sc
from anndata import AnnData


class MetadataProvider:  # pragma: no cover – new feature, tested manually
    """Generate summaries of clusters or arbitrary cell selections.

    Parameters
    ----------
    anndata_model
        Instance of :class:`app.models.anndata_model.AnnDataModel` that already
        holds an :class:`~anndata.AnnData` object (after data have been
        loaded).  The provider *does not* take ownership of the AnnData; it
        reads and writes directly to the same object so results are
        immediately visible to the rest of the application once integrated.
    """

    def __init__(self, anndata_model):
        self._am = anndata_model
        self.adata: Optional[AnnData] = getattr(anndata_model, "adata", None)
        if self.adata is None:
            raise ValueError("AnnDataModel does not contain loaded data.")

        self._has_clustered: bool = "clip_cluster" in self.adata.obs.columns

    # ------------------------------------------------------------------
    # Public helpers – implemented for chunks 0-2
    # ------------------------------------------------------------------
    def ensure_clip_clustering(self, resolution: float = 1.0, *, force: bool = False):
        """Cluster cells in CLIP-embedding space using Leiden.

        If `X_clip` is missing the method will ask the associated
        :class:`AnnDataModel` to compute embeddings first.
        """
        if self.adata is None:
            raise ValueError("No AnnData object available for clustering.")

        if "X_clip" not in self.adata.obsm:
            # This can take time; reuse existing AnnDataModel logic.
            print("X_clip not found ‑- calculating CLIP embeddings via AnnDataModel...")
            self._am.calculate_clip_embeddings()

        if force or not self._has_clustered:
            print("Running Leiden clustering on X_clip …")
            sc.pp.neighbors(self.adata, use_rep="X_clip")
            sc.tl.leiden(self.adata, resolution=resolution, key_added="clip_cluster")
            self._has_clustered = True
            print("Leiden clustering finished. Found",
                  len(self.adata.obs["clip_cluster"].unique()), "clusters.")
        else:
            print("Leiden clustering already present – skipping.")

    # ------------------------------------------------------------------
    # Placeholder methods – will be filled in future chunks
    # ------------------------------------------------------------------
    def summarise_categorical(self, series, max_levels: int = 8):
        """Return human-readable summary for a categorical Series.

        Parameters
        ----------
        series
            Pandas Series of dtype *object* or categorical.
        max_levels
            Keep at most this many most-frequent categories; the rest are
            aggregated into ``"other"``.
        """

        import pandas as pd
        from pandas.api.types import is_categorical_dtype

        if series.empty:
            return "(no data)"

        # Convert to categorical to ensure consistent handling of NaNs etc.
        if not (is_categorical_dtype(series) or series.dtype == "category"):
            series = series.astype("category")

        counts = series.value_counts(dropna=False)
        total = counts.sum()

        top = counts.head(max_levels)
        other = counts.iloc[max_levels:].sum()

        parts = [
            f"{idx}={cnt / total:.0%}" for idx, cnt in top.items()
        ]
        if other > 0:
            parts.append(f"other={other / total:.0%}")

        return ", ".join(parts)

    def summarise_numeric(self, series):
        """Return summary statistics for numeric or boolean Series."""

        import numpy as np
        from pandas.api.types import is_bool_dtype, is_numeric_dtype

        if series.empty:
            return "(no data)"

        if is_bool_dtype(series):
            true_cnt = int(series.sum())
            total = len(series)
            pct = true_cnt / total * 100 if total else 0
            return f"{pct:.1f}% true ({true_cnt}/{total})"

        if not is_numeric_dtype(series):
            # Fallback: return distinct count
            return f"non-numeric (unique={series.nunique()})"

        desc = series.describe()
        # pandas describe includes count/mean/std; we focus on  min/25/50/75/max
        return (
            f"min {desc['min']:.4g}, 25% {desc['25%']:.4g}, "
            f"median {desc['50%']:.4g}, 75% {desc['75%']:.4g}, max {desc['max']:.4g}"
        )

    def cluster_profile(self, cluster_id: str | int, *, top_genes_n: int = 20):  # noqa: D401
        """Return a human-readable profile string summarising one cluster.

        The profile includes:
        • size of the cluster,
        • per-column metadata summaries,
        • list of top expressed genes.
        """

        import pandas as pd
        from pandas.api.types import is_numeric_dtype, is_bool_dtype

        if self.adata is None:
            raise ValueError("AnnData not loaded.")

        # Boolean mask for target cluster
        mask = self.adata.obs["clip_cluster"] == cluster_id
        n_cells = int(mask.sum())
        if n_cells == 0:
            return f"Cluster {cluster_id}: (no cells)"

        lines: list[str] = [f"Cluster {cluster_id} (n={n_cells})"]

        # ---------------- Metadata ----------------
        for col in self.adata.obs.columns:
            series = self.adata.obs.loc[mask, col]
            # Skip synthetic columns we add later
            if col in {"clip_cluster", "marked", "queried", "queried_ever"}:
                continue

            if is_bool_dtype(series):
                summary = self.summarise_numeric(series)
            elif is_numeric_dtype(series):
                summary = self.summarise_numeric(series)
            else:
                summary = self.summarise_categorical(series, max_levels=8)

            lines.append(f"- {col}: {summary}")

        # ---------------- Top genes ----------------
        top20 = self.top_genes(mask, top_n=top_genes_n)
        gene_str = ", ".join(g for g, _ in top20)
        lines.append(f"- top_genes: {gene_str}")

        return "\n".join(lines)

    def dataset_cluster_summaries(self):
        """Return concatenated profiles for all clusters as one string."""

        if self.adata is None:
            raise ValueError("AnnData not loaded.")

        # Ensure clustering done
        if "clip_cluster" not in self.adata.obs.columns:
            self.ensure_clip_clustering()

        clusters = sorted(self.adata.obs["clip_cluster"].unique())
        profiles = [self.cluster_profile(cl) for cl in clusters]
        return "\n\n".join(profiles)

    def selection_summary(self, cell_ids):
        """Return summary for an arbitrary list of *cell_ids*.

        Behaviour:
        1. Build mask for the given cells.
        2. If the selection size is small (<= *recluster_threshold*), treat as
           one group and return a single profile.
        3. If larger, perform Leiden clustering *within the selection* on the
           CLIP space and generate a profile for each sub-cluster.
        """

        # ---------------- Parameters ----------------
        recluster_threshold = 300  # if selection larger than this, recluster
        resolution = 1.0  # leiden resolution for selection clustering
        top_genes_n = 20

        import numpy as np
        import scanpy as sc

        if self.adata is None:
            raise ValueError("AnnData not loaded.")

        # Build mask for selected cells
        selected_set = set(cell_ids)
        mask = self.adata.obs_names.isin(selected_set)

        n_sel = int(mask.sum())
        if n_sel == 0:
            return "(No selected cells found)"

        # Helper to build profile for arbitrary mask
        def _profile(mask_bool, label):
            n_cells = int(mask_bool.sum())
            if n_cells == 0:
                return ""
            lines = [f"{label} (n={n_cells})"]

            from pandas.api.types import is_numeric_dtype, is_bool_dtype

            for col in self.adata.obs.columns:
                if col in {"clip_cluster", "marked", "queried", "queried_ever"}:
                    continue
                series = self.adata.obs.loc[mask_bool, col]
                if is_bool_dtype(series):
                    summary = self.summarise_numeric(series)
                elif is_numeric_dtype(series):
                    summary = self.summarise_numeric(series)
                else:
                    summary = self.summarise_categorical(series, max_levels=8)
                lines.append(f"- {col}: {summary}")

            gene_list = self.top_genes(mask_bool, top_n=top_genes_n)
            gene_str = ", ".join(g for g, _ in gene_list)
            lines.append(f"- top_genes: {gene_str}")
            return "\n".join(lines)

        # Case 1: treat as one group
        if n_sel <= recluster_threshold:
            return _profile(mask, "Selection")

        # Case 2: recluster within selection
        adata_sel = self.adata[mask].copy()
        if "X_clip" not in adata_sel.obsm:
            # Copy slice of X_clip
            adata_sel.obsm["X_clip"] = self.adata.obsm["X_clip"][mask]

        sc.pp.neighbors(adata_sel, use_rep="X_clip")
        sc.tl.leiden(adata_sel, resolution=resolution, key_added="sel_leiden")

        # Map back to original adata indices
        sel_labels = adata_sel.obs["sel_leiden"].copy()
        profiles = []
        for lbl in sorted(sel_labels.unique()):
            mask_sub = np.zeros(len(self.adata), dtype=bool)
            # indices of selected cells for this label
            sub_idx = sel_labels[sel_labels == lbl].index
            mask_sub[self.adata.obs_names.isin(sub_idx)] = True
            profiles.append(_profile(mask_sub, f"Selection cluster {lbl}"))

        return "\n\n".join(profiles)

    # ------------------------------------------------------------------
    # Gene expression helper – chunk 5
    # ------------------------------------------------------------------
    def top_genes(self, cell_mask, top_n: int = 20):
        """Return the *top_n* genes by mean expression in the masked cells.

        Parameters
        ----------
        cell_mask
            Boolean mask (length n_obs) or list/array of indices specifying
            the cell subset of interest.
        top_n
            How many genes to return.

        Returns
        -------
        List[tuple[str, float]]
            Gene names (from ``adata.var_names``) with their mean expression
            values (unnormalised, whatever scale is stored in `adata.X`).
        """

        import numpy as np
        from scipy import sparse

        if self.adata is None:
            raise ValueError("AnnData not loaded.")

        # Ensure mask is a boolean ndarray
        mask_bool = np.asarray(cell_mask)
        if mask_bool.dtype != bool:
            mask_bool = mask_bool.astype(bool)

        X_sub = self.adata.X[mask_bool]

        # Compute mean expression per gene
        if sparse.issparse(X_sub):
            gene_means = np.asarray(X_sub.mean(axis=0)).ravel()
        else:
            gene_means = X_sub.mean(axis=0)

        # Identify indices of top genes
        top_idx = np.argsort(-gene_means)[:top_n]
        top_ids = self.adata.var_names[top_idx]
        top_values = gene_means[top_idx]

        # --------------------------------------------------
        # Map gene IDs -> human-readable symbols when possible
        # --------------------------------------------------
        id_to_symbol = None

        # 1) If AnnData.var carries a symbol column use it directly
        for col in ("gene_name", "gene_symbol", "symbol", "gene", "GeneSymbol"):
            if col in self.adata.var.columns:
                id_to_symbol = self.adata.var[col].to_dict()
                break

        # 2) Otherwise try the ClipModelHolder mapping (symbol ➜ Ensembl) reversed
        if id_to_symbol is None and hasattr(self._am, "clip_model") and self._am.clip_model is not None:
            mapping = getattr(self._am.clip_model, "gene_mapping_dict", None)
            if mapping:
                # mapping is symbol -> Ensembl; invert
                id_to_symbol = {ensembl: symbol for symbol, ensembl in mapping.items()}

        # 3) Fallback – identity mapping
        if id_to_symbol is None:
            id_to_symbol = {}

        symbols = [id_to_symbol.get(gid, gid) for gid in top_ids]

        return list(zip(symbols, top_values.tolist())) 