import scanpy as sc
import scvi
import os
import numpy as np
import random
from .clip_model_holder import ClipModelHolder
from app.utils.metadata_provider import MetadataProvider

class AnnDataModel:
    def __init__(self):
        self.adata = None
        self.model_path = "/home/wojciech/private/parda_v2/model"
        self.clip_model = None
        # Will hold MetadataProvider once embeddings are available
        self._metadata_provider = None

    # --------------------------------------------------
    # Expose metadata provider
    # --------------------------------------------------
    @property
    def metadata(self):
        """Return MetadataProvider if initialised, else None."""
        return self._metadata_provider

    def load_data(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            self.adata = sc.read_h5ad(file_path)
            return True
        except Exception as e:
            raise ValueError(f"Error loading h5ad file: {str(e)}")

    def calculate_scvi_embeddings(self):
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first.")

        try:
            print(f"Original adata shape: {self.adata.shape}")

            if "n_counts" not in self.adata.obs:
                self.adata.obs["n_counts"] = self.adata.X.sum(axis=1)
            if "batch" not in self.adata.obs:
                self.adata.obs["batch"] = "unassigned"

            # Prepare query
            scvi.model.SCVI.prepare_query_anndata(self.adata, self.model_path)
            vae_q = scvi.model.SCVI.load_query_data(self.adata, self.model_path)
            vae_q.is_trained = True

            # Latent representation
            latent = vae_q.get_latent_representation()
            self.adata.obsm["X_scvi"] = latent

            print(f"Final data shape after processing: {self.adata.shape}")
            return True
        except Exception as e:
            raise ValueError(f"Error calculating scVI embeddings: {str(e)}")

    def calculate_clip_embeddings(self):
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first.")

        try:
            print(f"Calculating CLIP embeddings for {self.adata.shape[0]} cells")

            # Initialize CLIP model if not already done
            if self.clip_model is None:
                self.clip_model = ClipModelHolder()

            # Calculate embeddings
            embeddings = self.clip_model.calculate_embeddings(self.adata)

            # --------------------------------------------------
            # Instantiate MetadataProvider after embeddings are ready
            # --------------------------------------------------
            self._metadata_provider = MetadataProvider(self)
            # Ensure clustering is ready for downstream usage
            self._metadata_provider.ensure_clip_clustering()

            print(f"CLIP embeddings calculated, shape: {embeddings.shape}")
            return True
        except Exception as e:
            raise ValueError(f"Error calculating CLIP embeddings: {str(e)}")

    def get_data(self):
        if self.adata is None:
            raise ValueError("No data loaded.")
        return self.adata

    # get_cells_metadata_by_id removed – the new MetadataProvider handles
    # selection summaries more comprehensively.

    def query_cells(self, query_text: str, top_k=5):
        """
        Query cells based on text similarity using the CLIP model.
        This performs semantic search to find cells most similar to the query text.

        Args:
            query_text: Natural language description of cells to search for
            top_k: Number of most similar cells to return

        Returns:
            List of dictionaries with cell information and similarity scores
        """
        if self.adata is None:
            return []

        # Initialize CLIP model if not already done
        if self.clip_model is None:
            self.clip_model = ClipModelHolder()

        # Ensure CLIP embeddings are calculated
        if "X_clip" not in self.adata.obsm:
            print("CLIP embeddings not found, calculating them now...")
            self.calculate_clip_embeddings()

        # Get embeddings for the query text
        print(f"Generating embeddings for query: {query_text}")
        query_embedding = self.clip_model.query_embeddings(query_text)

        # Get cell embeddings from the adata object
        cell_embeddings = self.adata.obsm["X_clip"]

        # Calculate cosine similarity between query and all cells
        # (dot product of normalized vectors = cosine similarity)
        similarity_scores = np.dot(cell_embeddings, query_embedding.T).flatten()

        # Get indices of top k most similar cells
        top_indices = np.argsort(-similarity_scores)[:top_k]
        top_scores = similarity_scores[top_indices]
        top_cell_ids = self.adata.obs_names[top_indices].tolist()

        # Mark queried cells in the observation dataframe
        self.adata.obs["queried"] = 0
        self.adata.obs.loc[top_cell_ids, "queried"] = 1
        # Also mark them in the 'queried_ever' column for persistent highlighting
        if "queried_ever" not in self.adata.obs:
            self.adata.obs["queried_ever"] = 0
        self.adata.obs.loc[top_cell_ids, "queried_ever"] = 1

        # Create a result list with cell info
        results = []
        for cell_id, score in zip(top_cell_ids, top_scores):
            cell_metadata = {}
            for col in self.adata.obs.columns:
                try:
                    cell_metadata[col] = str(self.adata.obs.loc[cell_id, col])
                except:
                    pass

            results.append({
                "cell_id": cell_id,
                "metadata": cell_metadata,
                "similarity_score": float(score)  # Convert to standard Python float for JSON serialization
            })

        return results

    def highlight_cells(self, query_text: str):
        if self.adata is None:
            return []

        # Initialize CLIP model if not already done
        if self.clip_model is None:
            self.clip_model = ClipModelHolder()

        # Ensure CLIP embeddings are calculated
        if "X_clip" not in self.adata.obsm:
            print("CLIP embeddings not found, calculating them now...")
            self.calculate_clip_embeddings()

        # Get embeddings for the query text
        print(f"Generating embeddings for query: {query_text}")
        query_embedding = self.clip_model.query_embeddings(query_text)

        # Get cell embeddings from the adata object
        cell_embeddings = self.adata.obsm["X_clip"]

        # Calculate cosine similarity between query and all cells
        # (dot product of normalized vectors = cosine similarity)
        similarity_scores = np.dot(cell_embeddings, query_embedding.T).flatten()

        # Initialize the "marked" column with zeros
        self.adata.obs["marked"] = similarity_scores
        return

    def prepare_additional_columns(self):
        if self.adata is None:
            raise ValueError("No data loaded.")
        self.adata.obs["marked"] = 0.0
        self.adata.obs['queried'] = 0.0
        self.adata.obs['queried_ever'] = 0.0


