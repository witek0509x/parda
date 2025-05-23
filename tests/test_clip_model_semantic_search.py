import sys
import os
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add the parent directory to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.anndata_model import AnnDataModel


def test_clip_semantic_search():
    # Path to h5ad file
    file_path = "/home/wojciech/private/parda_v2/tests/data/85a74c34-f6b3-4069-860e-58b10bf39a96.h5ad"

    # Create an instance of AnnDataModel
    model = AnnDataModel()

    # Load the h5ad file
    print("Loading h5ad file...")
    model.load_data(file_path)
    print("Data loaded successfully.")

    # Calculate scVI embeddings
    print("Calculating scVI embeddings...")
    model.calculate_scvi_embeddings()
    print("scVI embeddings calculated successfully.")

    # Calculate CLIP embeddings
    print("Calculating CLIP embeddings...")
    model.calculate_clip_embeddings()
    print("CLIP embeddings calculated successfully.")

    # Get the data with embeddings
    adata = model.get_data()

    # Define a list of semantic queries to test
    queries = [
        "T cells involved in immune response",
        "B cells producing antibodies",
        "Activated macrophages in inflammatory response",
        "Dendritic cells presenting antigens",
        "Stem cells with pluripotency markers"
    ]

    # Create a UMAP plot to visualize
    print("Computing UMAP from CLIP embeddings...")
    # Reduce dimensionality for visualization with PCA first
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(adata.obsm["X_clip"])

    # Then apply TSNE for final visualization
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=200)
    tsne_result = tsne.fit_transform(pca_result)

    # Store results for visualization
    adata.obsm["X_tsne"] = tsne_result

    # Perform semantic search with each query
    plt.figure(figsize=(20, 16))

    for i, query in enumerate(queries):
        # Create subplot
        plt.subplot(2, 3, i + 1)

        # Query cells
        print(f"\nQuerying: {query}")
        results = model.query_cells(query, top_k=10)

        # Print results
        print(f"Top matches for query: {query}")
        for j, result in enumerate(results):
            print(f"{j + 1}. Cell ID: {result['cell_id']}, Score: {result['similarity_score']:.4f}")
            if j < 3:  # Print some metadata for top 3 matches
                for key, value in result['metadata'].items():
                    if key in ['cell_type', 'cluster', 'condition', 'tissue']:
                        print(f"   {key}: {value}")
                print()

        # Get cell IDs of the results
        result_cell_ids = [result["cell_id"] for result in results]

        # Create mask for the results
        result_mask = np.zeros(adata.n_obs, dtype=bool)
        for cell_id in result_cell_ids:
            if cell_id in adata.obs_names:
                idx = np.where(adata.obs_names == cell_id)[0]
                if len(idx) > 0:
                    result_mask[idx[0]] = True

        # Plot all cells in gray
        plt.scatter(
            adata.obsm["X_tsne"][:, 0],
            adata.obsm["X_tsne"][:, 1],
            s=5,
            alpha=0.2,
            color='lightgray',
            label='All cells'
        )

        # Plot result cells in red
        plt.scatter(
            adata.obsm["X_tsne"][result_mask, 0],
            adata.obsm["X_tsne"][result_mask, 1],
            s=50,
            alpha=1.0,
            color='red',
            label='Query results'
        )

        plt.title(f"Query: {query}")
        plt.legend()
        plt.tight_layout()

    # Save the plot
    plt.show()

    print("\nTest completed successfully.")


if __name__ == "__main__":
    test_clip_semantic_search()