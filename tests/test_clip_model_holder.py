import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
print(os.path)

from app.models.clip_model_holder import ClipModelHolder

def create_mock_anndata():
    # Create a mock AnnData object with some cells
    n_obs = 100
    n_vars = 50
    X = np.random.normal(size=(n_obs, n_vars))
    cell_ids = [f"cell{i}" for i in range(n_obs)]
    gene_ids = [f"gene{i}" for i in range(n_vars)]
    
    gene_names = [f"GeneName{i}" for i in range(n_vars)]
    
    obs = pd.DataFrame({
        "cluster": np.random.choice(["T cells", "B cells", "Macrophages", "Dendritic cells"], size=n_obs),
        "condition": np.random.choice(["control", "treated"], size=n_obs),
        "n_genes": np.random.randint(100, 1000, size=n_obs)
    }, index=cell_ids)
    
    var = pd.DataFrame({
        "gene_type": np.random.choice(["protein_coding", "lincRNA", "miRNA"], size=n_vars),
        "gene_name": gene_names
    }, index=gene_ids)
    
    return sc.AnnData(X=X, obs=obs, var=var)

def test_clip_model_holder():
    print("Testing ClipModelHolder initialization...")
    clip_model = ClipModelHolder()
    print("ClipModelHolder initialized successfully.")
    
    print("Creating mock AnnData...")
    adata = create_mock_anndata()
    print(f"Mock AnnData created with shape: {adata.shape}")
    
    print("Testing tokenize_cell method...")
    tokenized_cells = clip_model.tokenize_cell(adata.X, adata.obs, adata.var)
    print(f"Tokenized {len(tokenized_cells)} cells")
    
    # Print sample of the first tokenized cell
    print("\nSample of first tokenized cell (top 5 genes):")
    if tokenized_cells and len(tokenized_cells) > 0:
        print(tokenized_cells[0].head())
    
    print("\nCalculating CLIP embeddings...")
    clip_embeddings = clip_model.calculate_embeddings(adata)
    print(f"CLIP embeddings calculated with shape: {clip_embeddings.shape}")
    
    # Verify the embeddings are in the adata object
    if "X_clip" in adata.obsm:
        print(f"CLIP embeddings found in adata.obsm with shape: {adata.obsm['X_clip'].shape}")
    else:
        print("Error: X_clip not found in adata.obsm")
        return
    
    # Visualize the embeddings using PCA
    print("\nVisualizing CLIP embeddings using PCA...")
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(adata.obsm["X_clip"])
    
    plt.figure(figsize=(10, 8))
    clusters = adata.obs["cluster"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    
    for i, cluster in enumerate(clusters):
        mask = adata.obs["cluster"] == cluster
        plt.scatter(
            pca_result[mask, 0], 
            pca_result[mask, 1],
            label=cluster,
            color=colors[i],
            s=10,
            alpha=0.7
        )
    
    plt.title('PCA of CLIP embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tests/clip_embeddings_pca.png', dpi=300)
    plt.show()
    
    print("\nTest completed successfully.")

if __name__ == "__main__":
    test_clip_model_holder() 