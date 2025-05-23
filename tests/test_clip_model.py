import sys
import os
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np

# Add the parent directory to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.anndata_model import AnnDataModel
from app.models.clip_model_holder import ClipModelHolder

def test_clip_model():
    # Path to h5ad file
    file_path = "/home/wojciech/private/parda_v2/tests/data/85a74c34-f6b3-4069-860e-58b10bf39a96.h5ad"
    
    # Create an instance of AnnDataModel
    model = AnnDataModel()
    
    # Load the h5ad file
    print("Loading h5ad file...")
    model.load_data(file_path)
    print("Data loaded successfully.")
    
    # Test direct ClipModelHolder instance
    print("Testing direct ClipModelHolder initialization...")
    clip_model = ClipModelHolder()
    print("ClipModelHolder initialized successfully.")
    
    # Generate embeddings using the clip model directly
    print("Calculating CLIP embeddings directly...")
    adata = model.get_data()
    clip_embeddings = clip_model.calculate_embeddings(adata)
    print(f"CLIP embeddings generated with shape: {clip_embeddings.shape}")
    
    # Test embeddings via AnnDataModel
    print("Calculating CLIP embeddings via AnnDataModel...")
    model.calculate_clip_embeddings()
    print("CLIP embeddings calculated successfully.")
    
    # Verify the embeddings are in the adata object
    adata = model.get_data()
    if "X_clip" in adata.obsm:
        print(f"CLIP embeddings found in adata.obsm with shape: {adata.obsm['X_clip'].shape}")
    else:
        print("Error: X_clip not found in adata.obsm")
        return
    
    # Calculate UMAP from CLIP embeddings
    print("Computing UMAP from CLIP embeddings...")
    # Create a temporary copy of adata for CLIP-based UMAP
    adata_clip = adata.copy()
    
    # Replace X_pca with X_clip for UMAP calculation
    adata_clip.obsm["X_pca"] = adata_clip.obsm["X_clip"]
    sc.pp.neighbors(adata_clip, use_rep='X_clip')
    sc.tl.umap(adata_clip)
    
    # Plot UMAP
    print("Plotting UMAP...")
    plt.figure(figsize=(10, 8))
    plt.scatter(adata_clip.obsm['X_umap'][:, 0], adata_clip.obsm['X_umap'][:, 1], s=1, alpha=0.5)
    plt.title('UMAP of CLIP embeddings')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    plt.show()

    print("Test completed successfully.")

if __name__ == "__main__":
    test_clip_model() 