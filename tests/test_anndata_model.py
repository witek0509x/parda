import sys
import os
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np

# Add the parent directory to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path)
from app.models.anndata_model import AnnDataModel

def test_load_and_visualize():
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
    print("Embeddings calculated successfully.")
    
    # Get the data with embeddings
    adata = model.get_data()
    
    # Calculate UMAP from scVI embeddings
    print("Computing UMAP from scVI embeddings...")
    sc.pp.neighbors(adata, use_rep='X_scvi')
    sc.tl.umap(adata)
    
    # Plot UMAP
    print("Plotting UMAP...")
    plt.figure(figsize=(10, 8))
    plt.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], s=1, alpha=0.5)
    plt.title('UMAP of scVI embeddings')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig('tests/umap_scvi_embeddings.png', dpi=300)
    plt.show()
    
    print("Test completed successfully.")

if __name__ == "__main__":
    test_load_and_visualize() 