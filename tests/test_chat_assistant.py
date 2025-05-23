import sys
import os
import pandas as pd
import numpy as np
import anndata as ad

# Add the parent directory to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.chat_assistant import ChatAssistant
from app.models.anndata_model import AnnDataModel

def mock_query_function(query_text):
    # Mock function that returns sample cells based on a query
    print(f"Mock query executed with text: {query_text}")
    return [
        {"cell_id": "cell1", "cluster": "T cells", "score": 0.95},
        {"cell_id": "cell2", "cluster": "B cells", "score": 0.85},
        {"cell_id": "cell3", "cluster": "Macrophages", "score": 0.75}
    ]

def create_mock_anndata():
    # Create a mock AnnData object with some cells
    n_obs = 100
    n_vars = 50
    X = np.random.normal(size=(n_obs, n_vars))
    cell_ids = [f"cell{i}" for i in range(n_obs)]
    gene_ids = [f"gene{i}" for i in range(n_vars)]
    
    obs = pd.DataFrame({
        "cluster": np.random.choice(["T cells", "B cells", "Macrophages", "Dendritic cells"], size=n_obs),
        "condition": np.random.choice(["control", "treated"], size=n_obs),
        "n_genes": np.random.randint(100, 1000, size=n_obs)
    }, index=cell_ids)
    
    var = pd.DataFrame({
        "gene_type": np.random.choice(["protein_coding", "lincRNA", "miRNA"], size=n_vars)
    }, index=gene_ids)
    
    return ad.AnnData(X=X, obs=obs, var=var)

def test_chat_assistant():
    # Replace with a valid API key for testing
    api_key = ""
    
    # Create a mock AnnData model
    model = AnnDataModel()
    model.adata = create_mock_anndata()
    
    # Initialize the assistant with the AnnData model
    assistant = ChatAssistant(api_key=api_key, anndata_model=model)
    assistant.initialize()
    
    # Send a message that doesn't require function calling but includes cell IDs
    print("Testing message with cell IDs:")
    response = assistant.send_message("Describe this cells?", ["cell1", "cell2", "cell3"])
    print(f"Response: {response}\n")
    
    # Send a message that might trigger function calling
    print("Testing message that might trigger function calling:")
    response = assistant.send_message("What cell types are present in this dataset")
    print(f"Response: {response}\n")
    
    # Send a message with too many cells to test the max_return feature
    many_cell_ids = [f"cell{i}" for i in range(30)]
    print("Testing message with many cell IDs:")
    response = assistant.send_message("Analyze these cells", many_cell_ids)
    print(f"Response: {response}\n")
    
    # Get conversation history
    print("Conversation history:")
    history = assistant.get_conversation_history()
    print(history)

if __name__ == "__main__":
    test_chat_assistant() 