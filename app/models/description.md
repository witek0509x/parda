# AnnDataModel Documentation

## Overview

The `AnnDataModel` class provides functionality for loading single-cell RNA sequencing (scRNA-seq) data from h5ad files and calculating scVI embeddings using a pretrained model. This is a core component of the scRNA-seq Explorer application, enabling dimensionality reduction and visualization of high-dimensional gene expression data.

## Methods

### `__init__()`

Initializes a new AnnDataModel instance with no data loaded yet. The pretrained model path is set to "/home/wojciech/private/parda_v2/model".

### `load_data(file_path)`

Loads an h5ad file from the specified path into the AnnDataModel.

**Parameters:**
- `file_path` (str): Path to the h5ad file to load.

**Returns:**
- `bool`: True if the file was loaded successfully.

**Raises:**
- `FileNotFoundError`: If the file does not exist.
- `ValueError`: If there is an error loading the file.

### `calculate_scvi_embeddings()`

Calculates scVI embeddings for the loaded data using a pretrained model. This method:
1. Adds required metadata if missing ('n_counts' and 'batch')
2. Prepares the AnnData object for query against the pretrained model
3. Loads the query data with the scVI model
4. Computes the latent representation
5. Stores the embeddings in `adata.obsm["X_scvi"]`

**Returns:**
- `bool`: True if embeddings were calculated successfully.

**Raises:**
- `ValueError`: If no data is loaded or there is an error calculating embeddings.

### `get_data()`

Retrieves the current AnnData object with calculated embeddings.

**Returns:**
- `anndata.AnnData`: The processed AnnData object.

**Raises:**
- `ValueError`: If no data is loaded.

### `get_cells_metadata_by_id(cell_ids, max_return=100)`

Retrieves metadata for specified cells by their IDs.

**Parameters:**
- `cell_ids` (List[str]): List of cell IDs to retrieve metadata for.
- `max_return` (int, optional): Maximum number of cells to return metadata for. If the input list is longer, cells will be randomly sampled. Default is 100.

**Returns:**
- `str`: CSV-formatted string containing the metadata for the requested cells.

**Raises:**
- `ValueError`: If no data is loaded.

## Requirements and Assumptions

1. **Pretrained scVI Model**: Assumes a pretrained scVI model is available at the specified path.

2. **Data Compatibility**: The h5ad files are assumed to contain gene expression data that is compatible with the pretrained model. The scVI query functionality handles gene mapping between the dataset and model.

3. **Required Metadata**: If not present, the model adds:
   - `n_counts`: Sum of gene expression counts per cell
   - `batch`: Batch identifier (set to "unassigned" if not provided)

4. **Dependencies**: Requires scanpy, scvi-tools, and numpy packages.

## Usage Example

```python
# Create an instance of AnnDataModel
model = AnnDataModel()

# Load h5ad file
model.load_data("/path/to/data.h5ad")

# Calculate scVI embeddings
model.calculate_scvi_embeddings()

# Get the processed data with embeddings
adata = model.get_data()

# Now you can use the embeddings for visualization or analysis
# For example, computing UMAP:
import scanpy as sc
sc.pp.neighbors(adata, use_rep='X_scvi')
sc.tl.umap(adata)

# Get metadata for specific cells
metadata_csv = model.get_cells_metadata_by_id(["cell1", "cell2", "cell3"], max_return=10)
print(metadata_csv)
```

## Technical Details

### scVI Embeddings

The scVI (single-cell Variational Inference) model is a deep generative model for scRNA-seq data that learns a low-dimensional representation of the high-dimensional gene expression space. The embeddings are calculated using a pretrained model, which has been trained on a reference dataset of mouse (mus musculus) data.

The embedding generation process uses scVI's reference mapping capability, which allows new datasets to be projected into the same latent space as the reference data, enabling integration and comparison of multiple datasets.

### Latent Space Structure

The latent representation is stored in the `obsm` attribute of the AnnData object with the key "X_scvi". These embeddings can be used for:

1. Dimensionality reduction and visualization
2. Cell type classification
3. Trajectory inference
4. Batch correction
5. Data integration

### Performance Considerations

- The embedding calculation process may be computationally intensive for large datasets.
- Memory usage scales with the number of cells in the dataset.
- The current implementation works with CPU only, but can be extended to use GPU acceleration.

# ChatAssistant Documentation

## Overview

The `ChatAssistant` class provides an AI-powered interface for interacting with single-cell RNA sequencing data. It wraps the OpenAI API to provide a conversational experience where users can query and analyze cell data.

## Methods

### `__init__(api_key, query_function, anndata_model=None)`

Initializes a new ChatAssistant instance.

**Parameters:**
- `api_key` (str): OpenAI API key for authentication.
- `query_function` (Callable): Function that takes a query string and returns a list of cell data dictionaries.
- `anndata_model` (AnnDataModel, optional): Instance of AnnDataModel for accessing cell metadata.

### `initialize(system_prompt=None)`

Initializes the conversation with a system prompt.

**Parameters:**
- `system_prompt` (str, optional): Custom system prompt. If None, a default prompt about scRNA-seq data analysis is used.

**Returns:**
- `bool`: True if initialization was successful.

### `send_message(text, cell_ids=None)`

Sends a message to the AI assistant and receives a response.

**Parameters:**
- `text` (str): The message text to send.
- `cell_ids` (List[str], optional): List of cell IDs to include with the message.

**Returns:**
- `str`: The assistant's response.

### `get_conversation_history(formatted=True)`

Retrieves the conversation history.

**Parameters:**
- `formatted` (bool, optional): If True, returns a nicely formatted string. If False, returns the raw message objects.

**Returns:**
- `str` or `List[Dict]`: The conversation history.

## Features

### Cell ID Integration

When cell IDs are provided with a message:
1. If an AnnDataModel is available, the assistant retrieves metadata for those cells (up to max_return)
2. The metadata is included in the message to the AI in CSV format
3. This gives the AI context about the actual cells being discussed

### Function Calling

The assistant supports function calling to query cells:
1. When the AI determines it needs to find cells based on a description, it can call the `query_cells` function
2. The function call is processed using the provided query_function
3. Results are returned to the AI to incorporate into its response

### Conversation Management

The assistant maintains the entire conversation history:
1. User messages (including cell metadata when provided)
2. AI responses
3. Function calls and their results
4. Can be retrieved as raw data or in a formatted string

## Usage Example

```python
from app.models.anndata_model import AnnDataModel
from app.models.chat_assistant import ChatAssistant

# Setup AnnData model
anndata_model = AnnDataModel()
anndata_model.load_data("path/to/data.h5ad")
anndata_model.calculate_scvi_embeddings()

# Define query function
def query_cells(query_text):
    # Implementation to find cells based on text query
    # Return list of cell metadata dictionaries
    pass

# Initialize assistant
assistant = ChatAssistant(
    api_key="your-openai-api-key",
    query_function=query_cells,
    anndata_model=anndata_model
)
assistant.initialize()

# Send messages
response = assistant.send_message("What cell types are in this dataset?")
print(response)

# Send message with cell IDs
cell_ids = ["cell1", "cell2", "cell3"]
response = assistant.send_message("Tell me about these cells", cell_ids)
print(response)

# Get conversation history
history = assistant.get_conversation_history()
print(history)
``` 