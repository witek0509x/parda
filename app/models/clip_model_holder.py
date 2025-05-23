import torch
import pickle
import os
import numpy as np
import pandas as pd
from typing import Dict, List
import sys

# Add scRNA to the path to import GenomicsCLIP
from scRNA.src.experiment.clip import GenomicsCLIP
from transformers import AutoTokenizer

class ClipModelHolder:
    def __init__(self):
        # Gene tokenization paths
        self.model_path = "/home/wojciech/private/parda_v2/data/mouse/weights"
        self.gene_mapping_file = os.path.join(self.model_path, "mouse-Geneformer/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl")
        self.gene_median_file = os.path.join(self.model_path, "mouse-Geneformer/mouse_gene_median_dictionary.pkl")
        self.token_dictionary_file = os.path.join(self.model_path, "mouse-Geneformer/MLM-re_token_dictionary_v1.pkl")
        
        # CLIP model parameters from config
        self.cell_vocab_size = 57000
        self.max_cell_tokens = 1200
        self.cell_embed_dim = 512
        self.cell_transformer_heads = 8
        self.cell_transformer_layers = 4
        self.text_model_name = "google-bert/bert-base-cased"
        self.max_text_tokens = 128
        self.text_proj_dim = 256
        self.projection_dim = 256
        self.dropout = 0.1
        
        # Data processing parameters
        self.target_sum = 10000
        self.chunk_size = 512
        
        # Device for model computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.load_tokenization_model()
        self.load_clip_model()
        
    def load_tokenization_model(self):
        """Load tokenization dictionaries for gene expression data"""
        with open(self.gene_mapping_file, "rb") as f:
            self.gene_mapping_dict = pickle.load(f)
        with open(self.gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)
        with open(self.token_dictionary_file, "rb") as f:
            self.token_dict = pickle.load(f)
            
        self.gene_keys = list(self.gene_median_dict.keys())
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))
        
        # Text tokenizer for queries
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        
    def load_clip_model(self):
        """Load the trained GenomicsCLIP model"""
        # Initialize model with config parameters
        self.clip_model = GenomicsCLIP(
            cell_vocab_size=self.cell_vocab_size,
            max_cell_tokens=self.max_cell_tokens,
            cell_embed_dim=self.cell_embed_dim,
            cell_transformer_heads=self.cell_transformer_heads,
            cell_transformer_layers=self.cell_transformer_layers,
            text_model_name=self.text_model_name,
            max_text_tokens=self.max_text_tokens,
            text_proj_dim=self.text_proj_dim,
            projection_dim=self.projection_dim,
            dropout=self.dropout,
            device=self.device
        )
        
        # Load saved weights
        weights_path = "/home/wojciech/private/parda_v2/model/model-weights"
        if os.path.exists(weights_path):
            print(f"Loading CLIP model weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.clip_model.load_state_dict(checkpoint["model_state_dict"])
            print(f"CLIP model loaded successfully (trained for {checkpoint['step']} steps)")
        else:
            print(f"Warning: Could not find model weights at {weights_path}")
            
        # Set model to evaluation mode
        self.clip_model.eval()
        
    def rank_genes(self, gene_vector, gene_tokens, gene_names):
        """Rank genes by expression level for tokenization"""
        sorted_indices = np.argsort(-gene_vector)[:2048]
        tokens = gene_tokens[sorted_indices]
        gene_names = gene_names.values[sorted_indices]
        return pd.Series(gene_names, index=tokens)
    
    def tokenize_cell(self, gene_expression_matrix, obs, var):
        """Tokenize cells for input to the CLIP model"""
        key_to_use = "gene_name" if "gene_name" in var.columns else "feature_name"
        ensemble_ids = var[key_to_use].map(self.gene_mapping_dict)
        
        coding_miRNA_loc = np.where([self.genelist_dict.get(i, False) for i in ensemble_ids])[0]
        norm_factor_vector = np.array([self.gene_median_dict[i] for i in ensemble_ids[coding_miRNA_loc]])
        coding_miRNA_ids = ensemble_ids.iloc[coding_miRNA_loc]
        coding_miRNA_tokens = np.array([self.token_dict[i] for i in coding_miRNA_ids])
        
        tokenized_cells = []
        positional_encodings = []
        
        for i in range(0, gene_expression_matrix.shape[0], self.chunk_size):
            idx = slice(i, i + self.chunk_size)
            
            col_to_use = "n_genes" if "n_genes" in obs.columns else "nFeature_RNA"
            if col_to_use not in obs.columns:
                n_counts = np.ones(gene_expression_matrix[idx].shape[0]) * 2500
            else:
                n_counts = obs[col_to_use].values[idx]
                
            X_view = gene_expression_matrix[idx][:, coding_miRNA_loc]
            X_norm = X_view / n_counts[:, np.newaxis] * self.target_sum / norm_factor_vector
            
            for j in range(X_norm.shape[0]):
                if isinstance(X_norm, np.ndarray):
                    cell_data = X_norm[j]
                    indices = np.nonzero(cell_data)[0]
                    values = cell_data[indices]
                else:  # Sparse matrix
                    cell_data = X_norm[j].toarray().flatten()
                    indices = np.nonzero(cell_data)[0]
                    values = cell_data[indices]
                
                tokenized_cell = self.rank_genes(
                    values,
                    coding_miRNA_tokens[indices],
                    var[key_to_use]
                )
                tokenized_cells.append(tokenized_cell)
                
                # Create tensor for model input
                positional_encoding = torch.tensor(
                    tokenized_cell.index.values,
                    dtype=torch.int32,
                )
                
                # Truncate or pad to max_cell_tokens
                cell_tokens = torch.zeros(self.max_cell_tokens, dtype=torch.int32)
                actual_length = min(len(positional_encoding), self.max_cell_tokens)
                cell_tokens[:actual_length] = positional_encoding[:actual_length]
                positional_encodings.append(cell_tokens)
                
        return tokenized_cells, positional_encodings
        
    def calculate_embeddings(self, adata):
        """Calculate embeddings using the CLIP model from AnnData object"""
        if adata is None:
            raise ValueError("No AnnData object provided")
            
        # Get gene expression matrix
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        
        # Tokenize cells
        print(f"Tokenizing {adata.n_obs} cells for CLIP embedding...")
        tokenized_cells, positional_encodings = self.tokenize_cell(X, adata.obs, adata.var)
        print(f"Tokenized {len(tokenized_cells)} cells")
        
        # Convert to tensor batch
        cell_tokens_batch = torch.stack(positional_encodings).to(self.device)
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 32
        embeddings = []
        
        print(f"Calculating CLIP embeddings using real model...")
        with torch.no_grad():
            for i in range(0, len(cell_tokens_batch), batch_size):
                batch = cell_tokens_batch[i:i+batch_size]
                
                # Get embeddings from cell encoder
                batch_embeddings = self.clip_model.encode_cells(batch)
                # Project to joint space
                batch_embeddings = self.clip_model.cell_proj(batch_embeddings)
                # Normalize
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
                
                embeddings.append(batch_embeddings.cpu().numpy())
                
        # Concatenate all batch embeddings
        embeddings = np.vstack(embeddings)
        print(f"Generated embeddings with shape {embeddings.shape}")
        
        # Store embeddings in adata object
        adata.obsm["X_clip"] = embeddings
        
        return embeddings
    
    def query_embeddings(self, query_text, normalize=True):
        """Generate embeddings for a text query to use for retrieval"""
        # Tokenize text
        text_tokens = self.text_tokenizer(
            query_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_tokens
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            # Get text features
            text_features = self.clip_model.encode_text(text_tokens)
            # Project to joint space
            text_embeddings = self.clip_model.text_proj(text_features)
            
            # Normalize if requested
            if normalize:
                text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
                
        return text_embeddings.cpu().numpy()

