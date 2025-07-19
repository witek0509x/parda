import numpy as np
import pandas as pd
import scanpy as sc
import sys
import os
from openai import OpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dotenv
dotenv.load_dotenv()
from app.models.anndata_model import AnnDataModel
from app.utils.fact_checker import FactChecker


def create_deterministic_anndata():
    """Create AnnData with exactly 10 cells expressing GeneName0."""
    n_obs = 20
    n_vars = 5

    # Initialize expression matrix with zeros
    X = np.zeros((n_obs, n_vars), dtype=float)
    # Make first 10 cells express gene 0 (value = 1)
    X[:10, 0] = 1.0

    cell_ids = [f"cell{i}" for i in range(n_obs)]
    gene_ids = [f"gene{i}" for i in range(n_vars)]
    gene_names = [f"GeneName{i}" for i in range(n_vars)]

    obs = pd.DataFrame(index=cell_ids)
    var = pd.DataFrame({"gene_name": gene_names}, index=gene_ids)

    return sc.AnnData(X=X, obs=obs, var=var)


def fact_checker():
    adata = create_deterministic_anndata()
    model = AnnDataModel()
    model.load_data("/home/wojciech/private/parda_v2/tests/data/E2f7-knockout.h5ad")
    fc = FactChecker(model)
    api_key = os.getenv("OPENAI_KEY")
    if api_key:
        fc.set_client(OpenAI(api_key=api_key))
    return fc


def test_gene_expression_count_valid(fact_checker):
    claim = {
        "type": "gene_expression_count",
        "gene": "GeneName0",
        "count": 10,
    }
    assert fact_checker._validate_claim(claim) is True


def test_gene_expression_count_invalid(fact_checker):
    claim = {
        "type": "gene_expression_count",
        "gene": "GeneName0",
        "count": 5,
    }
    assert fact_checker._validate_claim(claim) is False 

if __name__ == "__main__":
    fc = fact_checker()
    print(fc.validate_response("""The cells you provided are annotated as "Basal stem" cells and are identified as tracheal epithelial cells from mouse (Mus musculus) cell culture. Their metadata does not indicate that they are T cells. T cells would typically have cell_type or ontology term IDs related to T lymphocytes (e.g., CL:0000084 for T cell) and would express T cell marker genes such as CD3D, CD3E, CD4, or CD8A. These cells do not show such annotations or markers in the provided metadata. Instead, they are basal stem cells of the tracheal epithelium, a tissue-resident stem cell population, not immune T cells. Therefore, based on the metadata and cell type ontology terms, these cells are not T cells. If you want, I can help check expression of classical T cell markers to further confirm."""))