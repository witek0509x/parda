import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from transformers import BertModel


class GenomicsCLIP(nn.Module):
    def __init__(
        self,
        # Genomics encoder config
        cell_vocab_size: int = 57000,
        max_cell_tokens: int = 1200,
        cell_embed_dim: int = 512,
        cell_transformer_heads: int = 8,
        cell_transformer_layers: int = 4,
        # Text encoder config
        text_model_name: str = "google-bert/bert-base-cased",
        max_text_tokens: int = 128,
        text_proj_dim: int = 256,
        # Projection config
        projection_dim: int = 256,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.max_cell_tokens = max_cell_tokens
        self.max_text_tokens = max_text_tokens

        # ============= Genomics Encoder =============
        self.cell_embedding = nn.Embedding(
            cell_vocab_size, cell_embed_dim, device=device
        )
        self.cell_pos_embedding = nn.Parameter(
            torch.randn(1, max_cell_tokens, cell_embed_dim, device=device)
        )

        cell_encoder_layers = TransformerEncoderLayer(
            d_model=cell_embed_dim,
            nhead=cell_transformer_heads,
            dim_feedforward=cell_embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            device=device,
        )
        self.cell_encoder = TransformerEncoder(
            cell_encoder_layers, cell_transformer_layers
        )

        # ============= Text Encoder =============
        self.text_encoder = BertModel.from_pretrained(text_model_name).to(device)

        # Freeze BERT
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # ============= Projection Heads =============
        self.cell_proj = nn.Sequential(
            nn.Linear(cell_embed_dim, projection_dim, device=device),
            nn.GELU(),
            nn.LayerNorm(projection_dim, device=device),
            nn.Linear(projection_dim, projection_dim, device=device),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(
                self.text_encoder.config.hidden_size, text_proj_dim, device=device
            ),
            nn.GELU(),
            nn.LayerNorm(text_proj_dim, device=device),
            nn.Linear(text_proj_dim, projection_dim, device=device),
        )

        # Temperature parameter
        self.logit_scale = nn.Parameter(
            torch.ones([], device=device)
            * torch.log(torch.tensor(1 / 0.07, device=device))
        )

        self.device = device

    def encode_cells(self, cell_tokens: torch.Tensor) -> torch.Tensor:
        """Process tokenized cell data through genomics encoder"""
        # cell_tokens shape: (batch_size, seq_len)
        x = self.cell_embedding(cell_tokens)  # (batch, seq, embed)
        x = x + self.cell_pos_embedding[:, : cell_tokens.size(1), :]

        # Generate padding mask
        padding_mask = cell_tokens == 0

        x = self.cell_encoder(x, src_key_padding_mask=padding_mask)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, embed)
        return x

    def encode_text(self, text_tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        """Process tokenized text through BERT"""
        outputs = self.text_encoder(
            input_ids=text_tokens["input_ids"],
            attention_mask=text_tokens["attention_mask"],
        )
        # Use [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, batch: dict[str, list]) -> tuple[torch.Tensor, torch.Tensor]:
        cell_tokens = batch["cell_tokens"].to(self.device)
        text_tokens = batch["input_ids"].to(self.device)
        attention_masks = batch["attention_mask"].to(self.device)
        # Encode both modalities
        # cell_tokens = torch.stack(
        #     [self.tokenize_cells(cell) for cell in batch["cell_data"]]
        # )
        # text_tokens, attention_masks = zip(
        #     *[self.tokenize_text(text) for text in batch["text"]]
        # )

        # text_tokens = torch.stack(text_tokens)
        # attention_masks = torch.stack(attention_masks)

        cell_features = self.encode_cells(cell_tokens)
        text_features = self.encode_text(
            {"input_ids": text_tokens, "attention_mask": attention_masks}
        )

        # Project to joint space
        cell_embeddings = self.cell_proj(cell_features)
        text_embeddings = self.text_proj(text_features)

        # Normalize features
        cell_embeddings = cell_embeddings / cell_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        # Cosine similarity with temperature
        logit_scale = self.logit_scale.exp()
        logits_per_cell = logit_scale * cell_embeddings @ text_embeddings.t()
        logits_per_text = logits_per_cell.t()

        return logits_per_cell, logits_per_text

    def predict_similarity_matrix(self, batch: dict[str, list]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, logits_per_text = self.forward(batch)
            similarity_matrix = logits_per_text / self.logit_scale.exp()

        return similarity_matrix

    def predict_best_matches(self, batch: dict[str, list]) -> torch.Tensor:
        similarity_matrix = self.predict_similarity_matrix(batch)
        return similarity_matrix.argmax(dim=1)

    def accuracy_paired_batch(self, batch: dict[str, list]) -> float:
        assert len(batch["cell_tokens"]) == len(
            batch["input_ids"]
        )  # here we assume the batch contains paired text-cells
        y_hat = self.predict_best_matches(batch)
        y_true = torch.arange(len(batch["cell_tokens"]), device=self.device)

        return self.accuracy(y_hat, y_true)

    @staticmethod
    def accuracy(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
        return (y_hat == y_true).float().mean().item()

    # def tokenize_text(self, text):
    #     """Tokenize raw text"""
    #     encoding = self.text_tokenizer(
    #         text,
    #         return_tensors="pt",
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_text_tokens,
    #     )
    #     return encoding["input_ids"][0].to(self.device), encoding["attention_mask"][
    #         0
    #     ].to(self.device)

    # def tokenize_cells(self, cell_data):
    #     """Tokenize raw cell data"""
    #     x, obs, var = cell_data
    #     _, tokenized_cells = self.cell_tokenizer.tokenize_single_cell(x, obs, var)
    #     positional_encoding = torch.tensor(
    #         tokenized_cells[0].fillna(0).astype(int).values,
    #         dtype=torch.int32,
    #         device=self.device,
    #     )

    #     # Pad/truncate
    #     cell_tokens = torch.zeros(
    #         self.max_cell_tokens, dtype=torch.int32, device=self.device
    #     )
    #     actual_length = min(len(positional_encoding), self.max_cell_tokens)
    #     cell_tokens[:actual_length] = positional_encoding[:actual_length]
    #     return cell_tokens



