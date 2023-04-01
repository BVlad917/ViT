import torch


class PositionalEncoding(torch.nn.Module):
    """
    Implement the learnable positional encoding layer from the ViT paper. One learnable positional
    encoding for each patch + one for the classification token.
    """
    def __init__(self, num_patches, dropout_prob, embedding_dim):
        super().__init__()
        self.positional_encoding = torch.nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = x + self.positional_encoding  # broadcasting to use the same encoding for the entire batch
        x = self.dropout(x)
        return x