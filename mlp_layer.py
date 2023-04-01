import torch


class MLPLayer(torch.nn.Module):
    """
    Multi Layer Perceptron layer applied after the second layer normalization layer inside the ViT's encoder layer
    """
    def __init__(self, embed_size, hidden_size, dropout_prob):
        super().__init__()
        self.linear1 = torch.nn.Linear(embed_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, embed_size)
        self.relu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x):
        # x shape: (bs, seq_len, embed_size)
        x = self.dropout(self.relu(self.linear1(x)))  # (bs, seq_len, hidden_size)
        x = self.dropout(self.linear2(x))  # x shape: (bs, seq_len, embed_size)
        return x
