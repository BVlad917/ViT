import torch

from encoder_layer import EncoderLayer


class Encoder(torch.nn.Module):
    """
    Implements a ViT encoder. Consists of multiple EncoderLayer modules applied in sequence
    """
    def __init__(self, embed_size, num_heads, mlp_hidden_size, dropout_prob, num_layers):
        super().__init__()
        self.layers = torch.nn.Sequential(*[EncoderLayer(embed_size, num_heads, mlp_hidden_size, dropout_prob)
                                            for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x  # (bs, seq_len, embed_size)
