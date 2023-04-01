import torch

from layer_normalization import LayerNormalization
from mlp_layer import MLPLayer
from multi_headed_attention import MultiHeadedAttention


class EncoderLayer(torch.nn.Module):
    """
    Implements one ViT encoder layer consisting of: a MHA module, followed by a LayerNormalization module,
    followed by an MLP module, and finally another LayerNormalization layer. Also, a residual connection is used
    """
    def __init__(self, embed_size, num_heads, mlp_hidden_size, dropout_prob):
        super().__init__()
        self.attention = MultiHeadedAttention(embed_size=embed_size, num_heads=num_heads, dropout_prob=dropout_prob)
        self.norm1 = LayerNormalization(parameters_shape=[embed_size])
        self.mlp = MLPLayer(embed_size=embed_size, hidden_size=mlp_hidden_size, dropout_prob=dropout_prob)
        self.norm2 = LayerNormalization(parameters_shape=[embed_size])

    def forward(self, x):
        # first part
        residual_x = x
        x = self.attention(x)
        x = self.norm1(x + residual_x)
        # second part
        residual_x = x
        x = self.mlp(x)
        x = self.norm2(x + residual_x)
        return x
