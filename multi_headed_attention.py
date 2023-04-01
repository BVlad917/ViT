import torch
import numpy as np
import torch.nn.functional as F


class MultiHeadedAttention(torch.nn.Module):
    """
    Implementation of the dot product Multi-Headed attention mechanism from the original Transformer paper
    """
    def __init__(self, embed_size, num_heads, dropout_prob):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = self.embed_size // self.num_heads
        self.qkv_layer = torch.nn.Linear(embed_size, 3 * embed_size)
        self.linear_layer = torch.nn.Linear(embed_size, embed_size)
        self.out_dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x, mask=None):
        # x shape: (bs, seq_len, embed_size)
        batch_size, max_sequence_length, embed_size = x.size()
        qkv = self.qkv_layer(x)  # qkv shape: (bs, seq_len, 3 * embed_size)
        # qkv shape: (bs, seq_len, num_heads, 3 * head_size)
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_size)
        # qkv shape: (bs, num_heads, seq_len, 3 * head_size)
        qkv = qkv.permute(0, 2, 1, 3)
        # q, k, and v shapes: (bs, num_heads, seq_len, head_size)
        q, k, v = qkv.chunk(3, dim=-1)
        # values and attention shapes: (bs, num_heads, seq_len, head_size)
        values, attention = self.scaled_dot_product(q, k, v, mask)
        # values: (bs, seq_len, embed_size)
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_size)
        out = self.linear_layer(values)  # values: (bs, seq_len, embed_size)
        out = self.out_dropout(out)  # values: (bs, seq_len, embed_size)
        return out

    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        # NLP-inspired dot product attention; ignore the mask, not used in vision attention
        # q, k, and v shapes: (bs, num_heads, seq_len, head_size)
        d_k = q.size()[-1]  # d_k = head_size
        scaled = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)  # (bs, num_heads, seq_len, seq_len)
        if mask is not None:
            scaled += mask  # (bs, num_heads, seq_len, seq_len)
        attention = F.softmax(scaled, dim=-1)  # (bs, num_heads, seq_len, seq_len)
        values = torch.matmul(attention, v)  # (bs, num_heads, seq_len, head_size)
        return values, attention
