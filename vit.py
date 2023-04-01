import torch

from class_token import ClassToken
from encoder import Encoder
from patch_embeddings import PatchEmbedding
from positional_encoding import PositionalEncoding


class ViT(torch.nn.Module):
    """
    Vision Transformer implementation. Unifies all the layers used in the ViT under one module
    """
    def __init__(self, batch_size, in_channels, patch_size, embedding_dim, num_patches, dropout_prob, num_heads,
                 mlp_hidden_size, num_layers, num_classes):
        super().__init__()
        # BATCH_SIZE, NUM_PATCHES, EMBEDDING_DIM
        self.patcher = PatchEmbedding(in_channels, patch_size, embedding_dim)
        self.class_token = ClassToken(batch_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(num_patches, dropout_prob, embedding_dim)
        self.encoder = Encoder(embedding_dim, num_heads, mlp_hidden_size, dropout_prob, num_layers)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)
        self.all_vit_blocks = torch.nn.Sequential(*[
            self.patcher,
            self.class_token,
            self.pos_encoding,
            self.encoder,
        ])

    def forward(self, x):
        encoder_output = self.all_vit_blocks(x)  # (bs, seq_len, embed_size)
        # run classifier layer on the CLS token only, so on shape (bs, embed_size)
        logits = self.classifier(encoder_output[:, 0])  # (bs, num_classes)
        return logits
