import torch


# not used in our case currently
# treat more like a pseudocode; search better implementation if you want to use this
class PositionalMasking(torch.nn.Module):
    """
    Randomly mask patches from a input image. Used in the ViT paper and reported to improve ImageNet accuracy
    by up to 2%.
    """
    def __init__(self, batch_size, max_seq_len, embed_size):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.embed_size = embed_size
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, embed_size))  # (1, 1, embed_size)
        self.bool_masked_pos = torch.zeros((max_seq_len, ))  # (max_seq_len)

    def forward(self, x):
        # x shape: (batch_size, num_patches, embed_dim)
        pos_masked_indices = torch.multinomial(torch.ones(self.max_seq_len), num_samples=3, replacement=False)
        self.bool_masked_pos[pos_masked_indices] = 1.  # (max_seq_len)
        # mask_tokens shape: (batch_size, max_seq_len, embed_size)
        mask_tokens = self.mask_token.expand(self.batch_size, self.max_seq_len, -1)
        masked_positions = self.bool_masked_pos.unsqueeze(-1)  # (max_seq_len, 1)
        x = x * (1.0 - masked_positions) + mask_tokens * masked_positions
        return x
