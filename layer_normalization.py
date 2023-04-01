import torch


class LayerNormalization(torch.nn.Module):
    """
    Implementation of the layer normalization module used in the ViT paper. Finds the mean and variance of
    each word (i.e., image patch) and normalizes each patch. Then, add a learnable standard deviation and mean.
    Alternatively, use PyTorch's torch.nn.LayerNorm.
    """
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(parameters_shape))  # (embed_size)
        self.beta = torch.nn.Parameter(torch.zeros(parameters_shape))  # (embed_size)

    def forward(self, x):
        # x shape: (bs, seq_len, embed_size)
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]  # [-1] in our case
        mean = x.mean(dim=dims, keepdim=True)  # (bs, seq_len, 1)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)  # (bs, seq_len, 1)
        std = (var + self.eps).sqrt()  # (bs, seq_len, 1)
        y = (x - mean) / std  # (bs, seq_len, embed_size)
        out = self.gamma * y + self.beta  # (bs, seq_len, embed_size)
        return out
