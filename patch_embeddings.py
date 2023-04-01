import torch


class PatchEmbedding(torch.nn.Module):
    """
    Extract patches from an image and linearly project them to vectors of size <embedding_dim>
    """
    def __init__(self, in_channels, patch_size, embedding_dim):
        super().__init__()
        self.patcher = torch.nn.Conv2d(in_channels=in_channels,
                                       out_channels=embedding_dim,
                                       kernel_size=patch_size,
                                       stride=patch_size,
                                       padding=0)
        self.flatter = torch.nn.Flatten(start_dim=2)

    def forward(self, x):
        x = self.patcher(x)
        x = self.flatter(x)
        x = torch.permute(x, (0, 2, 1))
        return x
