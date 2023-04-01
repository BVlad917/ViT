import torch


class ClassToken(torch.nn.Module):
    """
    Learnable class token used for classifying an input image. Used in the ViT paper and inspired from the BERT
    paper. The classification token is repeated across the entire batch such that the same learnable parameter
    can be updated with backpropagation by the entire batch.
    """
    def __init__(self, batch_size, embedding_dim):
        super().__init__()
        self.batch_size = batch_size
        self.class_tokens = torch.nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, x):
        class_tokens = self.class_tokens.expand(self.batch_size, -1, -1)
        x = torch.cat([x, class_tokens], dim=1)
        return x
