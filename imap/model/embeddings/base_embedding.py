import torch


class BaseEmbedding(torch.nn.Module):
    def __init__(self, encoding_dimension):
        super().__init__()
        self.encoding_dimension = encoding_dimension

