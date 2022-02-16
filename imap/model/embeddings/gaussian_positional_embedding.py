import torch
import torch.nn as nn


class GaussianPositionalEmbedding(nn.Module):
    def __init__(self, sigma=25, encoding_dimension=93, use_only_sin=True, use_bias=False):
        super().__init__()
        self._use_only_sin = use_only_sin
        self.encoding_dimension = encoding_dimension

        self._sin_encoding_matrix = nn.Linear(3, encoding_dimension, bias=use_bias)
        nn.init.normal_(self._sin_encoding_matrix.weight, 0, sigma)
        if not self._use_only_sin:
            self._cos_encoding_matrix = nn.Linear(3, encoding_dimension, bias=use_bias)
            nn.init.normal_(self._cos_encoding_matrix.weight, 0, sigma)
            self.encoding_dimension *= 2

    def forward(self, x):
        sin_encodings = self._sin_encoding_matrix(x)
        if self._use_only_sin:
            return torch.sin(sin_encodings)

        cos_encodings = self._cos_encoding_matrix(x)
        cos = torch.cos(cos_encodings)
        sin = torch.sin(sin_encodings)
        return torch.cat([sin, cos], dim=1)
