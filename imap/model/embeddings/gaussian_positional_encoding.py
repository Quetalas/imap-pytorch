import torch
import torch.nn as nn


class GaussianPositionalEncoding(nn.Module):
    def __init__(self, sigma=25, encoding_dimension=93, use_only_sin=True):
        super().__init__()
        self._use_only_sin = use_only_sin
        self.encoding_dimension = encoding_dimension
        if not use_only_sin:
            assert encoding_dimension % 2 == 0  # support only even numbers
            encoding_dimension = encoding_dimension // 2
        self._b_encoding_matrix = nn.Linear(3, encoding_dimension, bias=False)
        nn.init.normal_(self._b_encoding_matrix.weight, 0, sigma)

    def forward(self, x):
        encodings = self._b_encoding_matrix(x)  # TODO: use independent neural layers for sin & cos
        if self._use_only_sin:
            return torch.sin(encodings)
        cos_encodings = torch.cos(encodings)
        sin_encodings = torch.sin(encodings)
        return torch.cat([cos_encodings, sin_encodings], dim=1)
