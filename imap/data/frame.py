import torch


class Frame:
    def __init__(self, image, depth, pixel_weights):
        assert len(image.shape) == 3
        assert len(depth.shape) == 2
        assert depth.shape == pixel_weights.shape
        self.color_image = image
        self.depth_image = depth
        self.pixel_weights = pixel_weights

    def get_pixel_probs(self):
        return self.pixel_weights.reshape(-1)

