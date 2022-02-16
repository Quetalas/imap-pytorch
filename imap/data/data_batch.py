import torch


class DataBatch:
    def __init__(self, pixels, colors, depths, camera_position):
        """
        :param pixels: np.array([x, y], dtype=np.float32)
        :param colors: self._color_images[image_index, y, x]
        :param depths: self._depth_images[image_index, y, x]
        :param camera_position: self._positions[image_index]
        """
        self.pixels = pixels
        self.colors = colors
        self.depths = depths
        self.camera_position = camera_position
        self.check_batch()

    def to(self, device):
        return DataBatch(self.pixels.to(device),
                         self.colors.to(device),
                         self.depths.to(device),
                         self.camera_position.to(device)
                         )

    def torch_from_numpy(self):
        return DataBatch(torch.from_numpy(self.pixels),
                         torch.from_numpy(self.colors),
                         torch.from_numpy(self.depths),
                         self.camera_position
                         )

    def check_batch(self):
        assert len(self.pixels.shape) == 2 and self.pixels.shape[1] == 2
        assert len(self.colors.shape) == 2 and self.colors.shape[1] == 3
        assert len(self.depths.shape) == 1
        assert len(self.camera_position.shape) == 2 and self.camera_position.shape[0] == 4 and \
               self.camera_position.shape[1] == 4
