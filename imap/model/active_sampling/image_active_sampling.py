import numpy as np
import torch
from imap.utils.utils import generate_image_meshgrid, get_ji_pixels_from_meshgrid
from imap.data.data_batch import DataBatch


class ImageActiveSampling:
    def __init__(self, camera, points_per_frame, grid_size=(8, 8)):
        self._grid_width = grid_size[0]
        self._grid_height = grid_size[1]
        self._image_width = camera.image_width
        self._image_height = camera.image_height
        self._points_per_frame = points_per_frame

        x, y = generate_image_meshgrid(self._image_width, self._image_height, scale=1)
        self._sampling_pixels = get_ji_pixels_from_meshgrid(x, y)
        self._random_generator = np.random.default_rng()

        self._grid_pattern = self.generate_grid_pattern().cuda()
        self._default_pixel_weights = np.ones(self._image_width * self._image_height, dtype=np.float32)

        self._grid_masks = self.get_grid_masks()

    def sample_batch(self, state, pixel_weights):
        y, x = self.sample_pixels(pixel_weights)
        return DataBatch(np.array([x, y], dtype=np.float32).T,
                         state.frame.color_image[y, x],
                         state.frame.depth_image[y, x],
                         state.get_matrix_position())

    def sample_pixels(self, p):
        """
        :param p: probabilities.shape = (self._image_width * self._image_height)
        :return: i(y), j(x) coordinates
        """
        pixels = self._random_generator.choice(self._sampling_pixels,
                                               self._points_per_frame, p=p, axis=0, shuffle=False)
        return pixels[:, 1], pixels[:, 0]

    def estimate_pixels_weights(self, pixels, losses, prior_probs=None):
        """
        Estimate new weights for image pixels based on their losses
        :param pixels: (x, y)
        :param losses:
        :param prior_probs:
        :return:
        """
        with torch.no_grad():
            for grid_square, mask in zip(self._grid_pattern, self._grid_masks):
                upper_left = grid_square[0]
                lower_right = grid_square[1]
                sampled_pixels_mask = torch.logical_and(
                    torch.all(pixels >= upper_left, dim=1),
                    torch.all(pixels <= lower_right, dim=1))

                average_region_loss = torch.mean(losses[sampled_pixels_mask])
                if torch.isnan(average_region_loss):
                    self._default_pixel_weights[mask] = 0
                else:
                    self._default_pixel_weights[mask] = average_region_loss.item()

            if prior_probs is not None:
                self._default_pixel_weights *= prior_probs

            self._default_pixel_weights /= np.sum(self._default_pixel_weights)
            return self._default_pixel_weights

    def generate_grid_pattern(self):
        """
        :return: output.shape = (self._grid_width * self._grid_height, 2, 2)
        """
        grid = []
        dx = self._image_width // self._grid_width
        dy = self._image_height // self._grid_height

        x = np.arange(self._image_width // dx) * dx + dx
        y = np.arange(self._image_height // dy) * dy + dy
        for i in range(self._image_height // dy):
            square = []
            for j in range(self._image_width // dx):
                square.append([])
                square[-1].append([x[j] - dx, y[i] - dy])   # upper left
                square[-1].append([x[j], y[i]])             # lower right

            grid.append(square)
        return torch.tensor(grid).reshape((-1, 4, 2))

    def get_grid_masks(self):
        masks = []
        for grid_square in self._grid_pattern:
            upper_left = grid_square[0].cpu().numpy()
            lower_right = grid_square[1].cpu().numpy()
            all_pixels_mask = np.logical_and(
                np.all(self._sampling_pixels >= upper_left, axis=1),
                np.all(self._sampling_pixels <= lower_right, axis=1))
            masks.append(all_pixels_mask)
        return masks


