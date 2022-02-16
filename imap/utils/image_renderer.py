import torch
import numpy as np
from tqdm.auto import tqdm

from imap.utils.utils import (
    generate_image_meshgrid,
    get_ji_pixels_from_meshgrid
)


class ImageRenderer:
    def __init__(self, camera, scale=1, batch_size=4096):
        self.height = camera.image_height
        self.width = camera.image_width
        self.mean = camera._color_mean
        self.std = camera._color_std
        self.scale = scale
        self.batch_size = batch_size

    def _generate_image_frame(self):
        x, y = generate_image_meshgrid(self.width, self.height, self.scale)
        pixels = torch.from_numpy(get_ji_pixels_from_meshgrid(x, y, self.scale).copy()).requires_grad_(False)
        return pixels

    def _yield_render(self, model, position):
        device = next(model.parameters()).device
        pixels = self._generate_image_frame()
        for i in range(pixels.shape[0] // self.batch_size):
            data_batch = pixels[i * self.batch_size:i * self.batch_size + self.batch_size].to(device)
            yield model(data_batch, position).to_numpy_cpu()

    @staticmethod
    def _check_mode(mode):
        assert mode == 'fine' or mode == 'coarse'

    def render_output(self, model, position):
        output_coarse_color = []
        output_coarse_depth = []
        output_fine_color = []
        output_fine_depth = []
        for predict in tqdm(self._yield_render(model, position)):
            output_coarse_color.append(predict.coarse_color)
            output_coarse_depth.append(predict.coarse_depths)
            output_fine_color.append(predict.fine_color)
            output_fine_depth.append(predict.fine_depths)

        output = [self._process_image_output(output_coarse_color),
                  self._process_depth_output(output_coarse_depth),
                  self._process_image_output(output_fine_color),
                  self._process_depth_output(output_fine_depth)]

        del output_coarse_color, output_coarse_depth, output_fine_color, output_fine_depth
        torch.cuda.empty_cache()
        return output

    def _process_image_output(self, image_output):
        return self.reshape_list_to_image(image_output)

    def _process_depth_output(self, depth_output):
        return self.reshape_list_to_depth_image(depth_output)

    def reconstruct_rendered_image(self, rendered_image):
        """
        Reconstruct rendered numpy image
        :param rendered_image:
        :return:
        """
        reconstructed_image = rendered_image * self.std[None, None] + self.mean[None, None]
        return np.clip(reconstructed_image / 255., 0, 1).astype(np.float32)

    def reshape_list_to_image(self, colors_list):
        return np.concatenate(colors_list, axis=0).reshape(self.height // self.scale,
                                                           self.width // self.scale,
                                                           3)

    def reshape_list_to_depth_image(self, depths_list):
        return np.concatenate(depths_list, axis=0).reshape(self.height // self.scale,
                                                           self.width // self.scale)
