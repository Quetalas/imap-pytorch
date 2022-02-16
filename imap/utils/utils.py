import numpy as np
import torch


def generate_image_meshgrid(width, height, scale=1):
    """
    :param height:
    :param width:
    :param scale:
    :return:    x.shape = (height // scale, width // scale),
                y.shape = (height // scale, width // scale)
    """
    x, y = np.meshgrid(range(width // scale), range(height // scale))
    return x, y


def get_ij_pixels_from_meshgrid(x_meshgrid, y_meshgrid, scale=1):
    """
    Return pixel coordinates in (i, j) format (i.e. (row, column) format)
    :param x_meshgrid:
    :param y_meshgrid:
    :param scale
    :return:
        array([[  0,   0],
               [  0,   1],
               [  0,   2],
               ...,
               [479, 637],
               [479, 638],
               [479, 639]])
    """
    return np.concatenate((y_meshgrid[:, :, None], x_meshgrid[:, :, None]), axis=-1).reshape(-1, 2) * scale


def get_ji_pixels_from_meshgrid(x_meshgrid, y_meshgrid, scale=1):
    """
    Return pixel coordinates in (j, i) format (i.e. (columns, row) format)
    :param x_meshgrid:
    :param y_meshgrid:
    :param scale:
    :return:
    """
    return np.flip(get_ij_pixels_from_meshgrid(x_meshgrid, y_meshgrid, scale), axis=1)


def get_image_values_in_pixels(image, pixels):
    return image[pixels[:, 1], pixels[:, 0]]


def get_depths_in_pixels(depth_image, pixels):
    return get_image_values_in_pixels(depth_image, pixels)


def get_colors_in_pixels(color_image, pixels):
    return get_image_values_in_pixels(color_image, pixels)


def reshape_points_list_to_image(points_list, height, width, scale):
    return torch.cat(points_list, dim=0).reshape(height // scale,
                                                 width // scale,
                                                 3)


def reshape_points_list_to_depth(points_list, height, width, scale):
    return torch.cat(points_list, dim=0).reshape(height // scale,
                                                 width // scale)
