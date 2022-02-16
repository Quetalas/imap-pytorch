import torch
import torch.nn.functional


def repeat_tensor(tensor, bins_count):
    result = torch.repeat_interleave(tensor[None], bins_count, dim=0)
    result = result.reshape(-1, *tensor.shape[1:])
    return result


def back_project_pixel(pixel, depth, camera_position, inverted_camera_matrix):
    """
    :param pixel: (batch_size, 2)
    :param depth: (ray_depths, batch_size)
    :param camera_position: [[R T],
                             [0 1]]
    :param inverted_camera_matrix: (3, 3)
    :return: (batch_size, 3)
    """
    batch_size = pixel.shape[0]
    inverted_camera_matrix = inverted_camera_matrix.to(pixel.device)
    homogeneous_pixel = torch.cat([pixel, torch.ones((batch_size, 1), device=pixel.device)], dim=1)
    homogeneous_keypoints = torch.matmul(inverted_camera_matrix, homogeneous_pixel[:, :, None])
    homogeneous_keypoints = repeat_tensor(homogeneous_keypoints, depth.shape[0])
    local_keypoints = (torch.nn.functional.normalize(homogeneous_keypoints, dim=1)) * depth.reshape(-1)[:, None, None]
    #local_keypoints = homogeneous_keypoints * depth.reshape(-1)[:, None, None]
    result = (camera_position[:3, :3] @ local_keypoints + camera_position[:3, 3:4])[:, :, 0]
    return result


def matrix_from_9d_position(position):
    matrix = torch.zeros(position.shape[0], 3, 4, device=position.device)
    matrix[:, :3, 3] = position[:, :3]
    matrix[:, :3, :3] = rotation_matrix_from_6d_parametrization(position[:, 3:9])
    return matrix


def rotation_matrix_from_6d_parametrization(parametrization):
    x = torch.nn.functional.normalize(parametrization[:, :3], dim=1)
    z = torch.nn.functional.normalize(torch.cross(parametrization[:, :3], parametrization[:, 3:], dim=1), dim=1)
    y = torch.cross(z, x, dim=1)
    matrix = torch.zeros(parametrization.shape[0], 3, 3, device=parametrization.device)
    matrix[:, 0, :3] = x
    matrix[:, 1, :3] = y
    matrix[:, 2, :3] = z
    return matrix


def position_9d_from_matrix(matrix):
    position_9d = torch.zeros((matrix.shape[0], 9), device=matrix.device)
    position_9d[:, :3] = matrix[:, :3, 3]
    position_9d[:, 3:6] = matrix[:, 0, :3]
    position_9d[:, 6:9] = matrix[:, 1, :3]
    return position_9d
