import numpy as np
import torch
import pytorch3d.transforms as transforms

from .frame import Frame


class State:
    def __init__(self,
                 color_image,
                 depth_image,
                 pixel_weights,
                 position,
                 ground_truth_position=None):
        """
        :param color_image:
        :param depth_image:
        :param pixel_weights:
        :param position:    [[R 0],
                            [T 1]]
        :param ground_truth_position:   [[R 0],
                                        [T 1]]
        """
        self.set_frame(color_image, depth_image, pixel_weights)
        self.set_ground_true_position(ground_truth_position)
        self.set_position(position)

    @staticmethod
    def check_position(position):
        assert position.shape == (4, 4)
        assert np.all(position[:3, 3] == 0)

    def train_position(self):
        self._position.requires_grad_(requires_grad=True)

    def freeze_position(self):
        self._position.requires_grad_(requires_grad=False)

    def get_matrix_position(self):
        return transforms.se3_exp_map(self._position[None])[0]

    def set_position(self, position):
        # self.position = position.T TODO: solve .T compatibility with legacy code
        State.check_position(position)
        with torch.no_grad():
            position_6d = transforms.se3_log_map(torch.tensor(position, device='cpu', dtype=torch.float32)[None])[0]
        self.set_position_6d(position_6d)

    def set_position_6d(self, position_6d):
        self._position = position_6d.clone().detach()
        self.freeze_position()

    def set_ground_true_position(self, gt_position):
        if gt_position is not None:
            State.check_position(gt_position)
            State.check_position(gt_position)
        self._ground_truth_position = gt_position

    def clear_memory(self):
        del self.frame
        self._position.cpu()
        self.freeze_position()

    def set_frame(self, color_image, depth_image, pixel_weights):
        self.frame = Frame(color_image, depth_image, pixel_weights)
